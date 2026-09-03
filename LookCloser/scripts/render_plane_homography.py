#!/usr/bin/env python3
"""Render one eval view from calibrated plane-induced train-image homographies.

Prediction construction does not load a mesh, depth, eval RGB, or an image
mask. The default plane passes through the normalized scene focus (world
origin) and is fronto-parallel to the target camera. Optional TSDF depth is
accepted only after rendering to exclude the unimportant room from metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.utils.data_utils import write_exr_image
from render_mesh_image_blend import (
    camera_parameters,
    grid_sample,
    load_depth,
    load_rgb,
    load_single_roi,
    masked_display_metrics,
    resolve_manifest_path,
    sha256,
    source_selection_fractions,
    write_prediction,
    write_source_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("mix", "nearest-fill"),
        default=("mix", "nearest-fill"),
    )
    parser.add_argument(
        "--plane-point",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Point on the world-space plane; the normalized scene focus is the dataset-agnostic default.",
    )
    parser.add_argument(
        "--plane-normal",
        type=float,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="World-space plane normal. By default the plane is fronto-parallel to the target camera.",
    )
    parser.add_argument(
        "--metric-surface-depth-manifest",
        type=Path,
        default=None,
        help="Optional mesh-depth manifest used after prediction only to score actor pixels.",
    )
    parser.add_argument("--roi-boxes-json", type=Path, nargs="*", default=())
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--center-method", default="focus")
    parser.add_argument("--auto-scale-poses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.metric_surface_depth_manifest is not None:
        args.metric_surface_depth_manifest = args.metric_surface_depth_manifest.expanduser().resolve()
    args.roi_boxes_json = tuple(path.expanduser().resolve() for path in args.roi_boxes_json)
    if not args.data.is_dir():
        parser.error("--data must exist")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.neighbors <= 0:
        parser.error("--neighbors must be positive")
    if args.metric_surface_depth_manifest is None and args.roi_boxes_json:
        parser.error("actor-only ROI metrics require --metric-surface-depth-manifest")
    if args.metric_surface_depth_manifest is not None and not args.metric_surface_depth_manifest.is_file():
        parser.error("--metric-surface-depth-manifest must exist")
    if any(not path.is_file() for path in args.roi_boxes_json):
        parser.error("all --roi-boxes-json files must exist")
    return args


def target_plane_world_points(
    *,
    height: int,
    width: int,
    target_c2w: torch.Tensor,
    target_intrinsics: dict[str, float],
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect target camera rays with one world-space plane."""

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=target_c2w.device),
        torch.arange(width, dtype=torch.float32, device=target_c2w.device),
        indexing="ij",
    )
    directions_gl = torch.stack(
        (
            (xx - target_intrinsics["cx"]) / target_intrinsics["fx"],
            -(yy - target_intrinsics["cy"]) / target_intrinsics["fy"],
            -torch.ones_like(xx),
        ),
        dim=-1,
    )
    directions_world = directions_gl @ target_c2w[:, :3].T
    origin = target_c2w[:, 3]
    denominator = (directions_world * plane_normal).sum(dim=-1)
    numerator = ((plane_point - origin) * plane_normal).sum()
    safe_denominator = torch.where(
        denominator.abs() > 1e-8,
        denominator,
        torch.ones_like(denominator),
    )
    distance = numerator / safe_denominator
    valid = (denominator.abs() > 1e-8) & (distance > 0.0)
    world = origin + distance.unsqueeze(-1) * directions_world
    return world, valid, distance


def project_world_to_camera(
    world: torch.Tensor,
    source_c2w: torch.Tensor,
    source_intrinsics: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points into a Nerfstudio OpenGL camera."""

    source_gl = (world - source_c2w[:, 3]) @ source_c2w[:, :3]
    depth = -source_gl[..., 2]
    safe_depth = depth.clamp_min(1e-8)
    u = source_intrinsics["fx"] * source_gl[..., 0] / safe_depth + source_intrinsics["cx"]
    v = source_intrinsics["fy"] * (-source_gl[..., 1]) / safe_depth + source_intrinsics["cy"]
    return u, v, depth


def aggregate_homographies(
    warped: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Either average all valid plane warps or fill holes in distance order."""

    if not warped or len(warped) != len(valid_masks):
        raise ValueError("warped and valid-mask lists must be non-empty and aligned")
    rgb_stack = torch.stack(warped)
    valid_stack = torch.stack(valid_masks)
    valid = valid_stack.any(dim=0)
    if mode == "mix":
        weights = valid_stack.float()
        rgb = (rgb_stack * weights.unsqueeze(1)).sum(dim=0) / weights.sum(dim=0).unsqueeze(0).clamp_min(1.0)
        selection = torch.argmax(weights, dim=0)
    elif mode == "nearest-fill":
        selection = torch.argmax(valid_stack.to(dtype=torch.int64), dim=0)
        gather_index = selection.unsqueeze(0).unsqueeze(0).expand(1, rgb_stack.shape[1], *selection.shape)
        rgb = torch.gather(rgb_stack, dim=0, index=gather_index)[0]
    else:
        raise ValueError(f"Unknown homography mode: {mode}")
    selection = torch.where(valid, selection, torch.full_like(selection, -1))
    return rgb, valid, selection


def metric_surface_mask(
    manifest_path: Path,
    data: Path,
    target_image: Path,
    device: torch.device,
) -> torch.Tensor:
    """Load a target mesh support mask after prediction construction."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    depth_by_image = {
        resolve_manifest_path(row["image"], data, manifest_path): resolve_manifest_path(
            row["depth"], data, manifest_path
        )
        for row in manifest["images"]
    }
    if target_image not in depth_by_image:
        raise KeyError(f"Metric manifest has no target depth for {target_image}")
    return torch.from_numpy(load_depth(depth_by_image[target_image])).to(device) > 0.0


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    parser_config = NerfstudioDataParserConfig(
        data=args.data,
        eval_mode=args.eval_mode,
        eval_interval=args.eval_interval,
        orientation_method=args.orientation_method,
        center_method=args.center_method,
        auto_scale_poses=args.auto_scale_poses,
        scale_factor=args.scale_factor,
        scene_scale=args.scene_scale,
        downscale_factor=args.downscale_factor,
        load_3D_points=False,
    )
    train = parser_config.setup().get_dataparser_outputs(split="train")
    target = parser_config.setup().get_dataparser_outputs(split="val")
    if train.mask_filenames is not None or target.mask_filenames is not None:
        raise ValueError("Plane homography renderer forbids image/person masks")
    if len(target.image_filenames) != 1:
        raise ValueError(f"Expected exactly one eval image, got {len(target.image_filenames)}")
    if args.neighbors > len(train.image_filenames):
        raise ValueError("Requested more neighbors than train cameras")

    target_image = Path(target.image_filenames[0]).resolve()
    target_c2w, target_intrinsics = camera_parameters(target.cameras, 0, device)
    train_cameras = [camera_parameters(train.cameras, index, device) for index in range(len(train.image_filenames))]
    centers = torch.stack([camera[0][:, 3] for camera in train_cameras])
    distances = torch.linalg.vector_norm(centers - target_c2w[:, 3], dim=-1)
    selected = torch.argsort(distances).tolist()[: args.neighbors]
    plane_point = torch.tensor(args.plane_point, dtype=torch.float32, device=device)
    plane_normal = (
        target_c2w[:, 2]
        if args.plane_normal is None
        else F.normalize(torch.tensor(args.plane_normal, dtype=torch.float32, device=device), dim=0)
    )
    height = int(target_intrinsics["height"])
    width = int(target_intrinsics["width"])
    world, target_plane_valid, ray_distance = target_plane_world_points(
        height=height,
        width=width,
        target_c2w=target_c2w,
        target_intrinsics=target_intrinsics,
        plane_point=plane_point,
        plane_normal=plane_normal,
    )

    warped: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    source_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for rank, source_index in enumerate(selected):
            source_image = Path(train.image_filenames[source_index]).resolve()
            source_c2w, source_intrinsics = train_cameras[source_index]
            u, v, source_depth = project_world_to_camera(world, source_c2w, source_intrinsics)
            valid = (
                target_plane_valid
                & (source_depth > 0.0)
                & (u >= 0.0)
                & (u <= source_intrinsics["width"] - 1.0)
                & (v >= 0.0)
                & (v <= source_intrinsics["height"] - 1.0)
            )
            warped.append(grid_sample(load_rgb(source_image, device), u, v))
            valid_masks.append(valid)
            source_rows.append(
                {
                    "rank": rank,
                    "train_index": source_index,
                    "source_image": str(source_image),
                    "camera_distance": float(distances[source_index].item()),
                    "valid_target_fraction": float(valid.float().mean().item()),
                }
            )

        args.output_dir.mkdir(parents=True)
        variants: list[dict[str, object]] = []
        variant_masks: dict[str, torch.Tensor] = {}
        for mode in dict.fromkeys(args.modes):
            prediction, valid, selection = aggregate_homographies(warped, valid_masks, mode=mode)
            prediction = torch.where(valid.unsqueeze(0), prediction, 0.0)
            name = f"homography_{mode.replace('-', '_')}{args.neighbors}"
            write_prediction(args.output_dir, name, prediction, None)
            variant_masks[name] = valid
            row: dict[str, object] = {
                "name": name,
                "mode": mode,
                "neighbors": args.neighbors,
                "valid_fraction": float(valid.float().mean().item()),
            }
            if mode == "nearest-fill":
                write_source_selection(args.output_dir, name, selection, args.neighbors)
                row["source_rank_fraction_on_valid"] = source_selection_fractions(selection, args.neighbors)
            else:
                valid_count = max(int(valid.sum().item()), 1)
                row["source_valid_fraction_on_valid"] = [
                    float((source_valid & valid).sum().item() / valid_count)
                    for source_valid in valid_masks
                ]
            variants.append(row)

    # Prediction construction is complete before either eval RGB or TSDF metric
    # support is loaded.
    ground_truth = args.output_dir / "eval_gt_0000.exr"
    target_rgb = load_rgb(target_image, device)
    write_exr_image(ground_truth, target_rgb.permute(1, 2, 0).detach().float().cpu())
    surface_mask = None
    metrics: dict[str, object] = {}
    if args.metric_surface_depth_manifest is not None:
        surface_mask = metric_surface_mask(
            args.metric_surface_depth_manifest,
            args.data,
            target_image,
            device,
        )
        Image.fromarray(surface_mask.byte().mul(255).cpu().numpy(), "L").save(
            args.output_dir / "metric_surface_mask.png", compress_level=3
        )
    if args.roi_boxes_json:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        with torch.inference_mode():
            for roi_path in args.roi_boxes_json:
                roi, roi_name = load_single_roi(roi_path)
                x0, y0, x1, y1 = roi
                roi_mask = torch.zeros_like(surface_mask, dtype=torch.bool)
                roi_mask[y0:y1, x0:x1] = True
                region_mask = surface_mask & roi_mask
                by_variant: dict[str, object] = {}
                for row in variants:
                    name = str(row["name"])
                    prediction = load_rgb(args.output_dir / name / "eval_pred_0000.exr", device)
                    by_variant[name] = {
                        "surface_roi": masked_display_metrics(
                            prediction,
                            target_rgb,
                            mask=region_mask,
                            lpips_model=lpips_model,
                        ),
                        "reprojected_surface_roi": masked_display_metrics(
                            prediction,
                            target_rgb,
                            mask=region_mask & variant_masks[name],
                            lpips_model=lpips_model,
                        ),
                    }
                metrics[roi_name] = {
                    "roi_bbox_xyxy": list(roi),
                    "roi_source": str(roi_path),
                    "variants": by_variant,
                }

    target_origin_gl = (plane_point - target_c2w[:, 3]) @ target_c2w[:, :3]
    audit = {
        "schema_version": 1,
        "method": "calibrated_plane_induced_homography",
        "data": str(args.data),
        "target_image": str(target_image),
        "uses_tsdf_for_prediction": False,
        "uses_eval_rgb_for_prediction": False,
        "uses_masks_for_prediction": False,
        "eval_rgb_use": "metrics_only",
        "metric_surface_depth_manifest": (
            None if args.metric_surface_depth_manifest is None else str(args.metric_surface_depth_manifest)
        ),
        "metric_surface_depth_manifest_sha256": (
            None if args.metric_surface_depth_manifest is None else sha256(args.metric_surface_depth_manifest)
        ),
        "tsdf_use": "metrics_only" if args.metric_surface_depth_manifest is not None else "none",
        "plane_point_world": plane_point.tolist(),
        "plane_normal_world": plane_normal.tolist(),
        "plane_depth_at_target_principal_ray": float(-target_origin_gl[2].item()),
        "target_plane_valid_fraction": float(target_plane_valid.float().mean().item()),
        "target_plane_ray_distance_median": float(torch.median(ray_distance[target_plane_valid]).item()),
        "sources": source_rows,
        "variants": variants,
        "metrics": metrics,
    }
    (args.output_dir / "homography_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
