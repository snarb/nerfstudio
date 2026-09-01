#!/usr/bin/env python3
"""Render an eval camera by calibrated train-image reprojection through one mesh.

This is a geometry/appearance causal gate, not a learned image refiner.  It
never reads eval RGB while constructing predictions, uses no image/person
mask, and admits pixels only when the same continuous mesh is visible in both
the target and source cameras.  A sharp result proves that mesh geometry can
support sharp image-based rendering; a displaced result localizes the failure
to geometry/calibration rather than a NeRF RGB head.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.utils.data_utils import load_exr_image, write_exr_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mesh-depth-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-prediction-exr", type=Path, default=None)
    parser.add_argument("--ground-truth-exr", type=Path, default=None)
    parser.add_argument("--neighbors", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument(
        "--blend-alphas",
        type=float,
        nargs="+",
        default=(1.0,),
        help="Geometric source-colour blend strengths; one fully replaces supported base pixels.",
    )
    parser.add_argument("--depth-log-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--camera-distance-power",
        type=float,
        default=2.0,
        help=(
            "Exponent p in the source-camera weight 1 / distance**p. "
            "Larger values keep the nearest source sharper while allowing farther sources to fill visibility holes."
        ),
    )
    parser.add_argument(
        "--detail-transfer-sigmas",
        type=float,
        nargs="*",
        default=(),
        help=(
            "Optional Gaussian sigmas for replacing only the base high-frequency band with the "
            "geometry-aligned source band while retaining base low-frequency colour."
        ),
    )
    parser.add_argument(
        "--detail-transfer-strengths",
        type=float,
        nargs="+",
        default=(1.0,),
    )
    parser.add_argument(
        "--score-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score display-domain PSNR/SSIM/LPIPS after every prediction has been constructed.",
    )
    parser.add_argument(
        "--roi-boxes-json",
        type=Path,
        default=None,
        help="Optional diagnostic JSON containing exactly one boxes_xyxy entry.",
    )
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
    args.mesh_depth_manifest = args.mesh_depth_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.base_prediction_exr is not None:
        args.base_prediction_exr = args.base_prediction_exr.expanduser().resolve()
    if args.ground_truth_exr is not None:
        args.ground_truth_exr = args.ground_truth_exr.expanduser().resolve()
    if args.roi_boxes_json is not None:
        args.roi_boxes_json = args.roi_boxes_json.expanduser().resolve()
    if not args.data.is_dir() or not args.mesh_depth_manifest.is_file():
        parser.error("--data and --mesh-depth-manifest must exist")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if not args.neighbors or min(args.neighbors) <= 0:
        parser.error("--neighbors must contain positive counts")
    if not args.blend_alphas or any(not 0.0 <= value <= 1.0 for value in args.blend_alphas):
        parser.error("--blend-alphas must contain values in [0, 1]")
    if args.depth_log_tolerance <= 0.0:
        parser.error("--depth-log-tolerance must be positive")
    if args.camera_distance_power < 0.0:
        parser.error("--camera-distance-power must be non-negative")
    if any(value <= 0.0 for value in args.detail_transfer_sigmas):
        parser.error("--detail-transfer-sigmas must be positive")
    if any(value < 0.0 for value in args.detail_transfer_strengths):
        parser.error("--detail-transfer-strengths must be non-negative")
    if args.roi_boxes_json is not None and not args.roi_boxes_json.is_file():
        parser.error("--roi-boxes-json must exist")
    if args.roi_boxes_json is not None and not args.score_metrics:
        parser.error("--roi-boxes-json requires --score-metrics")
    return args


def load_depth(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        depth = np.load(stream, allow_pickle=False)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError(f"Invalid non-negative depth: {path}")
    return depth.astype(np.float32, copy=False)


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix.lower() == ".exr":
        array = load_exr_image(path)[..., :3].astype(np.float32, copy=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).to(device)


def resolve_manifest_path(value: str, data: Path, manifest_path: Path) -> Path:
    """Resolve portable dataset-relative manifests without depending on cwd."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (data / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Prefer the dataset contract in the error that the eventual loader emits.
    return candidates[0].resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_parameters(cameras, index: int, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    c2w = cameras.camera_to_worlds[index].to(device=device, dtype=torch.float32)
    intrinsics = {
        "fx": float(cameras.fx[index].item()),
        "fy": float(cameras.fy[index].item()),
        "cx": float(cameras.cx[index].item()),
        "cy": float(cameras.cy[index].item()),
        "width": int(cameras.width[index].item()),
        "height": int(cameras.height[index].item()),
    }
    return c2w, intrinsics


def project_target_to_source(
    target_depth: torch.Tensor,
    target_c2w: torch.Tensor,
    target_intrinsics: dict[str, float],
    source_c2w: torch.Tensor,
    source_intrinsics: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project positive OpenCV camera-z depth between Nerfstudio cameras."""

    height, width = target_depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=target_depth.device),
        torch.arange(width, dtype=torch.float32, device=target_depth.device),
        indexing="ij",
    )
    z = target_depth
    x = (xx - target_intrinsics["cx"]) * z / target_intrinsics["fx"]
    y = (yy - target_intrinsics["cy"]) * z / target_intrinsics["fy"]
    target_gl = torch.stack((x, -y, -z), dim=-1)
    world = target_gl @ target_c2w[:, :3].T + target_c2w[:, 3]
    source_gl = (world - source_c2w[:, 3]) @ source_c2w[:, :3]
    source_z = -source_gl[..., 2]
    safe_z = source_z.clamp_min(1e-8)
    source_u = source_intrinsics["fx"] * source_gl[..., 0] / safe_z + source_intrinsics["cx"]
    source_v = source_intrinsics["fy"] * (-source_gl[..., 1]) / safe_z + source_intrinsics["cy"]
    return source_u, source_v, source_z


def grid_sample(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    grid = torch.stack(
        (2.0 * u / max(width - 1, 1) - 1.0, 2.0 * v / max(height - 1, 1) - 1.0),
        dim=-1,
    ).unsqueeze(0)
    return F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]


def normalized_gaussian_blur(image: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur CHW data inside a support mask without leaking zero across its boundary."""

    if image.ndim != 3 or valid.shape != (1, *image.shape[-2:]):
        raise ValueError("image must be CHW and valid must be 1HW")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    channels = image.shape[0]
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)

    def separable(value: torch.Tensor, groups: int) -> torch.Tensor:
        result = F.conv2d(
            value.unsqueeze(0),
            horizontal.expand(groups, 1, 1, -1),
            padding=(0, radius),
            groups=groups,
        )
        result = F.conv2d(
            result,
            vertical.expand(groups, 1, -1, 1),
            padding=(radius, 0),
            groups=groups,
        )
        return result[0]

    support = valid.to(dtype=image.dtype)
    numerator = separable(image * support, channels)
    denominator = separable(support, 1)
    return numerator / denominator.clamp_min(1e-6)


def surface_detail_transfer(
    *,
    base: torch.Tensor,
    source: torch.Tensor,
    valid: torch.Tensor,
    sigma: float,
    strength: float,
) -> torch.Tensor:
    """Replace the supported base high-pass band with aligned source detail."""

    base_low = normalized_gaussian_blur(base, valid, sigma)
    source_low = normalized_gaussian_blur(source, valid, sigma)
    replacement = base + strength * ((source - source_low) - (base - base_low))
    return torch.where(valid, replacement, base)


def load_single_roi(path: Path) -> tuple[tuple[int, int, int, int], str]:
    """Load the one eval ROI accepted by this single-target renderer."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_boxes = payload.get("boxes_xyxy")
    if raw_boxes is None and isinstance(payload.get("boxes"), list):
        raw_boxes = [row.get("xyxy") if isinstance(row, dict) else None for row in payload["boxes"]]
    if not isinstance(raw_boxes, list) or len(raw_boxes) != 1:
        raise ValueError(f"Expected exactly one ROI box in {path}")
    raw = raw_boxes[0]
    if not isinstance(raw, list) or len(raw) != 4 or not all(isinstance(value, int) for value in raw):
        raise ValueError(f"ROI must contain four integer xyxy coordinates: {raw!r}")
    x0, y0, x1, y1 = raw
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI box: {raw!r}")
    return (x0, y0, x1, y1), str(payload.get("name", "roi"))


def display_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    lpips_model,
    roi: tuple[int, int, int, int] | None = None,
) -> dict[str, float]:
    """Measure display-domain metrics without affecting prediction construction."""

    from torchmetrics.functional.image import structural_similarity_index_measure

    if prediction.shape != ground_truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and ground_truth must be matching CHW tensors")
    pred = prediction.float().clamp(0.0, 1.0).unsqueeze(0)
    gt = ground_truth.float().clamp(0.0, 1.0).unsqueeze(0)

    def measure(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, float]:
        mse = torch.mean((left - right).square())
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        ssim = structural_similarity_index_measure(left, right, data_range=1.0)
        lpips = lpips_model(left, right)
        return float(psnr.item()), float(ssim.item()), float(lpips.item())

    psnr, ssim, lpips = measure(pred, gt)
    result = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    if roi is not None:
        x0, y0, x1, y1 = roi
        height, width = pred.shape[-2:]
        if x1 > width or y1 > height:
            raise ValueError(f"ROI {roi} exceeds image width/height {(width, height)}")
        roi_psnr, roi_ssim, roi_lpips = measure(pred[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1])
        result.update({"roi_psnr": roi_psnr, "roi_ssim": roi_ssim, "roi_lpips": roi_lpips})
    return result


def score_written_variants(
    output_dir: Path,
    variants: list[dict[str, object]],
    ground_truth: Path,
    *,
    device: torch.device,
    roi_boxes_json: Path | None,
) -> dict[str, object]:
    """Score renderer outputs lazily so the optional path cannot affect legacy runs."""

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    roi = None
    roi_name = None
    if roi_boxes_json is not None:
        roi, roi_name = load_single_roi(roi_boxes_json)
    gt = load_rgb(ground_truth, device)
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    by_variant: dict[str, object] = {}
    with torch.inference_mode():
        for row in variants:
            name = str(row["name"])
            pred = load_rgb(output_dir / name / "eval_pred_0000.exr", device)
            aggregate = display_metrics(pred, gt, lpips_model=lpips_model, roi=roi)
            receipt = {
                "schema_version": 1,
                "metric_domain": "display-referred RGB clamped to [0, 1]",
                "roi_name": roi_name,
                "roi_bbox_xyxy": None if roi is None else list(roi),
                "aggregate": aggregate,
            }
            (output_dir / name / "metrics.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            row["metrics"] = aggregate
            by_variant[name] = receipt
    (output_dir / "metrics.json").write_text(
        json.dumps({"schema_version": 1, "variants": by_variant}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return by_variant


def write_prediction(directory: Path, name: str, rgb: torch.Tensor, ground_truth: Path | None) -> None:
    variant = directory / name
    variant.mkdir(parents=True, exist_ok=False)
    array = rgb.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu()
    write_exr_image(variant / "eval_pred_0000.exr", array)
    Image.fromarray(array.mul(255.0).add(0.5).byte().numpy(), "RGB").save(
        variant / "eval_pred_0000.png", compress_level=3
    )
    if ground_truth is not None:
        (variant / "eval_gt_0000.exr").symlink_to(ground_truth)


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    manifest = json.loads(args.mesh_depth_manifest.read_text(encoding="utf-8"))
    depth_by_image = {
        resolve_manifest_path(row["image"], args.data, args.mesh_depth_manifest): resolve_manifest_path(
            row["depth"], args.data, args.mesh_depth_manifest
        )
        for row in manifest["images"]
    }
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
        raise ValueError("Mesh reprojection diagnostic forbids image/person masks")
    if len(target.image_filenames) != 1:
        raise ValueError(f"Expected exactly one eval image, got {len(target.image_filenames)}")
    if max(args.neighbors) > len(train.image_filenames):
        raise ValueError("Requested more neighbors than train cameras")

    target_image = Path(target.image_filenames[0]).resolve()
    target_depth = torch.from_numpy(load_depth(depth_by_image[target_image])).to(device)
    target_depth = target_depth * float(target.dataparser_scale)
    target_c2w, target_intrinsics = camera_parameters(target.cameras, 0, device)
    train_cameras = [camera_parameters(train.cameras, index, device) for index in range(len(train.image_filenames))]
    centers = torch.stack([camera[0][:, 3] for camera in train_cameras])
    distances = torch.linalg.vector_norm(centers - target_c2w[:, 3], dim=-1)
    order = torch.argsort(distances).tolist()
    max_neighbors = max(args.neighbors)
    selected = order[:max_neighbors]

    warped: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    nearest_source_rgb: torch.Tensor | None = None
    source_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for rank, source_index in enumerate(selected):
            source_image = Path(train.image_filenames[source_index]).resolve()
            source_rgb = load_rgb(source_image, device)
            if rank == 0:
                nearest_source_rgb = source_rgb
            source_depth = torch.from_numpy(load_depth(depth_by_image[source_image])).to(device)
            source_depth = source_depth * float(train.dataparser_scale)
            source_c2w, source_intrinsics = train_cameras[source_index]
            u, v, projected_z = project_target_to_source(
                target_depth, target_c2w, target_intrinsics, source_c2w, source_intrinsics
            )
            sampled_rgb = grid_sample(source_rgb, u, v)
            sampled_depth = grid_sample(source_depth.unsqueeze(0), u, v)[0]
            in_bounds = (
                (target_depth > 0.0)
                & (projected_z > 0.0)
                & (u >= 0.0)
                & (u <= source_intrinsics["width"] - 1.0)
                & (v >= 0.0)
                & (v <= source_intrinsics["height"] - 1.0)
                & (sampled_depth > 0.0)
            )
            log_error = torch.abs(
                torch.log(projected_z.clamp_min(1e-6)) - torch.log(sampled_depth.clamp_min(1e-6))
            )
            valid = in_bounds & (log_error <= args.depth_log_tolerance)
            depth_weight = torch.exp(-0.5 * (log_error / args.depth_log_tolerance).square())
            pose_weight = 1.0 / max(
                float(distances[source_index].item()) ** args.camera_distance_power,
                1e-6,
            )
            weight = valid.float().unsqueeze(0) * depth_weight.unsqueeze(0) * pose_weight
            warped.append(sampled_rgb)
            weights.append(weight)
            source_rows.append(
                {
                    "rank": rank,
                    "train_index": source_index,
                    "source_image": str(source_image),
                    "camera_distance": float(distances[source_index].item()),
                    "valid_target_fraction": float(valid.float().mean().item()),
                    "valid_mesh_fraction": float(valid[target_depth > 0].float().mean().item()),
                    "median_log_depth_error_in_bounds": (
                        float(torch.median(log_error[in_bounds]).item()) if bool(in_bounds.any()) else None
                    ),
                }
            )

        if args.base_prediction_exr is None:
            base = torch.zeros_like(warped[0])
        else:
            base = load_rgb(args.base_prediction_exr, device)
            if base.shape != warped[0].shape:
                # Nerfstudio's ``eval_img`` PNG is a horizontal GT|prediction
                # review image.  Taking only the right half is deterministic
                # and avoids ever using its embedded held-out GT as input.
                if base.shape[:2] == warped[0].shape[:2] and base.shape[2] == 2 * warped[0].shape[2]:
                    base = base[:, :, warped[0].shape[2] :]
                else:
                    raise ValueError(
                        f"Base prediction shape {tuple(base.shape)} does not match target {tuple(warped[0].shape)}"
                    )
        args.output_dir.mkdir(parents=True)
        ground_truth = args.ground_truth_exr
        if ground_truth is None:
            # Held-out RGB is loaded only after every prediction input has
            # already been constructed.  It is written solely for metrics.
            ground_truth = args.output_dir / "eval_gt_0000.exr"
            target_rgb = load_rgb(target_image, device)
            write_exr_image(ground_truth, target_rgb.permute(1, 2, 0).detach().float().cpu())
        variants: list[dict[str, object]] = []
        assert nearest_source_rgb is not None
        write_prediction(args.output_dir, "unwarped_nearest_control", nearest_source_rgb, ground_truth)
        variants.append(
            {
                "name": "unwarped_nearest_control",
                "neighbors": 1,
                "control_without_geometry_reprojection": True,
                "valid_fraction": 1.0,
            }
        )
        for count in sorted(set(args.neighbors)):
            stack_rgb = torch.stack(warped[:count])
            stack_weight = torch.stack(weights[:count])
            weight_sum = stack_weight.sum(dim=0)
            blend = (stack_rgb * stack_weight).sum(dim=0) / weight_sum.clamp_min(1e-8)
            valid = weight_sum > 0.0
            for alpha in sorted(set(args.blend_alphas)):
                prediction = torch.where(valid, base * (1.0 - alpha) + blend * alpha, base)
                alpha_token = f"{alpha:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                name = f"blend{count}" if alpha == 1.0 else f"blend{count}_a{alpha_token}"
                write_prediction(args.output_dir, name, prediction, ground_truth)
                variants.append(
                    {
                        "name": name,
                        "neighbors": count,
                        "blend_alpha": alpha,
                        "valid_fraction": float(valid[0].float().mean().item()),
                        "valid_mesh_fraction": float(valid[0][target_depth > 0.0].float().mean().item()),
                    }
                )
            for sigma in sorted(set(args.detail_transfer_sigmas)):
                for strength in sorted(set(args.detail_transfer_strengths)):
                    prediction = surface_detail_transfer(
                        base=base,
                        source=blend,
                        valid=valid,
                        sigma=sigma,
                        strength=strength,
                    )
                    sigma_token = f"{sigma:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                    strength_token = f"{strength:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                    name = f"blend{count}_detail_s{sigma_token}_w{strength_token}"
                    write_prediction(args.output_dir, name, prediction, ground_truth)
                    variants.append(
                        {
                            "name": name,
                            "neighbors": count,
                            "detail_transfer_sigma": sigma,
                            "detail_transfer_strength": strength,
                            "valid_fraction": float(valid[0].float().mean().item()),
                            "valid_mesh_fraction": float(valid[0][target_depth > 0.0].float().mean().item()),
                        }
                    )

    metrics = (
        score_written_variants(
            args.output_dir,
            variants,
            ground_truth,
            device=device,
            roi_boxes_json=args.roi_boxes_json,
        )
        if args.score_metrics
        else None
    )

    audit = {
        "schema_version": 1,
        "method": "continuous_mesh_calibrated_train_image_reprojection",
        "uses_eval_rgb_for_prediction": False,
        "eval_rgb_use": "metrics_only",
        "uses_masks": False,
        "base_prediction": (
            None
            if args.base_prediction_exr is None
            else {
                "path": str(args.base_prediction_exr),
                "sha256": sha256(args.base_prediction_exr),
                "input_policy": "prediction_only_or_right_half_of_nerfstudio_gt_prediction_review",
            }
        ),
        "ground_truth": {
            "path": str(target_image if args.ground_truth_exr is None else args.ground_truth_exr),
            "use": "metrics_only",
        },
        "data": str(args.data),
        "mesh_depth_manifest": str(args.mesh_depth_manifest),
        "mesh_depth_manifest_sha256": sha256(args.mesh_depth_manifest),
        "target_image": str(target_image),
        "depth_log_tolerance": args.depth_log_tolerance,
        "camera_distance_power": args.camera_distance_power,
        "detail_transfer_sigmas": list(args.detail_transfer_sigmas),
        "detail_transfer_strengths": list(args.detail_transfer_strengths),
        "dataparser_scale": float(target.dataparser_scale),
        "sources": source_rows,
        "variants": variants,
        "metrics": metrics,
    }
    (args.output_dir / "reprojection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
