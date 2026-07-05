#!/usr/bin/env python3
"""Map structural artifact pixels to LookCloser/nerfacc occupancy-grid voxels.

This is an offline diagnostic: it loads a trained run, detects the largest
artifact blob in an eval render, projects pixels from that blob through the eval
camera, and reports whether the corresponding surface-depth and ray-path voxels
are occupied in the nerfacc grid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from detect_structural_artifacts import SSIM_SEVERE, detect_defects, load_pair
from nerfstudio.utils.eval_utils import eval_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--load-config", type=Path, help="Eval config.yml or eval_config_step_*.yml.")
    source.add_argument("--run-dir", type=Path, help="Run directory containing config.yml.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit step-*.ckpt to load.")
    parser.add_argument("--render-file", type=Path, required=True, help="Side-by-side GT|render eval image.")
    parser.add_argument("--eval-index", type=int, default=0, help="Eval camera index matching render-file.")
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--panels", type=int, default=2)
    parser.add_argument("--gt-panel", type=int, default=0)
    parser.add_argument("--cand-panel", type=int, default=1)
    parser.add_argument("--bbox", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=2000)
    parser.add_argument("--min-accumulation", type=float, default=0.05)
    parser.add_argument("--ray-samples", type=int, default=512)
    parser.add_argument("--near-surface-margin", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def config_path(args: argparse.Namespace) -> Path:
    if args.load_config is not None:
        return args.load_config
    assert args.run_dir is not None
    return args.run_dir / "config.yml"


def checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"step-(\d+)\.ckpt$", checkpoint.name)
    if match is None:
        raise ValueError(f"Could not parse checkpoint step from {checkpoint}")
    return int(match.group(1))


def selected_checkpoint(args: argparse.Namespace) -> Optional[Path]:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_dir is None:
        return None
    summary_path = args.run_dir / "run_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("selected_checkpoint",):
            value = data.get(key)
            if value and Path(value).exists():
                return Path(value)
        eval_data = data.get("eval") or {}
        value = eval_data.get("checkpoint")
        if value and Path(value).exists():
            return Path(value)
    checkpoints = sorted((args.run_dir / "nerfstudio_models").glob("step-*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def select_artifact_pixels(args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object], np.ndarray, np.ndarray]:
    load_args = SimpleNamespace(
        gt_file=None,
        cand_file=None,
        image=str(args.render_file),
        panels=args.panels,
        gt=args.gt_panel,
        cand=args.cand_panel,
        crop_top=0,
        crop_bottom=0,
        crop_left=0,
        crop_right=0,
    )
    gt, cand = load_pair(load_args, args.gt_panel, args.cand_panel)
    res = detect_defects(gt, cand)
    if args.bbox is None:
        regions = list(res["major"]) + list(res["minor"])
        if not regions:
            raise RuntimeError("No qualifying artifact bbox found. Pass --bbox to force an analysis region.")
        region = max(regions, key=lambda item: item[0])
        _, x0, y0, x1, y1, mean_severity = region
    else:
        x0, y0, x1, y1 = args.bbox
        mean_severity = float("nan")
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(cand.shape[1] - 1, int(x1))
    y1 = min(cand.shape[0] - 1, int(y1))

    err = res["error_map"]
    severe = err > (1.0 - SSIM_SEVERE)
    yy, xx = np.where(severe[y0 : y1 + 1, x0 : x1 + 1])
    if yy.size:
        yy = yy + y0
        xx = xx + x0
    else:
        ys = np.arange(y0, y1 + 1, max(1, args.pixel_stride))
        xs = np.arange(x0, x1 + 1, max(1, args.pixel_stride))
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        yy, xx = grid_y.reshape(-1), grid_x.reshape(-1)

    if args.pixel_stride > 1 and yy.size:
        keep = ((yy - y0) % args.pixel_stride == 0) & ((xx - x0) % args.pixel_stride == 0)
        yy, xx = yy[keep], xx[keep]
    if yy.size > args.max_pixels:
        indices = np.linspace(0, yy.size - 1, args.max_pixels).round().astype(np.int64)
        yy, xx = yy[indices], xx[indices]

    pixels = np.stack([yy, xx], axis=-1).astype(np.int64)
    artifact_info: Dict[str, object] = {
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_width": x1 - x0 + 1,
        "bbox_height": y1 - y0 + 1,
        "artifact_score": res["artifact_score"],
        "artifact_count": res["artifact_count"],
        "largest_area": res["largest_area"],
        "selected_mean_severity": None if math.isnan(mean_severity) else mean_severity,
        "selected_pixels": int(pixels.shape[0]),
    }
    return pixels, artifact_info, gt, cand


def generate_ray_bundle(pipeline, eval_index: int, pixels_yx: np.ndarray):
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader
    camera, _ = dataloader.get_camera(eval_index)
    coords = torch.from_numpy(pixels_yx).float()
    return camera.generate_rays(camera_indices=0, coords=coords, keep_shape=False)


def collided_ray_bundle(model, ray_bundle):
    ray_bundle = ray_bundle.to(model.device)
    if model.collider is not None:
        return model.collider(ray_bundle)
    return ray_bundle


def occupancy_tensors(model):
    grid = model.occupancy_grid
    binaries = grid.binaries.detach()
    occs = grid.occs.detach().view_as(binaries)
    aabb = model.scene_aabb.detach().view(2, 3).to(binaries.device)
    return binaries, occs, aabb


def query_occupancy(model, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
    binaries, occs, aabb = occupancy_tensors(model)
    resolution = binaries.shape[-1]
    rel = (positions - aabb[0]) / (aabb[1] - aabb[0]).clamp(min=1e-12)
    inside = torch.logical_and(rel >= 0.0, rel < 1.0).all(dim=-1)
    idx = torch.floor(rel * resolution).long().clamp(0, resolution - 1)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
    occupied = binaries[0, x, y, z] & inside
    occ_values = occs[0, x, y, z]
    occ_values = torch.where(inside, occ_values, torch.zeros_like(occ_values))
    linear = x * resolution * resolution + y * resolution + z
    return {
        "inside": inside,
        "occupied": occupied,
        "occ_values": occ_values,
        "indices": idx,
        "linear": linear,
    }


def summarize_bool(values: torch.Tensor) -> Dict[str, object]:
    if values.numel() == 0:
        return {"count": 0, "rate": None}
    return {"count": int(values.sum().item()), "rate": float(values.float().mean().item())}


def summarize_occ_values(values: torch.Tensor) -> Dict[str, Optional[float]]:
    if values.numel() == 0:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def analyze_sample_counts(model, outputs) -> Dict[str, object]:
    counts = outputs.get("num_samples_per_ray")
    if counts is None:
        return {"available": False}
    counts = counts.reshape(-1).detach().to(model.device)
    if counts.numel() == 0:
        return {"available": True, "count": 0}
    max_steps = int(getattr(model.config, "max_steps_per_ray", 0))
    result: Dict[str, object] = {
        "available": True,
        "count": int(counts.numel()),
        "min": float(counts.min().item()),
        "mean": float(counts.float().mean().item()),
        "max": float(counts.max().item()),
        "zero_count": int((counts == 0).sum().item()),
        "zero_rate": float((counts == 0).float().mean().item()),
        "configured_max_steps_per_ray": max_steps,
    }
    if max_steps > 0:
        saturated = counts >= max_steps
        result["saturated_count"] = int(saturated.sum().item())
        result["saturated_rate"] = float(saturated.float().mean().item())
    quantiles = torch.quantile(counts.float(), torch.tensor([0.5, 0.9, 0.99], device=counts.device))
    result["p50"] = float(quantiles[0].item())
    result["p90"] = float(quantiles[1].item())
    result["p99"] = float(quantiles[2].item())
    return result


def analyze_surface(model, ray_bundle, outputs, min_accumulation: float) -> Dict[str, object]:
    depth = outputs["depth"].reshape(-1).to(model.device)
    accumulation = outputs.get("accumulation", torch.ones_like(depth[:, None])).reshape(-1).to(model.device)
    valid = torch.isfinite(depth) & (depth > 0) & (accumulation >= min_accumulation)
    if not valid.any():
        return {"valid_surface_pixels": 0}
    origins = ray_bundle.origins.reshape(-1, 3)[valid]
    directions = ray_bundle.directions.reshape(-1, 3)[valid]
    points = origins + directions * depth[valid, None]
    occ = query_occupancy(model, points)
    occupied_values = occ["occ_values"][occ["inside"]]
    return {
        "valid_surface_pixels": int(valid.sum().item()),
        "inside_grid": summarize_bool(occ["inside"]),
        "occupied_surface_voxels": summarize_bool(occ["occupied"]),
        "unique_surface_voxels": int(torch.unique(occ["linear"][occ["inside"]]).numel()),
        "occ_values_inside": summarize_occ_values(occupied_values),
        "accumulation_mean": float(accumulation[valid].mean().item()),
        "depth_mean": float(depth[valid].mean().item()),
    }


def analyze_rays(model, ray_bundle, outputs, ray_samples: int, near_surface_margin: float) -> Dict[str, object]:
    origins = ray_bundle.origins.reshape(-1, 3)
    directions = ray_bundle.directions.reshape(-1, 3)
    nears = ray_bundle.nears.reshape(-1).to(model.device)
    fars = ray_bundle.fars.reshape(-1).to(model.device)
    depth = outputs["depth"].reshape(-1).to(model.device)
    finite_depth = torch.isfinite(depth) & (depth > 0)
    if finite_depth.any():
        fars = torch.minimum(fars, torch.where(finite_depth, depth + near_surface_margin, fars))
    valid = torch.isfinite(nears) & torch.isfinite(fars) & (fars > nears)
    if not valid.any():
        return {"valid_rays": 0}
    origins = origins[valid]
    directions = directions[valid]
    nears = nears[valid]
    fars = fars[valid]
    alphas = torch.linspace(0.0, 1.0, ray_samples, device=model.device)
    ts = nears[:, None] + (fars - nears)[:, None] * alphas[None, :]
    points = origins[:, None, :] + directions[:, None, :] * ts[..., None]
    flat_points = points.reshape(-1, 3)
    occ = query_occupancy(model, flat_points)
    occupied = occ["occupied"].view(origins.shape[0], ray_samples)
    inside = occ["inside"].view(origins.shape[0], ray_samples)
    any_occupied = occupied.any(dim=1)
    no_occupied_inside = (~any_occupied) & inside.any(dim=1)
    occupied_values = occ["occ_values"][occ["inside"]]
    return {
        "valid_rays": int(valid.sum().item()),
        "ray_samples": ray_samples,
        "rays_with_any_occupied_voxel": summarize_bool(any_occupied),
        "rays_inside_grid_with_no_occupied_voxel": summarize_bool(no_occupied_inside),
        "inside_sample_rate": float(inside.float().mean().item()),
        "occupied_sample_rate": float(occupied.float().mean().item()),
        "unique_ray_voxels_inside": int(torch.unique(occ["linear"][occ["inside"]]).numel()),
        "occ_values_inside": summarize_occ_values(occupied_values),
    }


def occupancy_global_stats(model) -> Dict[str, object]:
    binaries, occs, _ = occupancy_tensors(model)
    occ_mean = float(occs.mean().item())
    occ_thre = float(getattr(model.config, "occupancy_occ_thre", 1e-2))
    alpha_thre = float(getattr(model.config, "alpha_thre", 0.0))
    level_dims = tuple(range(1, binaries.ndim))
    ratios = binaries.float().mean(dim=level_dims)
    return {
        "grid_resolution": int(binaries.shape[-1]),
        "grid_levels": int(binaries.shape[0]),
        "occupancy_ratio": float(binaries.float().mean().item()),
        "occupancy_ratio_per_level": [float(value) for value in ratios.cpu()],
        "occs_mean": occ_mean,
        "occs_max": float(occs.max().item()),
        "configured_occ_thre": occ_thre,
        "effective_binary_threshold": min(occ_mean, occ_thre),
        "configured_alpha_thre": alpha_thre,
        "effective_alpha_thre": min(alpha_thre, occ_mean),
    }


def classify(surface: Dict[str, object], rays: Dict[str, object]) -> Dict[str, object]:
    surface_rate = ((surface.get("occupied_surface_voxels") or {}).get("rate") if surface else None)
    no_ray_rate = ((rays.get("rays_inside_grid_with_no_occupied_voxel") or {}).get("rate") if rays else None)
    grid_miss_likely = bool(
        (surface_rate is not None and surface_rate < 0.5)
        or (no_ray_rate is not None and no_ray_rate > 0.25)
    )
    field_issue_likely = bool(surface_rate is not None and surface_rate >= 0.8 and (no_ray_rate is None or no_ray_rate < 0.1))
    return {
        "grid_miss_likely": grid_miss_likely,
        "field_issue_likely": field_issue_likely,
        "read": (
            "surface/ray voxels are often unoccupied; occupancy traversal or grid update policy is a likely lever"
            if grid_miss_likely
            else "artifact pixels mostly map to occupied voxels; field quality, alpha integration, or checkpoint selection is more likely"
        ),
    }


def save_overlay(cand: np.ndarray, artifact: Dict[str, object], pixels_yx: np.ndarray, path: Path) -> None:
    image = Image.fromarray(cand.copy())
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = artifact["bbox_xyxy"]
    draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=3)
    for y, x in pixels_yx[:: max(1, len(pixels_yx) // 500)]:
        draw.point((int(x), int(y)), fill=(255, 255, 0))
    image.save(path)


def write_markdown(path: Path, data: Dict[str, object]) -> None:
    lines = [
        "# Artifact to Occupancy Grid Debug",
        "",
        "## Summary",
        "",
        f"- Render: `{data['render_file']}`",
        f"- Eval index: `{data['eval_index']}`",
        f"- Checkpoint: `{data['checkpoint']}`",
        f"- Artifact bbox: `{data['artifact']['bbox_xyxy']}`",
        f"- Artifact score: `{data['artifact']['artifact_score']}`",
        f"- Grid miss likely: `{data['classification']['grid_miss_likely']}`",
        f"- Field issue likely: `{data['classification']['field_issue_likely']}`",
        f"- Read: {data['classification']['read']}",
        "",
        "## Occupancy",
        "",
        "```json",
        json.dumps(data["occupancy_global"], indent=2, sort_keys=True),
        "```",
        "",
        "## Surface-depth voxel stats",
        "",
        "```json",
        json.dumps(data["surface"], indent=2, sort_keys=True),
        "```",
        "",
        "## Along-ray voxel stats",
        "",
        "```json",
        json.dumps(data["rays"], indent=2, sort_keys=True),
        "```",
        "",
        "## ARM sample counts in artifact pixels",
        "",
        "```json",
        json.dumps(data["sample_counts"], indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.out_dir or (args.render_file.parent / f"artifact_occupancy_eval{args.eval_index:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    load_config = config_path(args)
    checkpoint_arg = selected_checkpoint(args)

    pixels_yx, artifact, gt, cand = select_artifact_pixels(args)
    save_overlay(cand, artifact, pixels_yx, output_dir / "artifact_pixels_overlay.png")

    def update_config(config):
        if checkpoint_arg is not None:
            config.load_dir = checkpoint_arg.parent
            config.load_step = checkpoint_step(checkpoint_arg)
        return config

    _, pipeline, checkpoint, step = eval_setup(
        load_config,
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        update_config_callback=update_config,
    )
    pipeline.eval()
    model = pipeline.model
    if not hasattr(model, "occupancy_grid"):
        raise RuntimeError(f"Loaded model {type(model).__name__} does not expose occupancy_grid.")

    with torch.no_grad():
        ray_bundle = generate_ray_bundle(pipeline, args.eval_index, pixels_yx)
        ray_bundle = collided_ray_bundle(model, ray_bundle)
        outputs = model.get_outputs(ray_bundle)
        sample_counts = analyze_sample_counts(model, outputs)
        surface = analyze_surface(model, ray_bundle, outputs, args.min_accumulation)
        rays = analyze_rays(model, ray_bundle, outputs, args.ray_samples, args.near_surface_margin)
        occupancy = occupancy_global_stats(model)

    result: Dict[str, object] = {
        "render_file": str(args.render_file),
        "eval_index": args.eval_index,
        "config": str(load_config),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "artifact": artifact,
        "occupancy_global": occupancy,
        "surface": surface,
        "rays": rays,
        "sample_counts": sample_counts,
        "classification": classify(surface, rays),
        "overlay": str(output_dir / "artifact_pixels_overlay.png"),
    }
    (output_dir / "artifact_occupancy_debug.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "artifact_occupancy_debug.md", result)
    print(json.dumps(result["classification"], indent=2, sort_keys=True))
    print(f"summary={output_dir / 'artifact_occupancy_debug.md'}")
    print(f"json={output_dir / 'artifact_occupancy_debug.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
