#!/usr/bin/env python3
"""Project artifact surface points into train frequency maps.

This diagnostic answers a narrow question: when an eval artifact lies on a thin
structure, do the train views that see the corresponding 3D surface assign high
or low LookCloser frequency levels at those pixels?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from debug_artifact_occupancy_grid import (
    collided_ray_bundle,
    config_path,
    generate_ray_bundle,
    select_artifact_pixels,
    selected_checkpoint,
)
from nerfstudio.utils.eval_utils import eval_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--load-config", type=Path)
    source.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--render-file", type=Path, required=True)
    parser.add_argument("--eval-index", type=int, required=True)
    parser.add_argument("--bbox", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=2000)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--min-accumulation", type=float, default=0.05)
    parser.add_argument("--max-reprojection-error", type=float, default=8.0)
    parser.add_argument("--min-train-depth", type=float, default=0.01)
    parser.add_argument("--top-train-views", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def frequency_level(freq: torch.Tensor, metadata: Dict[str, object]) -> torch.Tensor:
    min_res = float(metadata["min_res"])
    max_res = float(metadata["max_res"])
    n_levels = int(metadata["n_levels"])
    b = math.exp((math.log(max_res) - math.log(min_res)) / max(n_levels - 1, 1))
    levels = torch.log(freq.float() / min_res) / math.log(b)
    return torch.clamp(torch.round(levels), 0, n_levels - 1).long()


def summarize(values: Iterable[float]) -> Dict[str, object]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0}
    arr = np.array(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def camera_values(cameras, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (
        cameras.camera_to_worlds[indices].float(),
        cameras.fx[indices].float().squeeze(-1),
        cameras.fy[indices].float().squeeze(-1),
        cameras.cx[indices].float().squeeze(-1),
        cameras.cy[indices].float().squeeze(-1),
        cameras.width[indices].long().squeeze(-1),
        cameras.height[indices].long().squeeze(-1),
    )


def project_points(
    points: torch.Tensor,
    cameras,
    indices: torch.Tensor,
    convention: str,
) -> Dict[str, torch.Tensor]:
    c2w, fx, fy, cx, cy, width, height = camera_values(cameras, indices)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3]
    rel = points[:, None, :] - translation[None, :, :]
    if convention == "row_R":
        cam = torch.einsum("pnc,nck->pnk", rel, rotation)
    elif convention == "row_RT":
        cam = torch.einsum("pnc,nkc->pnk", rel, rotation)
    else:
        raise ValueError(f"Unknown convention {convention!r}.")
    depth = -cam[..., 2]
    x = fx[None, :] * (cam[..., 0] / depth.clamp(min=1e-12)) + cx[None, :]
    y = fy[None, :] * (cam[..., 1] / depth.clamp(min=1e-12)) + cy[None, :]
    visible = (
        (depth > 0)
        & (x >= 0)
        & (y >= 0)
        & (x < width[None, :].float())
        & (y < height[None, :].float())
    )
    return {"x": x, "y": y, "depth": depth, "visible": visible}


def choose_projection_convention(points: torch.Tensor, pixels_yx: np.ndarray, eval_camera) -> Dict[str, object]:
    target = torch.from_numpy(pixels_yx[:, ::-1].copy()).float().to(points.device)
    indices = torch.zeros((1,), dtype=torch.long, device=points.device)
    rows = []
    for convention in ("row_R", "row_RT"):
        proj = project_points(points, eval_camera.to(points.device), indices, convention)
        xy = torch.stack([proj["x"][:, 0], proj["y"][:, 0]], dim=-1)
        err = torch.linalg.norm(xy - target, dim=-1)
        rows.append(
            {
                "convention": convention,
                "median_error": float(err.median().item()),
                "mean_error": float(err.mean().item()),
                "p90_error": float(torch.quantile(err, 0.9).item()),
            }
        )
    best = min(rows, key=lambda row: row["median_error"])
    return {"selected": best["convention"], "candidates": rows}


def load_frequency_maps(train_dataset, freq_dir: Path) -> Dict[int, Dict[str, object]]:
    maps: Dict[int, Dict[str, object]] = {}
    for idx, image_path in enumerate(train_dataset.image_filenames):
        map_path = freq_dir / f"{image_path.stem}.pt"
        meta_path = map_path.with_suffix(".json")
        if not map_path.exists() or not meta_path.exists():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        freq_map = torch.load(map_path, map_location="cpu").float()
        levels = frequency_level(freq_map, metadata)
        maps[idx] = {
            "image": str(image_path),
            "map_path": str(map_path),
            "metadata": metadata,
            "freq_map": freq_map,
            "levels": levels,
        }
    return maps


def lookup_frequency(
    projections: Dict[str, torch.Tensor],
    train_dataset,
    freq_maps: Dict[int, Dict[str, object]],
    convention: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    del convention
    visible = projections["visible"].cpu()
    x = projections["x"].cpu()
    y = projections["y"].cpu()
    depth = projections["depth"].cpu()
    rows: List[Dict[str, object]] = []
    per_view: List[Dict[str, object]] = []
    for img_idx, info in freq_maps.items():
        mask = visible[:, img_idx]
        if not bool(mask.any()):
            continue
        metadata = info["metadata"]
        stride = int(metadata.get("stride", metadata.get("patch_size", 8)))
        patch_size = int(metadata.get("patch_size", 8))
        freq_map = info["freq_map"]
        levels = info["levels"]
        covered_h = (freq_map.shape[0] - 1) * stride + patch_size
        covered_w = (freq_map.shape[1] - 1) * stride + patch_size
        ys = y[mask, img_idx].long().clamp(0, covered_h - 1)
        xs = x[mask, img_idx].long().clamp(0, covered_w - 1)
        map_y = torch.clamp(ys // stride, 0, freq_map.shape[0] - 1)
        map_x = torch.clamp(xs // stride, 0, freq_map.shape[1] - 1)
        vals = freq_map[map_y, map_x]
        lvls = levels[map_y, map_x]
        deps = depth[mask, img_idx]
        f = ((train_dataset.cameras.fx[img_idx] + train_dataset.cameras.fy[img_idx]) * 0.5).item()
        f3d = vals * float(f) / deps.clamp(min=1e-12)
        image_name = Path(str(train_dataset.image_filenames[img_idx])).name
        per_view.append(
            {
                "train_index": int(img_idx),
                "image": image_name,
                "visible_points": int(mask.sum().item()),
                "level": summarize(lvls.tolist()),
                "scalar_resolution": summarize(vals.tolist()),
                "f3d": summarize(f3d.tolist()),
                "depth": summarize(deps.tolist()),
                "mean_x": float(x[mask, img_idx].mean().item()),
                "mean_y": float(y[mask, img_idx].mean().item()),
            }
        )
        for local_i in range(vals.numel()):
            rows.append(
                {
                    "train_index": int(img_idx),
                    "image": image_name,
                    "x": float(x[mask, img_idx][local_i].item()),
                    "y": float(y[mask, img_idx][local_i].item()),
                    "depth": float(deps[local_i].item()),
                    "scalar_resolution": float(vals[local_i].item()),
                    "level": int(lvls[local_i].item()),
                    "f3d": float(f3d[local_i].item()),
                }
            )
    return rows, per_view


def draw_overlay(train_dataset, per_view: List[Dict[str, object]], projections, out_dir: Path, top_n: int) -> List[str]:
    out_paths: List[str] = []
    visible = projections["visible"].cpu()
    x = projections["x"].cpu()
    y = projections["y"].cpu()
    ranked = sorted(per_view, key=lambda row: row["visible_points"], reverse=True)[:top_n]
    for row in ranked:
        idx = int(row["train_index"])
        image_path = Path(str(train_dataset.image_filenames[idx]))
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        mask = visible[:, idx]
        xs = x[mask, idx].numpy()
        ys = y[mask, idx].numpy()
        for px, py in zip(xs, ys):
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(255, 255, 0), width=2)
        if xs.size:
            draw.rectangle((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), outline=(255, 0, 0), width=3)
        out_path = out_dir / f"train_{idx:03d}_{image_path.stem}_projection.png"
        image.save(out_path)
        out_paths.append(str(out_path))
    return out_paths


def write_markdown(path: Path, data: Dict[str, object]) -> None:
    lines = [
        "# Artifact Frequency Projection Audit",
        "",
        "## Summary",
        "",
        f"- Render: `{data['render_file']}`",
        f"- Eval index: `{data['eval_index']}`",
        f"- Checkpoint: `{data['checkpoint']}`",
        f"- Artifact bbox: `{data['artifact']['bbox_xyxy']}`",
        f"- Surface points: `{data['surface_points']}`",
        f"- Projection convention: `{data['projection']['selected']}`",
        f"- Eval reprojection median px: `{data['projection']['selected_stats']['median_error']:.3f}`",
        f"- Train observations: `{data['train_observation_count']}`",
        f"- Visible train views: `{data['visible_train_views']}`",
        "",
        "## Frequency Summary",
        "",
        "```json",
        json.dumps(data["frequency_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Top Train Views",
        "",
        "| Train idx | Image | Visible pts | Median level | P90 level | Median f3d | Median depth | Mean xy |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in data["top_train_views"]:
        level = row["level"]
        f3d = row["f3d"]
        depth = row["depth"]
        lines.append(
            f"| {row['train_index']} | `{row['image']}` | {row['visible_points']} | "
            f"{level.get('median', 'n/a')} | {level.get('p90', 'n/a')} | "
            f"{f3d.get('median', 'n/a')} | {depth.get('median', 'n/a')} | "
            f"({row['mean_x']:.1f}, {row['mean_y']:.1f}) |"
        )
    lines.extend(
        [
            "",
            "## Overlays",
            "",
            *[f"- `{p}`" for p in data["overlay_paths"]],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_config = config_path(args)
    checkpoint_arg = args.checkpoint or selected_checkpoint(args)
    if checkpoint_arg is None:
        raise RuntimeError("Could not determine checkpoint; pass --checkpoint.")

    pixel_args = SimpleNamespace(**vars(args), panels=2, gt_panel=0, cand_panel=1)
    pixels_yx, artifact, _gt, _cand = select_artifact_pixels(pixel_args)

    def update_config(config):
        config.load_dir = checkpoint_arg.parent
        step_text = checkpoint_arg.stem.split("-")[-1]
        config.load_step = int(step_text)
        return config

    _, pipeline, checkpoint, step = eval_setup(
        load_config,
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        update_config_callback=update_config,
    )
    pipeline.eval()
    model = pipeline.model
    with torch.no_grad():
        ray_bundle = generate_ray_bundle(pipeline, args.eval_index, pixels_yx)
        ray_bundle = collided_ray_bundle(model, ray_bundle)
        outputs = model.get_outputs(ray_bundle)
        depth = outputs["depth"].reshape(-1).to(model.device)
        accumulation = outputs.get("accumulation", torch.ones_like(depth[:, None])).reshape(-1).to(model.device)
        valid = torch.isfinite(depth) & (depth > 0) & (accumulation >= args.min_accumulation)
        if not bool(valid.any()):
            raise RuntimeError("No valid surface points in artifact region.")
        valid_pixels = pixels_yx[valid.detach().cpu().numpy()]
        points = ray_bundle.origins.reshape(-1, 3)[valid] + ray_bundle.directions.reshape(-1, 3)[valid] * depth[valid, None]
        eval_camera, _ = pipeline.datamanager.fixed_indices_eval_dataloader.get_camera(args.eval_index)
        projection = choose_projection_convention(points, valid_pixels, eval_camera)
        selected_stats = min(projection["candidates"], key=lambda row: row["median_error"])
        projection["selected_stats"] = selected_stats
        train_dataset = pipeline.datamanager.train_dataset
        train_cameras = train_dataset.cameras.to(model.device)
        train_indices = torch.arange(len(train_dataset), dtype=torch.long, device=model.device)
        projections = project_points(points, train_cameras, train_indices, projection["selected"])

    data_path = pipeline.datamanager.config.data
    freq_dir = Path(data_path) / pipeline.config.frequency_map_dir
    freq_maps = load_frequency_maps(train_dataset, freq_dir)
    observation_rows, per_view = lookup_frequency(projections, train_dataset, freq_maps, projection["selected"])
    overlay_paths = draw_overlay(train_dataset, per_view, projections, args.out_dir, args.top_train_views)

    level_values = [row["level"] for row in observation_rows]
    f2d_values = [row["scalar_resolution"] for row in observation_rows]
    f3d_values = [row["f3d"] for row in observation_rows]
    result: Dict[str, object] = {
        "render_file": str(args.render_file),
        "eval_index": args.eval_index,
        "config": str(load_config),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "artifact": artifact,
        "surface_points": int(points.shape[0]),
        "projection": projection,
        "frequency_dir": str(freq_dir),
        "train_observation_count": len(observation_rows),
        "visible_train_views": len(per_view),
        "frequency_summary": {
            "level": summarize(level_values),
            "scalar_resolution": summarize(f2d_values),
            "f3d": summarize(f3d_values),
            "fraction_level_ge_4": float(np.mean(np.array(level_values) >= 4)) if level_values else None,
            "fraction_level_ge_8": float(np.mean(np.array(level_values) >= 8)) if level_values else None,
            "fraction_level_ge_12": float(np.mean(np.array(level_values) >= 12)) if level_values else None,
        },
        "top_train_views": sorted(per_view, key=lambda row: row["visible_points"], reverse=True)[: args.top_train_views],
        "overlay_paths": overlay_paths,
    }
    (args.out_dir / "artifact_frequency_projection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "artifact_frequency_projection.md", result)
    print(json.dumps({
        "surface_points": result["surface_points"],
        "train_observation_count": result["train_observation_count"],
        "visible_train_views": result["visible_train_views"],
        "frequency_summary": result["frequency_summary"],
        "projection": result["projection"],
        "markdown": str(args.out_dir / "artifact_frequency_projection.md"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
