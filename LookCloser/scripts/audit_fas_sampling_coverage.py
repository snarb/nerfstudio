#!/usr/bin/env python3
"""Audit expected LookCloser FAS sampling coverage from frequency maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from PIL import Image


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_FRAMES = ("frame_train_00029", "frame_train_00047", "frame_train_00056", "frame_train_00062")
DEFAULT_CROP = (320, 0, 617, 530)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frequency-map-dir", default="lookcloser_frequencies")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", nargs="*", default=list(DEFAULT_FRAMES))
    parser.add_argument("--crop-xyxy", nargs=4, type=int, default=list(DEFAULT_CROP))
    parser.add_argument("--num-levels", type=int, default=16)
    parser.add_argument("--sampling-ramp-start", type=float, default=1.0)
    parser.add_argument("--sampling-ramp-end", type=float, default=3.0)
    parser.add_argument("--fas-level-count-alpha", type=float, default=0.0)
    parser.add_argument("--fas-strength", type=float, default=0.35)
    parser.add_argument("--fas-max-sampling-level", type=int, default=-1)
    return parser.parse_args()


def read_levels(freq_path: Path, num_levels: int, max_sampling_level: int) -> Tuple[np.ndarray, Dict]:
    metadata = json.loads(freq_path.with_suffix(".json").read_text(encoding="utf-8"))
    freq_map = torch.load(freq_path, map_location="cpu").float()
    min_res = float(metadata["min_res"])
    max_res = float(metadata["max_res"])
    n_levels = int(metadata["n_levels"])
    scale = math.exp((math.log(max_res) - math.log(min_res)) / (n_levels - 1))
    levels = torch.round(torch.log(freq_map / min_res) / math.log(scale))
    levels = torch.clamp(levels, 0, num_levels - 1).long()
    if max_sampling_level >= 0:
        levels = torch.clamp(levels, 0, min(max_sampling_level, num_levels - 1))
    return levels.numpy(), metadata


def iter_frequency_maps(freq_dir: Path) -> Iterable[Path]:
    yield from sorted(freq_dir.glob("frame_train_*.pt"))


def colorize_relative(values: np.ndarray) -> Image.Image:
    clipped = np.clip(values, 0.0, 2.0) / 2.0
    red = (255 * clipped).astype(np.uint8)
    blue = (255 * (1.0 - clipped)).astype(np.uint8)
    green = (255 * (1.0 - np.abs(clipped - 0.5) * 2.0)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def overlay(rgb: Image.Image, heatmap: Image.Image, alpha: float = 0.45) -> Image.Image:
    return Image.blend(rgb.convert("RGB"), heatmap.convert("RGB"), alpha)


def main() -> int:
    args = parse_args()
    freq_dir = args.data / args.frequency_map_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    levels_by_frame: Dict[str, np.ndarray] = {}
    metadata_by_frame: Dict[str, Dict] = {}
    level_counts = np.zeros(args.num_levels, dtype=np.float64)
    for freq_path in iter_frequency_maps(freq_dir):
        levels, metadata = read_levels(freq_path, args.num_levels, args.fas_max_sampling_level)
        levels_by_frame[freq_path.stem] = levels
        metadata_by_frame[freq_path.stem] = metadata
        level_counts += np.bincount(levels.reshape(-1), minlength=args.num_levels)

    non_empty = level_counts > 0
    ramp = np.linspace(args.sampling_ramp_start, args.sampling_ramp_end, args.num_levels)
    count_alpha = max(float(args.fas_level_count_alpha), 0.0)
    if count_alpha > 0.0:
        weights = ramp * np.where(non_empty, np.power(np.maximum(level_counts, 1.0), count_alpha), 0.0)
    else:
        weights = ramp * np.where(non_empty, 1.0, 0.0)
    if weights.sum() <= 0:
        weights = ramp
    probs = weights / weights.sum()

    total_cells = float(level_counts.sum())
    fas_relative = np.divide(probs * total_cells, level_counts, out=np.zeros_like(probs), where=level_counts > 0)
    active_strength = float(np.clip(args.fas_strength, 0.0, 1.0))
    mixed_relative = (1.0 - active_strength) + active_strength * fas_relative

    level_rows = []
    for level in range(args.num_levels):
        level_rows.append(
            {
                "level": level,
                "bucket_cells": int(level_counts[level]),
                "probability": float(probs[level]),
                "fas_relative_to_uniform": float(fas_relative[level]),
                "mixed_relative_to_uniform": float(mixed_relative[level]),
            }
        )

    with (args.output_dir / "level_sampling_weights.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(level_rows[0].keys()))
        writer.writeheader()
        writer.writerows(level_rows)

    x0, y0, x1, y1 = args.crop_xyxy
    crop_rows = []
    for frame in args.frames:
        levels = levels_by_frame[frame]
        metadata = metadata_by_frame[frame]
        patch_size = int(metadata["patch_size"])
        stride = int(metadata.get("stride", patch_size))
        py0 = max(y0 // stride, 0)
        px0 = max(x0 // stride, 0)
        py1 = min(math.ceil((y1 - patch_size) / stride) + 1, levels.shape[0])
        px1 = min(math.ceil((x1 - patch_size) / stride) + 1, levels.shape[1])
        crop_levels = levels[py0:py1, px0:px1]
        hist = np.bincount(crop_levels.reshape(-1), minlength=args.num_levels)
        uniform_share = crop_levels.size / total_cells
        fas_share = sum(probs[level] * hist[level] / level_counts[level] for level in range(args.num_levels) if level_counts[level] > 0)
        mixed_share = (1.0 - active_strength) * uniform_share + active_strength * fas_share
        crop_rows.append(
            {
                "frame": frame,
                "crop_patch_yx": f"{py0}:{py1},{px0}:{px1}",
                "crop_cells": int(crop_levels.size),
                "uniform_share": float(uniform_share),
                "fas_share": float(fas_share),
                "mixed_share": float(mixed_share),
                "fas_relative_to_uniform": float(fas_share / uniform_share),
                "mixed_relative_to_uniform": float(mixed_share / uniform_share),
            }
        )

        rel_map = mixed_relative[levels]
        heat = colorize_relative(rel_map)
        image_shape = metadata.get("image_shape")
        if image_shape is not None:
            image_h, image_w = int(image_shape[0]), int(image_shape[1])
            heat = heat.resize((image_w, image_h), resample=Image.Resampling.NEAREST)
            rgb = Image.open(args.data / "images" / f"{frame}.jpg").convert("RGB")
            overlay(rgb, heat).save(args.output_dir / f"{frame}_mixed_relative_overlay.png")
            heat.save(args.output_dir / f"{frame}_mixed_relative_heatmap.png")
            rgb.crop((x0, y0, x1, y1)).save(args.output_dir / f"{frame}_crop_rgb.png")
            overlay(rgb, heat).crop((x0, y0, x1, y1)).save(args.output_dir / f"{frame}_crop_mixed_relative_overlay.png")

    with (args.output_dir / "crop_sampling_coverage.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(crop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(crop_rows)

    summary = {
        "data": str(args.data),
        "frequency_map_dir": str(freq_dir),
        "num_maps": len(levels_by_frame),
        "crop_xyxy": [x0, y0, x1, y1],
        "sampling": {
            "sampling_ramp_start": args.sampling_ramp_start,
            "sampling_ramp_end": args.sampling_ramp_end,
            "fas_level_count_alpha": args.fas_level_count_alpha,
            "fas_strength": args.fas_strength,
            "fas_max_sampling_level": args.fas_max_sampling_level,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"output_dir={args.output_dir}")
    print(f"level_weights={args.output_dir / 'level_sampling_weights.csv'}")
    print(f"crop_coverage={args.output_dir / 'crop_sampling_coverage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
