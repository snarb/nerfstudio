#!/usr/bin/env python3
"""Inspect LookCloser scalar-resolution frequency maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frequency-map-dir", default="lookcloser_frequencies")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def level_hist(freq_map: torch.Tensor, metadata: Dict) -> List[int]:
    min_res = float(metadata["min_res"])
    max_res = float(metadata["max_res"])
    n_levels = int(metadata["n_levels"])
    b = math.exp((math.log(max_res) - math.log(min_res)) / max(n_levels - 1, 1))
    levels = torch.log(freq_map.float() / min_res) / math.log(b)
    levels = torch.clamp(torch.round(levels), 0, n_levels - 1).long()
    return torch.bincount(levels.flatten(), minlength=n_levels).tolist()


def inspect_map(path: Path) -> Dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        freq_map = torch.load(path, map_location="cpu", weights_only=True).float()
    except TypeError:
        freq_map = torch.load(path, map_location="cpu").float()
    hist = level_hist(freq_map, metadata)
    total = int(sum(hist))
    return {
        "path": str(path),
        "shape": list(freq_map.shape),
        "min": float(freq_map.min().item()),
        "max": float(freq_map.max().item()),
        "mean": float(freq_map.mean().item()),
        "median": float(freq_map.median().item()),
        "non_empty_levels": int(sum(1 for count in hist if count > 0)),
        "fraction_min_level": float(hist[0] / total),
        "fraction_max_level": float(hist[-1] / total),
        "histogram": hist,
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    freq_dir = args.data / args.frequency_map_dir
    paths = sorted(freq_dir.glob("*.pt"))
    if args.limit > 0:
        paths = paths[: args.limit]
    rows = [inspect_map(path) for path in paths]
    summary = {
        "frequency_dir": str(freq_dir),
        "map_count": len(rows),
        "metadata_count": len(list(freq_dir.glob("*.json"))) if freq_dir.exists() else 0,
        "mean_non_empty_levels": mean(row["non_empty_levels"] for row in rows) if rows else None,
        "mean_fraction_min_level": mean(row["fraction_min_level"] for row in rows) if rows else None,
        "mean_fraction_max_level": mean(row["fraction_max_level"] for row in rows) if rows else None,
        "max_fraction_max_level": max((row["fraction_max_level"] for row in rows), default=None),
        "rows": rows,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
