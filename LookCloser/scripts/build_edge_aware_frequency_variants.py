#!/usr/bin/env python3
"""Derive conservative, edge-aware map variants from cached EXR recovery maps.

This is intentionally cheap: it does not refit the per-image 2D HashGrid.  It
combines the already selected threshold-free knee map, the independently
calibrated crossing map, and the PQ structural proxy while retaining scene-
adaptive level statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--structural-fraction", type=float, default=0.20)
    parser.add_argument("--floor-quantile", type=float, default=0.75)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 < args.structural_fraction < 1 or not 0 <= args.floor_quantile <= 1:
        parser.error("fractions/quantiles must lie in their valid unit intervals")
    if args.dilation_radius < 0:
        parser.error("dilation-radius must be non-negative")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolutions_to_levels(values: torch.Tensor, min_res: float, max_res: float, n_levels: int) -> torch.Tensor:
    scale = torch.log(torch.tensor(float(max_res) / float(min_res))) / float(n_levels - 1)
    return torch.round(torch.log(values.float() / float(min_res)) / scale).long().clamp(0, n_levels - 1)


def levels_to_resolutions(values: torch.Tensor, min_res: float, max_res: float, n_levels: int) -> torch.Tensor:
    scale = torch.exp(torch.log(torch.tensor(float(max_res) / float(min_res))) / float(n_levels - 1))
    return float(min_res) * torch.pow(scale, values.float())


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask
    width = 2 * radius + 1
    return F.max_pool2d(mask[None, None].float(), kernel_size=width, stride=1, padding=radius)[0, 0] > 0


def variants(
    knee: torch.Tensor,
    calibrated: torch.Tensor,
    proxy: torch.Tensor,
    *,
    structural_fraction: float,
    floor_level: int,
    dilation_radius: int,
    n_levels: int,
) -> dict[str, torch.Tensor]:
    cutoff = torch.quantile(proxy.float(), 1.0 - float(structural_fraction))
    structural = dilate(proxy >= cutoff, dilation_radius)
    floor = torch.full_like(knee, int(floor_level))
    return {
        "knee_plus1": (knee + 1).clamp_max(n_levels - 1),
        "knee_edge_floor": torch.where(structural, torch.maximum(knee, floor), knee),
        "knee_calibrated_union": torch.maximum(knee, calibrated),
        "knee_calibrated_edges": torch.where(structural, torch.maximum(knee, calibrated), knee),
    }


def main() -> int:
    args = parse_args()
    provenance_path = args.map_root / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    parameters = provenance["parameters"]
    min_res = 16.0
    max_res = float(parameters["max_res"])
    n_levels = 16
    knee_paths = sorted((args.map_root / "knee").glob("*.pt"))
    if not knee_paths:
        raise FileNotFoundError(args.map_root / "knee")

    all_knee_levels = []
    inputs = []
    for knee_path in knee_paths:
        calibrated_path = args.map_root / "calibrated" / knee_path.name
        proxy_path = args.map_root / "recovery" / f"{knee_path.stem}.proxy_pq.pt"
        if not calibrated_path.is_file() or not proxy_path.is_file():
            raise FileNotFoundError(f"Missing calibrated/proxy pair for {knee_path.name}")
        knee = resolutions_to_levels(torch.load(knee_path, map_location="cpu", weights_only=True), min_res, max_res, n_levels)
        all_knee_levels.append(knee.flatten())
        inputs.append((knee_path, calibrated_path, proxy_path, knee))
    floor_level = int(torch.quantile(torch.cat(all_knee_levels).float(), args.floor_quantile).round().item())

    output_hashes: dict[str, dict[str, str]] = {}
    output_levels: dict[str, list[torch.Tensor]] = {}
    for knee_path, calibrated_path, proxy_path, knee in inputs:
        calibrated = resolutions_to_levels(
            torch.load(calibrated_path, map_location="cpu", weights_only=True), min_res, max_res, n_levels
        )
        proxy = torch.load(proxy_path, map_location="cpu", weights_only=True)
        if knee.shape != calibrated.shape or knee.shape != proxy.shape:
            raise ValueError(f"Shape mismatch for {knee_path.stem}: {knee.shape}, {calibrated.shape}, {proxy.shape}")
        image_variants = variants(
            knee,
            calibrated,
            proxy,
            structural_fraction=args.structural_fraction,
            floor_level=floor_level,
            dilation_radius=args.dilation_radius,
            n_levels=n_levels,
        )
        for name, levels in image_variants.items():
            output_dir = args.map_root / name
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / knee_path.name
            if output_path.exists() and not args.force:
                raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")
            torch.save(levels_to_resolutions(levels, min_res, max_res, n_levels), output_path)
            output_hashes.setdefault(name, {})[knee_path.name] = sha256_file(output_path)
            output_levels.setdefault(name, []).append(levels.flatten())

    statistics = {}
    source_levels = torch.cat(all_knee_levels)
    for name, parts in output_levels.items():
        values = torch.cat(parts)
        counts = torch.bincount(values, minlength=n_levels).float()
        probabilities = counts / counts.sum()
        statistics[name] = {
            "mean_level": float(values.float().mean()),
            "changed_fraction_from_knee": float((values != source_levels).float().mean()),
            "top_level_fraction": float(probabilities[-1]),
            "nonempty_bins": int((counts > 0).sum()),
        }

    manifest = {
        "schema": 1,
        "source_provenance": str(provenance_path.resolve()),
        "source_provenance_sha256": sha256_file(provenance_path),
        "parameters": {
            "structural_fraction": args.structural_fraction,
            "floor_quantile": args.floor_quantile,
            "floor_level": floor_level,
            "dilation_radius": args.dilation_radius,
            "min_res": min_res,
            "max_res": max_res,
            "n_levels": n_levels,
        },
        "image_count": len(inputs),
        "outputs": output_hashes,
        "statistics": statistics,
    }
    atomic_json(args.map_root / "edge_aware_variants.json", manifest)
    print(f"images={len(inputs)} floor_level={floor_level} variants={','.join(sorted(output_hashes))}")
    print(f"manifest={args.map_root / 'edge_aware_variants.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
