#!/usr/bin/env python3
"""Build EXR structural maps for occupancy-independent 3D geometry probing.

The maps deliberately encode image-space evidence only.  During training each
selected pixel is fixed-probed along its camera ray before any voxel is marked
as geometry-supported, so texture, shadow, and highlight edges remain attached
to an actual predicted surface instead of opening an entire viewing ray.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nerfstudio.data.utils.data_utils import load_exr_image
from nerfstudio.utils.hdr import BT709_LUMA, hdr_display_preview, scene_linear_to_pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--frequency-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--glob", default="frame_train_*.exr")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--ridge-scales", default="5,9,17")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.ridge_scales = tuple(int(value) for value in args.ridge_scales.split(",") if value)
    if args.patch_size <= 0 or not args.ridge_scales:
        parser.error("patch-size and ridge-scales must be positive")
    if any(value <= 1 or value % 2 == 0 for value in args.ridge_scales):
        parser.error("ridge-scales must contain odd integers greater than one")
    return args


def rank_normalize(values: torch.Tensor) -> torch.Tensor:
    """Percentile mid-ranks with exact ties sharing one value."""
    flat = values.flatten()
    _, inverse, counts = torch.unique(flat, sorted=True, return_inverse=True, return_counts=True)
    preceding = torch.cumsum(counts, dim=0) - counts
    denominator = max(flat.numel() - 1, 1)
    midranks = (preceding.float() + 0.5 * (counts.float() - 1.0)) / float(denominator)
    return midranks[inverse].reshape_as(values)


def structural_maps(pq_rgb: torch.Tensor, patch_size: int, ridge_scales: tuple[int, ...]) -> dict[str, torch.Tensor]:
    luminance = torch.tensordot(
        pq_rgb[..., :3],
        torch.tensor(BT709_LUMA, device=pq_rgb.device, dtype=pq_rgb.dtype),
        dims=([-1], [0]),
    )[None, None]
    scharr_x = torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        device=pq_rgb.device,
        dtype=pq_rgb.dtype,
    )[None, None] / 16.0
    scharr_y = scharr_x.transpose(-1, -2)
    gx = F.conv2d(luminance, scharr_x, padding=1)
    gy = F.conv2d(luminance, scharr_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-12)

    dark_ridges = []
    for scale in ridge_scales:
        dilated = F.max_pool2d(luminance, kernel_size=scale, stride=1, padding=scale // 2)
        closed = -F.max_pool2d(-dilated, kernel_size=scale, stride=1, padding=scale // 2)
        dark_ridges.append((closed - luminance).clamp_min(0.0))
    ridge = torch.stack(dark_ridges).amax(dim=0)

    edge_patch = F.max_pool2d(edge, kernel_size=patch_size, stride=patch_size)[0, 0]
    ridge_patch = F.max_pool2d(ridge, kernel_size=patch_size, stride=patch_size)[0, 0]
    edge_rank = rank_normalize(edge_patch)
    ridge_rank = rank_normalize(ridge_patch)
    return {
        "edge": edge_rank,
        "edge_ridge": torch.maximum(edge_rank, ridge_rank),
        "ridge": ridge_rank,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_preview(image: torch.Tensor, score: torch.Tensor, path: Path) -> None:
    preview = hdr_display_preview(image.detach().cpu()).numpy()
    heat = F.interpolate(
        score[None, None],
        size=preview.shape[:2],
        mode="nearest",
    )[0, 0].detach().cpu().numpy()
    selected = heat >= float(torch.quantile(score, 0.8).item())
    overlay = preview.copy()
    overlay[..., 0] = np.where(selected, 0.45 * overlay[..., 0] + 0.55, overlay[..., 0])
    overlay[..., 1] = np.where(selected, 0.45 * overlay[..., 1], overlay[..., 1])
    overlay[..., 2] = np.where(selected, 0.45 * overlay[..., 2], overlay[..., 2])
    side_by_side = np.concatenate((preview, overlay), axis=1)
    Image.fromarray((side_by_side.clip(0.0, 1.0) * 255.0).astype("uint8")).save(path, quality=92)


def main() -> int:
    args = parse_args()
    paths = sorted(args.images_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No images matched {args.images_dir / args.glob}")
    provenance = json.loads((args.frequency_root / "provenance.json").read_text(encoding="utf-8"))
    calibration = provenance["hdr_calibration"]
    device = torch.device(args.device)
    preview_indices = set(
        torch.linspace(0, len(paths) - 1, min(args.preview_count, len(paths))).round().long().tolist()
    )
    hashes: dict[str, dict[str, str]] = {"edge": {}, "edge_ridge": {}, "ridge": {}}
    statistics: dict[str, list[torch.Tensor]] = {name: [] for name in hashes}
    preview_dir = args.out / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(paths):
        array = load_exr_image(image_path)[..., :3]
        image = torch.from_numpy(array).to(device=device, dtype=torch.float32)
        pq = scene_linear_to_pq(
            image,
            nits_per_scene_unit=float(calibration["nits_per_scene_unit"]),
            black_nits=float(calibration["black_nits"]),
        )
        maps = structural_maps(pq, args.patch_size, args.ridge_scales)
        for name, score in maps.items():
            output_dir = args.out / name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}.pt"
            if output_path.exists() and not args.force:
                raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")
            torch.save(score.cpu(), output_path)
            hashes[name][output_path.name] = sha256_file(output_path)
            statistics[name].append(score.detach().cpu().flatten())
        if index in preview_indices:
            save_preview(image, maps["edge_ridge"], preview_dir / f"{image_path.stem}.jpg")
        print(f"[{index + 1}/{len(paths)}] {image_path.stem}", flush=True)

    summary: dict[str, Any] = {}
    for name, values in statistics.items():
        joined = torch.cat(values)
        summary[name] = {
            "mean": float(joined.mean()),
            "q80": float(torch.quantile(joined, 0.8)),
            "q90": float(torch.quantile(joined, 0.9)),
            "q95": float(torch.quantile(joined, 0.95)),
        }
    manifest = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images_dir": str(args.images_dir.resolve()),
        "frequency_provenance": str((args.frequency_root / "provenance.json").resolve()),
        "parameters": {
            "patch_size": args.patch_size,
            "stride": args.patch_size,
            "ridge_scales": list(args.ridge_scales),
            "domain": "pq_bt709_luminance",
            "edge": "Scharr patch maximum",
            "ridge": "multi-scale dark morphological closing patch maximum",
        },
        "statistics": summary,
        "outputs": hashes,
    }
    temporary = args.out / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out / "manifest.json")
    print(f"manifest={args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
