#!/usr/bin/env python3
"""Build frequency maps from luminance-preserving JPEG-422-normalized RGB inputs.

The original training/evaluation JPEGs are never modified. Only the temporary tensor seen by the
2D frequency estimator has its Cb/Cr channels horizontally downsampled by two and reconstructed,
matching the chroma sampling ratio of the canonical 007740 JPEGs. Map sidecars remain bound to the
original ``frame_train_*.jpg`` names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F

import fast_freqmap
from nerfstudio.scripts.lookcloser_preprocess import load_image_as_tensor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--glob", default="frame_train_*.jpg")
    parser.add_argument("--steps-per-level", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--eval-patch-batch", type=int, default=16384)
    parser.add_argument("--max-res", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--ssim-threshold", type=float, default=0.95)
    parser.add_argument("--ssim-window", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def rgb_luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def normalize_chroma_422(image: torch.Tensor) -> torch.Tensor:
    """Return RGB with horizontal 2x Cb/Cr subsampling while preserving full-resolution Y."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {tuple(image.shape)}")
    red, green, blue = image.unbind(dim=-1)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5
    chroma = torch.stack((cb, cr), dim=0).unsqueeze(0)
    half = F.avg_pool2d(chroma, kernel_size=(1, 2), stride=(1, 2))
    restored = F.interpolate(half, size=image.shape[:2], mode="bilinear", align_corners=False)[0]
    cb_delta, cr_delta = restored[0] - 0.5, restored[1] - 0.5
    normalized = torch.stack(
        (
            y + 1.402 * cr_delta,
            y - 0.344136 * cb_delta - 0.714136 * cr_delta,
            y + 1.772 * cb_delta,
        ),
        dim=-1,
    )
    normalized = normalized.clamp(0.0, 1.0)
    # Gamut clipping can move Y for saturated colors. Adding the same residual to
    # RGB preserves chroma and changes luminance one-for-one because the Y weights
    # sum to one. A second pass handles channels pinned by the first correction.
    for _ in range(2):
        residual = y - rgb_luminance(normalized)
        normalized = (normalized + residual.unsqueeze(-1)).clamp(0.0, 1.0)
    return normalized


def build_maps(args: argparse.Namespace) -> Dict[str, Any]:
    paths = sorted(args.images_dir.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No images matched {args.images_dir / args.glob}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    records = []
    started = time.monotonic()
    print(f"processing {len(paths)} chroma-normalized images -> {args.out}", flush=True)
    for index, path in enumerate(paths, 1):
        map_path = args.out / f"{path.stem}.pt"
        metadata_path = args.out / f"{path.stem}.json"
        if not args.force and map_path.is_file() and metadata_path.is_file():
            print(f"  [{index}/{len(paths)}] {path.stem} existing", flush=True)
            records.append({"image": path.name, "image_sha256": sha256_file(path), "status": "existing"})
            continue
        image = load_image_as_tensor(path, device)
        normalized = normalize_chroma_422(image)
        y_delta = float(torch.mean(torch.abs(rgb_luminance(normalized) - rgb_luminance(image))).item())
        chroma_delta = float(torch.mean(torch.abs(normalized - image)).item())
        image_started = time.monotonic()
        frequency_map = fast_freqmap.process_image(
            normalized,
            args.steps_per_level,
            args.train_batch_size,
            args.eval_patch_batch,
            args.max_res,
            args.patch_size,
            args.ssim_threshold,
            args.ssim_window,
            lr=args.lr,
            amp=False,
        )
        torch.save(frequency_map, map_path)
        fast_freqmap.save_metadata(
            metadata_path, path.name, (image.shape[0], image.shape[1]), args.patch_size, args.max_res
        )
        seconds = time.monotonic() - image_started
        records.append(
            {
                "image": path.name,
                "image_sha256": sha256_file(path),
                "map_sha256": sha256_file(map_path),
                "mean_luminance_delta": y_delta,
                "mean_rgb_delta": chroma_delta,
                "seconds": seconds,
                "status": "generated",
            }
        )
        del image, normalized, frequency_map
        torch.cuda.empty_cache()
        print(
            f"  [{index}/{len(paths)}] {path.stem} {seconds:.1f}s "
            f"y_delta={y_delta:.8f} rgb_delta={chroma_delta:.8f}",
            flush=True,
        )
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "luminance_preserving_horizontal_chroma_2x_lowpass",
        "images_dir": str(args.images_dir.resolve()),
        "output_dir": str(args.out.resolve()),
        "parameters": {
            "steps_per_level": args.steps_per_level,
            "train_batch_size": args.train_batch_size,
            "eval_patch_batch": args.eval_patch_batch,
            "max_res": args.max_res,
            "patch_size": args.patch_size,
            "ssim_threshold": args.ssim_threshold,
            "ssim_window": args.ssim_window,
            "lr": args.lr,
            "seed": args.seed,
        },
        "source_sha256": {
            "builder": sha256_file(Path(__file__)),
            "fast_freqmap": sha256_file(Path(fast_freqmap.__file__)),
        },
        "records": records,
        "total_seconds": time.monotonic() - started,
    }
    atomic_json(args.out.parent / f"{args.out.name}.provenance.json", provenance)
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    provenance = build_maps(parse_args(argv))
    print(
        f"DONE {len(provenance['records'])} images in {provenance['total_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
