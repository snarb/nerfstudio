#!/usr/bin/env python3
"""Create a provenance-recorded model-only field interpolation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


FIELD_KEYS = (
    "_model.field.encoding.params",
    "_model.field.mlp_geo.params",
    "_model.field.mlp_color.params",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Checkpoint supplying buffers and alpha=0 field weights")
    parser.add_argument("right", type=Path, help="Checkpoint supplying alpha=1 field weights")
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, required=True, help="Right-checkpoint field weight in [0, 1]")
    parser.add_argument("--step", type=int, required=True, help="Unique evaluator step stored in the output")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.step < 0:
        raise ValueError("--step must be non-negative")
    if args.output.exists():
        raise FileExistsError(args.output)

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    left_pipeline = left["pipeline"]
    right_pipeline = right["pipeline"]
    if set(left_pipeline) != set(right_pipeline):
        raise ValueError("Pipeline state keys differ between checkpoints")

    output_pipeline = dict(left_pipeline)
    tensor_stats = {}
    for key in FIELD_KEYS:
        left_tensor = left_pipeline[key]
        right_tensor = right_pipeline[key]
        if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
            raise ValueError(f"Field tensor mismatch for {key}")
        if not torch.is_floating_point(left_tensor):
            raise TypeError(f"Field tensor {key} is not floating point")
        output_pipeline[key] = torch.lerp(left_tensor, right_tensor, args.alpha)
        tensor_stats[key] = {
            "shape": list(left_tensor.shape),
            "dtype": str(left_tensor.dtype),
            "left_right_max_abs": float((left_tensor - right_tensor).abs().max()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": args.step, "pipeline": output_pipeline}, args.output)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "left_checkpoint": str(args.left),
        "left_sha256": sha256_file(args.left),
        "right_checkpoint": str(args.right),
        "right_sha256": sha256_file(args.right),
        "output_checkpoint": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_step": args.step,
        "alpha_right": args.alpha,
        "interpolated_keys": list(FIELD_KEYS),
        "buffer_policy": "left checkpoint unchanged",
        "tensor_stats": tensor_stats,
    }
    sidecar = args.output.with_suffix(args.output.suffix + ".interpolation.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
