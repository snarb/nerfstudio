#!/usr/bin/env python3
"""Create an eval-ready LookCloser run from interpolated field checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch


FIELD_PARAMETER_KEYS = (
    "_model.field.encoding.params",
    "_model.field.mlp_geo.params",
    "_model.field.mlp_color.params",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--out-run-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint filename step. Defaults to the base checkpoint's saved step.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step-(\d+)\.ckpt$", path.name)
    if match:
        return int(match.group(1))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["step"])


def rewrite_config(template: Path, out_run_dir: Path) -> None:
    if out_run_dir.parent.name != "lookcloser":
        raise ValueError(f"Expected out run dir to end with <experiment>/lookcloser/<timestamp>: {out_run_dir}")
    experiment_name = out_run_dir.parent.parent.name
    timestamp = out_run_dir.name
    text = template.read_text(encoding="utf-8")
    text = re.sub(r"^experiment_name:.*$", f"experiment_name: {experiment_name}", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^timestamp:.*$", f"timestamp: {timestamp}", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^load_step:.*$", "load_step: null", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^max_num_iterations:.*$", "max_num_iterations: 200000", text, count=1, flags=re.MULTILINE)
    out_run_dir.mkdir(parents=True, exist_ok=True)
    (out_run_dir / "config.yml").write_text(text, encoding="utf-8")


def interpolate(args: argparse.Namespace) -> Path:
    alpha = float(args.alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    # Local experiment checkpoints are trusted and contain NumPy/Python metadata
    # that PyTorch 2.6's weights-only loader intentionally rejects.
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    target = torch.load(args.target_checkpoint, map_location="cpu", weights_only=False)
    base_pipeline = base["pipeline"]
    target_pipeline = target["pipeline"]

    interpolated_keys = []
    for key in FIELD_PARAMETER_KEYS:
        if key not in base_pipeline or key not in target_pipeline:
            raise KeyError(f"Missing field parameter key {key}")
        base_value = base_pipeline[key]
        target_value = target_pipeline[key]
        if base_value.shape != target_value.shape:
            raise ValueError(f"Shape mismatch for {key}: {base_value.shape} != {target_value.shape}")
        if not torch.is_floating_point(base_value):
            raise TypeError(f"Cannot interpolate non-floating tensor {key}")
        base_pipeline[key] = base_value.lerp(target_value.to(dtype=base_value.dtype), alpha)
        interpolated_keys.append(key)

    step = int(args.step if args.step is not None else base.get("step", checkpoint_step(args.base_checkpoint)))
    base["step"] = step

    model_dir = args.out_run_dir / "nerfstudio_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_checkpoint = model_dir / f"step-{step:09d}.ckpt"
    torch.save(base, out_checkpoint)

    metadata = {
        "base_checkpoint": str(args.base_checkpoint),
        "target_checkpoint": str(args.target_checkpoint),
        "alpha": alpha,
        "step": step,
        "interpolated_keys": interpolated_keys,
        "out_checkpoint": str(out_checkpoint),
    }
    (args.out_run_dir / "interpolation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_checkpoint


def main() -> None:
    args = parse_args()
    if not args.base_checkpoint.exists():
        raise FileNotFoundError(args.base_checkpoint)
    if not args.target_checkpoint.exists():
        raise FileNotFoundError(args.target_checkpoint)
    if not args.template_config.exists():
        raise FileNotFoundError(args.template_config)
    if args.out_run_dir.exists():
        shutil.rmtree(args.out_run_dir)
    rewrite_config(args.template_config, args.out_run_dir)
    out_checkpoint = interpolate(args)
    print(out_checkpoint, flush=True)


if __name__ == "__main__":
    main()
