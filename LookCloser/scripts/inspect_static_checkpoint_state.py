#!/usr/bin/env python3
"""Inspect optimizer, scheduler, and AMP exposure in trusted LookCloser checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    return str(value)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash a checkpoint RNG byte tensor without serializing Python metadata."""

    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def object_sha256(value: Any) -> str:
    """Hash Python/NumPy RNG metadata using a fixed local pickle protocol."""

    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def inspect(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    step = int(checkpoint["step"])
    pipeline = checkpoint.get("pipeline", {})
    optimizer = checkpoint["optimizers"]["fields"]
    adam_steps = sorted(
        {int(scalar(state["step"])) for state in optimizer["state"].values() if "step" in state}
    )
    scheduler = checkpoint["schedulers"]["fields"]
    scaler = checkpoint.get("scalers", {})
    rng_state = checkpoint.get("rng_state")
    rng_summary = None
    if isinstance(rng_state, dict):
        torch_cpu = rng_state.get("torch_cpu")
        torch_cuda = rng_state.get("torch_cuda", [])
        if isinstance(torch_cpu, torch.Tensor) and isinstance(torch_cuda, (list, tuple)):
            rng_summary = {
                "python_sha256": object_sha256(rng_state.get("python")),
                "numpy_sha256": object_sha256(rng_state.get("numpy")),
                "torch_cpu_sha256": tensor_sha256(torch_cpu),
                "torch_cuda_sha256": [
                    tensor_sha256(state) for state in torch_cuda if isinstance(state, torch.Tensor)
                ],
            }
    optimizer_updates = min(adam_steps) if adam_steps else None
    training_iterations = step + 1
    return {
        "checkpoint": str(path),
        "bytes": path.stat().st_size,
        "trainer_step": step,
        "cumulative_point_samples": scalar(pipeline.get("cumulative_point_samples")),
        "fas_sample_count_state": scalar(pipeline.get("fas_sample_count_state")),
        "adam_steps": adam_steps,
        "optimizer_updates": optimizer_updates,
        "trainer_optimizer_gap": None if optimizer_updates is None else step - optimizer_updates,
        "training_iterations": training_iterations,
        "skipped_optimizer_updates": (
            None if optimizer_updates is None else training_iterations - optimizer_updates
        ),
        "optimizer_lrs": [scalar(group.get("lr")) for group in optimizer["param_groups"]],
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_step_count": int(scheduler["_step_count"]),
        "scheduler_last_lrs": scalar(scheduler.get("_last_lr")),
        "grad_scaler_scale": scalar(scaler.get("scale")),
        "grad_scaler_growth_tracker": scalar(scaler.get("_growth_tracker")),
        "grad_scaler_growth_factor": scalar(scaler.get("growth_factor")),
        "grad_scaler_backoff_factor": scalar(scaler.get("backoff_factor")),
        "grad_scaler_growth_interval": scalar(scaler.get("growth_interval")),
        "rng_state_present": isinstance(rng_state, dict),
        "rng_state": rng_summary,
    }


def main() -> int:
    args = parse_args()
    results = [inspect(path) for path in args.checkpoints]
    payload = {"checkpoints": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "| Checkpoint | Trainer step | Adam updates | Gap | Scheduler epoch | LR | "
        "Skipped updates | AMP scale | AMP growth interval | AMP tracker |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['checkpoint']} | {row['trainer_step']} | {row['optimizer_updates']} | "
            f"{row['trainer_optimizer_gap']} | {row['scheduler_last_epoch']} | "
            f"{row['optimizer_lrs'][0]:.9g} | {row['skipped_optimizer_updates']} | "
            f"{row['grad_scaler_scale']:.9g} | "
            f"{row['grad_scaler_growth_interval']} | {row['grad_scaler_growth_tracker']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
