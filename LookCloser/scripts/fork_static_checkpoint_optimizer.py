#!/usr/bin/env python3
"""Create a provenance-recorded optimizer/scheduler fork of a trusted checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--scheduler-time-scale",
        type=float,
        default=1.0,
        help="Scale loaded scheduler epoch while preserving its current LR (for point-time batch changes).",
    )
    parser.add_argument("--reset-adam", action="store_true")
    parser.add_argument("--restart-scheduler", action="store_true")
    parser.add_argument("--reset-scaler", action="store_true")
    parser.add_argument(
        "--reset-torch-cpu-rng-seed",
        type=int,
        default=None,
        help="Replace only the checkpointed Torch CPU RNG stream; preserve CUDA/Python/NumPy streams.",
    )
    parser.add_argument(
        "--drop-rng-state",
        action="store_true",
        help=(
            "Remove the persisted RNG snapshot so a resumed trainer retains the newly seeded "
            "post-setup process streams, matching a historical checkpoint restart."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return str(value)


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def snapshot(checkpoint: dict[str, Any]) -> dict[str, Any]:
    optimizer = checkpoint["optimizers"]["fields"]
    scheduler = checkpoint["schedulers"]["fields"]
    adam_steps = sorted(
        {int(native(state["step"])) for state in optimizer["state"].values() if "step" in state}
    )
    rng_state = checkpoint.get("rng_state")
    rng_summary = None
    if isinstance(rng_state, dict):
        rng_summary = {
            "torch_cpu_sha256": tensor_sha256(rng_state["torch_cpu"]),
            "torch_cuda_sha256": [tensor_sha256(state) for state in rng_state.get("torch_cuda", [])],
        }
    return {
        "trainer_step": int(checkpoint["step"]),
        "adam_steps": adam_steps,
        "adam_state_entries": len(optimizer["state"]),
        "optimizer_lrs": [native(group["lr"]) for group in optimizer["param_groups"]],
        "optimizer_initial_lrs": [native(group.get("initial_lr")) for group in optimizer["param_groups"]],
        "scheduler_base_lrs": native(scheduler.get("base_lrs")),
        "scheduler_last_lrs": native(scheduler.get("_last_lr")),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_step_count": int(scheduler["_step_count"]),
        "grad_scaler": {key: native(value) for key, value in checkpoint.get("scalers", {}).items()},
        "rng_state_present": any("rng" in key.lower() for key in checkpoint),
        "rng_state": rng_summary,
    }


def main() -> int:
    args = parse_args()
    if args.lr_multiplier <= 0:
        raise ValueError("--lr-multiplier must be positive")
    if args.scheduler_time_scale <= 0:
        raise ValueError("--scheduler-time-scale must be positive")
    if args.restart_scheduler and args.scheduler_time_scale != 1.0:
        raise ValueError("--scheduler-time-scale cannot be combined with --restart-scheduler")
    if args.drop_rng_state and args.reset_torch_cpu_rng_seed is not None:
        raise ValueError("--drop-rng-state cannot be combined with --reset-torch-cpu-rng-seed")
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("Refusing to overwrite the source checkpoint")
    if args.output.exists():
        raise FileExistsError(args.output)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    before = snapshot(checkpoint)
    optimizer = checkpoint["optimizers"]["fields"]
    scheduler = checkpoint["schedulers"]["fields"]

    for group in optimizer["param_groups"]:
        group["lr"] = native(group["lr"]) * args.lr_multiplier
        if group.get("initial_lr") is not None:
            group["initial_lr"] = native(group["initial_lr"]) * args.lr_multiplier
    scheduler["base_lrs"] = [native(lr) * args.lr_multiplier for lr in scheduler["base_lrs"]]
    scheduler["_last_lr"] = [native(lr) * args.lr_multiplier for lr in scheduler["_last_lr"]]

    if args.scheduler_time_scale != 1.0:
        old_epoch = int(scheduler["last_epoch"])
        old_step_count = int(scheduler["_step_count"])
        new_epoch = int(round(old_epoch * args.scheduler_time_scale))
        scheduler["last_epoch"] = new_epoch
        # PyTorch normally keeps _step_count one ahead of last_epoch. Preserve
        # the observed offset instead of scaling it and introducing a fake gap.
        scheduler["_step_count"] = new_epoch + (old_step_count - old_epoch)

    if args.reset_adam:
        optimizer["state"] = {}
    if args.restart_scheduler:
        current_lrs = [native(group["lr"]) for group in optimizer["param_groups"]]
        for group, lr in zip(optimizer["param_groups"], current_lrs):
            group["initial_lr"] = lr
        scheduler["base_lrs"] = current_lrs
        scheduler["last_epoch"] = -1
        scheduler["_step_count"] = 0
        scheduler["_get_lr_called_within_step"] = False
        scheduler["_last_lr"] = current_lrs
    if args.reset_scaler:
        checkpoint["scalers"] = {
            "scale": 65_536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2_000,
            "_growth_tracker": 0,
        }
    if args.reset_torch_cpu_rng_seed is not None:
        if "rng_state" not in checkpoint or "torch_cpu" not in checkpoint["rng_state"]:
            raise ValueError("Checkpoint has no persisted Torch CPU RNG state to reset selectively")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(args.reset_torch_cpu_rng_seed))
        checkpoint["rng_state"]["torch_cpu"] = generator.get_state()
    if args.drop_rng_state:
        if "rng_state" not in checkpoint:
            raise ValueError("Checkpoint has no persisted RNG state to drop")
        del checkpoint["rng_state"]

    after = snapshot(checkpoint)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output),
        "actions": {
            "lr_multiplier": args.lr_multiplier,
            "scheduler_time_scale": args.scheduler_time_scale,
            "reset_adam": args.reset_adam,
            "restart_scheduler": args.restart_scheduler,
            "reset_scaler": args.reset_scaler,
            "reset_torch_cpu_rng_seed": args.reset_torch_cpu_rng_seed,
            "drop_rng_state": args.drop_rng_state,
        },
        "before": before,
        "after": after,
        "note": (
            "Legacy checkpoints omit RNG state; --drop-rng-state deliberately reproduces their "
            "new-process post-setup stream instead of an exact continuation."
        ),
        "status": "dry_run" if args.dry_run else "initialized",
    }
    if args.dry_run:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    provenance["source_sha256"] = sha256_file(args.checkpoint)
    provenance["output_sha256"] = sha256_file(args.output)
    provenance["status"] = "complete"
    sidecar = args.output.with_suffix(args.output.suffix + ".fork.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
