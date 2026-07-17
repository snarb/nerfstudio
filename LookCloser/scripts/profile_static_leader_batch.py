#!/usr/bin/env python3
"""Solo mature-checkpoint throughput profile for a frozen leader batch scale."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import run_static_leader_e2e as leader


DEFAULT_CHECKPOINT = Path(
    "/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/lookcloser/"
    "20260715_005006/nerfstudio_models/step-000091128.ckpt"
)
DEFAULT_OUTPUT = Path("/home/brans/lookcloser_leader_speed_profiles")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-scale", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument(
        "--train-rays-per-batch",
        type=int,
        default=None,
        help="Optional intermediate ray batch; point-normalized cadence is derived from 4096 rays.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--discard-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--staged-speed",
        action="store_true",
        help="Profile the reviewed static-cache/fused/JIT/replay step-91128 recipe.",
    )
    parser.add_argument(
        "--cpu-fas-prefetch",
        action="store_true",
        help="Enable the default-off one-batch CPU FAS prefetch on the staged recipe.",
    )
    parser.add_argument(
        "--load-training-state",
        action="store_true",
        help="Restore Adam, scheduler and scaler instead of resetting them for throughput-only profiling.",
    )
    parser.add_argument("--skip-provenance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def gpu_sample() -> dict[str, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    utilization, memory, power = (float(value.strip()) for value in output.split(","))
    return {"utilization_percent": utilization, "memory_mib": memory, "power_w": power}


def metric_rows(path: Path, minimum_step: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            step = int(raw["step"])
            if step < minimum_step or not raw.get("iter_time_s") or not raw.get("train_num_samples_per_batch"):
                continue
            rows.append(
                {
                    "step": float(step),
                    "iter_time_s": float(raw["iter_time_s"]),
                    "point_samples": float(raw["train_num_samples_per_batch"]),
                    "cumulative_point_samples": float(raw.get("cumulative_point_samples") or 0),
                    "gpu_mem_mb": float(raw.get("gpu_mem_mb") or 0),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_steps <= args.discard_steps:
        raise ValueError("--num-steps must exceed --discard-steps")
    if args.cpu_fas_prefetch and not args.staged_speed:
        raise ValueError("--cpu-fas-prefetch requires --staged-speed")
    if args.staged_speed and args.batch_scale != 1:
        raise ValueError("--staged-speed preserves the reviewed fixed B4096 recipe")
    train_rays = args.train_rays_per_batch or 4096 * args.batch_scale
    if train_rays < 4096 or train_rays > 16384 or train_rays % 256:
        raise ValueError("--train-rays-per-batch must be a multiple of 256 in [4096, 16384]")
    point_scale = train_rays / 4096.0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = args.name or f"batch{train_rays}_{timestamp}"
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=False)

    leader_argv = [
            "--historical-worktree",
            str(leader.DEFAULT_SPEED_WORKTREE),
            "--output-dir",
            str(args.output_dir),
            "--eval-num-rays-per-chunk",
            "16384",
            "--poll-seconds",
            "1",
            "--skip-provenance",
        ]
    if args.staged_speed:
        leader_argv.extend(
            [
                "--tcnn-overlay",
                str(leader.DEFAULT_JIT_TCNN_OVERLAY),
                "--cache-train-rays",
                "--fused-adam-switch-step",
                "15189",
                "--tcnn-network-jit-switch-step",
                "15189",
                "--tcnn-network-jit-scope",
                "color",
                "--tcnn-network-jit-second-switch-step",
                "30377",
                "--tcnn-network-jit-second-switch-scope",
                "geometry",
                "--replay-eval-trajectory",
                "--historical-stage-boundary-rng-reset",
                "--speed-final-step",
                "91128",
            ]
        )
        if args.cpu_fas_prefetch:
            leader_argv.append("--cpu-fas-prefetch")
    else:
        leader_argv.extend(
            [
                "--batch-scale",
                str(args.batch_scale),
                "--speed-stop-at-accepted-boundary",
            ]
        )
    leader_args = leader.parse_args(leader_argv)
    recipe = leader.resolve_recipe(leader_args)
    env = leader.historical_environment(leader_args)

    # This performs the same commit/dirty-path/source/TCNN/runtime validation as a full speed run.
    if args.staged_speed:
        preflight_command = [
            str(leader_args.venv / "bin" / "python"),
            str(Path(__file__).with_name("run_static_leader_e2e.py")),
            *leader_argv,
            "--campaign-name",
            f"{name}_preflight",
            "--dry-run",
        ]
    else:
        preflight_command = [
            str(leader_args.venv / "bin" / "python"),
            str(Path(__file__).with_name("run_static_leader_speed_e2e.py")),
            "--batch-scale",
            str(args.batch_scale),
            "--campaign-name",
            f"{name}_preflight",
            "--dry-run",
            "--skip-provenance",
        ]
    preflight = json.loads(subprocess.check_output(preflight_command, text=True, env=env))
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")

    if not args.skip_provenance:
        provenance_command = [
            str(leader_args.venv / "bin" / "python"),
            str(leader.DEFAULT_PROVENANCE_SCRIPT),
            "--local",
            str(leader_args.data),
            "--output",
            str(output / "dataset_provenance.json"),
        ]
        subprocess.run(provenance_command, check=True, env=env, stdout=subprocess.DEVNULL)

    checkpoint_step = int(args.checkpoint.stem.rsplit("-", 1)[-1])
    experiment = f"profile_{name}"
    command = leader.common_runner_args(leader_args, recipe, 42, experiment, timestamp)
    replace_option(command, "--train-num-rays-per-batch", str(train_rays))
    replace_option(command, "--adaptive-warmup-steps", str(round(4096 / point_scale)))
    replace_option(command, "--occupancy-warmup-steps", str(round(4096 / point_scale)))
    replace_option(command, "--occupancy-binary-warmup-steps", str(round(4096 / point_scale)))
    replace_option(command, "--occupancy-update-interval", str(max(1, round(16 / point_scale))))
    replace_option(command, "--grid-update-interval", str(max(1, round(1024 / point_scale))))
    replace_option(command, "--depth-loss-steps", str(round(5000 / point_scale)))
    replace_option(command, "--fields-scheduler-max-steps", str(round(200000 / point_scale)))
    replace_option(command, "--step-interval", "10")
    replace_option(command, "--save-interval", "1000000")
    command.extend(
        [
            "--eval-batch-interval",
            "1000000",
            "--eval-image-interval",
            "1000000",
            "--eval-all-interval",
            "1000000",
            "--feature-reweighting-strength",
            "0.3",
            "--max-num-iterations",
            str(checkpoint_step + args.num_steps + 1),
            "--load-checkpoint",
            str(args.checkpoint),
            "--no-render-final",
        ]
    )
    if not args.load_training_state:
        command.extend(["--no-load-optimizers", "--no-load-scheduler"])
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"output": str(output), "recipe": leader.asdict(recipe), "command": command}, indent=2))
        return 0

    log_path = output / "controller_stdout.log"
    samples: list[dict[str, float]] = []
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                samples.append(gpu_sample())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(0.25)
    controller_seconds = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(f"Profile runner exited {process.returncode}; see {log_path}")

    run_path = leader.run_path(args.output_dir, experiment, timestamp)
    metrics_path = run_path / "metrics_compact.csv"
    rows = metric_rows(metrics_path, checkpoint_step + args.discard_steps)
    if not rows:
        raise RuntimeError(f"No mature timing rows found in {metrics_path}")
    iter_times = [row["iter_time_s"] for row in rows]
    point_counts = [row["point_samples"] for row in rows]
    per_row_throughput = [points / seconds for points, seconds in zip(point_counts, iter_times)]
    run_summary_path = run_path / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    result = {
        "name": name,
        "batch_scale": point_scale,
        "train_rays_per_batch": train_rays,
        "staged_speed": bool(args.staged_speed),
        "cpu_fas_prefetch": bool(args.cpu_fas_prefetch),
        "load_training_state": bool(args.load_training_state),
        "resolved_recipe": leader.asdict(recipe),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "profile_steps": args.num_steps,
        "discard_steps": args.discard_steps,
        "timing_row_count": len(rows),
        "median_iter_time_s": statistics.median(iter_times),
        "p95_iter_time_s": sorted(iter_times)[max(0, int(0.95 * len(iter_times)) - 1)],
        "median_point_samples_per_step": statistics.median(point_counts),
        "median_point_samples_per_second": statistics.median(per_row_throughput),
        "peak_memory_mib": max((sample["memory_mib"] for sample in samples), default=0.0),
        "median_gpu_utilization_percent": statistics.median(
            [sample["utilization_percent"] for sample in samples]
        ) if samples else None,
        "peak_power_w": max((sample["power_w"] for sample in samples), default=0.0),
        "runner_train_seconds": float(run_summary["train_seconds"]),
        "controller_seconds": controller_seconds,
        "metrics": str(metrics_path),
        "run_summary": str(run_summary_path),
        "run_path": str(run_path),
        "gpu_samples": samples,
    }
    (output / "profile.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "gpu_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
