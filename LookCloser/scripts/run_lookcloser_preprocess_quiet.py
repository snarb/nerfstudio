#!/usr/bin/env python3
"""Quiet LookCloser frequency-map preprocessing runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_LOG = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/lookcloser_preprocess_stdout.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-name", default="lookcloser_frequencies")
    parser.add_argument("--train-steps-per-level", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=8192)
    parser.add_argument("--ssim-threshold", type=float, default=0.95)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument("--high-frequency-level", type=int, default=13)
    parser.add_argument("--n-levels", type=int, default=16)
    parser.add_argument("--min-res", type=int, default=16)
    parser.add_argument("--max-res", type=int, default=None)
    parser.add_argument("--max-res-base", type=int, default=2048)
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--scale-factor", type=float, default=1.15)
    parser.add_argument("--center-method", default="focus")
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--debug-save", action="store_true")
    parser.add_argument("--debug-max-images", type=int, default=2)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def preprocess_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "nerfstudio.scripts.lookcloser_preprocess",
        "--output-name",
        args.output_name,
        "--train-steps-per-level",
        str(args.train_steps_per_level),
        "--train-batch-size",
        str(args.train_batch_size),
        "--ssim-threshold",
        str(args.ssim_threshold),
        "--patch-size",
        str(args.patch_size),
        "--ssim-window-size",
        str(args.ssim_window_size),
        "--high-frequency-level",
        str(args.high_frequency_level),
        "--n-levels",
        str(args.n_levels),
        "--min-res",
        str(args.min_res),
        "--max-res-base",
        str(args.max_res_base),
    ]
    if args.max_res is not None:
        cmd.extend(["--max-res", str(args.max_res)])
    if args.debug_save:
        cmd.extend(["--debug-save", "--debug-max-images", str(args.debug_max_images)])
    if args.force_recompute:
        cmd.append("--force-recompute")
    cmd.extend(
        [
            "nerfstudio-data",
            "--data",
            str(args.data),
            "--eval-mode",
            args.eval_mode,
            "--eval-interval",
            str(args.eval_interval),
            "--orientation-method",
            args.orientation_method,
            "--center-method",
            args.center_method,
            "--auto-scale-poses",
            "--scene-scale",
            str(args.scene_scale),
            "--downscale-factor",
            "1",
        ]
    )
    if args.scale_factor is not None:
        cmd.extend(["--scale-factor", str(args.scale_factor)])
    return cmd


def main() -> int:
    args = parse_args()
    cmd = preprocess_command(args)
    freq_dir = args.data / args.output_name
    print(f"data={args.data}", flush=True)
    print(f"frequency_dir={freq_dir}", flush=True)
    print(f"log={args.log_path}", flush=True)
    print(f"command={' '.join(cmd)}", flush=True)
    if args.dry_run:
        return 0
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.monotonic() - started
    pt_count = len(list(freq_dir.glob("*.pt"))) if freq_dir.exists() else 0
    json_count = len(list(freq_dir.glob("*.json"))) if freq_dir.exists() else 0
    print(f"preprocess_exit={proc.returncode}", flush=True)
    print(f"preprocess_seconds={seconds:.3f}", flush=True)
    print(f"frequency_maps={pt_count}", flush=True)
    print(f"frequency_metadata={json_count}", flush=True)
    print(f"log={args.log_path}", flush=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
