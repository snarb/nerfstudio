#!/usr/bin/env python3
"""Batch frequency-map preprocessor for the temporal per-frame dataset.

Iterates over all frame subdirectories in the temporal dataset root and runs
run_lookcloser_preprocess_quiet.py for each frame, using the same hyperparameters
that were used for the static leader (007740, PSNR 29.618).

Usage:
    # Dry-run to verify commands:
    python LookCloser/scripts/run_temporal_preprocess.py --dry-run

    # Run all frames sequentially (run inside tmux):
    python LookCloser/scripts/run_temporal_preprocess.py

    # Resume from a specific frame (skip already-done frames):
    python LookCloser/scripts/run_temporal_preprocess.py --start-from 007747
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


TEMPORAL_ROOT = Path(
    "/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/temporal_perframe_stride7_45f"
)
LOG_DIR = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs/temporal_preprocess")
PREPROCESS_RUNNER = Path(__file__).parent / "run_lookcloser_preprocess_quiet.py"

# Hyperparameters matching the static 007740 leader preprocessing exactly.
STATIC_LEADER_PARAMS = {
    "--output-name": "lookcloser_frequencies",
    "--train-steps-per-level": "1000",
    "--train-batch-size": "8192",
    "--ssim-threshold": "0.95",
    "--patch-size": "8",
    "--ssim-window-size": "7",
    "--high-frequency-level": "13",
    "--n-levels": "16",
    "--min-res": "16",
    "--max-res-base": "2048",
    "--scene-scale": "2.0",
    "--scale-factor": "1.15",
    "--center-method": "focus",
    "--orientation-method": "up",
    "--eval-mode": "filename",
}

EXPECTED_MAPS_PER_FRAME = 66


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-root", type=Path, default=TEMPORAL_ROOT,
                   help="Root directory containing per-frame subdirectories.")
    p.add_argument("--log-dir", type=Path, default=LOG_DIR,
                   help="Directory for per-frame log files.")
    p.add_argument("--start-from", default=None, metavar="FRAME_ID",
                   help="Skip frames before this frame ID (e.g. '007747' to resume after 007740).")
    p.add_argument("--only-frame", default=None, metavar="FRAME_ID",
                   help="Process only this single frame (useful for verification).")
    p.add_argument("--force-recompute", action="store_true",
                   help="Pass --force-recompute to the preprocessor (re-run even if maps exist).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    return p.parse_args()


def get_frame_dirs(root: Path) -> list[Path]:
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    return dirs


def maps_complete(frame_dir: Path, output_name: str, expected: int) -> bool:
    freq_dir = frame_dir / output_name
    if not freq_dir.exists():
        return False
    return len(list(freq_dir.glob("*.pt"))) >= expected


def build_command(frame_dir: Path, log_path: Path, force_recompute: bool) -> list[str]:
    cmd = [sys.executable, str(PREPROCESS_RUNNER)]
    for flag, value in STATIC_LEADER_PARAMS.items():
        cmd.extend([flag, value])
    cmd.extend(["--data", str(frame_dir)])
    cmd.extend(["--log-path", str(log_path)])
    if force_recompute:
        cmd.append("--force-recompute")
    return cmd


def main() -> int:
    args = parse_args()
    all_frames = get_frame_dirs(args.frames_root)
    if not all_frames:
        print(f"ERROR: no frame directories found in {args.frames_root}", flush=True)
        return 1

    output_name = STATIC_LEADER_PARAMS["--output-name"]

    if args.only_frame:
        frames = [args.frames_root / args.only_frame]
        if not frames[0].exists():
            print(f"ERROR: frame dir not found: {frames[0]}", flush=True)
            return 1
    else:
        frames = all_frames
        if args.start_from:
            start_ids = [f.name for f in frames]
            if args.start_from not in start_ids:
                print(f"ERROR: --start-from {args.start_from!r} not found in frames: {start_ids}", flush=True)
                return 1
            frames = [f for f in frames if f.name >= args.start_from]

    args.log_dir.mkdir(parents=True, exist_ok=True)

    total = len(frames)
    print(f"frames_root={args.frames_root}", flush=True)
    print(f"log_dir={args.log_dir}", flush=True)
    print(f"frames_to_process={total} (of {len(all_frames)} total)", flush=True)
    print(f"output_name={output_name}", flush=True)
    print(f"dry_run={args.dry_run}", flush=True)
    print("", flush=True)

    batch_start = time.monotonic()
    skipped = 0
    done = 0
    failed_frames: list[str] = []

    for i, frame_dir in enumerate(frames, 1):
        frame_id = frame_dir.name

        if not args.force_recompute and maps_complete(frame_dir, output_name, EXPECTED_MAPS_PER_FRAME):
            pt_count = len(list((frame_dir / output_name).glob("*.pt")))
            print(f"[{i}/{total}] frame={frame_id} SKIP (already has {pt_count} maps)", flush=True)
            skipped += 1
            continue

        log_path = args.log_dir / f"{frame_id}_preprocess.log"
        cmd = build_command(frame_dir, log_path, args.force_recompute)

        print(f"[{i}/{total}] frame={frame_id} starting ...", flush=True)
        if args.dry_run:
            print(f"  CMD: {' '.join(cmd)}", flush=True)
            continue

        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=False)
        elapsed = time.monotonic() - t0

        pt_count = len(list((frame_dir / output_name).glob("*.pt"))) if (frame_dir / output_name).exists() else 0
        status = "OK" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
        print(
            f"[{i}/{total}] frame={frame_id} {status} maps={pt_count} time={elapsed:.0f}s log={log_path}",
            flush=True,
        )

        if proc.returncode != 0:
            failed_frames.append(frame_id)
        else:
            done += 1

    total_elapsed = time.monotonic() - batch_start
    print("", flush=True)
    print(f"=== DONE: processed={done} skipped={skipped} failed={len(failed_frames)} "
          f"total_time={total_elapsed:.0f}s ===", flush=True)
    if failed_frames:
        print(f"FAILED frames: {failed_frames}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
