#!/usr/bin/env python3
"""Quiet bounded Instant-NGP experiment runner for LookCloser datasets."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_OUTPUT = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs")
DEFAULT_EXPERIMENT = "007740_hd_aabb4_multicamera_eval3_ns_focus_scene15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--scene-scale", type=float, default=1.5)
    parser.add_argument("--center-method", default="focus")
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--step-interval", type=int, default=15188)
    parser.add_argument("--max-num-iterations", type=int, default=60752)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.set_defaults(stop_on_no_improve=True, render_final=True)
    parser.add_argument("--no-stop-on-no-improve", dest="stop_on_no_improve", action="store_false")
    parser.add_argument("--no-render-final", dest="render_final", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.experiment_name / "instant-ngp-bounded" / args.timestamp


def train_command(args: argparse.Namespace) -> List[str]:
    interval = str(args.step_interval)
    return [
        "ns-train",
        "instant-ngp-bounded",
        "--output-dir",
        str(args.output_dir),
        "--experiment-name",
        args.experiment_name,
        "--timestamp",
        args.timestamp,
        "--vis",
        "tensorboard",
        "--viewer.quit-on-train-completion",
        "True",
        "--steps-per-eval-batch",
        interval,
        "--steps-per-eval-image",
        interval,
        "--steps-per-eval-all-images",
        interval,
        "--steps-per-save",
        interval,
        "--max-num-iterations",
        str(args.max_num_iterations),
        "--save-only-latest-checkpoint",
        "False",
        "--logging.local-writer.enable",
        "False",
        "--logging.csv-writer.enable",
        "True",
        "--logging.csv-writer.write-interval",
        interval,
        "--logging.csv-writer.improvement-tolerance",
        "0.0",
        "--logging.profiler",
        "none",
        "--pipeline.datamanager.cache-images-type",
        "uint8",
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
        "True",
        "--scene-scale",
        str(args.scene_scale),
        "--downscale-factor",
        "1",
    ]


def read_csv_rows(metrics_path: Path) -> List[Dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eval_rows(metrics_path: Path) -> List[Dict[str, str]]:
    return [row for row in read_csv_rows(metrics_path) if row.get("eval_loss")]


def latest_train_step(metrics_path: Path) -> Optional[str]:
    rows = read_csv_rows(metrics_path)
    return rows[-1]["step"] if rows else None


def print_eval_row(row: Dict[str, str]) -> None:
    print(
        "eval "
        f"step={row.get('step')} "
        f"loss={row.get('eval_loss')} "
        f"psnr={row.get('eval_all_psnr')} "
        f"ssim={row.get('eval_all_ssim')} "
        f"delta={row.get('eval_loss_delta')} "
        f"status={row.get('status')}",
        flush=True,
    )


def latest_checkpoint(model_dir: Path) -> Optional[Path]:
    checkpoints = sorted(model_dir.glob("step-*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_final_eval(run_path: Path) -> None:
    config = run_path / "config.yml"
    ckpt = latest_checkpoint(run_path / "nerfstudio_models")
    if ckpt is None:
        raise RuntimeError(f"No checkpoint found in {run_path / 'nerfstudio_models'}")
    output_json = run_path / f"eval_last_{ckpt.stem}.json"
    render_dir = run_path / f"renders_last_{ckpt.stem}"
    log_path = run_path / "eval_stdout.log"
    cmd = [
        "ns-eval",
        "--load-config",
        str(config),
        "--output-path",
        str(output_json),
        "--render-output-path",
        str(render_dir),
    ]
    print(f"running final eval: {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(output_json.read_text(encoding="utf-8"))
    results = data["results"]
    print(
        "final "
        f"checkpoint={data['checkpoint']} "
        f"psnr={results.get('psnr'):.6f} "
        f"ssim={results.get('ssim'):.6f} "
        f"lpips={results.get('lpips'):.6f}",
        flush=True,
    )
    print(f"renders={render_dir}", flush=True)
    print(f"eval_json={output_json}", flush=True)
    print(f"eval_log={log_path}", flush=True)


def main() -> int:
    args = parse_args()
    run_path = run_dir(args)
    metrics_path = run_path / "metrics_compact.csv"
    train_log = run_path / "train_stdout.log"
    model_dir = run_path / "nerfstudio_models"
    cmd = train_command(args)

    print(f"data={args.data}", flush=True)
    print(f"run_dir={run_path}", flush=True)
    print(f"train_log={train_log}", flush=True)
    print(f"command={' '.join(cmd)}", flush=True)
    if args.dry_run:
        return 0

    run_path.mkdir(parents=True, exist_ok=True)
    with train_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        seen_eval_count = 0
        stopped_for_plateau = False
        try:
            while proc.poll() is None:
                time.sleep(args.poll_seconds)
                step = latest_train_step(metrics_path)
                if step is not None:
                    print(f"step={step}", flush=True)
                current_evals = eval_rows(metrics_path)
                for row in current_evals[seen_eval_count:]:
                    print_eval_row(row)
                if len(current_evals) > seen_eval_count:
                    seen_eval_count = len(current_evals)
                    if args.stop_on_no_improve and len(current_evals) >= 2:
                        prev = float(current_evals[-2]["eval_loss"])
                        last = float(current_evals[-1]["eval_loss"])
                        if last >= prev:
                            print(f"stopping: eval loss did not improve ({last:.8g} >= {prev:.8g})", flush=True)
                            stopped_for_plateau = True
                            stop_process(proc)
                            break
        except KeyboardInterrupt:
            print("interrupted: stopping train process", flush=True)
            stop_process(proc)
            raise

    ckpt = latest_checkpoint(model_dir)
    print(f"train_exit={proc.returncode}", flush=True)
    print(f"latest_checkpoint={ckpt}", flush=True)
    print(f"metrics_csv={metrics_path}", flush=True)
    print(f"train_log={train_log}", flush=True)

    if args.render_final and ckpt is not None:
        run_final_eval(run_path)
    if stopped_for_plateau:
        return 0
    return 0 if proc.returncode in (0, -signal.SIGINT) else int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
