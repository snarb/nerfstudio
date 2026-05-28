#!/usr/bin/env python3
"""Quiet bounded Instant-NGP experiment runner for LookCloser datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_OUTPUT = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs")
DEFAULT_EXPERIMENT = "007740_hd_aabb4_multicamera_eval3_ns_focus_scene15"
DEFAULT_SUMMARY = Path(__file__).resolve().parents[1] / "experiments" / "bounded_ngp_param_sweep.md"


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-interval", type=int, default=15188)
    parser.add_argument("--max-num-iterations", type=int, default=60752)
    parser.add_argument("--train-num-rays-per-batch", type=int, default=8192)
    parser.add_argument("--background-color", choices=("random", "black", "white"), default="black")
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1000.0)
    parser.add_argument("--alpha-thre", type=float, default=0.0)
    parser.add_argument("--cone-angle", type=float, default=0.0)
    parser.add_argument("--render-step-size", type=float, default=None)
    parser.add_argument("--render-step-size-mult", type=float, default=None)
    parser.add_argument("--loss-type", choices=("mse", "instant_ngp_huber"), default="mse")
    parser.add_argument("--use-gradient-scaling", action="store_true")
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--no-update-summary", dest="update_summary", action="store_false")
    parser.add_argument("--eval-checkpoint", choices=("best", "latest"), default="best")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.set_defaults(stop_on_no_improve=True, render_final=True, update_summary=True)
    parser.add_argument("--no-stop-on-no-improve", dest="stop_on_no_improve", action="store_false")
    parser.add_argument("--no-render-final", dest="render_final", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.experiment_name / "instant-ngp-bounded" / args.timestamp


def train_command(args: argparse.Namespace) -> List[str]:
    interval = str(args.step_interval)
    cmd = [
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
        "--machine.seed",
        str(args.seed),
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
    ]
    if args.load_dir is not None:
        cmd.extend(["--load-dir", str(args.load_dir)])
    if args.load_step is not None:
        cmd.extend(["--load-step", str(args.load_step)])
    if args.load_checkpoint is not None:
        cmd.extend(["--load-checkpoint", str(args.load_checkpoint)])
    cmd.extend(
        [
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
        "--pipeline.datamanager.train-num-rays-per-batch",
        str(args.train_num_rays_per_batch),
        "--pipeline.model.background-color",
        args.background_color,
        "--pipeline.model.near-plane",
        str(args.near_plane),
        "--pipeline.model.far-plane",
        str(args.far_plane),
        "--pipeline.model.alpha-thre",
        str(args.alpha_thre),
        "--pipeline.model.cone-angle",
        str(args.cone_angle),
        "--pipeline.model.loss-type",
        args.loss_type,
        "--pipeline.model.use-gradient-scaling",
        str(args.use_gradient_scaling),
        ]
    )
    render_step_size = resolved_render_step_size(args)
    if render_step_size is not None:
        cmd.extend(["--pipeline.model.render-step-size", str(render_step_size)])
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
            "True",
            "--scene-scale",
            str(args.scene_scale),
            "--downscale-factor",
            "1",
        ]
    )
    if args.scale_factor is not None:
        cmd.extend(["--scale-factor", str(args.scale_factor)])
    return cmd


def resolved_render_step_size(args: argparse.Namespace) -> Optional[float]:
    if args.render_step_size is not None and args.render_step_size_mult is not None:
        raise ValueError("Use only one of --render-step-size or --render-step-size-mult")
    if args.render_step_size is not None:
        return args.render_step_size
    if args.render_step_size_mult is None:
        return None
    return (2.0 * args.scene_scale * math.sqrt(3.0) / 1000.0) * args.render_step_size_mult


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


def checkpoint_step(checkpoint: Path) -> int:
    return int(checkpoint.stem.split("-")[-1])


def best_eval_checkpoint(metrics_path: Path, model_dir: Path) -> Tuple[Optional[Path], str]:
    rows = eval_rows(metrics_path)
    checkpoints = sorted(model_dir.glob("step-*.ckpt"))
    if not checkpoints:
        return None, "missing"
    if not rows:
        return checkpoints[-1], "latest_no_eval_rows"

    best_row = min(rows, key=lambda row: float(row["eval_loss"]))
    target_step = int(best_row["step"])
    by_step = {checkpoint_step(ckpt): ckpt for ckpt in checkpoints}
    if target_step in by_step:
        return by_step[target_step], f"best_eval_loss_step_{target_step}"

    earlier_or_equal = [step for step in by_step if step <= target_step]
    if earlier_or_equal:
        step = max(earlier_or_equal)
        return by_step[step], f"nearest_saved_checkpoint_for_best_eval_loss_step_{target_step}"
    return checkpoints[-1], f"latest_no_checkpoint_for_best_eval_loss_step_{target_step}"


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


def eval_config_for_step(config: Path, checkpoint: Path) -> Path:
    step = checkpoint_step(checkpoint)
    eval_config = config.with_name(f"eval_config_step_{step}.yml")
    text = config.read_text(encoding="utf-8")
    if re.search(r"^load_step:", text, flags=re.MULTILINE):
        text = re.sub(r"^load_step:.*$", f"load_step: {step}", text, count=1, flags=re.MULTILINE)
    else:
        text = text.replace("load_scheduler:", f"load_step: {step}\nload_scheduler:", 1)
    eval_config.write_text(text, encoding="utf-8")
    return eval_config


def run_final_eval(run_path: Path, checkpoint: Path, eval_label: str) -> Dict[str, object]:
    config = run_path / "config.yml"
    eval_config = eval_config_for_step(config, checkpoint)
    output_json = run_path / f"eval_{eval_label}_{checkpoint.stem}.json"
    render_dir = run_path / f"renders_{eval_label}_{checkpoint.stem}"
    log_path = run_path / "eval_stdout.log"
    cmd = [
        "ns-eval",
        "--load-config",
        str(eval_config),
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
    return {
        "checkpoint": data["checkpoint"],
        "results": results,
        "render_dir": str(render_dir),
        "eval_json": str(output_json),
        "eval_log": str(log_path),
        "eval_config": str(eval_config),
    }


def summarize_params(args: argparse.Namespace) -> str:
    params = {
        "seed": args.seed,
        "scene_scale": args.scene_scale,
        "center_method": args.center_method,
        "orientation_method": args.orientation_method,
        "scale_factor": args.scale_factor,
        "train_num_rays_per_batch": args.train_num_rays_per_batch,
        "background_color": args.background_color,
        "near_plane": args.near_plane,
        "far_plane": args.far_plane,
        "alpha_thre": args.alpha_thre,
        "cone_angle": args.cone_angle,
        "render_step_size": resolved_render_step_size(args),
        "render_step_size_mult": args.render_step_size_mult,
        "loss_type": args.loss_type,
        "use_gradient_scaling": args.use_gradient_scaling,
        "load_dir": str(args.load_dir) if args.load_dir is not None else None,
        "load_step": args.load_step,
        "load_checkpoint": str(args.load_checkpoint) if args.load_checkpoint is not None else None,
    }
    return json.dumps(params, sort_keys=True)


def update_summary(args: argparse.Namespace, run_path: Path, selection: str, eval_data: Dict[str, object]) -> None:
    summary_path = args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        summary_path.write_text(
            "# Bounded Instant-NGP Parameter Sweep\n\n"
            "Winner selection: lowest combined rank of PSNR descending and SSIM descending; LPIPS is a secondary tie-breaker.\n\n"
            "| Timestamp | Selection | Params | Checkpoint | PSNR | SSIM | LPIPS | Eval JSON | Renders |\n"
            "|---|---|---|---|---:|---:|---:|---|---|\n",
            encoding="utf-8",
        )

    results = eval_data["results"]
    assert isinstance(results, dict)
    row = (
        f"| {args.timestamp} "
        f"| {selection} "
        f"| `{summarize_params(args)}` "
        f"| `{eval_data['checkpoint']}` "
        f"| {float(results.get('psnr')):.6f} "
        f"| {float(results.get('ssim')):.6f} "
        f"| {float(results.get('lpips')):.6f} "
        f"| `{eval_data['eval_json']}` "
        f"| `{eval_data['render_dir']}` |\n"
    )
    text = summary_path.read_text(encoding="utf-8")
    ranking_marker = "\n## Scene Scale Ranking\n"
    if ranking_marker in text:
        text = text.replace(ranking_marker, row + ranking_marker, 1)
        summary_path.write_text(text, encoding="utf-8")
    else:
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(row)
    print(f"summary={summary_path}", flush=True)


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
    best_ckpt, best_selection = best_eval_checkpoint(metrics_path, model_dir)
    if args.eval_checkpoint == "latest":
        selected_ckpt, selection = ckpt, "latest"
    else:
        selected_ckpt, selection = best_ckpt, best_selection
    print(f"train_exit={proc.returncode}", flush=True)
    print(f"latest_checkpoint={ckpt}", flush=True)
    print(f"best_eval_checkpoint={best_ckpt}", flush=True)
    print(f"best_eval_checkpoint_reason={best_selection}", flush=True)
    print(f"selected_checkpoint={selected_ckpt}", flush=True)
    print(f"selected_checkpoint_reason={selection}", flush=True)
    print(f"metrics_csv={metrics_path}", flush=True)
    print(f"train_log={train_log}", flush=True)

    if args.render_final and selected_ckpt is not None:
        eval_data = run_final_eval(run_path, selected_ckpt, args.eval_checkpoint)
        if args.update_summary:
            update_summary(args, run_path, selection, eval_data)
    if stopped_for_plateau:
        return 0
    return 0 if proc.returncode in (0, -signal.SIGINT) else int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
