#!/usr/bin/env python3
"""Quiet LookCloser experiment runner for the HD bounded dataset."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_EXPERIMENT = "007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid"
DEFAULT_SUMMARY = Path(__file__).resolve().parents[1] / "experiments" / "lookcloser_frequency_grid_optimization.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--scene-scale", type=float, default=2.0)
    parser.add_argument("--scale-factor", type=float, default=1.15)
    parser.add_argument("--center-method", default="focus")
    parser.add_argument("--orientation-method", default="up")
    parser.add_argument("--eval-mode", default="filename")
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-interval", type=int, default=15188)
    parser.add_argument("--eval-batch-interval", type=int, default=None)
    parser.add_argument("--eval-image-interval", type=int, default=None)
    parser.add_argument("--eval-all-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--max-num-iterations", type=int, default=60752)
    parser.add_argument("--train-num-rays-per-batch", type=int, default=4096)
    parser.add_argument("--eval-num-rays-per-batch", type=int, default=4096)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--background-color", choices=("random", "last_sample", "black", "white"), default="black")
    parser.add_argument("--reconstruction-loss-type", choices=("charbonnier", "mse", "huber"), default="charbonnier")

    parser.add_argument("--frequency-map-dir", default="lookcloser_frequencies")
    parser.add_argument("--frequency-patch-size", type=int, default=8)
    parser.add_argument("--frequency-stride", type=int, default=8)
    parser.add_argument("--allow-missing-frequency-maps", action="store_true")

    parser.add_argument("--grid-resolution", type=int, default=128)
    parser.add_argument("--num-frequency-levels", type=int, default=16)
    parser.add_argument("--min-res", type=float, default=16.0)
    parser.add_argument("--max-res", type=float, default=None)
    parser.add_argument("--max-res-base", type=float, default=2048.0)
    parser.add_argument("--fallback-frequency-level", type=float, default=0.0)
    parser.add_argument("--grid-update-interval", type=int, default=1024)
    parser.add_argument("--grid-update-batch-size", type=int, default=2048)

    parser.add_argument("--disable-frequency-grid", action="store_true")
    parser.add_argument("--disable-feature-reweighting", action="store_true")
    parser.add_argument("--disable-adaptive-ray-marching", action="store_true")
    parser.add_argument("--disable-fas", action="store_true")
    parser.add_argument("--sampling-ramp-start", type=float, default=1.0)
    parser.add_argument("--sampling-ramp-end", type=float, default=3.0)
    parser.add_argument("--fas-strength", type=float, default=1.0)
    parser.add_argument("--fas-warmup-steps", type=int, default=0)
    parser.add_argument("--fas-ramp-steps", type=int, default=0)
    parser.add_argument("--fas-level-count-alpha", type=float, default=0.0)
    parser.add_argument("--fas-patch-group-size", type=int, default=1)
    parser.add_argument("--fas-max-sampling-level", type=int, default=-1)

    parser.add_argument("--hash-features-per-level", type=int, default=2)
    parser.add_argument("--log2-hashmap-size", type=int, default=23)
    parser.add_argument("--field-hidden-dim", type=int, default=64)
    parser.add_argument("--geo-num-layers", type=int, default=1)
    parser.add_argument("--color-num-layers", type=int, default=2)
    parser.add_argument("--appearance-embedding-dim", type=int, default=0)
    parser.add_argument("--sh-degree", type=int, default=4)
    parser.add_argument("--fixed-num-samples-per-ray", type=int, default=256)
    parser.add_argument("--max-steps-per-ray", type=int, default=1024)
    parser.add_argument("--adaptive-min-step-size", type=float, default=1e-4)
    parser.add_argument("--adaptive-max-step-size", type=float, default=0.1)
    parser.add_argument("--adaptive-coarse-step-size", type=float, default=None)
    parser.add_argument("--adaptive-min-frequency-level", type=float, default=0.0)
    parser.add_argument("--adaptive-max-frequency-level", type=float, default=None)
    parser.add_argument("--adaptive-warmup-steps", type=int, default=0)
    parser.add_argument("--near-plane", type=float, default=0.02)
    parser.add_argument("--far-plane", type=float, default=1000.0)
    parser.add_argument("--alpha-thre", type=float, default=0.0025)
    parser.add_argument("--cone-angle", type=float, default=0.0)
    parser.add_argument("--render-step-size", type=float, default=None)
    parser.add_argument("--render-step-size-mult", type=float, default=0.75)
    parser.add_argument("--use-gradient-scaling", action="store_true")
    parser.add_argument("--distortion-loss-mult", type=float, default=0.01)
    parser.add_argument("--depth-loss-mult", type=float, default=0.001)
    parser.add_argument("--depth-loss-steps", type=int, default=5000)

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
    parser.add_argument("--keep-all-checkpoints", dest="prune_checkpoints", action="store_false")
    parser.set_defaults(prune_checkpoints=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.experiment_name / "lookcloser" / args.timestamp


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def train_command(args: argparse.Namespace) -> List[str]:
    interval = str(args.step_interval)
    eval_batch_interval = str(args.eval_batch_interval or args.step_interval)
    eval_image_interval = str(args.eval_image_interval or args.step_interval)
    eval_all_interval = str(args.eval_all_interval or args.step_interval)
    save_interval = str(args.save_interval or args.step_interval)
    enable_frequency_grid = not args.disable_frequency_grid
    enable_fas = enable_frequency_grid and not args.disable_fas
    cmd = [
        "ns-train",
        "lookcloser",
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
            eval_batch_interval,
            "--steps-per-eval-image",
            eval_image_interval,
            "--steps-per-eval-all-images",
            eval_all_interval,
            "--steps-per-save",
            save_interval,
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
            "--pipeline.datamanager.eval-num-rays-per-batch",
            str(args.eval_num_rays_per_batch),
            "--pipeline.datamanager.pixel-sampler.frequency-map-dir",
            args.frequency_map_dir,
            "--pipeline.datamanager.pixel-sampler.enable-fas",
            bool_text(enable_fas),
            "--pipeline.datamanager.pixel-sampler.num-levels",
            str(args.num_frequency_levels),
            "--pipeline.datamanager.pixel-sampler.min-res",
            str(args.min_res),
            "--pipeline.datamanager.pixel-sampler.max-res",
            str(args.max_res if args.max_res is not None else args.max_res_base),
            "--pipeline.datamanager.pixel-sampler.sampling-ramp-start",
            str(args.sampling_ramp_start),
            "--pipeline.datamanager.pixel-sampler.sampling-ramp-end",
            str(args.sampling_ramp_end),
            "--pipeline.datamanager.pixel-sampler.fas-strength",
            str(args.fas_strength),
            "--pipeline.datamanager.pixel-sampler.fas-warmup-steps",
            str(args.fas_warmup_steps),
            "--pipeline.datamanager.pixel-sampler.fas-ramp-steps",
            str(args.fas_ramp_steps),
            "--pipeline.datamanager.pixel-sampler.fas-level-count-alpha",
            str(args.fas_level_count_alpha),
            "--pipeline.datamanager.pixel-sampler.fas-patch-group-size",
            str(args.fas_patch_group_size),
            "--pipeline.datamanager.pixel-sampler.fas-max-sampling-level",
            str(args.fas_max_sampling_level),
            "--pipeline.datamanager.pixel-sampler.patch-size",
            str(args.frequency_patch_size),
            "--pipeline.datamanager.pixel-sampler.stride",
            str(args.frequency_stride),
            "--pipeline.frequency-map-dir",
            args.frequency_map_dir,
            "--pipeline.enable-frequency-grid",
            bool_text(enable_frequency_grid),
            "--pipeline.grid-update-interval",
            str(args.grid_update_interval),
            "--pipeline.grid-update-batch-size",
            str(args.grid_update_batch_size),
            "--pipeline.frequency-patch-size",
            str(args.frequency_patch_size),
            "--pipeline.frequency-stride",
            str(args.frequency_stride),
            "--pipeline.model.eval-num-rays-per-chunk",
            str(args.eval_num_rays_per_chunk),
            "--pipeline.model.background-color",
            args.background_color,
            "--pipeline.model.reconstruction-loss-type",
            args.reconstruction_loss_type,
            "--pipeline.model.enable-frequency-grid",
            bool_text(enable_frequency_grid),
            "--pipeline.model.grid-resolution",
            str(args.grid_resolution),
            "--pipeline.model.num-frequency-levels",
            str(args.num_frequency_levels),
            "--pipeline.model.min-res",
            str(args.min_res),
            "--pipeline.model.max-res-base",
            str(args.max_res_base),
            "--pipeline.model.fallback-frequency-level",
            str(args.fallback_frequency_level),
            "--pipeline.model.enable-feature-reweighting",
            bool_text(not args.disable_feature_reweighting),
            "--pipeline.model.hash-features-per-level",
            str(args.hash_features_per_level),
            "--pipeline.model.log2-hashmap-size",
            str(args.log2_hashmap_size),
            "--pipeline.model.field-hidden-dim",
            str(args.field_hidden_dim),
            "--pipeline.model.geo-num-layers",
            str(args.geo_num_layers),
            "--pipeline.model.color-num-layers",
            str(args.color_num_layers),
            "--pipeline.model.appearance-embedding-dim",
            str(args.appearance_embedding_dim),
            "--pipeline.model.sh-degree",
            str(args.sh_degree),
            "--pipeline.model.enable-adaptive-ray-marching",
            bool_text(not args.disable_adaptive_ray_marching),
            "--pipeline.model.fixed-num-samples-per-ray",
            str(args.fixed_num_samples_per_ray),
            "--pipeline.model.max-steps-per-ray",
            str(args.max_steps_per_ray),
            "--pipeline.model.adaptive-min-step-size",
            str(args.adaptive_min_step_size),
            "--pipeline.model.adaptive-max-step-size",
            str(args.adaptive_max_step_size),
            "--pipeline.model.adaptive-min-frequency-level",
            str(args.adaptive_min_frequency_level),
            "--pipeline.model.adaptive-warmup-steps",
            str(args.adaptive_warmup_steps),
            "--pipeline.model.near-plane",
            str(args.near_plane),
            "--pipeline.model.far-plane",
            str(args.far_plane),
            "--pipeline.model.alpha-thre",
            str(args.alpha_thre),
            "--pipeline.model.cone-angle",
            str(args.cone_angle),
            "--pipeline.model.render-step-size-mult",
            str(args.render_step_size_mult),
            "--pipeline.model.use-gradient-scaling",
            bool_text(args.use_gradient_scaling),
            "--pipeline.model.distortion-loss-mult",
            str(args.distortion_loss_mult),
            "--pipeline.model.depth-loss-mult",
            str(args.depth_loss_mult),
            "--pipeline.model.depth-loss-steps",
            str(args.depth_loss_steps),
        ]
    )
    if args.max_res is not None:
        cmd.extend(["--pipeline.model.max-res", str(args.max_res)])
    if args.render_step_size is not None:
        cmd.extend(["--pipeline.model.render-step-size", str(args.render_step_size)])
    if args.adaptive_coarse_step_size is not None:
        cmd.extend(["--pipeline.model.adaptive-coarse-step-size", str(args.adaptive_coarse_step_size)])
    if args.adaptive_max_frequency_level is not None:
        cmd.extend(["--pipeline.model.adaptive-max-frequency-level", str(args.adaptive_max_frequency_level)])

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


def check_frequency_maps(args: argparse.Namespace) -> None:
    if args.allow_missing_frequency_maps or args.disable_frequency_grid:
        return
    freq_dir = args.data / args.frequency_map_dir
    if not freq_dir.exists():
        raise FileNotFoundError(
            f"Frequency map directory missing: {freq_dir}. "
            "Run ns-process-lookcloser-freqs before LookCloser frequency-grid experiments, "
            "or pass --allow-missing-frequency-maps for a deliberate fallback smoke test."
        )
    pt_count = len(list(freq_dir.glob("*.pt")))
    if pt_count == 0:
        raise FileNotFoundError(f"No .pt frequency maps found in {freq_dir}.")


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
        f"lpips={row.get('eval_all_lpips')} "
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


def eval_config_for_step(config: Path, checkpoint: Path, eval_num_rays_per_chunk: Optional[int]) -> Path:
    step = checkpoint_step(checkpoint)
    eval_config = config.with_name(f"eval_config_step_{step}.yml")
    text = config.read_text(encoding="utf-8")
    if re.search(r"^load_step:", text, flags=re.MULTILINE):
        text = re.sub(r"^load_step:.*$", f"load_step: {step}", text, count=1, flags=re.MULTILINE)
    else:
        text = text.replace("load_scheduler:", f"load_step: {step}\nload_scheduler:", 1)
    if eval_num_rays_per_chunk is not None:
        text = re.sub(
            r"^(\s*eval_num_rays_per_chunk:\s*).*$",
            rf"\g<1>{eval_num_rays_per_chunk}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    eval_config.write_text(text, encoding="utf-8")
    return eval_config


def run_final_eval(
    run_path: Path,
    checkpoint: Path,
    eval_label: str,
    eval_num_rays_per_chunk: Optional[int],
) -> Dict[str, object]:
    config = run_path / "config.yml"
    eval_config = eval_config_for_step(config, checkpoint, eval_num_rays_per_chunk)
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
        f"psnr={format_metric(results.get('psnr'))} "
        f"ssim={format_metric(results.get('ssim'))} "
        f"lpips={format_metric(results.get('lpips'))}",
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


def format_metric(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.6f}"


def summarize_params(args: argparse.Namespace) -> str:
    params = {
        "seed": args.seed,
        "scene_scale": args.scene_scale,
        "scale_factor": args.scale_factor,
        "center_method": args.center_method,
        "orientation_method": args.orientation_method,
        "train_num_rays_per_batch": args.train_num_rays_per_batch,
        "background_color": args.background_color,
        "reconstruction_loss_type": args.reconstruction_loss_type,
        "frequency_map_dir": args.frequency_map_dir,
        "grid_resolution": args.grid_resolution,
        "num_frequency_levels": args.num_frequency_levels,
        "min_res": args.min_res,
        "max_res": args.max_res,
        "max_res_base": args.max_res_base,
        "fallback_frequency_level": args.fallback_frequency_level,
        "grid_update_interval": args.grid_update_interval,
        "grid_update_batch_size": args.grid_update_batch_size,
        "geo_num_layers": args.geo_num_layers,
        "color_num_layers": args.color_num_layers,
        "appearance_embedding_dim": args.appearance_embedding_dim,
        "enable_frequency_grid": not args.disable_frequency_grid,
        "enable_feature_reweighting": not args.disable_feature_reweighting,
        "enable_adaptive_ray_marching": not args.disable_adaptive_ray_marching,
        "enable_fas": not args.disable_fas,
        "sampling_ramp_start": args.sampling_ramp_start,
        "sampling_ramp_end": args.sampling_ramp_end,
        "fas_strength": args.fas_strength,
        "fas_warmup_steps": args.fas_warmup_steps,
        "fas_ramp_steps": args.fas_ramp_steps,
        "fas_level_count_alpha": args.fas_level_count_alpha,
        "fas_patch_group_size": args.fas_patch_group_size,
        "fas_max_sampling_level": args.fas_max_sampling_level,
        "near_plane": args.near_plane,
        "alpha_thre": args.alpha_thre,
        "cone_angle": args.cone_angle,
        "render_step_size": args.render_step_size,
        "render_step_size_mult": args.render_step_size_mult,
        "adaptive_coarse_step_size": args.adaptive_coarse_step_size,
        "adaptive_min_frequency_level": args.adaptive_min_frequency_level,
        "adaptive_max_frequency_level": args.adaptive_max_frequency_level,
        "adaptive_warmup_steps": args.adaptive_warmup_steps,
        "use_gradient_scaling": args.use_gradient_scaling,
    }
    return json.dumps(params, sort_keys=True)


def write_run_summary(
    run_path: Path,
    args: argparse.Namespace,
    train_seconds: float,
    train_returncode: Optional[int],
    selection: str,
    selected_ckpt: Optional[Path],
    eval_data: Optional[Dict[str, object]],
) -> None:
    summary = {
        "timestamp": args.timestamp,
        "params": json.loads(summarize_params(args)),
        "train_seconds": train_seconds,
        "train_returncode": train_returncode,
        "selected_checkpoint": str(selected_ckpt) if selected_ckpt is not None else None,
        "selected_checkpoint_reason": selection,
        "eval": eval_data,
    }
    (run_path / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prune_nonselected_checkpoints(model_dir: Path, selected_ckpt: Optional[Path]) -> None:
    if selected_ckpt is None or not model_dir.exists():
        return
    selected = selected_ckpt.resolve()
    removed = 0
    kept = 0
    for checkpoint in sorted(model_dir.glob("step-*.ckpt")):
        if checkpoint.resolve() == selected:
            kept += 1
            continue
        checkpoint.unlink()
        removed += 1
    print(f"checkpoint_prune kept={kept} removed={removed}", flush=True)


def update_summary(
    args: argparse.Namespace,
    run_path: Path,
    selection: str,
    eval_data: Dict[str, object],
    train_seconds: float,
) -> None:
    summary_path = args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        summary_path.write_text(
            "# LookCloser Frequency Grid Optimization\n\n"
            "| Timestamp | Selection | Train Seconds | Params | Checkpoint | PSNR | SSIM | LPIPS | Eval JSON | Renders |\n"
            "|---|---|---:|---|---|---:|---:|---:|---|---|\n",
            encoding="utf-8",
        )
    results = eval_data["results"]
    assert isinstance(results, dict)
    row = (
        f"| {args.timestamp} "
        f"| {selection} "
        f"| {train_seconds:.3f} "
        f"| `{summarize_params(args)}` "
        f"| `{eval_data['checkpoint']}` "
        f"| {format_metric(results.get('psnr'))} "
        f"| {format_metric(results.get('ssim'))} "
        f"| {format_metric(results.get('lpips'))} "
        f"| `{eval_data['eval_json']}` "
        f"| `{eval_data['render_dir']}` |\n"
    )
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

    check_frequency_maps(args)
    run_path.mkdir(parents=True, exist_ok=True)
    stopped_for_plateau = False
    train_start = time.monotonic()
    proc: Optional[subprocess.Popen] = None
    with train_log.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        seen_eval_count = 0
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
    train_seconds = time.monotonic() - train_start

    ckpt = latest_checkpoint(model_dir)
    best_ckpt, best_selection = best_eval_checkpoint(metrics_path, model_dir)
    if args.eval_checkpoint == "latest":
        selected_ckpt, selection = ckpt, "latest"
    else:
        selected_ckpt, selection = best_ckpt, best_selection
    print(f"train_exit={proc.returncode if proc is not None else None}", flush=True)
    print(f"train_seconds={train_seconds:.3f}", flush=True)
    print(f"latest_checkpoint={ckpt}", flush=True)
    print(f"best_eval_checkpoint={best_ckpt}", flush=True)
    print(f"best_eval_checkpoint_reason={best_selection}", flush=True)
    print(f"selected_checkpoint={selected_ckpt}", flush=True)
    print(f"selected_checkpoint_reason={selection}", flush=True)
    print(f"metrics_csv={metrics_path}", flush=True)
    print(f"train_log={train_log}", flush=True)

    eval_data = None
    if args.render_final and selected_ckpt is not None:
        eval_data = run_final_eval(run_path, selected_ckpt, args.eval_checkpoint, args.eval_num_rays_per_chunk)
        if args.update_summary:
            update_summary(args, run_path, selection, eval_data, train_seconds)
    write_run_summary(
        run_path=run_path,
        args=args,
        train_seconds=train_seconds,
        train_returncode=proc.returncode if proc is not None else None,
        selection=selection,
        selected_ckpt=selected_ckpt,
        eval_data=eval_data,
    )
    if args.prune_checkpoints:
        prune_nonselected_checkpoints(model_dir, selected_ckpt)
    if stopped_for_plateau:
        return 0
    return 0 if proc is not None and proc.returncode in (0, -signal.SIGINT) else int(proc.returncode or 1)


if __name__ == "__main__":
    sys.exit(main())
