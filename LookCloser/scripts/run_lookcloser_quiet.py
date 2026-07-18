#!/usr/bin/env python3
"""Quiet LookCloser experiment runner for the HD bounded dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


DEFAULT_DATA = Path("/home/brans/temporal_perframe_stride7_45f/007740")
DEFAULT_OUTPUT = Path("/home/brans/lookcloser_leader_repro_runs")
DEFAULT_EXPERIMENT = "007740_static_leader_stage_a"
DEFAULT_SUMMARY = Path(__file__).resolve().parents[1] / "experiments" / "lookcloser_frequency_grid_optimization.md"
ARTIFACT_DETECTOR = Path(__file__).resolve().parent / "detect_structural_artifacts.py"
ROI_ARTIFACT_SCORER = Path(__file__).resolve().parent / "score_artifact_rois.py"
DEFAULT_ARTIFACT_ROI_CROPS = (
    "left_stand_connector_eval0,left_stand_eval0,left_hand_background_eval0,"
    "left_hand_outlet_stand_eval0,floor_crack_eval0,fingers_right_tight_eval1,"
    "stand_label_eval2,tangled_cable_eval2,fingers_center_eval2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--scene-scale", type=float, default=1.5)
    parser.add_argument("--scale-factor", type=float, default=1.0)
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
    parser.add_argument("--max-num-iterations", type=int, default=75941)
    parser.add_argument("--train-num-rays-per-batch", type=int, default=4096)
    parser.add_argument(
        "--cache-train-rays",
        action="store_true",
        help="Precompute static training rays on device; opt-in speed path for fixed cameras.",
    )
    parser.add_argument("--cache-train-rays-chunk-size", type=int, default=1 << 20)
    parser.add_argument(
        "--cpu-fas-prefetch",
        action="store_true",
        help="Opt in to private-generator one-batch CPU FAS prefetch (fixed B4096/static-ray-cache only).",
    )
    parser.add_argument("--train-rays-switch-step", type=int, default=None)
    parser.add_argument("--train-rays-after-switch", type=int, default=None)
    parser.add_argument("--feature-reweighting-switch-step", type=int, default=None)
    parser.add_argument("--feature-reweighting-after-switch", type=float, default=None)
    parser.add_argument("--eval-num-rays-per-batch", type=int, default=4096)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--background-color", choices=("random", "last_sample", "black", "white"), default="black")
    parser.add_argument("--reconstruction-loss-type", choices=("charbonnier", "mse", "huber"), default="charbonnier")
    parser.add_argument("--huber-delta", type=float, default=0.1)

    parser.add_argument("--frequency-map-dir", default="lookcloser_frequencies")
    parser.add_argument("--frequency-patch-size", type=int, default=8)
    parser.add_argument("--frequency-stride", type=int, default=8)
    parser.add_argument("--allow-missing-frequency-maps", action="store_true")
    parser.add_argument(
        "--independent-rng-streams",
        action="store_true",
        help="Opt in to step-addressed pixel/FAS, occupancy, and frequency-grid RNG streams.",
    )

    parser.add_argument("--grid-resolution", type=int, default=128)
    parser.add_argument("--occupancy-grid-levels", type=int, default=1)
    parser.add_argument("--num-frequency-levels", type=int, default=16)
    parser.add_argument("--min-res", type=float, default=16.0)
    parser.add_argument("--max-res", type=float, default=8192.0)
    parser.add_argument("--max-res-base", type=float, default=2048.0)
    parser.add_argument("--fallback-frequency-level", type=float, default=0.0)
    parser.add_argument("--grid-update-interval", type=int, default=1024)
    parser.add_argument("--grid-update-batch-size", type=int, default=2048)
    parser.add_argument(
        "--target-num-samples-per-batch",
        type=int,
        default=0,
        help="Dynamic field-point budget; non-positive preserves the historical fixed ray batch.",
    )
    parser.add_argument("--target-num-samples-switch-step", type=int, default=None)
    parser.add_argument("--target-num-samples-after-switch", type=int, default=None)
    parser.add_argument("--dynamic-rays-ema", type=float, default=0.9)
    parser.add_argument("--dynamic-rays-start-step", type=int, default=0)
    parser.add_argument("--dynamic-rays-min", type=int, default=256)
    parser.add_argument("--dynamic-rays-max", type=int, default=32768)
    parser.add_argument("--dynamic-rays-change-limit", type=float, default=1.25)

    parser.add_argument("--disable-frequency-grid", action="store_true")
    parser.add_argument("--disable-feature-reweighting", action="store_true")
    parser.add_argument("--feature-reweighting-strength", type=float, default=1.0)
    parser.add_argument(
        "--ray-sampling-mode",
        choices=("auto", "adaptive", "occupancy", "fixed"),
        default="auto",
        help="Model ray sampling mode. 'auto' preserves --disable-adaptive-ray-marching compatibility.",
    )
    parser.add_argument("--disable-adaptive-ray-marching", action="store_true")
    parser.add_argument("--disable-fas", action="store_true")
    parser.add_argument("--sampling-ramp-start", type=float, default=1.0)
    parser.add_argument("--sampling-ramp-end", type=float, default=3.0)
    parser.add_argument("--fas-strength", type=float, default=1.0)
    parser.add_argument("--fas-warmup-steps", type=int, default=0)
    parser.add_argument("--fas-ramp-steps", type=int, default=0)
    parser.add_argument("--fas-decay-start-steps", type=int, default=-1)
    parser.add_argument("--fas-decay-steps", type=int, default=0)
    parser.add_argument("--fas-level-count-alpha", type=float, default=0.0)
    parser.add_argument("--fas-patch-group-size", type=int, default=1)
    parser.add_argument("--fas-max-sampling-level", type=int, default=-1)
    parser.add_argument(
        "--fas-consolidate-h2d",
        action="store_true",
        help="Preserve FAS RNG order while consolidating selected-cell CPU-to-CUDA copies.",
    )

    parser.add_argument("--hash-features-per-level", type=int, default=2)
    parser.add_argument("--log2-hashmap-size", type=int, default=23)
    parser.add_argument("--field-hidden-dim", type=int, default=64)
    parser.add_argument("--geo-num-layers", type=int, default=1)
    parser.add_argument("--color-num-layers", type=int, default=2)
    parser.add_argument("--appearance-embedding-dim", type=int, default=0)
    parser.add_argument("--sh-degree", type=int, default=4)
    parser.add_argument("--tcnn-network-jit", action="store_true")
    parser.add_argument(
        "--tcnn-network-jit-scope",
        choices=("both", "geometry", "color"),
        default="both",
        help="TCNN MLP subset affected by initial or scheduled JIT enablement.",
    )
    parser.add_argument(
        "--tcnn-network-jit-switch-step",
        type=int,
        default=None,
        help="Enable the selected TCNN field MLP scope in-process before this trainer update.",
    )
    parser.add_argument(
        "--tcnn-network-jit-second-switch-step",
        type=int,
        default=None,
        help="Enable an additional TCNN field MLP scope at this later trainer update.",
    )
    parser.add_argument(
        "--tcnn-network-jit-second-switch-scope",
        choices=("both", "geometry", "color"),
        default=None,
        help="TCNN MLP subset enabled by the optional second live JIT switch.",
    )
    parser.add_argument("--fixed-num-samples-per-ray", type=int, default=256)
    parser.add_argument("--max-steps-per-ray", type=int, default=1024)
    parser.add_argument("--adaptive-min-step-size", type=float, default=1e-4)
    parser.add_argument("--adaptive-max-step-size", type=float, default=0.1)
    parser.add_argument("--adaptive-coarse-step-size", type=float, default=None)
    parser.add_argument("--adaptive-min-frequency-level", type=float, default=0.0)
    parser.add_argument("--adaptive-max-frequency-level", type=float, default=None)
    parser.add_argument("--adaptive-interval-level-mode", choices=("midpoint", "max3"), default="midpoint")
    parser.add_argument(
        "--corrected-arm-allocator",
        action="store_true",
        help="Use deterministic minimum-one/largest-remainder ARM capping without tail truncation.",
    )
    parser.add_argument("--adaptive-warmup-steps", type=int, default=4096)
    parser.add_argument("--adaptive-fixed-fallback-samples-per-ray", type=int, default=0)
    parser.add_argument("--transmittance-threshold", type=float, default=0.0)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1000.0)
    parser.add_argument("--alpha-thre", type=float, default=0.0)
    parser.add_argument("--cone-angle", type=float, default=0.0)
    parser.add_argument("--render-step-size", type=float, default=None)
    parser.add_argument("--render-step-size-mult", type=float, default=1.0)
    parser.add_argument("--occupancy-occ-thre", type=float, default=1e-2)
    parser.add_argument("--occupancy-ema-decay", type=float, default=0.95)
    parser.add_argument("--occupancy-warmup-steps", type=int, default=4096)
    parser.add_argument("--occupancy-update-interval", type=int, default=16)
    parser.add_argument("--occupancy-update-step-size", type=float, default=None)
    parser.add_argument("--occupancy-thre-clamp-mult", type=float, default=1.0)
    parser.add_argument("--occupancy-dilation-radius", type=int, default=0)
    parser.add_argument("--occupancy-binary-warmup-steps", type=int, default=4096)
    parser.add_argument("--occupancy-fixed-fallback-samples-per-ray", type=int, default=0)
    parser.add_argument(
        "--stable-occupancy-reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the accepted duplicate-cell max reducer; disable only for a legacy forensic control.",
    )
    parser.add_argument(
        "--occupancy-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Collect occupancy update metrics; --no-occupancy-diagnostics removes their hot-path reductions.",
    )
    parser.add_argument("--use-gradient-scaling", action="store_true")
    parser.add_argument("--distortion-loss-mult", type=float, default=0.01)
    parser.add_argument("--depth-loss-mult", type=float, default=0.001)
    parser.add_argument("--depth-loss-steps", type=int, default=5000)
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-load-mode",
        choices=("resume", "model_parameters_only"),
        default="resume",
        help="Full-state resume or fresh local training initialized only from fields parameters.",
    )
    parser.add_argument(
        "--resume-fields-lr-override",
        type=float,
        default=None,
        help="Full-resume fields LR override that preserves Adam/scaler/RNG and scheduler progress.",
    )
    parser.add_argument("--no-load-scheduler", dest="load_scheduler", action="store_false")
    parser.add_argument("--no-load-optimizers", dest="load_optimizers", action="store_false")
    parser.add_argument("--fields-lr", type=float, default=None)
    parser.add_argument("--fields-lr-final", type=float, default=None)
    parser.add_argument("--fields-scheduler-max-steps", type=int, default=None)
    parser.add_argument(
        "--grad-scaler-init-scale",
        type=float,
        default=None,
        help="Override AMP GradScaler initial scale; omitted preserves the historical 65536 default.",
    )
    parser.add_argument(
        "--grad-scaler-growth-interval",
        type=int,
        default=None,
        help="Override successful updates between AMP scale growth; omitted preserves 2000.",
    )
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument(
        "--fused-adam-switch-step",
        type=int,
        default=None,
        help="Enable fused Adam in-process before this trainer update, preserving optimizer state.",
    )
    parser.add_argument(
        "--replay-eval-trajectory",
        action="store_true",
        help="Replay scheduled eval sampler/RNG side effects without intermediate model eval or renders.",
    )
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--no-update-summary", dest="update_summary", action="store_false")
    parser.add_argument("--eval-checkpoint", choices=("best", "latest", "artifact", "roi"), default="best")
    parser.add_argument("--artifact-render-name", default="eval_img_0000.png")
    parser.add_argument(
        "--artifact-render-names",
        default=None,
        help="Comma-separated render filenames for artifact scoring; overrides --artifact-render-name.",
    )
    parser.add_argument("--artifact-crop-top", type=int, default=0)
    parser.add_argument("--artifact-crop-bottom", type=int, default=0)
    parser.add_argument("--artifact-crop-left", type=int, default=0)
    parser.add_argument("--artifact-crop-right", type=int, default=0)
    parser.add_argument(
        "--artifact-detector-preset",
        choices=("legacy", "significant", "micro"),
        default="legacy",
        help="Threshold preset passed to full-frame and ROI artifact detectors.",
    )
    parser.add_argument("--artifact-roi-drop-border-components", type=int, default=0)
    parser.add_argument(
        "--artifact-roi-crop-names",
        default=DEFAULT_ARTIFACT_ROI_CROPS,
        help="Comma-separated ROI crop names for ROI artifact scoring; use 'all' to score every ROI in score_artifact_rois.py.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.set_defaults(render_final=True, update_summary=True, artifact_score=True, artifact_roi_score=True)
    parser.add_argument(
        "--stop-on-no-improve",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional diagnostic early stop; the accepted leader trajectory never enables it.",
    )
    parser.add_argument("--no-render-final", dest="render_final", action="store_false")
    parser.add_argument("--no-artifact-score", dest="artifact_score", action="store_false")
    parser.add_argument("--no-artifact-roi-score", dest="artifact_roi_score", action="store_false")
    parser.add_argument(
        "--prune-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete non-selected checkpoints after final evaluation; off for leader provenance.",
    )
    parser.add_argument("--keep-all-checkpoints", dest="prune_checkpoints", action="store_false")
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
    if args.cpu_fas_prefetch:
        if not args.cache_train_rays:
            raise ValueError("--cpu-fas-prefetch requires --cache-train-rays")
        if args.train_num_rays_per_batch != 4096:
            raise ValueError("--cpu-fas-prefetch v1 requires --train-num-rays-per-batch 4096")
        if not enable_fas:
            raise ValueError("--cpu-fas-prefetch requires FAS")
        if (
            args.train_rays_switch_step is not None
            or args.target_num_samples_per_batch > 0
            or args.target_num_samples_switch_step is not None
        ):
            raise ValueError("--cpu-fas-prefetch v1 does not support ray-batch or dynamic point-target schedules")
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
        bool_text(args.prune_checkpoints),
    ]
    if args.load_dir is not None:
        cmd.extend(["--load-dir", str(args.load_dir)])
    if args.load_step is not None:
        cmd.extend(["--load-step", str(args.load_step)])
    if args.load_checkpoint is not None:
        cmd.extend(["--load-checkpoint", str(args.load_checkpoint)])
    if args.checkpoint_load_mode != "resume":
        cmd.extend(["--checkpoint-load-mode", args.checkpoint_load_mode])
    if args.resume_fields_lr_override is not None:
        if not math.isfinite(args.resume_fields_lr_override) or args.resume_fields_lr_override <= 0:
            raise ValueError("--resume-fields-lr-override must be finite and positive")
        cmd.extend(["--resume-fields-lr-override", str(args.resume_fields_lr_override)])
    if not args.load_scheduler:
        cmd.extend(["--load-scheduler", "False"])
    if not args.load_optimizers:
        cmd.extend(["--load-optimizers", "False"])
    if args.fields_lr is not None:
        cmd.extend(["--optimizers.fields.optimizer.lr", str(args.fields_lr)])
    if args.fields_lr_final is not None:
        cmd.extend(["--optimizers.fields.scheduler.lr-final", str(args.fields_lr_final)])
    if args.fields_scheduler_max_steps is not None:
        cmd.extend(["--optimizers.fields.scheduler.max-steps", str(args.fields_scheduler_max_steps)])
    if args.grad_scaler_init_scale is not None:
        if not math.isfinite(args.grad_scaler_init_scale) or args.grad_scaler_init_scale <= 0:
            raise ValueError("--grad-scaler-init-scale must be finite and positive")
        cmd.extend(["--grad-scaler-init-scale", str(args.grad_scaler_init_scale)])
    if args.grad_scaler_growth_interval is not None:
        if args.grad_scaler_growth_interval <= 0:
            raise ValueError("--grad-scaler-growth-interval must be positive")
        cmd.extend(["--grad-scaler-growth-interval", str(args.grad_scaler_growth_interval)])
    if args.fused_adam and args.fused_adam_switch_step is not None:
        raise ValueError("--fused-adam and --fused-adam-switch-step are mutually exclusive")
    if args.fused_adam:
        cmd.extend(["--optimizers.fields.optimizer.fused", "True"])
    if args.fused_adam_switch_step is not None:
        cmd.extend(["--fused-adam-switch-step", str(args.fused_adam_switch_step)])
    if args.replay_eval_trajectory:
        cmd.extend(["--replay-eval-trajectory", "True"])
    if (args.train_rays_switch_step is None) != (args.train_rays_after_switch is None):
        raise ValueError("--train-rays-switch-step and --train-rays-after-switch must be set together")
    if args.train_rays_switch_step is not None:
        cmd.extend(
            [
                "--pipeline.train-rays-switch-step",
                str(args.train_rays_switch_step),
                "--pipeline.train-rays-after-switch",
                str(args.train_rays_after_switch),
            ]
        )
    if (args.feature_reweighting_switch_step is None) != (args.feature_reweighting_after_switch is None):
        raise ValueError(
            "--feature-reweighting-switch-step and --feature-reweighting-after-switch must be set together"
        )
    if args.feature_reweighting_switch_step is not None:
        cmd.extend(
            [
                "--pipeline.feature-reweighting-switch-step",
                str(args.feature_reweighting_switch_step),
                "--pipeline.feature-reweighting-after-switch",
                str(args.feature_reweighting_after_switch),
            ]
        )
    if args.tcnn_network_jit and args.tcnn_network_jit_switch_step is not None:
        raise ValueError("--tcnn-network-jit and --tcnn-network-jit-switch-step are mutually exclusive")
    second_jit_switch = (
        args.tcnn_network_jit_second_switch_step,
        args.tcnn_network_jit_second_switch_scope,
    )
    if (second_jit_switch[0] is None) != (second_jit_switch[1] is None):
        raise ValueError(
            "--tcnn-network-jit-second-switch-step and --tcnn-network-jit-second-switch-scope "
            "must be set together"
        )
    if second_jit_switch[0] is not None:
        if args.tcnn_network_jit_switch_step is None:
            raise ValueError("A second TCNN network JIT switch requires --tcnn-network-jit-switch-step")
        if second_jit_switch[0] <= args.tcnn_network_jit_switch_step:
            raise ValueError("--tcnn-network-jit-second-switch-step must be strictly greater than the first switch")
    if args.tcnn_network_jit_switch_step is not None:
        cmd.extend(
            ["--pipeline.tcnn-network-jit-switch-step", str(args.tcnn_network_jit_switch_step)]
        )
    if second_jit_switch[0] is not None:
        cmd.extend(
            [
                "--pipeline.tcnn-network-jit-second-switch-step",
                str(second_jit_switch[0]),
                "--pipeline.tcnn-network-jit-second-switch-scope",
                str(second_jit_switch[1]),
            ]
        )
    if (args.target_num_samples_switch_step is None) != (args.target_num_samples_after_switch is None):
        raise ValueError(
            "--target-num-samples-switch-step and --target-num-samples-after-switch must be set together"
        )
    if args.target_num_samples_switch_step is not None:
        cmd.extend(
            [
                "--pipeline.target-num-samples-switch-step",
                str(args.target_num_samples_switch_step),
                "--pipeline.target-num-samples-after-switch",
                str(args.target_num_samples_after_switch),
            ]
        )

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
            "--pipeline.datamanager.cache-train-rays",
            bool_text(args.cache_train_rays),
            "--pipeline.datamanager.cache-train-rays-chunk-size",
            str(args.cache_train_rays_chunk_size),
            "--pipeline.datamanager.cpu-fas-prefetch",
            bool_text(args.cpu_fas_prefetch),
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
            "--pipeline.datamanager.pixel-sampler.fas-decay-start-steps",
            str(args.fas_decay_start_steps),
            "--pipeline.datamanager.pixel-sampler.fas-decay-steps",
            str(args.fas_decay_steps),
            "--pipeline.datamanager.pixel-sampler.fas-level-count-alpha",
            str(args.fas_level_count_alpha),
            "--pipeline.datamanager.pixel-sampler.fas-patch-group-size",
            str(args.fas_patch_group_size),
            "--pipeline.datamanager.pixel-sampler.fas-max-sampling-level",
            str(args.fas_max_sampling_level),
            "--pipeline.datamanager.pixel-sampler.fas-consolidate-h2d",
            bool_text(args.fas_consolidate_h2d),
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
            "--pipeline.target-num-samples-per-batch",
            str(args.target_num_samples_per_batch),
            "--pipeline.dynamic-rays-ema",
            str(args.dynamic_rays_ema),
            "--pipeline.dynamic-rays-start-step",
            str(args.dynamic_rays_start_step),
            "--pipeline.dynamic-rays-min",
            str(args.dynamic_rays_min),
            "--pipeline.dynamic-rays-max",
            str(args.dynamic_rays_max),
            "--pipeline.dynamic-rays-change-limit",
            str(args.dynamic_rays_change_limit),
            "--pipeline.model.eval-num-rays-per-chunk",
            str(args.eval_num_rays_per_chunk),
            "--pipeline.model.background-color",
            args.background_color,
            "--pipeline.model.reconstruction-loss-type",
            args.reconstruction_loss_type,
            "--pipeline.model.huber-delta",
            str(args.huber_delta),
            "--pipeline.model.enable-frequency-grid",
            bool_text(enable_frequency_grid),
            "--pipeline.model.grid-resolution",
            str(args.grid_resolution),
            "--pipeline.model.occupancy-grid-levels",
            str(args.occupancy_grid_levels),
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
            "--pipeline.model.feature-reweighting-strength",
            str(args.feature_reweighting_strength),
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
            "--pipeline.model.tcnn-network-jit",
            bool_text(args.tcnn_network_jit),
            "--pipeline.model.tcnn-network-jit-scope",
            args.tcnn_network_jit_scope,
            "--pipeline.model.enable-adaptive-ray-marching",
            bool_text(not args.disable_adaptive_ray_marching),
            "--pipeline.model.ray-sampling-mode",
            args.ray_sampling_mode,
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
            "--pipeline.model.adaptive-interval-level-mode",
            args.adaptive_interval_level_mode,
            "--pipeline.model.corrected-arm-allocator",
            bool_text(args.corrected_arm_allocator),
            "--pipeline.model.adaptive-warmup-steps",
            str(args.adaptive_warmup_steps),
            "--pipeline.model.adaptive-fixed-fallback-samples-per-ray",
            str(args.adaptive_fixed_fallback_samples_per_ray),
            "--pipeline.model.transmittance-threshold",
            str(args.transmittance_threshold),
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
            "--pipeline.model.occupancy-occ-thre",
            str(args.occupancy_occ_thre),
            "--pipeline.model.occupancy-ema-decay",
            str(args.occupancy_ema_decay),
            "--pipeline.model.occupancy-warmup-steps",
            str(args.occupancy_warmup_steps),
            "--pipeline.model.occupancy-update-interval",
            str(args.occupancy_update_interval),
            "--pipeline.model.occupancy-thre-clamp-mult",
            str(args.occupancy_thre_clamp_mult),
            "--pipeline.model.occupancy-dilation-radius",
            str(args.occupancy_dilation_radius),
            "--pipeline.model.occupancy-binary-warmup-steps",
            str(args.occupancy_binary_warmup_steps),
            "--pipeline.model.occupancy-fixed-fallback-samples-per-ray",
            str(args.occupancy_fixed_fallback_samples_per_ray),
            "--pipeline.model.stable-occupancy-reduction",
            bool_text(args.stable_occupancy_reduction),
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
    if args.independent_rng_streams:
        cmd.extend(
            [
                "--pipeline.training-seed",
                str(args.seed),
                "--pipeline.independent-rng-streams",
                "True",
                "--pipeline.model.training-seed",
                str(args.seed),
                "--pipeline.model.independent-rng-streams",
                "True",
            ]
        )
    if args.occupancy_diagnostics is False:
        cmd.extend(["--pipeline.model.occupancy-diagnostics", "False"])
    if args.max_res is not None:
        cmd.extend(["--pipeline.model.max-res", str(args.max_res)])
    if args.render_step_size is not None:
        cmd.extend(["--pipeline.model.render-step-size", str(args.render_step_size)])
    if args.occupancy_update_step_size is not None:
        cmd.extend(["--pipeline.model.occupancy-update-step-size", str(args.occupancy_update_step_size)])
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


PSNR_TIE_THRESHOLD_DB = 0.07  # checkpoints within this PSNR range are considered tied; LPIPS breaks the tie


def eval_rows(metrics_path: Path) -> List[Dict[str, str]]:
    return [row for row in read_csv_rows(metrics_path) if row.get("eval_all_psnr") or row.get("eval_loss")]


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
        f"delta={row.get('eval_psnr_delta') or row.get('eval_loss_delta')} "
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

    psnr_rows = [row for row in rows if row.get("eval_all_psnr")]
    if psnr_rows:
        best_psnr = max(float(r["eval_all_psnr"]) for r in psnr_rows)
        # All checkpoints within PSNR_TIE_THRESHOLD_DB of the best are tied.
        # Among ties, prefer lowest LPIPS (better perceptual quality, fewer artifacts).
        tied = [r for r in psnr_rows if best_psnr - float(r["eval_all_psnr"]) < PSNR_TIE_THRESHOLD_DB]
        lpips_tied = [r for r in tied if r.get("eval_all_lpips")]
        if lpips_tied:
            best_row = min(lpips_tied, key=lambda r: float(r["eval_all_lpips"]))
            reason = f"best_psnr{best_psnr:.3f}_lpips_tiebreak"
        else:
            best_row = max(psnr_rows, key=lambda r: float(r["eval_all_psnr"]))
            reason = "best_eval_all_psnr"
    else:
        # Fallback: no full-image PSNR available, use batch eval_loss
        best_row = min(rows, key=lambda r: float(r["eval_loss"]))
        reason = "best_eval_loss_fallback"

    target_step = int(best_row["step"])
    by_step = {checkpoint_step(ckpt): ckpt for ckpt in checkpoints}
    if target_step in by_step:
        return by_step[target_step], f"{reason}_step_{target_step}"
    earlier_or_equal = [step for step in by_step if step <= target_step]
    if earlier_or_equal:
        step = max(earlier_or_equal)
        return by_step[step], f"nearest_saved_for_{reason}_step_{target_step}"
    return checkpoints[-1], f"latest_no_checkpoint_for_{reason}_step_{target_step}"


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


def eval_config_for_step(
    config: Path,
    checkpoint: Path,
    eval_num_rays_per_chunk: Optional[int],
    *,
    cache_train_rays: Optional[bool] = None,
    filename_tag: Optional[str] = None,
) -> Path:
    step = checkpoint_step(checkpoint)
    if filename_tag is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", filename_tag) is None:
        raise ValueError(f"Unsafe eval config filename tag: {filename_tag!r}")
    tag = f"_{filename_tag}" if filename_tag is not None else ""
    eval_config = config.with_name(f"eval_config{tag}_step_{step}.yml")
    loaded = yaml.load(config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    loaded.load_dir = None
    # Historical eval_utils reconstructs load_dir from the run config and
    # selects by load_step; it does not honor TrainerConfig.load_checkpoint.
    # Record both so old and patched evaluators bind the same exact file.
    loaded.load_step = step
    loaded.load_checkpoint = checkpoint
    if eval_num_rays_per_chunk is not None:
        loaded.pipeline.model.eval_num_rays_per_chunk = int(eval_num_rays_per_chunk)
    if cache_train_rays is not None:
        loaded.pipeline.datamanager.cache_train_rays = bool(cache_train_rays)
        if not cache_train_rays and hasattr(loaded.pipeline.datamanager, "cpu_fas_prefetch"):
            loaded.pipeline.datamanager.cpu_fas_prefetch = False
    eval_config.write_text(yaml.dump(loaded), encoding="utf-8")
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
    ns_eval = Path(sys.executable).with_name("ns-eval")
    if not ns_eval.is_file():
        raise FileNotFoundError(f"ns-eval was not found beside the active Python interpreter: {ns_eval}")
    cmd = [
        str(ns_eval),
        "--load-config",
        str(eval_config),
        "--output-path",
        str(output_json),
        "--render-output-path",
        str(render_dir),
    ]
    print(f"running final eval: {' '.join(cmd)}", flush=True)
    eval_start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    eval_seconds = time.monotonic() - eval_start
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
    print(f"eval_seconds={eval_seconds:.3f}", flush=True)
    return {
        "checkpoint": data["checkpoint"],
        "results": results,
        "render_dir": str(render_dir),
        "eval_json": str(output_json),
        "eval_log": str(log_path),
        "eval_config": str(eval_config),
        "eval_seconds": eval_seconds,
    }


def candidate_checkpoints(model_dir: Path) -> List[Path]:
    return sorted(model_dir.glob("step-*.ckpt"))


def artifact_selection_key(eval_data: Dict[str, object], mode: str) -> Tuple[float, ...]:
    results = eval_data.get("results") or {}
    artifact = eval_data.get("artifact") or {}
    roi = artifact.get("roi") if isinstance(artifact, dict) else {}
    if roi is None:
        roi = {}
    full_score = artifact.get("artifact_score") if isinstance(artifact, dict) else None
    roi_score = roi.get("roi_artifact_score") if isinstance(roi, dict) else None
    roi_serious = roi.get("roi_serious_count") if isinstance(roi, dict) else None
    stand_score = roi.get("stand_connector_score") if isinstance(roi, dict) else None
    serious_score = artifact.get("serious_artifact_score") if isinstance(artifact, dict) else None
    # Higher image metrics are represented as negative values so normal tuple
    # sorting still chooses the best remaining quality on artifact ties.
    psnr = results.get("psnr")
    ssim = results.get("ssim")
    lpips = results.get("lpips")
    if mode == "roi":
        return (
            float(roi_serious) if roi_serious is not None else float("inf"),
            float(stand_score) if stand_score is not None else float("inf"),
            float(roi_score) if roi_score is not None else float("inf"),
            float(serious_score) if serious_score is not None else float("inf"),
            float(full_score) if full_score is not None else float("inf"),
            float(lpips) if lpips is not None else float("inf"),
            -float(ssim) if ssim is not None else float("inf"),
            -float(psnr) if psnr is not None else float("inf"),
        )
    return (
        float(full_score) if full_score is not None else float("inf"),
        float(serious_score) if serious_score is not None else float("inf"),
        float(roi_serious) if roi_serious is not None else float("inf"),
        float(stand_score) if stand_score is not None else float("inf"),
        float(roi_score) if roi_score is not None else float("inf"),
        float(lpips) if lpips is not None else float("inf"),
        -float(ssim) if ssim is not None else float("inf"),
        -float(psnr) if psnr is not None else float("inf"),
    )


def select_by_artifact(
    run_path: Path,
    model_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Optional[Path], str, Optional[Dict[str, object]], List[Dict[str, object]]]:
    mode = args.eval_checkpoint
    assert mode in {"artifact", "roi"}
    evaluated: List[Dict[str, object]] = []
    checkpoints = candidate_checkpoints(model_dir)
    if not checkpoints:
        return None, "missing", None, evaluated
    for checkpoint in checkpoints:
        label = f"{mode}_selection"
        eval_data = run_final_eval(run_path, checkpoint, label, args.eval_num_rays_per_chunk)
        eval_data["artifact"] = run_artifact_detector(run_path, eval_data, args)
        key = artifact_selection_key(eval_data, mode)
        evaluated.append(
            {
                "checkpoint": str(checkpoint),
                "step": checkpoint_step(checkpoint),
                "selection_key": list(key),
                "eval": eval_data,
            }
        )
        artifact = eval_data.get("artifact") or {}
        roi = artifact.get("roi") if isinstance(artifact, dict) else {}
        if roi is None:
            roi = {}
        print(
            "candidate "
            f"step={checkpoint_step(checkpoint)} "
            f"artifact={format_metric(artifact.get('artifact_score') if isinstance(artifact, dict) else None)} "
            f"serious_artifact={format_metric(artifact.get('serious_artifact_score') if isinstance(artifact, dict) else None)} "
            f"roi={format_metric(roi.get('roi_artifact_score') if isinstance(roi, dict) else None)} "
            f"roi_serious={format_metric(roi.get('roi_serious_count') if isinstance(roi, dict) else None)} "
            f"stand={format_metric(roi.get('stand_connector_score') if isinstance(roi, dict) else None)}",
            flush=True,
        )
    best = min(evaluated, key=lambda item: tuple(item["selection_key"]))
    selected = Path(str(best["checkpoint"]))
    selection = f"best_{mode}_checkpoint_step_{checkpoint_step(selected)}"
    return selected, selection, best["eval"], evaluated


def parse_artifact_stdout(text: str) -> Dict[str, object]:
    data: Dict[str, object] = {}
    match = re.search(
        r"\[candidate\]\s+serious=(\w+)\s+artifact_score=([0-9.]+)\s+count=(\d+)\s+largest=(\d+)px",
        text,
    )
    if match:
        data.update(
            {
                "serious": match.group(1) == "True",
                "artifact_score": float(match.group(2)),
                "artifact_count": int(match.group(3)),
                "largest_area": int(match.group(4)),
            }
        )
    serious_score = re.search(r"serious_artifact_score=([0-9.]+)", text)
    if serious_score:
        data["serious_artifact_score"] = float(serious_score.group(1))
    sanity = re.search(r"\[gt vs gt sanity\]\s+serious=(\w+)\s+artifact_score=([0-9.]+)", text)
    if sanity:
        data["gt_sanity_artifact_score"] = float(sanity.group(2))
    return data


def artifact_render_names(args: argparse.Namespace) -> List[str]:
    if args.artifact_render_names:
        return [name.strip() for name in args.artifact_render_names.split(",") if name.strip()]
    return [args.artifact_render_name]


def artifact_roi_crop_names(args: argparse.Namespace) -> List[str]:
    value = str(args.artifact_roi_crop_names or "").strip()
    if not value or value.lower() == "all":
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def artifact_roi_all_rois(args: argparse.Namespace) -> bool:
    return str(args.artifact_roi_crop_names or "").strip().lower() == "all"


def _run_artifact_view(
    render_dir: Path,
    artifact_dir: Path,
    render_name: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Score one rendered view; callers may safely execute independent views concurrently."""

    view_start = time.monotonic()
    render_file = render_dir / render_name
    log_path = artifact_dir / f"{render_file.stem}_artifact_stdout.log"
    if not render_file.exists():
        return {
            "status": "missing_render",
            "render_name": render_name,
            "render_file": str(render_file),
            "artifact_seconds": time.monotonic() - view_start,
        }
    out_prefix = artifact_dir / render_file.stem
    cmd = [
        sys.executable,
        str(ARTIFACT_DETECTOR),
        str(render_file),
        "--panels",
        "2",
        "--gt",
        "0",
        "--cand",
        "1",
        "--crop-top",
        str(args.artifact_crop_top),
        "--crop-bottom",
        str(args.artifact_crop_bottom),
        "--crop-left",
        str(args.artifact_crop_left),
        "--crop-right",
        str(args.artifact_crop_right),
        "--preset",
        args.artifact_detector_preset,
        "--out",
        str(out_prefix),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        log_path.write_text(proc.stdout, encoding="utf-8")
        view = parse_artifact_stdout(proc.stdout)
        view.update(
            {
                "status": "complete" if proc.returncode == 0 and "artifact_score" in view else "failed",
                "returncode": proc.returncode,
                "render_name": render_name,
                "render_file": str(render_file),
                "artifact_log": str(log_path),
                "artifact_seconds": time.monotonic() - view_start,
            }
        )
        return view
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "render_name": render_name,
            "render_file": str(render_file),
            "artifact_seconds": time.monotonic() - view_start,
        }


def run_artifact_detector(run_path: Path, eval_data: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    artifact_start = time.monotonic()
    render_dir = Path(str(eval_data["render_dir"]))
    if not args.artifact_score:
        return {"status": "disabled", "artifact_seconds": 0.0}
    artifact_dir = run_path / f"artifact_{render_dir.name}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    render_names = artifact_render_names(args)
    artifact_gate_only = bool(getattr(args, "artifact_gate_only", False))
    if artifact_gate_only:
        if len(render_names) != 3 or len(set(render_names)) != 3:
            raise ValueError("artifact_gate_only requires exactly three unique render names")
        # executor.map preserves input order even when subprocesses finish out of order.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="artifact-view") as executor:
            per_view = list(
                executor.map(
                    lambda name: _run_artifact_view(render_dir, artifact_dir, name, args),
                    render_names,
                )
            )
    else:
        per_view = [
            _run_artifact_view(render_dir, artifact_dir, render_name, args)
            for render_name in render_names
        ]
    completed = [view for view in per_view if view.get("status") == "complete" and view.get("artifact_score") is not None]
    missing = [view for view in per_view if view.get("status") == "missing_render"]
    if artifact_gate_only:
        aggregate_status = "complete" if len(completed) == len(per_view) else ("missing_render" if missing else "failed")
    else:
        aggregate_status = "complete" if completed else ("missing_render" if missing else "failed")
    result = {
        "status": aggregate_status,
        "artifact_score": max(float(view["artifact_score"]) for view in completed) if completed else None,
        "artifact_score_mean": sum(float(view["artifact_score"]) for view in completed) / len(completed) if completed else None,
        "serious_artifact_score": (
            max(float(view["serious_artifact_score"]) for view in completed if view.get("serious_artifact_score") is not None)
            if any(view.get("serious_artifact_score") is not None for view in completed)
            else None
        ),
        "serious_artifact_score_mean": (
            sum(float(view["serious_artifact_score"]) for view in completed if view.get("serious_artifact_score") is not None)
            / sum(1 for view in completed if view.get("serious_artifact_score") is not None)
            if any(view.get("serious_artifact_score") is not None for view in completed)
            else None
        ),
        "artifact_count": sum(int(view.get("artifact_count") or 0) for view in completed),
        "artifact_views_scored": len(completed),
        "artifact_views_requested": len(per_view),
        "views": per_view,
        "artifact_dir": str(artifact_dir),
        "artifact_seconds": None,
    }
    roi_result = run_roi_artifact_scorer(render_dir, artifact_dir, args)
    if roi_result is not None:
        result["roi"] = roi_result
    result["artifact_seconds"] = time.monotonic() - artifact_start
    print(
        "artifact "
        f"status={result.get('status')} "
        f"score={format_metric(result.get('artifact_score'))} "
        f"serious_score={format_metric(result.get('serious_artifact_score'))} "
        f"roi_score={format_metric((result.get('roi') or {}).get('roi_artifact_score'))} "
        f"seconds={format_metric(result.get('artifact_seconds'))}",
        flush=True,
    )
    return result


def run_roi_artifact_scorer(
    render_dir: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    if not args.artifact_roi_score:
        return {"status": "disabled", "roi_artifact_seconds": 0.0}
    if not ROI_ARTIFACT_SCORER.exists():
        return {"status": "missing_script", "script": str(ROI_ARTIFACT_SCORER), "roi_artifact_seconds": 0.0}
    roi_start = time.monotonic()
    roi_dir = artifact_dir / "roi_scores"
    log_path = roi_dir / "roi_artifact_stdout.log"
    cmd = [
        sys.executable,
        str(ROI_ARTIFACT_SCORER),
        "--render-dir",
        str(render_dir),
        "--out-dir",
        str(roi_dir),
        "--drop-border-components",
        str(args.artifact_roi_drop_border_components),
        "--preset",
        args.artifact_detector_preset,
    ]
    if not bool(getattr(args, "artifact_gate_only", False)):
        cmd.append("--write-images")
    if artifact_roi_all_rois(args):
        cmd.append("--all-rois")
    else:
        for crop_name in artifact_roi_crop_names(args):
            cmd.extend(["--crop-name", crop_name])
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        roi_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
        json_path = roi_dir / "roi_artifact_scores.json"
        rows = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
        completed = proc.returncode == 0 and isinstance(rows, list)
        scores = [float(row.get("artifact_score") or 0.0) for row in rows] if completed else []
        serious_scores = [float(row.get("serious_artifact_score") or 0.0) for row in rows] if completed else []
        serious_count = sum(1 for row in rows if row.get("serious")) if completed else 0
        stand_connector = None
        for row in rows:
            if row.get("crop") == "left_stand_connector_eval0":
                stand_connector = float(row.get("artifact_score") or 0.0)
                break
        return {
            "status": "complete" if completed else "failed",
            "returncode": proc.returncode,
            "roi_artifact_score": max(scores) if scores else None,
            "roi_artifact_score_mean": (sum(scores) / len(scores)) if scores else None,
            "roi_serious_artifact_score": max(serious_scores) if serious_scores else None,
            "roi_serious_artifact_score_mean": (
                (sum(serious_scores) / len(serious_scores)) if serious_scores else None
            ),
            "roi_serious_count": serious_count,
            "roi_count": len(rows) if completed else 0,
            "stand_connector_score": stand_connector,
            "roi_artifact_seconds": time.monotonic() - roi_start,
            "roi_dir": str(roi_dir),
            "roi_json": str(json_path),
            "roi_csv": str(roi_dir / "roi_artifact_scores.csv"),
            "roi_log": str(log_path),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "roi_artifact_seconds": time.monotonic() - roi_start,
            "roi_dir": str(roi_dir),
            "roi_log": str(log_path),
        }


def git_value(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def path_fingerprint(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode("utf-8", errors="ignore"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "manifest_hash": digest.hexdigest(),
    }


def provenance(args: argparse.Namespace, run_path: Path) -> Dict[str, object]:
    return {
        "git_branch": git_value(["branch", "--show-current"]),
        "git_sha": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--short"])),
        "data_fingerprint": path_fingerprint(args.data),
        "frequency_map_fingerprint": path_fingerprint(args.data / args.frequency_map_dir),
        "run_path": str(run_path),
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
        "cache_train_rays": args.cache_train_rays,
        "cache_train_rays_chunk_size": args.cache_train_rays_chunk_size,
        "train_rays_switch_step": args.train_rays_switch_step,
        "train_rays_after_switch": args.train_rays_after_switch,
        "feature_reweighting_switch_step": args.feature_reweighting_switch_step,
        "feature_reweighting_after_switch": args.feature_reweighting_after_switch,
        "target_num_samples_per_batch": args.target_num_samples_per_batch,
        "target_num_samples_switch_step": args.target_num_samples_switch_step,
        "target_num_samples_after_switch": args.target_num_samples_after_switch,
        "eval_num_rays_per_batch": args.eval_num_rays_per_batch,
        "eval_num_rays_per_chunk": args.eval_num_rays_per_chunk,
        "step_interval": args.step_interval,
        "max_num_iterations": args.max_num_iterations,
        "background_color": args.background_color,
        "reconstruction_loss_type": args.reconstruction_loss_type,
        "huber_delta": args.huber_delta,
        "frequency_map_dir": args.frequency_map_dir,
        "artifact_render_names": artifact_render_names(args),
        "artifact_crop_top": args.artifact_crop_top,
        "artifact_crop_bottom": args.artifact_crop_bottom,
        "artifact_crop_left": args.artifact_crop_left,
        "artifact_crop_right": args.artifact_crop_right,
        "artifact_roi_score": args.artifact_roi_score,
        "artifact_detector_preset": args.artifact_detector_preset,
        "artifact_roi_drop_border_components": args.artifact_roi_drop_border_components,
        "artifact_roi_crop_names": artifact_roi_crop_names(args) or "all",
        "grid_resolution": args.grid_resolution,
        "occupancy_grid_levels": args.occupancy_grid_levels,
        "num_frequency_levels": args.num_frequency_levels,
        "min_res": args.min_res,
        "max_res": args.max_res,
        "max_res_base": args.max_res_base,
        "fallback_frequency_level": args.fallback_frequency_level,
        "grid_update_interval": args.grid_update_interval,
        "grid_update_batch_size": args.grid_update_batch_size,
        "load_scheduler": args.load_scheduler,
        "load_optimizers": args.load_optimizers,
        "checkpoint_load_mode": args.checkpoint_load_mode,
        "resume_fields_lr_override": args.resume_fields_lr_override,
        "fields_lr": args.fields_lr,
        "fields_lr_final": args.fields_lr_final,
        "fields_scheduler_max_steps": args.fields_scheduler_max_steps,
        "grad_scaler_init_scale": args.grad_scaler_init_scale,
        "grad_scaler_growth_interval": args.grad_scaler_growth_interval,
        "fused_adam": args.fused_adam,
        "fused_adam_switch_step": args.fused_adam_switch_step,
        "replay_eval_trajectory": args.replay_eval_trajectory,
        "geo_num_layers": args.geo_num_layers,
        "color_num_layers": args.color_num_layers,
        "appearance_embedding_dim": args.appearance_embedding_dim,
        "tcnn_network_jit": args.tcnn_network_jit,
        "tcnn_network_jit_scope": args.tcnn_network_jit_scope,
        "tcnn_network_jit_switch_step": args.tcnn_network_jit_switch_step,
        "tcnn_network_jit_second_switch_step": args.tcnn_network_jit_second_switch_step,
        "tcnn_network_jit_second_switch_scope": args.tcnn_network_jit_second_switch_scope,
        "enable_frequency_grid": not args.disable_frequency_grid,
        "enable_feature_reweighting": not args.disable_feature_reweighting,
        "feature_reweighting_strength": args.feature_reweighting_strength,
        "ray_sampling_mode": args.ray_sampling_mode,
        "enable_adaptive_ray_marching": not args.disable_adaptive_ray_marching,
        "enable_fas": not args.disable_fas,
        "sampling_ramp_start": args.sampling_ramp_start,
        "sampling_ramp_end": args.sampling_ramp_end,
        "fas_strength": args.fas_strength,
        "fas_warmup_steps": args.fas_warmup_steps,
        "fas_ramp_steps": args.fas_ramp_steps,
        "fas_decay_start_steps": args.fas_decay_start_steps,
        "fas_decay_steps": args.fas_decay_steps,
        "fas_level_count_alpha": args.fas_level_count_alpha,
        "fas_patch_group_size": args.fas_patch_group_size,
        "fas_max_sampling_level": args.fas_max_sampling_level,
        "fas_consolidate_h2d": args.fas_consolidate_h2d,
        "near_plane": args.near_plane,
        "far_plane": args.far_plane,
        "alpha_thre": args.alpha_thre,
        "cone_angle": args.cone_angle,
        "render_step_size": args.render_step_size,
        "render_step_size_mult": args.render_step_size_mult,
        "occupancy_occ_thre": args.occupancy_occ_thre,
        "occupancy_ema_decay": args.occupancy_ema_decay,
        "occupancy_warmup_steps": args.occupancy_warmup_steps,
        "occupancy_update_interval": args.occupancy_update_interval,
        "occupancy_update_step_size": args.occupancy_update_step_size,
        "occupancy_thre_clamp_mult": args.occupancy_thre_clamp_mult,
        "occupancy_dilation_radius": args.occupancy_dilation_radius,
        "occupancy_binary_warmup_steps": args.occupancy_binary_warmup_steps,
        "occupancy_fixed_fallback_samples_per_ray": args.occupancy_fixed_fallback_samples_per_ray,
        "stable_occupancy_reduction": args.stable_occupancy_reduction,
        "adaptive_coarse_step_size": args.adaptive_coarse_step_size,
        "adaptive_min_step_size": args.adaptive_min_step_size,
        "adaptive_max_step_size": args.adaptive_max_step_size,
        "max_steps_per_ray": args.max_steps_per_ray,
        "fixed_num_samples_per_ray": args.fixed_num_samples_per_ray,
        "adaptive_min_frequency_level": args.adaptive_min_frequency_level,
        "adaptive_max_frequency_level": args.adaptive_max_frequency_level,
        "adaptive_interval_level_mode": args.adaptive_interval_level_mode,
        "adaptive_warmup_steps": args.adaptive_warmup_steps,
        "adaptive_fixed_fallback_samples_per_ray": args.adaptive_fixed_fallback_samples_per_ray,
        "transmittance_threshold": args.transmittance_threshold,
        "use_gradient_scaling": args.use_gradient_scaling,
    }
    if args.independent_rng_streams:
        params["independent_rng_streams"] = True
    if args.occupancy_diagnostics is not None:
        params["occupancy_diagnostics"] = args.occupancy_diagnostics
    return json.dumps(params, sort_keys=True)


def write_run_summary(
    run_path: Path,
    args: argparse.Namespace,
    train_seconds: float,
    train_returncode: Optional[int],
    selection: str,
    selected_ckpt: Optional[Path],
    eval_data: Optional[Dict[str, object]],
    total_seconds: float,
    artifact_candidate_evals: Optional[List[Dict[str, object]]] = None,
) -> None:
    artifact_data = eval_data.get("artifact") if eval_data else None
    eval_seconds = eval_data.get("eval_seconds") if eval_data else None
    artifact_seconds = artifact_data.get("artifact_seconds") if isinstance(artifact_data, dict) else None
    summary = {
        "timestamp": args.timestamp,
        "params": json.loads(summarize_params(args)),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "artifact_seconds": artifact_seconds,
        "total_seconds": total_seconds,
        "artifact": artifact_data,
        "provenance": provenance(args, run_path),
        "train_returncode": train_returncode,
        "selected_checkpoint": str(selected_ckpt) if selected_ckpt is not None else None,
        "selected_checkpoint_reason": selection,
        "artifact_candidate_evals": artifact_candidate_evals,
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
    total_seconds: float,
) -> None:
    summary_path = args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        summary_path.write_text(
            "# LookCloser Frequency Grid Optimization\n\n"
            "| Timestamp | Selection | Train Seconds | Eval Seconds | Artifact Seconds | Total Seconds | Artifact Score | Serious Artifact Score | ROI Artifact Score | ROI Serious Score | ROI Serious Count | Stand Connector | Params | Checkpoint | PSNR | SSIM | LPIPS | Eval JSON | Renders |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|\n",
            encoding="utf-8",
        )
    elif "Serious Artifact Score" not in summary_path.read_text(encoding="utf-8", errors="ignore"):
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(
                "\n## Runs With Artifact ROI And Runtime Metrics\n\n"
                "| Timestamp | Selection | Train Seconds | Eval Seconds | Artifact Seconds | Total Seconds | Artifact Score | Serious Artifact Score | ROI Artifact Score | ROI Serious Score | ROI Serious Count | Stand Connector | Params | Checkpoint | PSNR | SSIM | LPIPS | Eval JSON | Renders |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|\n"
            )
    results = eval_data["results"]
    assert isinstance(results, dict)
    artifact = eval_data.get("artifact")
    artifact_score = artifact.get("artifact_score") if isinstance(artifact, dict) else None
    serious_artifact_score = artifact.get("serious_artifact_score") if isinstance(artifact, dict) else None
    artifact_seconds = artifact.get("artifact_seconds") if isinstance(artifact, dict) else None
    roi = artifact.get("roi") if isinstance(artifact, dict) else None
    if roi is None:
        roi = {}
    roi_score = roi.get("roi_artifact_score") if isinstance(roi, dict) else None
    roi_serious_score = roi.get("roi_serious_artifact_score") if isinstance(roi, dict) else None
    roi_serious = roi.get("roi_serious_count") if isinstance(roi, dict) else None
    stand_score = roi.get("stand_connector_score") if isinstance(roi, dict) else None
    row = (
        f"| {args.timestamp} "
        f"| {selection} "
        f"| {train_seconds:.3f} "
        f"| {format_metric(eval_data.get('eval_seconds'))} "
        f"| {format_metric(artifact_seconds)} "
        f"| {total_seconds:.3f} "
        f"| {format_metric(artifact_score)} "
        f"| {format_metric(serious_artifact_score)} "
        f"| {format_metric(roi_score)} "
        f"| {format_metric(roi_serious_score)} "
        f"| {format_metric(roi_serious)} "
        f"| {format_metric(stand_score)} "
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
    total_start = time.monotonic()
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
                        prev_row = current_evals[-2]
                        last_row = current_evals[-1]
                        if prev_row.get("eval_all_psnr") and last_row.get("eval_all_psnr"):
                            prev = float(prev_row["eval_all_psnr"])
                            last = float(last_row["eval_all_psnr"])
                            if last <= prev:
                                print(f"stopping: eval psnr did not improve ({last:.6g} <= {prev:.6g})", flush=True)
                                stopped_for_plateau = True
                                stop_process(proc)
                                break
                        else:
                            prev = float(prev_row["eval_loss"])
                            last = float(last_row["eval_loss"])
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
    artifact_candidate_evals: Optional[List[Dict[str, object]]] = None
    eval_data = None
    if args.eval_checkpoint in {"artifact", "roi"}:
        if not args.render_final:
            selected_ckpt, selection = best_ckpt, f"{best_selection}_artifact_selection_skipped_no_render_final"
        else:
            selected_ckpt, selection, eval_data, artifact_candidate_evals = select_by_artifact(run_path, model_dir, args)
    elif args.eval_checkpoint == "latest":
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

    if args.render_final and selected_ckpt is not None and eval_data is None:
        eval_data = run_final_eval(run_path, selected_ckpt, args.eval_checkpoint, args.eval_num_rays_per_chunk)
        eval_data["artifact"] = run_artifact_detector(run_path, eval_data, args)
        total_seconds = time.monotonic() - total_start
    else:
        total_seconds = time.monotonic() - total_start
    if args.update_summary and eval_data is not None:
        update_summary(args, run_path, selection, eval_data, train_seconds, total_seconds)
    print(f"total_seconds={total_seconds:.3f}", flush=True)
    write_run_summary(
        run_path=run_path,
        args=args,
        train_seconds=train_seconds,
        train_returncode=proc.returncode if proc is not None else None,
        selection=selection,
        selected_ckpt=selected_ckpt,
        eval_data=eval_data,
        total_seconds=total_seconds,
        artifact_candidate_evals=artifact_candidate_evals,
    )
    if args.prune_checkpoints:
        prune_nonselected_checkpoints(model_dir, selected_ckpt)
    if stopped_for_plateau:
        return 0
    return 0 if proc is not None and proc.returncode in (0, -signal.SIGINT) else int(proc.returncode or 1)


if __name__ == "__main__":
    sys.exit(main())
