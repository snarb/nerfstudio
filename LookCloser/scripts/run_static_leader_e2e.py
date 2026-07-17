#!/usr/bin/env python3
"""Reproduce the archived static LookCloser leader through its exact two-stage ancestry.

This controller intentionally uses the historical executable worktree.  Stage A trains
from scratch through step 75940 with FR=1 and FAS=1.  Stage A_fw03 then restores the
model, Adam, and exponential scheduler, lowers FR to 0.3, and continues through step
106316.  The shorter trainer limits include the historical boundary step while leaving
the optimizer scheduler's 200000-step horizon unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import secrets
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


HISTORICAL_COMMIT = "85818149"
HISTORICAL_TCNN_COMMIT = "2e757bbe781db59c4980d389d7dccbf5edc09669"
ALLOWED_TCNN_COMPATIBILITY_PATCHES = {"CMakeLists.txt", "bindings/torch/setup.py"}
TCNN_GRID_GRAD_FP32_PATCH = "include/tiny-cuda-nn/encodings/grid.h"
EXPECTED_TCNN_GRID_GRAD_FP32_SHA256 = "f74500ff76838b012f3fbb77324c9e282b1cc2e1c4250f9b28ea5cd812c2f362"
EXPECTED_ACCEPTED_SOURCE_FINGERPRINT = "69d4f36cc1e06256a8dcd5a1e9dd6c4a465bb81e8cee09a3d8b188358857b252"
EXPECTED_SPEED_SOURCE_FINGERPRINT = "c31ab574ac7b9b51796b65ceb6e517eefd2611f17d3bed97b33e4b3646561b55"
EXPECTED_CONTROLLER_PROTOCOL_FINGERPRINT = "1027adfb9086d508109efb5563347527099947568c41354da42df5f3121a9eaf"
EXPECTED_ACCEPTED_TCNN_SOURCE_DIFF = "441f8877df4bbcc665dd1072c23d4cec8063f18ed14c909b598fde3a95a41673"
EXPECTED_ACCEPTED_TCNN_BUILD_PROVENANCE = "566e6dd9caba605ab053408794c9bbc854dedd0d171c1b1f99e77abe95180b5f"
EXPECTED_ACCEPTED_TCNN_BINDING = "f2163346afd103c27e78b9f56f8d82b6eeb3317c1ce11caf57d45f0216aece36"
EXPECTED_JIT_TCNN_OVERLAY_PROVENANCE = "e5d67f9750465112e3996b13c74e43175a142986590dea90503116dd8aa29606"
EXPECTED_TORCH_VERSION = "2.7.1+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_PYTHON_VERSION = "3.10.20"
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
ALLOWED_COMPATIBILITY_PATCHES = {
    "nerfstudio/data/utils/data_utils.py",
    "nerfstudio/engine/trainer.py",
    "nerfstudio/utils/eval_utils.py",
}
STABLE_OCCUPANCY_PATCHES = {
    "LookCloser/scripts/run_lookcloser_quiet.py",
    "nerfstudio/model_components/lookcloser_occupancy.py",
    "nerfstudio/models/lookcloser.py",
    "tests/model_components/test_lookcloser_occupancy.py",
}
INDEPENDENT_RNG_PATCHES = {
    "LookCloser/scripts/run_lookcloser_quiet.py",
    "nerfstudio/models/lookcloser.py",
    "nerfstudio/pipelines/lookcloser_pipeline.py",
    "nerfstudio/utils/lookcloser_rng.py",
    "tests/utils/test_lookcloser_rng.py",
}
SPEED_TELEMETRY_PATCHES = {
    "LookCloser/experiments/lookcloser_frequency_grid_optimization.md",
    "nerfstudio/data/datamanagers/base_datamanager.py",
    "nerfstudio/data/datamanagers/cpu_batch_prefetch.py",
    "nerfstudio/engine/trainer.py",
    "nerfstudio/engine/optimizers.py",
    "nerfstudio/fields/lookcloser_field.py",
    "nerfstudio/lookcloser_pixel_sampler.py",
    "nerfstudio/model_components/ray_generators.py",
    "nerfstudio/model_components/lookcloser_samplers.py",
    "nerfstudio/models/lookcloser.py",
    "nerfstudio/pipelines/lookcloser_pipeline.py",
    "nerfstudio/utils/writer.py",
    "tests/data/test_lookcloser_pixel_sampler_consolidated_h2d.py",
    "tests/data/test_lookcloser_cpu_fas_prefetch.py",
    "tests/data/test_lookcloser_pixel_sampler_lut.py",
    "tests/engine/test_fused_adam.py",
    "tests/fields/test_lookcloser_field_weights.py",
    "tests/fields/test_lookcloser_tcnn_jit_scope.py",
    "tests/model_components/test_lookcloser_sampler.py",
    "tests/model_components/test_ray_generator_cache.py",
    "tests/models/test_lookcloser_packed_render.py",
    "tests/pipelines/test_lookcloser_dynamic_target_schedule.py",
    "tests/pipelines/test_lookcloser_tcnn_jit_pipeline.py",
    "tests/scripts/test_run_lookcloser_quiet_eval_config.py",
    "tests/test_trainer_rng_state.py",
}
EXPECTED_TRANSFORMS_SHA256 = "022f8748a1a039861a754e68ab3ef830beeb3e5dd94ccb00457a630d28f64aa1"
PARENT_STEP = 75_940
FINAL_STEP = 106_316
ACCEPTED_STEP = 91_128
SCHEDULER_MAX_STEPS = 200_000
FIELDS_LR = 0.01
FIELDS_LR_FINAL = 0.0001
ADAPTIVE_WARMUP_STEPS = 4_096
TRAIN_RAYS_PER_BATCH = 4_096
FIXED_SAMPLES_PER_RAY = 256
FIXED_WARMUP_POINT_SAMPLES = ADAPTIVE_WARMUP_STEPS * TRAIN_RAYS_PER_BATCH * FIXED_SAMPLES_PER_RAY
REVIEWED_STAGED_SPEED_RECIPE = {
    "cache_train_rays": True,
    "fused_adam_switch_step": 15_189,
    "tcnn_network_jit_switch_step": 15_189,
    "tcnn_network_jit_scope": "color",
    "tcnn_network_jit_second_switch_step": 30_377,
    "tcnn_network_jit_second_switch_scope": "geometry",
    "replay_eval_trajectory": True,
    "historical_stage_boundary_rng_reset": True,
    "speed_final_step": ACCEPTED_STEP,
}

DEFAULT_WORKTREE = Path("/home/brans/repos/nerfstudio_leader_repro")
DEFAULT_VENV = Path("/home/brans/repos/nerfstudio/.venv")
DEFAULT_DATA = Path("/home/brans/temporal_perframe_stride7_45f/007740")
DEFAULT_OUTPUT = Path("/home/brans/lookcloser_leader_repro_runs")
DEFAULT_TORCH_EXTENSIONS = Path("/home/brans/.cache/torch_extensions_lookcloser")
DEFAULT_ACCEPTED_WORKTREE = Path("/home/brans/repos/nerfstudio_leader_stable_occ")
DEFAULT_SPEED_WORKTREE = Path("/home/brans/repos/nerfstudio_leader_speed")
DEFAULT_SPEED_OUTPUT = Path("/home/brans/lookcloser_leader_speed_runs")
DEFAULT_HISTORICAL_TCNN_OVERLAY = Path("/home/brans/deps/tcnn_2e757_py310")
DEFAULT_JIT_TCNN_OVERLAY = Path("/home/brans/deps/tcnn_2e757_py310_jit_rtc")
DEFAULT_HISTORICAL_TCNN_SOURCE = Path("/home/brans/deps/tiny-cuda-nn-2e757")
LEADER_GATES = {"psnr": 29.617964, "ssim": 0.668450, "lpips": 0.231135}
QUALITY_FAILURE_EXIT_CODE = 2
INCOMPLETE_OR_INFRASTRUCTURE_EXIT_CODE = 3
DEFAULT_PROVENANCE_SCRIPT = Path(
    "/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/scripts/check_static_dataset_provenance.py"
)
PROTOCOL_EXTERNAL_DETAIL_SCORER = Path(
    "/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/scripts/score_static_detail_rois.py"
)
PROTOCOL_EXTERNAL_DETAIL_REFERENCE = Path(
    "/home/brans/repos/nerfstudio_static_lookcloser/LookCloser/experiments/static_archive_detail_reference.json"
)
STAGE_BOUNDARY_CHECKPOINT_FORK = Path(__file__).resolve().with_name("fork_static_checkpoint_optimizer.py")


@dataclass(frozen=True)
class ResolvedRecipe:
    speed_mode: bool
    batch_scale: int
    lr_scale: float
    fields_lr: float
    fields_lr_final: float
    train_rays_per_batch: int
    adaptive_warmup_steps: int
    occupancy_warmup_steps: int
    occupancy_binary_warmup_steps: int
    occupancy_update_interval: int
    grid_update_interval: int
    depth_loss_steps: int
    scheduler_max_steps: int
    checkpoint_interval: int
    save_interval: int
    parent_step: int
    final_step: int
    fixed_warmup_point_samples: int
    target_num_samples_per_batch: int
    corrected_arm_allocator: bool
    cache_train_rays: bool
    cpu_fas_prefetch: bool
    fused_adam: bool
    fused_adam_switch_step: int | None
    tcnn_network_jit_switch_step: int | None
    tcnn_network_jit_scope: str | None
    tcnn_network_jit_second_switch_step: int | None
    tcnn_network_jit_second_switch_scope: str | None
    feature_reweighting_switch_step: int | None
    feature_reweighting_after_switch: float | None
    replay_eval_trajectory: bool
    historical_stage_boundary_rng_reset: bool
    hard_candidate_only: bool
    wall_milestone_seconds: int | None


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-worktree", type=Path, default=DEFAULT_ACCEPTED_WORKTREE)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tcnn-overlay",
        type=Path,
        default=DEFAULT_HISTORICAL_TCNN_OVERLAY,
        help="Optional isolated tiny-cuda-nn package directory prepended to PYTHONPATH.",
    )
    parser.add_argument(
        "--tcnn-source-worktree",
        type=Path,
        default=DEFAULT_HISTORICAL_TCNN_SOURCE,
        help="Source worktree used to build --tcnn-overlay; its revision/diff are recorded.",
    )
    parser.add_argument("--campaign-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Generate one random seed, record it, and use it for both ancestry stages.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--batch-scale",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Point-normalized speed recipe scale. Values above one activate the fingerprinted speed worktree.",
    )
    parser.add_argument(
        "--speed-stop-at-accepted-boundary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop at the point-normalized historical step-91128 boundary instead of full step 106316.",
    )
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help=(
            "Speed ablation: multiply both the historical fields LR (0.01) and exponential "
            "scheduler endpoint (0.0001). Only valid in point-normalized speed mode."
        ),
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=0,
        help=(
            "Speed ablation: dynamically adjust rays so each optimizer update evaluates approximately this many "
            "field points. Zero preserves the historical fixed ray batch."
        ),
    )
    parser.add_argument(
        "--corrected-arm-allocator",
        action="store_true",
        help="Speed/quality ablation: use deterministic full-tail ARM budget allocation.",
    )
    parser.add_argument(
        "--fused-adam",
        action="store_true",
        help="Speed ablation: use PyTorch fused CUDA Adam with checkpoint-state migration.",
    )
    parser.add_argument(
        "--fused-adam-switch-step",
        type=int,
        default=None,
        help="Reviewed speed path: enable fused Adam in-process before this update.",
    )
    parser.add_argument(
        "--cache-train-rays",
        action="store_true",
        help="Reviewed speed path: precompute static training-camera rays during setup.",
    )
    parser.add_argument(
        "--cpu-fas-prefetch",
        action="store_true",
        help=(
            "Reviewed staged-speed extension: prefetch exactly one private-generator CPU FAS batch. "
            "Default off; requires the complete fixed-B4096 static-ray-cache recipe."
        ),
    )
    parser.add_argument(
        "--tcnn-network-jit-switch-step",
        type=int,
        default=None,
        help="Reviewed speed path: enable the selected TCNN field MLP JIT scope before this update.",
    )
    parser.add_argument(
        "--tcnn-network-jit-scope",
        choices=("both", "geometry", "color"),
        default=None,
        help="TCNN MLP scope enabled by --tcnn-network-jit-switch-step.",
    )
    parser.add_argument(
        "--tcnn-network-jit-second-switch-step",
        type=int,
        default=None,
        help="Reviewed speed path: enable a second TCNN field MLP JIT scope before this later update.",
    )
    parser.add_argument(
        "--tcnn-network-jit-second-switch-scope",
        choices=("both", "geometry", "color"),
        default=None,
        help="TCNN MLP scope enabled by --tcnn-network-jit-second-switch-step.",
    )
    parser.add_argument(
        "--feature-reweighting-switch-step",
        type=int,
        default=None,
        help="Reviewed hard-speed path: change FR strength in-process before this update.",
    )
    parser.add_argument(
        "--feature-reweighting-after-switch",
        type=float,
        default=None,
        help="FR strength applied by --feature-reweighting-switch-step.",
    )
    parser.add_argument(
        "--replay-eval-trajectory",
        action="store_true",
        help="Replace intermediate eval forwards with an exact sampler/dataloader trajectory replay.",
    )
    parser.add_argument(
        "--historical-stage-boundary-rng-reset",
        action="store_true",
        help=(
            "At the Stage-A -> FR0.3 process boundary, remove only the checkpoint RNG snapshot so "
            "the new seeded process keeps its post-setup streams, matching the archived leader restart."
        ),
    )
    parser.add_argument(
        "--speed-final-step",
        type=int,
        choices=(ACCEPTED_STEP,),
        default=None,
        help="Reviewed <=60-minute speed candidate boundary; selects only hard checkpoint step 91128.",
    )
    parser.add_argument("--eval-num-rays-per-chunk", type=int, choices=(2048, 4096, 8192, 16384), default=2048)
    parser.add_argument(
        "--stable-occupancy-reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the accepted stable reducer that updates duplicate nerfacc occupancy cells once by per-cell max. "
            "Disable only together with the legacy historical worktree for forensic controls."
        ),
    )
    parser.add_argument(
        "--tcnn-grid-grad-fp32",
        action="store_true",
        help="Precision ablation: require an overlay that accumulates TCNN hash-grid gradients in FP32.",
    )
    parser.add_argument(
        "--independent-rng-streams",
        action="store_true",
        help="Variance ablation: derive separate per-step pixel, occupancy, and frequency-grid RNG streams.",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--automatic-finalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fresh-evaluate scheduled numeric candidates in order and record the first automatic clean pass.",
    )
    parser.add_argument("--skip-provenance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_recipe(args: argparse.Namespace) -> ResolvedRecipe:
    scale = int(args.batch_scale)
    target_points = int(args.target_points)
    lr_scale = float(args.lr_scale)
    if target_points < 0:
        raise ValueError("--target-points must be non-negative")
    if target_points > 0 and scale > 1:
        raise ValueError("Dynamic point-budget control and point-normalized fixed batch scaling are separate ablations")

    first_jit_switch = (args.tcnn_network_jit_switch_step, args.tcnn_network_jit_scope)
    if (first_jit_switch[0] is None) != (first_jit_switch[1] is None):
        raise ValueError(
            "--tcnn-network-jit-switch-step and --tcnn-network-jit-scope must be set together"
        )
    second_jit_switch = (
        args.tcnn_network_jit_second_switch_step,
        args.tcnn_network_jit_second_switch_scope,
    )
    if (second_jit_switch[0] is None) != (second_jit_switch[1] is None):
        raise ValueError(
            "--tcnn-network-jit-second-switch-step and "
            "--tcnn-network-jit-second-switch-scope must be set together"
        )
    if second_jit_switch[0] is not None and first_jit_switch[0] is None:
        raise ValueError("The second TCNN JIT switch requires the first TCNN JIT switch")

    live_switches = (args.fused_adam_switch_step, args.tcnn_network_jit_switch_step)
    if (live_switches[0] is None) != (live_switches[1] is None):
        raise ValueError("The reviewed live speed path requires both fused Adam and TCNN JIT switches")

    fr_switch = (args.feature_reweighting_switch_step, args.feature_reweighting_after_switch)
    if (fr_switch[0] is None) != (fr_switch[1] is None):
        raise ValueError(
            "--feature-reweighting-switch-step and --feature-reweighting-after-switch must be set together"
        )
    if fr_switch[0] is not None:
        raise ValueError("A live feature-reweighting switch is not part of the reviewed staged speed recipe")

    reviewed_candidate_requested = bool(
        args.cache_train_rays
        or args.cpu_fas_prefetch
        or live_switches[0] is not None
        or first_jit_switch[1] is not None
        or second_jit_switch[0] is not None
        or second_jit_switch[1] is not None
        or args.replay_eval_trajectory
        or args.historical_stage_boundary_rng_reset
        or args.speed_final_step is not None
    )
    if reviewed_candidate_requested:
        staged_recipe = {
            "cache_train_rays": bool(args.cache_train_rays),
            "fused_adam_switch_step": (
                int(live_switches[0]) if live_switches[0] is not None else None
            ),
            "tcnn_network_jit_switch_step": (
                int(first_jit_switch[0]) if first_jit_switch[0] is not None else None
            ),
            "tcnn_network_jit_scope": first_jit_switch[1],
            "tcnn_network_jit_second_switch_step": (
                int(second_jit_switch[0]) if second_jit_switch[0] is not None else None
            ),
            "tcnn_network_jit_second_switch_scope": second_jit_switch[1],
            "replay_eval_trajectory": bool(args.replay_eval_trajectory),
            "historical_stage_boundary_rng_reset": bool(args.historical_stage_boundary_rng_reset),
            "speed_final_step": (
                int(args.speed_final_step) if args.speed_final_step is not None else None
            ),
        }
        if staged_recipe != REVIEWED_STAGED_SPEED_RECIPE:
            mismatched = {
                key: {"actual": staged_recipe[key], "expected": expected}
                for key, expected in REVIEWED_STAGED_SPEED_RECIPE.items()
                if staged_recipe[key] != expected
            }
            raise ValueError(
                "Only the complete reviewed staged speed recipe is supported; "
                f"mismatched={mismatched}"
            )
        if args.fused_adam:
            raise ValueError("Initial fused Adam and delayed fused Adam are mutually exclusive")
        if (
            scale != 1
            or target_points > 0
            or args.corrected_arm_allocator
            or args.speed_stop_at_accepted_boundary
            or lr_scale != 1.0
        ):
            raise ValueError(
                "The reviewed staged speed recipe requires the exact historical B4096 optimizer schedule"
            )
        if args.tcnn_overlay != DEFAULT_JIT_TCNN_OVERLAY:
            raise ValueError(f"The live JIT path requires --tcnn-overlay {DEFAULT_JIT_TCNN_OVERLAY}")
        if args.cpu_fas_prefetch and not args.cache_train_rays:
            raise ValueError("--cpu-fas-prefetch requires the reviewed static training-ray cache")

    speed_mode = bool(
        args.speed_stop_at_accepted_boundary
        or scale > 1
        or target_points > 0
        or args.corrected_arm_allocator
        or args.fused_adam
        or args.cache_train_rays
        or args.cpu_fas_prefetch
        or live_switches[0] is not None
        or first_jit_switch[1] is not None
        or second_jit_switch[0] is not None
        or second_jit_switch[1] is not None
        or fr_switch[0] is not None
        or args.replay_eval_trajectory
        or args.historical_stage_boundary_rng_reset
        or args.speed_final_step is not None
    )
    if not 0.0 < lr_scale <= 4.0:
        raise ValueError("--lr-scale must be in (0, 4]")
    if not speed_mode and lr_scale != 1.0:
        raise ValueError("--lr-scale is an explicit speed ablation and requires speed mode")
    expected_interval = 15_188 // scale if speed_mode else 15_188
    if args.checkpoint_interval is not None and int(args.checkpoint_interval) != expected_interval:
        raise ValueError(
            "The point-normalized recipe derives checkpoint cadence; "
            f"expected {expected_interval}, got {args.checkpoint_interval}."
        )
    warmup = ADAPTIVE_WARMUP_STEPS // scale if speed_mode else ADAPTIVE_WARMUP_STEPS
    rays = TRAIN_RAYS_PER_BATCH * scale if speed_mode else TRAIN_RAYS_PER_BATCH
    fixed_points = warmup * rays * FIXED_SAMPLES_PER_RAY
    if fixed_points != FIXED_WARMUP_POINT_SAMPLES:
        raise AssertionError("Point-normalized warmup must preserve the exact fixed point exposure.")
    if target_points > 0:
        # The dynamic controller changes rays during the warmup, so only the exact
        # checkpointed cumulative counter is valid for this ablation.
        fixed_points = 0
    return ResolvedRecipe(
        speed_mode=speed_mode,
        batch_scale=scale,
        lr_scale=lr_scale,
        fields_lr=FIELDS_LR * lr_scale,
        fields_lr_final=FIELDS_LR_FINAL * lr_scale,
        train_rays_per_batch=rays,
        adaptive_warmup_steps=warmup,
        occupancy_warmup_steps=warmup,
        occupancy_binary_warmup_steps=warmup,
        occupancy_update_interval=16 // scale if speed_mode else 16,
        grid_update_interval=1024 // scale if speed_mode else 1024,
        depth_loss_steps=5000 // scale if speed_mode else 5000,
        scheduler_max_steps=SCHEDULER_MAX_STEPS // scale if speed_mode else SCHEDULER_MAX_STEPS,
        checkpoint_interval=expected_interval,
        save_interval=(
            int(args.speed_final_step) + 1
            if args.speed_final_step is not None
            else expected_interval
        ),
        parent_step=PARENT_STEP // scale if speed_mode else PARENT_STEP,
        final_step=(
            int(args.speed_final_step)
            if args.speed_final_step is not None
            else ACCEPTED_STEP // scale
            if speed_mode
            else FINAL_STEP
        ),
        fixed_warmup_point_samples=fixed_points,
        target_num_samples_per_batch=target_points,
        corrected_arm_allocator=bool(args.corrected_arm_allocator),
        cache_train_rays=bool(args.cache_train_rays),
        cpu_fas_prefetch=bool(args.cpu_fas_prefetch),
        fused_adam=bool(args.fused_adam),
        fused_adam_switch_step=(int(live_switches[0]) if live_switches[0] is not None else None),
        tcnn_network_jit_switch_step=(int(live_switches[1]) if live_switches[1] is not None else None),
        tcnn_network_jit_scope=first_jit_switch[1],
        tcnn_network_jit_second_switch_step=(
            int(second_jit_switch[0]) if second_jit_switch[0] is not None else None
        ),
        tcnn_network_jit_second_switch_scope=second_jit_switch[1],
        feature_reweighting_switch_step=(int(fr_switch[0]) if fr_switch[0] is not None else None),
        feature_reweighting_after_switch=(float(fr_switch[1]) if fr_switch[1] is not None else None),
        replay_eval_trajectory=bool(args.replay_eval_trajectory),
        historical_stage_boundary_rng_reset=bool(args.historical_stage_boundary_rng_reset),
        hard_candidate_only=args.speed_final_step is not None,
        wall_milestone_seconds=3600 if args.speed_final_step is not None else None,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_identity(path: Path) -> Dict[str, int]:
    """Return the stable local-file identity used to bind a previously computed hash."""
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def sha256_checkpoint(path: Path) -> tuple[str, Dict[str, int]]:
    """Hash a checkpoint and fail if its local-file identity changes while reading it."""
    before = checkpoint_file_identity(path)
    digest = sha256_file(path)
    after = checkpoint_file_identity(path)
    if before != after:
        raise RuntimeError(f"Checkpoint changed while hashing: {path}")
    return digest, after


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_output(command: Iterable[str], *, cwd: Path, env: Dict[str, str]) -> str:
    return subprocess.check_output(list(command), cwd=cwd, env=env, text=True).strip()


def require_frozen_values(label: str, actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """Fail closed when any reviewed frozen provenance value is absent or differs."""
    mismatched = {
        name: {"actual": actual.get(name), "expected": expected_value}
        for name, expected_value in expected.items()
        if actual.get(name) != expected_value
    }
    if mismatched:
        raise RuntimeError(f"{label} fingerprint mismatch: {mismatched}")


def validate_stage_boundary_rng_reset_provenance(
    provenance: Dict[str, Any], *, source_sha256: str, output_sha256: str
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Prove that the checkpoint fork removed only the persisted RNG snapshot."""
    before = provenance.get("before")
    after = provenance.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError("Stage-boundary RNG reset sidecar omitted checkpoint state snapshots")
    expected_after = dict(before)
    expected_after["rng_state_present"] = False
    expected_after["rng_state"] = None
    if before.get("rng_state_present") is not True or after != expected_after:
        raise RuntimeError("Stage-boundary fork changed state beyond removing the persisted RNG snapshot")
    if provenance.get("source_sha256") != source_sha256 or provenance.get("output_sha256") != output_sha256:
        raise RuntimeError("Stage-boundary RNG reset sidecar hashes do not bind the actual checkpoints")
    actions = provenance.get("actions")
    if not isinstance(actions, dict) or actions.get("drop_rng_state") is not True:
        raise RuntimeError("Stage-boundary RNG reset sidecar does not record --drop-rng-state")
    disallowed_actions = {
        "reset_adam",
        "restart_scheduler",
        "reset_scaler",
        "reset_torch_cpu_rng_seed",
    }
    if any(actions.get(name) not in (False, None) for name in disallowed_actions):
        raise RuntimeError("Stage-boundary RNG reset sidecar records an additional state mutation")
    if float(actions.get("lr_multiplier", 1.0)) != 1.0 or float(actions.get("scheduler_time_scale", 1.0)) != 1.0:
        raise RuntimeError("Stage-boundary RNG reset sidecar records an LR/scheduler mutation")
    return before, after


def controller_protocol_fingerprint() -> tuple[str, Dict[str, str]]:
    """Hash selection/finalization code and references without a circular self-hash."""
    scripts = Path(__file__).resolve().parent
    paths = {
        "controller": Path(__file__).resolve(),
        "speed_controller": scripts / "run_static_leader_speed_e2e.py",
        "candidate_evaluator": scripts / "evaluate_static_leader_candidate.py",
        "candidate_recorder": scripts / "record_static_leader_candidate.py",
        "retry_finalizer": scripts / "finalize_static_leader_campaign.py",
        "dataset_provenance": DEFAULT_PROVENANCE_SCRIPT,
        "detail_scorer": PROTOCOL_EXTERNAL_DETAIL_SCORER,
        "detail_reference": PROTOCOL_EXTERNAL_DETAIL_REFERENCE,
        "checkpoint_fork": STAGE_BOUNDARY_CHECKPOINT_FORK,
    }
    digest = hashlib.sha256()
    source_hashes: Dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise RuntimeError(f"Controller protocol input is missing: {path}")
        payload = path.read_bytes()
        if name == "controller":
            payload, replacements = re.subn(
                rb'EXPECTED_CONTROLLER_PROTOCOL_FINGERPRINT = "[^"]+"',
                b'EXPECTED_CONTROLLER_PROTOCOL_FINGERPRINT = "<normalized-self-hash>"',
                payload,
                count=1,
            )
            if replacements != 1:
                raise RuntimeError("Could not normalize controller protocol self-hash")
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest(), source_hashes


def accepted_stable_fp16_mode(args: argparse.Namespace) -> bool:
    return (
        bool(args.stable_occupancy_reduction)
        and not bool(args.tcnn_grid_grad_fp32)
        and not bool(args.independent_rng_streams)
        and not speed_mode(args)
    )


def speed_mode(args: argparse.Namespace) -> bool:
    return bool(
        args.speed_stop_at_accepted_boundary
        or int(args.batch_scale) > 1
        or int(args.target_points) > 0
        or args.corrected_arm_allocator
        or args.fused_adam
        or args.cache_train_rays
        or args.cpu_fas_prefetch
        or args.fused_adam_switch_step is not None
        or args.tcnn_network_jit_switch_step is not None
        or args.tcnn_network_jit_scope is not None
        or args.tcnn_network_jit_second_switch_step is not None
        or args.tcnn_network_jit_second_switch_scope is not None
        or args.feature_reweighting_switch_step is not None
        or args.feature_reweighting_after_switch is not None
        or args.replay_eval_trajectory
        or args.historical_stage_boundary_rng_reset
        or args.speed_final_step is not None
    )


def runtime_provenance(args: argparse.Namespace, env: Dict[str, str]) -> Dict[str, Any]:
    payload = command_output(
        [
            str(args.venv / "bin" / "python"),
            "-c",
            (
                "import json,platform,torch; "
                "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
                "'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),"
                "'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,"
                "'cudnn_tf32':torch.backends.cudnn.allow_tf32,"
                "'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}))"
            ),
        ],
        cwd=args.historical_worktree,
        env=env,
    )
    result = json.loads(payload)
    require_frozen_values(
        "Accepted runtime",
        result,
        {
            "python": EXPECTED_PYTHON_VERSION,
            "torch": EXPECTED_TORCH_VERSION,
            "torch_cuda": EXPECTED_TORCH_CUDA,
            "gpu": EXPECTED_GPU_NAME,
            "matmul_tf32": False,
            "cudnn_tf32": True,
            "deterministic_algorithms": False,
        },
    )
    return result


def historical_environment(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    venv_bin = args.venv / "bin"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    python_paths = [str(args.historical_worktree)]
    if args.tcnn_overlay is not None:
        python_paths.insert(0, str(args.tcnn_overlay))
    prior_pythonpath = env.get("PYTHONPATH", "")
    if prior_pythonpath:
        python_paths.append(prior_pythonpath)
    env["PYTHONPATH"] = ":".join(python_paths)
    env.setdefault("PYTHONHASHSEED", "0")
    # CUDA 12.6 does not expose native sm_120 code generation.  The already
    # validated Blackwell path compiles compute_90 PTX and lets the driver JIT
    # it for sm_120.  Keep the cache path stable across stages and repeats.
    env["CUDA_HOME"] = "/usr/local/cuda-12.6"
    env["PATH"] = f"/usr/local/cuda-12.6/bin:{env['PATH']}"
    env["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
    env["TORCH_EXTENSIONS_DIR"] = str(DEFAULT_TORCH_EXTENSIONS)
    return env


def common_runner_args(
    args: argparse.Namespace,
    recipe: ResolvedRecipe,
    seed: int,
    experiment: str,
    timestamp: str,
) -> List[str]:
    runner = args.historical_worktree / "LookCloser" / "scripts" / "run_lookcloser_quiet.py"
    command = [
        str(args.venv / "bin" / "python"),
        str(runner),
        "--data",
        str(args.data),
        "--output-dir",
        str(args.output_dir),
        "--experiment-name",
        experiment,
        "--timestamp",
        timestamp,
        "--seed",
        str(seed),
        "--scene-scale",
        "1.5",
        "--scale-factor",
        "1.0",
        "--max-res",
        "8192",
        "--ray-sampling-mode",
        "adaptive",
        "--max-steps-per-ray",
        "1024",
        "--adaptive-coarse-step-size",
        "0.00625",
        "--adaptive-warmup-steps",
        str(recipe.adaptive_warmup_steps),
        "--reconstruction-loss-type",
        "charbonnier",
        "--distortion-loss-mult",
        "0.01",
        "--depth-loss-steps",
        str(recipe.depth_loss_steps),
        "--grid-resolution",
        "128",
        "--grid-update-interval",
        str(recipe.grid_update_interval),
        "--background-color",
        "black",
        "--occupancy-warmup-steps",
        str(recipe.occupancy_warmup_steps),
        "--occupancy-binary-warmup-steps",
        str(recipe.occupancy_binary_warmup_steps),
        "--occupancy-update-interval",
        str(recipe.occupancy_update_interval),
        "--train-num-rays-per-batch",
        str(recipe.train_rays_per_batch),
        "--eval-num-rays-per-chunk",
        str(args.eval_num_rays_per_chunk),
        "--step-interval",
        str(recipe.checkpoint_interval),
        "--save-interval",
        str(recipe.save_interval),
        "--no-stop-on-no-improve",
        "--eval-checkpoint",
        "latest",
        "--keep-all-checkpoints",
        "--no-update-summary",
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.stable_occupancy_reduction:
        command.append("--stable-occupancy-reduction")
    if args.independent_rng_streams:
        command.append("--independent-rng-streams")
    if recipe.speed_mode:
        command.extend(
            [
                "--fields-lr",
                str(recipe.fields_lr),
                "--fields-lr-final",
                str(recipe.fields_lr_final),
                "--fields-scheduler-max-steps",
                str(recipe.scheduler_max_steps),
            ]
        )
    if recipe.target_num_samples_per_batch > 0:
        command.extend(
            [
                "--target-num-samples-per-batch",
                str(recipe.target_num_samples_per_batch),
                "--dynamic-rays-start-step",
                str(recipe.adaptive_warmup_steps),
            ]
        )
    if recipe.corrected_arm_allocator:
        command.append("--corrected-arm-allocator")
    if recipe.cache_train_rays:
        command.append("--cache-train-rays")
    if recipe.cpu_fas_prefetch:
        command.append("--cpu-fas-prefetch")
    if recipe.fused_adam:
        command.append("--fused-adam")
    if recipe.fused_adam_switch_step is not None:
        command.extend(["--fused-adam-switch-step", str(recipe.fused_adam_switch_step)])
    if recipe.tcnn_network_jit_switch_step is not None:
        command.extend(
            ["--tcnn-network-jit-switch-step", str(recipe.tcnn_network_jit_switch_step)]
        )
    if recipe.tcnn_network_jit_scope is not None:
        command.extend(["--tcnn-network-jit-scope", recipe.tcnn_network_jit_scope])
    if recipe.tcnn_network_jit_second_switch_step is not None:
        command.extend(
            [
                "--tcnn-network-jit-second-switch-step",
                str(recipe.tcnn_network_jit_second_switch_step),
            ]
        )
    if recipe.tcnn_network_jit_second_switch_scope is not None:
        command.extend(
            [
                "--tcnn-network-jit-second-switch-scope",
                recipe.tcnn_network_jit_second_switch_scope,
            ]
        )
    if recipe.replay_eval_trajectory:
        command.append("--replay-eval-trajectory")
    return command


def stage_a_phase_args(recipe: ResolvedRecipe) -> List[str]:
    """Build the FR ancestry for Stage A without leaking a completed live switch into Stage B."""
    command: List[str] = []
    if recipe.feature_reweighting_switch_step is not None:
        command.extend(
            [
                "--feature-reweighting-switch-step",
                str(recipe.feature_reweighting_switch_step),
                "--feature-reweighting-after-switch",
                str(recipe.feature_reweighting_after_switch),
            ]
        )
    command.extend(
        [
            "--feature-reweighting-strength",
            "1.0",
            "--max-num-iterations",
            str(recipe.parent_step + 1),
            "--no-render-final",
        ]
    )
    return command


def run_path(output: Path, experiment: str, timestamp: str) -> Path:
    return output / experiment / "lookcloser" / timestamp


def checkpoint_path(path: Path, step: int) -> Path:
    return path / "nerfstudio_models" / f"step-{step:09d}.ckpt"


def read_eval_trajectory(metrics_path: Path) -> List[Dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("eval_all_psnr"):
                continue
            rows.append(
                {
                    "step": int(row["step"]),
                    "psnr": float(row["eval_all_psnr"]),
                    "ssim": float(row["eval_all_ssim"]),
                    "lpips": float(row["eval_all_lpips"]),
                    "lr": float(row["lr_fields"]) if row.get("lr_fields") else None,
                }
            )
    return rows


def passes_leader_numeric_gate(row: Dict[str, Any]) -> bool:
    """Require one checkpoint to meet all archived leader metrics simultaneously."""
    return (
        float(row["psnr"]) >= LEADER_GATES["psnr"]
        and float(row["ssim"]) >= LEADER_GATES["ssim"]
        and float(row["lpips"]) <= LEADER_GATES["lpips"]
    )


def estimate_legacy_adaptive_point_samples(metrics_path: Path) -> int:
    """Match the archive's sparse `logged count * logging interval` exposure estimate.

    The fixed marcher used before step 4096 did not emit sample-count telemetry.  The
    historical ~250.035B figure excludes that warmup and sums each 10-step ARM sample,
    so retain the same convention explicitly instead of extrapolating across the gap.
    """
    if not metrics_path.exists():
        return 0
    samples: List[tuple[int, float]] = []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("train_num_samples_per_batch"):
                samples.append((int(row["step"]), float(row["train_num_samples_per_batch"])))
    if not samples:
        return 0
    intervals = [right[0] - left[0] for left, right in zip(samples, samples[1:]) if right[0] > left[0]]
    logging_interval = int(statistics.median(intervals)) if intervals else 10
    return int(sum(count * logging_interval for _, count in samples))


def read_exact_cumulative_point_samples(metrics_path: Path) -> int | None:
    """Read the telemetry-only exact counter emitted by the speed worktree, if present."""
    if not metrics_path.exists():
        return None
    latest: int | None = None
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("cumulative_point_samples")
            if raw:
                latest = int(float(raw))
    return latest


def run_logged(command: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def gpu_snapshot(env: Dict[str, str], cwd: Path) -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate",
        "--format=csv,noheader",
    ]
    return command_output(command, cwd=cwd, env=env)


class CandidateFinalizationError(RuntimeError):
    """A candidate was not completely and validly evaluated; this is not a quality result."""


def load_completed_candidate_summary(summary_path: Path, returncode: int, step: int) -> Dict[str, Any]:
    """Load a complete gate result or raise so the controller cannot skip an unevaluated candidate."""
    if returncode != 0:
        raise CandidateFinalizationError(f"candidate evaluator exited {returncode} at step {step}")
    if not summary_path.is_file():
        raise CandidateFinalizationError(f"candidate evaluator wrote no summary at step {step}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateFinalizationError(f"invalid candidate summary at step {step}: {exc}") from exc
    required = {
        "numeric_pass",
        "automatic_pass",
        "automatic_gate_complete",
        "detail_pass",
        "detail_gate_complete",
        "quality_pass",
    }
    missing = sorted(required - summary.keys())
    if missing:
        raise CandidateFinalizationError(f"candidate summary at step {step} is missing {missing}")
    if summary["automatic_gate_complete"] is not True:
        raise CandidateFinalizationError(f"automatic artifact/ROI gate incomplete at step {step}")
    if summary["detail_gate_complete"] is not True:
        raise CandidateFinalizationError(f"micro-detail gate incomplete at step {step}")
    expected_quality = bool(
        summary["numeric_pass"] and summary["automatic_pass"] and summary["detail_pass"]
    )
    if summary["quality_pass"] is not expected_quality:
        raise CandidateFinalizationError(f"inconsistent quality gate summary at step {step}")
    return summary


def controller_exit_code(accepted: bool, status: str) -> int:
    """Keep accepted, quality-failed, and not-evaluated/infrastructure outcomes distinct."""
    if accepted:
        return 0
    if status in {"complete_no_accepted_candidate", "complete_quality_pass_wall_fail"}:
        return QUALITY_FAILURE_EXIT_CODE
    return INCOMPLETE_OR_INFRASTRUCTURE_EXIT_CODE


def main() -> int:
    controller_started = time.monotonic()
    args = parse_args()
    recipe = resolve_recipe(args)
    protocol_fingerprint, protocol_sources = controller_protocol_fingerprint()
    require_frozen_values(
        "Controller protocol",
        {"fingerprint": protocol_fingerprint},
        {"fingerprint": EXPECTED_CONTROLLER_PROTOCOL_FINGERPRINT},
    )
    if recipe.speed_mode:
        if args.historical_worktree == DEFAULT_ACCEPTED_WORKTREE:
            args.historical_worktree = DEFAULT_SPEED_WORKTREE
        if args.output_dir == DEFAULT_OUTPUT:
            args.output_dir = DEFAULT_SPEED_OUTPUT
        if not args.stable_occupancy_reduction or args.tcnn_grid_grad_fp32 or args.independent_rng_streams:
            raise ValueError("Batch speed mode freezes stable occupancy, historical FP16 TCNN and shared RNG policy.")
        if not args.automatic_finalization:
            raise ValueError("Batch speed mode requires automatic end-to-end finalization.")
    seed = secrets.randbelow(2**31 - 1) if args.random_seed else args.seed
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_name = args.campaign_name or (
        f"007740_leader_speed_b{recipe.train_rays_per_batch}_seed{seed}_{timestamp}"
        if recipe.speed_mode
        else f"007740_leader_e2e_seed{seed}_{timestamp}"
    )
    campaign_dir = args.output_dir / "campaigns" / campaign_name
    stage_a_experiment = f"{campaign_name}_A"
    stage_b_experiment = f"{campaign_name}_A_fw03"
    stage_a_path = run_path(args.output_dir, stage_a_experiment, timestamp)
    stage_b_path = run_path(args.output_dir, stage_b_experiment, timestamp)
    parent_checkpoint = checkpoint_path(stage_a_path, recipe.parent_step)
    stage_b_parent_checkpoint = (
        campaign_dir / f"stage_a_step-{recipe.parent_step:09d}_historical_rng_reset.ckpt"
        if recipe.historical_stage_boundary_rng_reset
        else parent_checkpoint
    )
    final_checkpoint = checkpoint_path(stage_b_path, recipe.final_step)
    env = historical_environment(args)

    if (args.tcnn_overlay is None) != (args.tcnn_source_worktree is None):
        raise RuntimeError("--tcnn-overlay and --tcnn-source-worktree must be provided together")
    if args.tcnn_grid_grad_fp32 and args.tcnn_overlay is None:
        raise RuntimeError("--tcnn-grid-grad-fp32 requires --tcnn-overlay and --tcnn-source-worktree")
    if args.independent_rng_streams and not args.stable_occupancy_reduction:
        raise RuntimeError("--independent-rng-streams currently requires --stable-occupancy-reduction")
    tcnn_provenance: Dict[str, Any]
    if args.tcnn_overlay is not None and args.tcnn_source_worktree is not None:
        if not args.tcnn_overlay.is_dir():
            raise RuntimeError(f"tiny-cuda-nn overlay does not exist: {args.tcnn_overlay}")
        tcnn_commit = command_output(
            ["git", "rev-parse", "HEAD"], cwd=args.tcnn_source_worktree, env=env
        )
        expected_tcnn_commit = command_output(
            ["git", "rev-parse", HISTORICAL_TCNN_COMMIT], cwd=args.tcnn_source_worktree, env=env
        )
        if tcnn_commit != expected_tcnn_commit:
            raise RuntimeError(f"tiny-cuda-nn source is {tcnn_commit}, expected {expected_tcnn_commit}")
        tcnn_dirty_lines = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=args.tcnn_source_worktree,
            env=env,
            text=True,
        ).splitlines()
        tcnn_dirty_paths = {line[3:] for line in tcnn_dirty_lines if line}
        allowed_tcnn_patches = set(ALLOWED_TCNN_COMPATIBILITY_PATCHES)
        if args.tcnn_grid_grad_fp32:
            allowed_tcnn_patches.add(TCNN_GRID_GRAD_FP32_PATCH)
            grid_grad_source = args.tcnn_source_worktree / TCNN_GRID_GRAD_FP32_PATCH
            if sha256_file(grid_grad_source) != EXPECTED_TCNN_GRID_GRAD_FP32_SHA256:
                raise RuntimeError("TCNN FP32 grid-gradient source hash does not match the reviewed ablation")
        if tcnn_dirty_paths != allowed_tcnn_patches:
            raise RuntimeError(
                f"Unexpected tiny-cuda-nn worktree changes: {sorted(tcnn_dirty_paths)}; "
                f"allowed={sorted(allowed_tcnn_patches)}"
            )
        tcnn_diff = subprocess.check_output(
            ["git", "diff", "--binary"], cwd=args.tcnn_source_worktree, env=env
        )
        tcnn_diff_sha256 = hashlib.sha256(tcnn_diff).hexdigest()
        build_provenance_path = args.tcnn_overlay / "build_provenance.json"
        if not build_provenance_path.is_file():
            raise RuntimeError(f"Missing tiny-cuda-nn build provenance: {build_provenance_path}")
        build_provenance = json.loads(build_provenance_path.read_text(encoding="utf-8"))
        if build_provenance.get("source_commit") != tcnn_commit:
            raise RuntimeError("tiny-cuda-nn build provenance source commit does not match worktree")
        if build_provenance.get("source_diff_sha256") != tcnn_diff_sha256:
            raise RuntimeError("tiny-cuda-nn build provenance diff does not match worktree")
        tcnn_provenance = {
            "mode": "historical_overlay",
            "commit": tcnn_commit,
            "source_worktree": str(args.tcnn_source_worktree),
            "source_diff_sha256": tcnn_diff_sha256,
            "compatibility_patch_paths": sorted(allowed_tcnn_patches),
            "grid_grad_fp32": bool(args.tcnn_grid_grad_fp32),
            "overlay": str(args.tcnn_overlay),
            "build_provenance": str(build_provenance_path),
            "build_provenance_sha256": sha256_file(build_provenance_path),
        }
        if recipe.tcnn_network_jit_switch_step is not None:
            rtc_provenance_path = args.tcnn_overlay / "rtc_overlay_provenance.json"
            if not rtc_provenance_path.is_file():
                raise RuntimeError(f"Missing TCNN RTC overlay provenance: {rtc_provenance_path}")
            rtc_provenance_sha256 = sha256_file(rtc_provenance_path)
            if rtc_provenance_sha256 != EXPECTED_JIT_TCNN_OVERLAY_PROVENANCE:
                raise RuntimeError("TCNN RTC overlay provenance hash does not match the reviewed live-JIT path")
            rtc_provenance = json.loads(rtc_provenance_path.read_text(encoding="utf-8"))
            require_frozen_values(
                "TCNN RTC overlay",
                rtc_provenance,
                {
                    "source_commit": HISTORICAL_TCNN_COMMIT,
                    "binding_sha256": EXPECTED_ACCEPTED_TCNN_BINDING,
                    "status": "complete",
                },
            )
            tcnn_provenance["rtc_overlay_provenance"] = str(rtc_provenance_path)
            tcnn_provenance["rtc_overlay_provenance_sha256"] = rtc_provenance_sha256
    else:
        tcnn_provenance = {"mode": "environment_default"}
    tcnn_imports = command_output(
        [
            str(args.venv / "bin" / "python"),
            "-c",
            "import tinycudann,tinycudann.modules; "
            "print(tinycudann.__file__); print(tinycudann.modules._C.__file__)",
        ],
        cwd=args.historical_worktree,
        env=env,
    ).splitlines()
    tcnn_provenance["import"] = tcnn_imports
    if args.tcnn_overlay is not None:
        binding_path = Path(tcnn_imports[-1])
        binding_sha256 = sha256_file(binding_path)
        if build_provenance.get("binding_sha256") != binding_sha256:
            raise RuntimeError("tiny-cuda-nn imported binding hash does not match build provenance")
        tcnn_provenance["binding_sha256"] = binding_sha256
        if not args.tcnn_grid_grad_fp32:
            require_frozen_values(
                "Accepted stable-FP16 TCNN",
                {
                    "source_diff_sha256": tcnn_diff_sha256,
                    "build_provenance_sha256": tcnn_provenance["build_provenance_sha256"],
                    "binding_sha256": binding_sha256,
                },
                {
                    "source_diff_sha256": EXPECTED_ACCEPTED_TCNN_SOURCE_DIFF,
                    "build_provenance_sha256": EXPECTED_ACCEPTED_TCNN_BUILD_PROVENANCE,
                    "binding_sha256": EXPECTED_ACCEPTED_TCNN_BINDING,
                },
            )

    commit = command_output(
        ["git", "rev-parse", "HEAD"], cwd=args.historical_worktree, env=env
    )
    expected_commit = command_output(
        ["git", "rev-parse", HISTORICAL_COMMIT], cwd=args.historical_worktree, env=env
    )
    if commit != expected_commit:
        raise RuntimeError(f"Historical worktree is {commit}, expected {expected_commit}")
    # Do not strip porcelain output: the leading column is part of Git's XY
    # status and stripping it corrupts the first path (" M file" -> "M file").
    dirty_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=args.historical_worktree,
        env=env,
        text=True,
    ).splitlines()
    dirty_paths = {line[3:] for line in dirty_lines if line}
    allowed_patches = set(ALLOWED_COMPATIBILITY_PATCHES)
    if args.stable_occupancy_reduction:
        allowed_patches.update(STABLE_OCCUPANCY_PATCHES)
    if args.independent_rng_streams:
        allowed_patches.update(INDEPENDENT_RNG_PATCHES)
    if recipe.speed_mode:
        allowed_patches.update(SPEED_TELEMETRY_PATCHES)
    if dirty_paths != allowed_patches:
        raise RuntimeError(
            f"Unexpected historical worktree changes: {sorted(dirty_paths)}; "
            f"allowed={sorted(allowed_patches)}"
        )
    compatibility_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *sorted(allowed_patches)],
        cwd=args.historical_worktree,
        env=env,
    )
    source_sha256 = {
        path: sha256_file(args.historical_worktree / path) for path in sorted(allowed_patches)
    }
    compatibility_fingerprint = compatibility_diff
    if args.stable_occupancy_reduction or args.independent_rng_streams:
        # git diff omits untracked helper/test files. Include every allowed file
        # hash without changing the exact-control fingerprint used by default.
        compatibility_fingerprint += json.dumps(source_sha256, sort_keys=True).encode("utf-8")
    compatibility_patch_sha256 = hashlib.sha256(compatibility_fingerprint).hexdigest()
    if accepted_stable_fp16_mode(args):
        require_frozen_values(
            "Accepted stable-FP16 source",
            {"compatibility_patch_sha256": compatibility_patch_sha256},
            {"compatibility_patch_sha256": EXPECTED_ACCEPTED_SOURCE_FINGERPRINT},
        )
    if recipe.speed_mode:
        require_frozen_values(
            "Speed worktree source",
            {"compatibility_patch_sha256": compatibility_patch_sha256},
            {"compatibility_patch_sha256": EXPECTED_SPEED_SOURCE_FINGERPRINT},
        )
    transforms = args.data / "transforms.json"
    transforms_sha = sha256_file(transforms)
    if transforms_sha != EXPECTED_TRANSFORMS_SHA256:
        raise RuntimeError(f"Unexpected transforms.json SHA-256: {transforms_sha}")

    stage_a_command = common_runner_args(args, recipe, seed, stage_a_experiment, timestamp)
    stage_a_command.extend(stage_a_phase_args(recipe))
    stage_b_command = common_runner_args(args, recipe, seed, stage_b_experiment, timestamp)
    stage_b_command.extend(
        [
            "--feature-reweighting-strength",
            "0.3",
            "--max-num-iterations",
            str(recipe.final_step + 1),
            "--load-checkpoint",
            str(stage_b_parent_checkpoint),
            "--artifact-render-names",
            "eval_img_0000.png,eval_img_0001.png,eval_img_0002.png",
            "--artifact-detector-preset",
            "significant",
        ]
    )
    if args.automatic_finalization:
        # The selector below renders the first scheduled clean numeric candidate.
        # Avoid a redundant unconditional render of the latest checkpoint here.
        stage_b_command.append("--no-render-final")

    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "status": "dry_run" if args.dry_run else "initialized",
        "created_at": utc_now(),
        "campaign_name": campaign_name,
        "seed": seed,
        "seed_policy": "random_recorded" if args.random_seed else "explicit",
        "historical_commit": commit,
        "historical_worktree": str(args.historical_worktree),
        "compatibility_patch_paths": sorted(allowed_patches),
        "compatibility_patch_sha256": compatibility_patch_sha256,
        "compatibility_source_sha256": source_sha256,
        "controller_protocol_fingerprint": protocol_fingerprint,
        "controller_protocol_source_sha256": protocol_sources,
        "algorithmic_ablation": {
            "stable_occupancy_reduction": bool(args.stable_occupancy_reduction),
            "tcnn_grid_grad_fp32": bool(args.tcnn_grid_grad_fp32),
            "independent_rng_streams": bool(args.independent_rng_streams),
            "point_normalized_batch_scale": recipe.batch_scale if recipe.speed_mode else None,
            "fields_lr_scale": recipe.lr_scale if recipe.speed_mode else None,
            "target_num_samples_per_batch": (
                recipe.target_num_samples_per_batch if recipe.target_num_samples_per_batch > 0 else None
            ),
            "corrected_arm_allocator": recipe.corrected_arm_allocator,
            "cache_train_rays": recipe.cache_train_rays,
            "cpu_fas_prefetch": recipe.cpu_fas_prefetch,
            "fused_adam": recipe.fused_adam,
            "fused_adam_switch_step": recipe.fused_adam_switch_step,
            "tcnn_network_jit_switch_step": recipe.tcnn_network_jit_switch_step,
            "tcnn_network_jit_scope": recipe.tcnn_network_jit_scope,
            "tcnn_network_jit_second_switch_step": recipe.tcnn_network_jit_second_switch_step,
            "tcnn_network_jit_second_switch_scope": recipe.tcnn_network_jit_second_switch_scope,
            "feature_reweighting_switch_step": recipe.feature_reweighting_switch_step,
            "feature_reweighting_after_switch": recipe.feature_reweighting_after_switch,
            "replay_eval_trajectory": recipe.replay_eval_trajectory,
            "historical_stage_boundary_rng_reset": recipe.historical_stage_boundary_rng_reset,
            "hard_candidate_only": recipe.hard_candidate_only,
            "wall_milestone_seconds": recipe.wall_milestone_seconds,
        },
        "resolved_recipe": asdict(recipe),
        "scheduler_max_steps": recipe.scheduler_max_steps,
        "cuda_home": env["CUDA_HOME"],
        "torch_cuda_arch_list": env["TORCH_CUDA_ARCH_LIST"],
        "torch_extensions_dir": env["TORCH_EXTENSIONS_DIR"],
        "tiny_cuda_nn": tcnn_provenance,
        "runtime": runtime_provenance(args, env),
        "data": str(args.data),
        "transforms_sha256": transforms_sha,
        "checkpoint_interval": recipe.checkpoint_interval,
        "save_interval": recipe.save_interval,
        "numeric_gates": LEADER_GATES,
        "automatic_finalization": bool(args.automatic_finalization),
        "point_exposure_definition": {
            "legacy_adaptive": "sum(logged train_num_samples_per_batch * median logging interval)",
            "fixed_warmup_point_samples": recipe.fixed_warmup_point_samples,
            "dynamic_point_budget": (
                "checkpointed exact cumulative_point_samples is authoritative"
                if recipe.target_num_samples_per_batch > 0
                else None
            ),
        },
        "stage_a": {
            "target_step": recipe.parent_step,
            "run_path": str(stage_a_path),
            "checkpoint": str(parent_checkpoint),
            "command": stage_a_command,
        },
        "stage_a_fw03": {
            "target_step": recipe.final_step,
            "run_path": str(stage_b_path),
            "input_checkpoint": str(stage_b_parent_checkpoint),
            "checkpoint": str(final_checkpoint),
            "command": stage_b_command,
        },
    }
    if recipe.historical_stage_boundary_rng_reset:
        manifest["stage_boundary_rng_reset"] = {
            "status": "planned" if args.dry_run else "pending",
            "source_checkpoint": str(parent_checkpoint),
            "output_checkpoint": str(stage_b_parent_checkpoint),
            "command": [
                str(args.venv / "bin" / "python"),
                str(STAGE_BOUNDARY_CHECKPOINT_FORK),
                str(parent_checkpoint),
                str(stage_b_parent_checkpoint),
                "--drop-rng-state",
            ],
        }
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return 0

    campaign_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = campaign_dir / "campaign.json"
    manifest["gpu_start"] = gpu_snapshot(env, args.historical_worktree)
    write_json(manifest_path, manifest)
    print(f"campaign={campaign_dir}", flush=True)
    print(f"seed={seed} policy={manifest['seed_policy']}", flush=True)
    print(f"gpu={manifest['gpu_start']}", flush=True)

    if not args.skip_provenance:
        provenance_command = [
            str(args.venv / "bin" / "python"),
            str(DEFAULT_PROVENANCE_SCRIPT),
            "--local",
            str(args.data),
            "--output",
            str(campaign_dir / "dataset_provenance.json"),
        ]
        print("provenance=running", flush=True)
        rc = run_logged(
            provenance_command,
            cwd=args.historical_worktree,
            env=env,
            log_path=campaign_dir / "provenance_stdout.log",
        )
        if rc != 0:
            manifest["status"] = "provenance_failed"
            manifest["provenance_returncode"] = rc
            write_json(manifest_path, manifest)
            raise RuntimeError(f"Dataset provenance failed with exit code {rc}")
        print("provenance=match", flush=True)

    manifest["status"] = "stage_a_running"
    manifest["stage_a"]["started_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"stage=A target_step={recipe.parent_step}", flush=True)
    started = time.monotonic()
    rc = run_logged(
        stage_a_command,
        cwd=args.historical_worktree,
        env=env,
        log_path=campaign_dir / "stage_a_controller.log",
    )
    manifest["stage_a"]["wall_seconds"] = time.monotonic() - started
    manifest["stage_a"]["returncode"] = rc
    manifest["stage_a"]["finished_at"] = utc_now()
    if rc != 0 or not parent_checkpoint.is_file():
        manifest["status"] = "stage_a_failed"
        write_json(manifest_path, manifest)
        raise RuntimeError(f"Stage A failed (returncode={rc}, checkpoint={parent_checkpoint.is_file()})")
    stage_a_sha256, stage_a_identity = sha256_checkpoint(parent_checkpoint)
    manifest["stage_a"]["checkpoint_sha256"] = stage_a_sha256
    manifest["stage_a"]["checkpoint_file_identity"] = stage_a_identity
    manifest["stage_a"]["trajectory"] = read_eval_trajectory(stage_a_path / "metrics_compact.csv")
    stage_a_adaptive_points = estimate_legacy_adaptive_point_samples(stage_a_path / "metrics_compact.csv")
    manifest["stage_a"]["estimated_adaptive_point_samples_legacy"] = stage_a_adaptive_points
    manifest["stage_a"]["fixed_warmup_point_samples"] = recipe.fixed_warmup_point_samples
    manifest["stage_a"]["estimated_total_point_samples"] = (
        stage_a_adaptive_points + recipe.fixed_warmup_point_samples
    )
    stage_a_exact_points = read_exact_cumulative_point_samples(stage_a_path / "metrics_compact.csv")
    manifest["stage_a"]["exact_cumulative_point_samples"] = stage_a_exact_points
    manifest["stage_a"]["authoritative_point_samples"] = (
        stage_a_exact_points
        if recipe.target_num_samples_per_batch > 0 and stage_a_exact_points is not None
        else manifest["stage_a"]["estimated_total_point_samples"]
    )
    write_json(manifest_path, manifest)
    print(
        f"stage=A complete seconds={manifest['stage_a']['wall_seconds']:.1f} "
        f"adaptive_points_legacy={stage_a_adaptive_points} "
        f"total_points={manifest['stage_a']['authoritative_point_samples']}",
        flush=True,
    )

    if recipe.historical_stage_boundary_rng_reset:
        fork_record = manifest["stage_boundary_rng_reset"]
        fork_record["status"] = "running"
        fork_record["started_at"] = utc_now()
        write_json(manifest_path, manifest)
        started = time.monotonic()
        rc = run_logged(
            fork_record["command"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            log_path=campaign_dir / "stage_boundary_rng_reset_controller.log",
        )
        fork_record["wall_seconds"] = time.monotonic() - started
        fork_record["returncode"] = rc
        fork_record["finished_at"] = utc_now()
        sidecar = stage_b_parent_checkpoint.with_suffix(stage_b_parent_checkpoint.suffix + ".fork.json")
        if rc != 0 or not stage_b_parent_checkpoint.is_file() or not sidecar.is_file():
            fork_record["status"] = "failed"
            manifest["status"] = "stage_boundary_rng_reset_failed"
            write_json(manifest_path, manifest)
            raise RuntimeError(
                "Historical Stage-B RNG reset fork failed "
                f"(returncode={rc}, checkpoint={stage_b_parent_checkpoint.is_file()}, sidecar={sidecar.is_file()})"
            )
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        source_sha = sha256_file(parent_checkpoint)
        output_sha = sha256_file(stage_b_parent_checkpoint)
        before, after = validate_stage_boundary_rng_reset_provenance(
            provenance, source_sha256=source_sha, output_sha256=output_sha
        )
        fork_record.update(
            {
                "status": "complete",
                "sidecar": str(sidecar),
                "source_sha256": source_sha,
                "output_sha256": output_sha,
                "before": before,
                "after": after,
            }
        )
        write_json(manifest_path, manifest)
        print(
            f"stage_boundary_rng_reset=complete seconds={fork_record['wall_seconds']:.1f} "
            f"checkpoint={stage_b_parent_checkpoint}",
            flush=True,
        )

    manifest["status"] = "stage_a_fw03_running"
    manifest["stage_a_fw03"]["started_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"stage=A_fw03 target_step={recipe.final_step}", flush=True)
    started = time.monotonic()
    rc = run_logged(
        stage_b_command,
        cwd=args.historical_worktree,
        env=env,
        log_path=campaign_dir / "stage_a_fw03_controller.log",
    )
    manifest["stage_a_fw03"]["wall_seconds"] = time.monotonic() - started
    manifest["stage_a_fw03"]["returncode"] = rc
    manifest["stage_a_fw03"]["finished_at"] = utc_now()
    if rc != 0 or not final_checkpoint.is_file():
        manifest["status"] = "stage_a_fw03_failed"
        write_json(manifest_path, manifest)
        raise RuntimeError(
            f"Stage A_fw03 failed (returncode={rc}, checkpoint={final_checkpoint.is_file()})"
        )
    final_sha256, final_identity = sha256_checkpoint(final_checkpoint)
    manifest["stage_a_fw03"]["checkpoint_sha256"] = final_sha256
    manifest["stage_a_fw03"]["checkpoint_file_identity"] = final_identity
    manifest["stage_a_fw03"]["trajectory"] = read_eval_trajectory(stage_b_path / "metrics_compact.csv")
    continuation_points = estimate_legacy_adaptive_point_samples(stage_b_path / "metrics_compact.csv")
    manifest["stage_a_fw03"]["estimated_adaptive_point_samples_legacy"] = continuation_points
    manifest["stage_a_fw03"]["estimated_total_point_samples"] = continuation_points
    final_exact_points = read_exact_cumulative_point_samples(stage_b_path / "metrics_compact.csv")
    manifest["stage_a_fw03"]["exact_cumulative_point_samples"] = final_exact_points
    manifest["total_estimated_adaptive_point_samples_legacy"] = stage_a_adaptive_points + continuation_points
    manifest["total_estimated_point_samples"] = (
        manifest["total_estimated_adaptive_point_samples_legacy"] + recipe.fixed_warmup_point_samples
    )
    # A resumed dynamic-point run restores the cumulative counter, so the final
    # Stage B value is already the exact campaign total and must not be added to
    # Stage A again. The legacy estimator remains useful for historical runs
    # whose old CSV schema has no exact cumulative counter.
    if recipe.target_num_samples_per_batch > 0:
        if final_exact_points is None:
            raise RuntimeError("Dynamic point-budget run completed without an exact cumulative point counter")
        manifest["total_point_samples"] = final_exact_points
        manifest["point_sample_accounting"] = "exact_checkpointed_cumulative"
    else:
        manifest["total_point_samples"] = manifest["total_estimated_point_samples"]
        manifest["point_sample_accounting"] = "legacy_csv_estimate_plus_fixed_warmup"
    manifest["gpu_end"] = gpu_snapshot(env, args.historical_worktree)
    manifest["status"] = "training_complete"
    write_json(manifest_path, manifest)

    accepted = False
    if args.automatic_finalization:
        finalization_started = time.monotonic()
        evaluator = Path(__file__).with_name("evaluate_static_leader_candidate.py")
        candidate_specs: List[tuple[Path, Path, Dict[str, Any]]] = []
        if recipe.hard_candidate_only:
            candidate_specs.append(
                (
                    stage_b_path,
                    final_checkpoint,
                    {"step": recipe.final_step, "selection": "predeclared_hard_candidate"},
                )
            )
        else:
            for stage_name, stage_path in (("stage_a", stage_a_path), ("stage_a_fw03", stage_b_path)):
                for row in manifest[stage_name]["trajectory"]:
                    if passes_leader_numeric_gate(row):
                        candidate_specs.append(
                            (stage_path, checkpoint_path(stage_path, int(row["step"])), row)
                        )
        candidate_specs.sort(key=lambda item: int(item[2]["step"]))
        attempts: List[Dict[str, Any]] = []
        finalization_error: str | None = None
        manifest["status"] = "candidate_finalization_running"
        write_json(manifest_path, manifest)
        for candidate_run, candidate_checkpoint, row in candidate_specs:
            if not candidate_checkpoint.is_file():
                attempts.append(
                    {
                        "step": int(row["step"]),
                        "checkpoint": str(candidate_checkpoint),
                        "status": "missing_checkpoint",
                    }
                )
                finalization_error = f"missing candidate checkpoint: {candidate_checkpoint}"
                break
            candidate_log = campaign_dir / f"candidate_step_{int(row['step']):09d}_controller.log"
            evaluator_command = [
                str(args.venv / "bin" / "python"),
                str(evaluator),
                "--run-dir",
                str(candidate_run),
                "--checkpoint",
                str(candidate_checkpoint),
                "--campaign",
                str(manifest_path),
                "--historical-runner",
                str(args.historical_worktree / "LookCloser" / "scripts" / "run_lookcloser_quiet.py"),
                "--eval-num-rays-per-chunk",
                str(args.eval_num_rays_per_chunk),
            ]
            rc = run_logged(
                evaluator_command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                log_path=candidate_log,
            )
            summary_path = candidate_run / f"candidate_evaluation_step-{int(row['step']):09d}.json"
            try:
                summary = load_completed_candidate_summary(summary_path, rc, int(row["step"]))
            except CandidateFinalizationError as exc:
                finalization_error = str(exc)
                attempts.append(
                    {
                        "step": int(row["step"]),
                        "checkpoint": str(candidate_checkpoint),
                        "returncode": rc,
                        "summary": str(summary_path),
                        "controller_log": str(candidate_log),
                        "status": "infrastructure_failed",
                    }
                )
                break
            attempt = {
                "step": int(row["step"]),
                "checkpoint": str(candidate_checkpoint),
                "returncode": rc,
                "summary": str(summary_path),
                "controller_log": str(candidate_log),
                "numeric_pass": bool(summary.get("numeric_pass")),
                "automatic_pass": bool(summary.get("automatic_pass")),
                "detail_pass": bool(summary.get("detail_pass")),
                "quality_pass": bool(summary.get("quality_pass")),
            }
            attempts.append(attempt)
            if rc == 0 and attempt["quality_pass"]:
                accepted = True
                break
        # The candidate recorder writes accepted_candidate into the same manifest.
        # Reload before adding controller fields so that record is never overwritten.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["candidate_finalization_attempts"] = attempts
        manifest["finalization_seconds"] = time.monotonic() - finalization_started
        manifest["finalization_error"] = finalization_error
        manifest["status"] = (
            "complete" if accepted else "finalization_failed" if finalization_error else "complete_no_accepted_candidate"
        )
    else:
        manifest["status"] = "complete_unfinalized"
        accepted = False

    manifest["controller_wall_seconds"] = time.monotonic() - controller_started
    manifest["quality_accepted"] = accepted
    if recipe.wall_milestone_seconds is not None:
        manifest["wall_milestone_seconds"] = recipe.wall_milestone_seconds
        manifest["wall_milestone_pass"] = (
            manifest["controller_wall_seconds"] <= recipe.wall_milestone_seconds
        )
        if accepted and not manifest["wall_milestone_pass"]:
            accepted = False
            manifest["status"] = "complete_quality_pass_wall_fail"
    manifest["accepted"] = accepted
    manifest["finished_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(
        f"complete checkpoint={final_checkpoint} total_points={manifest['total_point_samples']} "
        f"accepted={accepted} wall_seconds={manifest['controller_wall_seconds']:.1f} manifest={manifest_path}",
        flush=True,
    )
    return controller_exit_code(accepted, str(manifest["status"]))


if __name__ == "__main__":
    raise SystemExit(main())
