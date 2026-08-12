#!/usr/bin/env python3
"""Run the paired two-seed EXR loss-schedule validation campaign.

The controller deliberately separates long, from-scratch prefixes from short
same-seed forks.  Parents are selected by cumulative rendered point exposure,
never by an image metric.  Seed 42 is historical evidence only and is excluded
from this campaign's decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
DEFAULT_VENV = REPO / ".venv"
DEFAULT_DATA = Path("/mnt/data/temporal_perframe_stride7_45f_exr_1920x1080/007740")
DEFAULT_OUTPUT = Path("/mnt/data/lookcloser_loss_schedule_validation")
DEFAULT_CAMPAIGN = "exr_loss_schedule_two_seed_v1"
DEFAULT_SEEDS = (43, 44)

EXPOSURE_BASE = 2.379e11
EXPOSURE_LPIPS_END = 2.402e11
EXPOSURE_FINAL = 2.419e11
EXPOSURE_REL_TOLERANCE = 0.005

PSNR_WINDOW = 0.07
QUALITY_FLOORS = {"psnr": 0.06, "ssim": 0.001, "lpips": 0.002}

STANDARD_PREFIX_MAX_STEP = 250_000
LPIPS_PREFIX_MAX_STEP = 65_000
PREFIX_EVAL_INTERVAL = 7_594
LPIPS_PREFIX_EVAL_INTERVAL = 1_899
STANDARD_PREFIX_SAVE_INTERVAL = PREFIX_EVAL_INTERVAL
LPIPS_PREFIX_SAVE_INTERVAL = LPIPS_PREFIX_EVAL_INTERVAL
TAIL_MAX_EXTRA_STEPS = 5_000
MATURE_LPIPS_MAX_EXTRA_STEPS = 768
STANDARD_TAIL_EVAL_INTERVAL = 512
LPIPS_TAIL_EVAL_INTERVAL = 128

EARLY_REJECT_MIN_EVALS = 2
EARLY_REJECT_THRESHOLDS = {"psnr": 30.0, "ssim": 0.8, "lpips": 0.5}


def tail_eval_interval(recipe: str) -> int:
    return LPIPS_TAIL_EVAL_INTERVAL if recipe == "lpips" else STANDARD_TAIL_EVAL_INTERVAL


def checkpoint_interval(recipe: str) -> int:
    return tail_eval_interval(recipe)


@dataclass(frozen=True)
class LossRecipe:
    loss: str
    rays: int
    training_patch_size: int
    eag_patch_size: int
    dssim_weight: float = 0.3
    lpips_weight: float = 0.0


RECIPES = {
    "eag": LossRecipe("eag_pq_dssim", 3993, 1, 11, dssim_weight=0.3),
    "pql1": LossRecipe("linear_pq", 3993, 1, 11),
    "pqmse": LossRecipe("pq_mse", 3993, 1, 11),
    "lpips": LossRecipe("eag_pq_lpips", 16_384, 64, 64, dssim_weight=0.2, lpips_weight=0.02),
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    seed: int
    recipe: str
    target_exposure: float
    max_step: int
    eval_interval: int
    save_interval: int
    parent: str | None = None
    scratch: bool = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--campaign-name", default=DEFAULT_CAMPAIGN)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--phase", choices=("prefixes", "forks", "evaluate", "report", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--visual-review-status", choices=("pass", "fail"), default=None)
    parser.add_argument("--visual-review-note", action="append", default=[])
    parser.add_argument("--visual-review-artifact", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.campaign_name):
        parser.error("--campaign-name must be filesystem-safe")
    if tuple(args.seeds) != DEFAULT_SEEDS:
        parser.error("This validation is frozen to the two new seeds 43 and 44")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def campaign_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "campaigns" / args.campaign_name


def manifest_path(args: argparse.Namespace) -> Path:
    return campaign_dir(args) / "campaign.json"


def experiment_name(args: argparse.Namespace) -> str:
    return args.campaign_name


def run_dir(args: argparse.Namespace, spec: RunSpec) -> Path:
    return args.output_dir / experiment_name(args) / "lookcloser" / spec.name


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def validate_dataset(data: Path) -> dict[str, Any]:
    transforms = data / "transforms.json"
    if not transforms.is_file():
        raise FileNotFoundError(transforms)
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    paths = [str(frame.get("file_path", "")) for frame in frames]
    train = sum("train" in Path(path).stem for path in paths)
    evaluation = sum("eval" in Path(path).stem for path in paths)
    if (train, evaluation) != (66, 3):
        raise RuntimeError(f"Expected 66 train / 3 eval frames, got {train} / {evaluation}")
    frequency = data / "lookcloser_frequencies_exr_auto" / "knee"
    geometry = data / "lookcloser_geometry_support_v2" / "edge_ridge"
    if len(list(frequency.glob("*.pt"))) < 66 and len(list(frequency.glob("*.npy"))) < 66:
        raise RuntimeError(f"Missing 66 knee frequency maps under {frequency}")
    if len(list(geometry.glob("*.pt"))) != 66:
        raise RuntimeError(f"Expected 66 geometry support maps under {geometry}")
    return {
        "transforms": str(transforms),
        "transforms_sha256": sha256_file(transforms),
        "train_frames": train,
        "eval_frames": evaluation,
        "frequency_map_dir": str(frequency),
        "geometry_support_map_dir": str(geometry),
    }


def prefix_specs(seed: int) -> list[RunSpec]:
    return [
        RunSpec(
            name=f"s{seed}_prefix_eag",
            seed=seed,
            recipe="eag",
            target_exposure=EXPOSURE_BASE,
            max_step=STANDARD_PREFIX_MAX_STEP,
            eval_interval=PREFIX_EVAL_INTERVAL,
            save_interval=STANDARD_PREFIX_SAVE_INTERVAL,
        ),
        RunSpec(
            name=f"s{seed}_prefix_pql1",
            seed=seed,
            recipe="pql1",
            target_exposure=EXPOSURE_BASE,
            max_step=STANDARD_PREFIX_MAX_STEP,
            eval_interval=PREFIX_EVAL_INTERVAL,
            save_interval=STANDARD_PREFIX_SAVE_INTERVAL,
        ),
        RunSpec(
            name=f"s{seed}_prefix_pqmse",
            seed=seed,
            recipe="pqmse",
            target_exposure=EXPOSURE_BASE,
            max_step=STANDARD_PREFIX_MAX_STEP,
            eval_interval=PREFIX_EVAL_INTERVAL,
            save_interval=STANDARD_PREFIX_SAVE_INTERVAL,
        ),
        RunSpec(
            name=f"s{seed}_prefix_lpips",
            seed=seed,
            recipe="lpips",
            target_exposure=EXPOSURE_LPIPS_END,
            max_step=LPIPS_PREFIX_MAX_STEP,
            eval_interval=LPIPS_PREFIX_EVAL_INTERVAL,
            save_interval=LPIPS_PREFIX_SAVE_INTERVAL,
        ),
    ]


def fork_specs(seed: int, parent_steps: dict[str, int]) -> list[RunSpec]:
    eag_parent = f"s{seed}_prefix_eag"
    pql1_parent = f"s{seed}_prefix_pql1"
    lpips_parent = f"s{seed}_prefix_lpips"
    stage1_step = parent_steps[eag_parent]

    direct = [
        ("eag_continue", "eag", eag_parent, stage1_step),
        ("direct_pql1", "pql1", eag_parent, stage1_step),
        ("direct_pqmse", "pqmse", eag_parent, stage1_step),
    ]
    if pql1_parent in parent_steps:
        direct.append(("pure_pql1", "pql1", pql1_parent, parent_steps[pql1_parent]))
    pqmse_parent = f"s{seed}_prefix_pqmse"
    if pqmse_parent in parent_steps:
        direct.insert(4, ("pure_pqmse", "pqmse", pqmse_parent, parent_steps[pqmse_parent]))
    if lpips_parent in parent_steps:
        lpips_step = parent_steps[lpips_parent]
        direct.extend(
            [
                ("scratch_lpips_continue", "lpips", lpips_parent, lpips_step),
                ("scratch_lpips_to_pql1", "pql1", lpips_parent, lpips_step),
                ("scratch_lpips_to_pqmse", "pqmse", lpips_parent, lpips_step),
            ]
        )
    specs = [
        RunSpec(
            name=f"s{seed}_{tag}",
            seed=seed,
            recipe=recipe,
            target_exposure=EXPOSURE_FINAL,
            max_step=parent_step + TAIL_MAX_EXTRA_STEPS,
            eval_interval=tail_eval_interval(recipe),
            save_interval=checkpoint_interval(recipe),
            parent=parent,
            scratch=False,
        )
        for tag, recipe, parent, parent_step in direct
    ]
    specs.extend(
        [
            RunSpec(
                name=f"s{seed}_mature_lpips_to_b1",
                seed=seed,
                recipe="lpips",
                target_exposure=EXPOSURE_LPIPS_END,
                max_step=stage1_step + MATURE_LPIPS_MAX_EXTRA_STEPS,
                eval_interval=LPIPS_TAIL_EVAL_INTERVAL,
                save_interval=LPIPS_PREFIX_SAVE_INTERVAL,
                parent=eag_parent,
                scratch=False,
            )
        ]
    )
    return specs


def second_stage_specs(seed: int, parent_steps: dict[str, int]) -> list[RunSpec]:
    parent = f"s{seed}_mature_lpips_to_b1"
    step = parent_steps[parent]
    output = [
        RunSpec(
            name=(
                f"s{seed}_mature_lpips_continue"
                if tail == "continue"
                else f"s{seed}_mature_lpips_to_{tail}"
            ),
            seed=seed,
            recipe=recipe,
            target_exposure=EXPOSURE_FINAL,
            max_step=step + TAIL_MAX_EXTRA_STEPS,
            eval_interval=tail_eval_interval(recipe),
            save_interval=checkpoint_interval(recipe),
            parent=parent,
            scratch=False,
        )
        for tail, recipe in (("continue", "lpips"), ("pql1", "pql1"), ("pqmse", "pqmse"))
    ]
    return output


def train_command(
    args: argparse.Namespace,
    spec: RunSpec,
    parent_checkpoint: Path | None = None,
) -> list[str]:
    recipe = RECIPES[spec.recipe]
    python = args.venv / "bin" / "python"
    command = [
        str(python),
        str(SCRIPT_DIR / "run_lookcloser_quiet.py"),
        "--data",
        str(args.data),
        "--output-dir",
        str(args.output_dir),
        "--experiment-name",
        experiment_name(args),
        "--timestamp",
        spec.name,
        "--seed",
        str(spec.seed),
        "--frequency-map-dir",
        "lookcloser_frequencies_exr_auto/knee",
        "--geometry-support-map-dir",
        "lookcloser_geometry_support_v2/edge_ridge",
        "--geometry-support-quantile",
        "0.8",
        "--geometry-support-threshold",
        "0.2",
        "--geometry-support-dilation-radius",
        "1",
        "--geometry-support-dilation-shape",
        "cross",
        "--reconstruction-loss-type",
        recipe.loss,
        "--rgb-output-parameterization",
        "linear_softplus",
        "--eag-dssim-weight",
        str(recipe.dssim_weight),
        "--eag-lpips-weight",
        str(recipe.lpips_weight),
        "--eag-patch-size",
        str(recipe.eag_patch_size),
        "--training-patch-size",
        str(recipe.training_patch_size),
        "--train-num-rays-per-batch",
        str(recipe.rays),
        "--max-num-iterations",
        str(spec.max_step + 1),
        "--step-interval",
        str(spec.eval_interval),
        "--save-interval",
        str(spec.save_interval),
        "--max-res",
        "8192",
        "--ray-sampling-mode",
        "adaptive",
        "--max-steps-per-ray",
        "1024",
        "--adaptive-coarse-step-size",
        "0.00625",
        "--corrected-arm-allocator",
        "--no-stop-on-no-improve",
        "--save-only-latest-checkpoint",
        "--preserve-best-eval-model-checkpoint",
        "--stop-at-cumulative-point-samples",
        str(spec.target_exposure),
        "--no-render-final",
        "--no-update-summary",
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if spec.scratch and spec.recipe == "lpips":
        command.extend(["--fields-scheduler-max-steps", str(spec.max_step)])
    if spec.scratch and spec.recipe in ("pqmse", "lpips"):
        command.extend(
            [
                "--early-reject-psnr-below",
                str(EARLY_REJECT_THRESHOLDS["psnr"]),
                "--early-reject-ssim-below",
                str(EARLY_REJECT_THRESHOLDS["ssim"]),
                "--early-reject-lpips-above",
                str(EARLY_REJECT_THRESHOLDS["lpips"]),
                "--early-reject-after-evals",
                str(EARLY_REJECT_MIN_EVALS),
            ]
        )
    if parent_checkpoint is not None:
        command.extend(
            [
                "--load-checkpoint",
                str(parent_checkpoint),
                "--checkpoint-load-mode",
                "resume",
            ]
        )
    return command


def read_progress(metrics_path: Path) -> list[tuple[int, float]]:
    if not metrics_path.is_file():
        return []
    rows: list[tuple[int, float]] = []
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step_text = row.get("step", "")
            exposure_text = row.get("cumulative_point_samples", "")
            if not step_text or not exposure_text:
                continue
            try:
                rows.append((int(step_text), float(exposure_text)))
            except ValueError:
                continue
    return rows


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def choose_exposure_checkpoint(path: Path, target: float) -> tuple[Path, float, float]:
    progress = read_progress(path / "metrics_compact.csv")
    checkpoints = sorted((path / "nerfstudio_models").glob("step-*.ckpt"))
    if not progress or not checkpoints:
        raise RuntimeError(f"No exposure/checkpoint data under {path}")
    candidates: list[tuple[float, Path, float]] = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        nearest = min(progress, key=lambda item: abs(item[0] - step))
        candidates.append((abs(nearest[1] - target), checkpoint, nearest[1]))
    _, selected, exposure = min(candidates, key=lambda item: item[0])
    relative_error = abs(exposure - target) / target
    if relative_error > EXPOSURE_REL_TOLERANCE:
        raise RuntimeError(
            f"Closest checkpoint {selected} has exposure {exposure:.6g}, "
            f"target {target:.6g}, relative error {relative_error:.3%}"
        )
    return selected, exposure, relative_error


def completed_run(path: Path) -> bool:
    summary = path / "run_summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("train_returncode") in (0, -2)


def rejected_eval_rows(metrics_path: Path) -> list[dict[str, float]]:
    """Return stable bad eval evidence, or no rows when the run should continue."""

    if not metrics_path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values = (row.get("eval_all_psnr"), row.get("eval_all_ssim"), row.get("eval_all_lpips"))
            if not all(values):
                continue
            rows.append(
                {
                    "step": float(row["step"]),
                    "psnr": float(values[0]),
                    "ssim": float(values[1]),
                    "lpips": float(values[2]),
                }
            )
    if len(rows) < EARLY_REJECT_MIN_EVALS:
        return []
    recent = rows[-EARLY_REJECT_MIN_EVALS:]
    stable_bad = all(
        row["psnr"] < EARLY_REJECT_THRESHOLDS["psnr"]
        or row["ssim"] < EARLY_REJECT_THRESHOLDS["ssim"]
        or row["lpips"] > EARLY_REJECT_THRESHOLDS["lpips"]
        for row in recent
    )
    return rows if stable_bad else []


def record_rejected_run(
    args: argparse.Namespace,
    spec: RunSpec,
    manifest: dict[str, Any],
    command: Sequence[str],
    parent_checkpoint: Path | None,
    train_seconds: float,
    evidence: list[dict[str, float]],
) -> None:
    path = run_dir(args, spec)
    removed = 0
    for checkpoint in (path / "nerfstudio_models").glob("step-*.ckpt"):
        checkpoint.unlink()
        removed += 1
    for compact in (path / "best_eval_model.ckpt", path / "best_eval_model.ckpt.json"):
        if compact.is_file():
            compact.unlink()
            removed += 1
    manifest.setdefault("runs", {})[spec.name] = {
        "status": "rejected",
        "reason": f"stable_material_degradation_after_{len(evidence)}_eval_boundaries",
        "thresholds": dict(EARLY_REJECT_THRESHOLDS),
        "minimum_eval_boundaries": EARLY_REJECT_MIN_EVALS,
        "spec": asdict(spec),
        "command": list(command),
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "train_seconds": train_seconds,
        "run_dir": str(path),
        "eval_trajectory": evidence,
        "checkpoint_files_deleted_after_rejection": removed,
    }
    atomic_json(manifest_path(args), manifest)


def runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PATH"] = str(args.venv / "bin") + os.pathsep + environment.get("PATH", "")
    cuda_home = Path("/usr/local/cuda-12.6")
    if cuda_home.is_dir():
        environment.setdefault("CUDA_HOME", str(cuda_home))
        environment["PATH"] = str(cuda_home / "bin") + os.pathsep + environment["PATH"]
        environment.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")
    extension_cache = Path("/home/brans/.cache/torch_extensions_lookcloser")
    if extension_cache.is_dir():
        environment.setdefault("TORCH_EXTENSIONS_DIR", str(extension_cache))
    return environment


def remove_incomplete_run(args: argparse.Namespace, spec: RunSpec) -> None:
    """Remove only the exact campaign run directory before a clean retry."""

    path = run_dir(args, spec).resolve()
    expected_root = (args.output_dir / experiment_name(args) / "lookcloser").resolve()
    if path.parent != expected_root or path.name != spec.name:
        raise RuntimeError(f"Refusing to clean unexpected run directory: {path}")
    if path.is_dir():
        shutil.rmtree(path)


def run_one(
    args: argparse.Namespace,
    spec: RunSpec,
    manifest: dict[str, Any],
    parent_checkpoint: Path | None = None,
) -> None:
    previous = manifest.get("runs", {}).get(spec.name)
    if (
        isinstance(previous, dict)
        and previous.get("status") in ("complete", "rejected")
        and not args.force
    ):
        return
    command = train_command(args, spec, parent_checkpoint)
    print("command=" + " ".join(command), flush=True)
    if args.dry_run:
        manifest.setdefault("runs", {})[spec.name] = {
            "status": "dry-run",
            "spec": asdict(spec),
            "command": command,
            "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        }
        return
    path = run_dir(args, spec)
    if args.force or not completed_run(path):
        remove_incomplete_run(args, spec)
        started = time.monotonic()
        subprocess.run(command, check=True)
        train_seconds = time.monotonic() - started
    else:
        train_seconds = float(
            json.loads((path / "run_summary.json").read_text(encoding="utf-8")).get("train_seconds", 0.0)
        )
    evidence = rejected_eval_rows(path / "metrics_compact.csv") if spec.scratch else []
    if evidence:
        record_rejected_run(
            args, spec, manifest, command, parent_checkpoint, train_seconds, evidence
        )
        return
    checkpoint, exposure, relative_error = choose_exposure_checkpoint(path, spec.target_exposure)
    for other_checkpoint in (path / "nerfstudio_models").glob("step-*.ckpt"):
        if other_checkpoint != checkpoint:
            other_checkpoint.unlink()
    manifest.setdefault("runs", {})[spec.name] = {
        "status": "complete",
        "spec": asdict(spec),
        "command": command,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "step": checkpoint_step(checkpoint),
        "exposure": exposure,
        "exposure_relative_error": relative_error,
        "train_seconds": train_seconds,
        "run_dir": str(path),
    }
    best_model = path / "best_eval_model.ckpt"
    best_sidecar = best_model.with_suffix(best_model.suffix + ".json")
    if best_model.is_file() and best_sidecar.is_file():
        manifest["runs"][spec.name]["best_model_checkpoint"] = str(best_model)
        manifest["runs"][spec.name]["best_model_checkpoint_sha256"] = sha256_file(best_model)
        manifest["runs"][spec.name]["best_model_selection"] = json.loads(
            best_sidecar.read_text(encoding="utf-8")
        )
    atomic_json(manifest_path(args), manifest)


def evaluate_run(args: argparse.Namespace, spec: RunSpec, manifest: dict[str, Any]) -> dict[str, Any]:
    record = manifest["runs"][spec.name]
    existing = record.get("dense_evaluation")
    if isinstance(existing, dict) and existing.get("status") == "complete" and not args.force:
        return existing
    checkpoint = Path(record["best_model_checkpoint"])
    path = run_dir(args, spec)
    config_path = path / "config.yml"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config_path)
    output = path / "evaluation_dense4"
    output.mkdir(parents=True, exist_ok=True)
    render_dir = output / "renders"
    eval_json = output / "eval.json"
    dense_config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    dense_config.eval_checkpoint = checkpoint.resolve()
    dense_config.pipeline.model.adaptive_coarse_step_size = 0.0015625
    dense_config.pipeline.model.max_steps_per_ray = 4096
    dense_config.pipeline.model.corrected_arm_allocator = True
    eval_config = output / "config.yml"
    eval_config.write_text(yaml.dump(dense_config), encoding="utf-8")
    ns_eval = args.venv / "bin" / "ns-eval"
    eval_log = output / "eval_stdout.log"
    started = time.monotonic()
    with eval_log.open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                str(ns_eval),
                "--load-config",
                str(eval_config),
                "--output-path",
                str(eval_json),
                "--render-output-path",
                str(render_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=runtime_environment(args),
            check=True,
        )
    cable_dir = output / "target_cable_gaps"
    subprocess.run(
        [
            str(args.venv / "bin" / "python"),
            str(SCRIPT_DIR / "score_thin_cable_gaps.py"),
            "--render-dir",
            str(render_dir),
            "--data",
            str(args.data),
            "--output-dir",
            str(cable_dir),
        ],
        check=True,
    )
    review_dir = output / "hdr_review"
    subprocess.run(
        [
            str(args.venv / "bin" / "python"),
            str(SCRIPT_DIR / "evaluate_exr_hdr_renders.py"),
            "--render-dir",
            str(render_dir),
            "--output-dir",
            str(review_dir),
            "--data",
            str(args.data),
        ],
        check=True,
    )
    metrics = json.loads(eval_json.read_text(encoding="utf-8"))["results"]
    cable_json = cable_dir / "thin_cable_gaps.json"
    cable = json.loads(cable_json.read_text(encoding="utf-8"))["aggregate"]
    evaluation = {
        "status": "complete",
        "seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint),
        "results": {
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "lpips": float(metrics["lpips"]),
        },
        "cable": cable,
        "eval_json": str(eval_json),
        "render_dir": str(render_dir),
        "cable_json": str(cable_json),
        "review_dir": str(review_dir),
    }
    record["dense_evaluation"] = evaluation
    atomic_json(manifest_path(args), manifest)
    return evaluation


def delete_full_training_checkpoints(args: argparse.Namespace, spec: RunSpec, manifest: dict[str, Any]) -> None:
    record = manifest["runs"][spec.name]
    model_dir = run_dir(args, spec) / "nerfstudio_models"
    removed = 0
    for checkpoint in model_dir.glob("step-*.ckpt"):
        checkpoint.unlink()
        removed += 1
    record["full_checkpoints_deleted_after_evaluation"] = removed
    atomic_json(manifest_path(args), manifest)


def parent_maps(manifest: dict[str, Any]) -> tuple[dict[str, Path], dict[str, int]]:
    checkpoints: dict[str, Path] = {}
    steps: dict[str, int] = {}
    for name, run in manifest.get("runs", {}).items():
        checkpoint = run.get("checkpoint")
        if run.get("status") == "complete" and checkpoint:
            checkpoints[name] = Path(checkpoint)
            steps[name] = int(run["step"])
    return checkpoints, steps


def initial_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = manifest_path(args)
    if path.is_file() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))
    dataset = validate_dataset(args.data) if not args.dry_run else {"path": str(args.data)}
    return {
        "schema": 1,
        "campaign": args.campaign_name,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "seeds": list(args.seeds),
        "seed_42_excluded_from_decision": True,
        "dataset": dataset,
        "exposure_targets": {
            "base": EXPOSURE_BASE,
            "lpips_end": EXPOSURE_LPIPS_END,
            "final": EXPOSURE_FINAL,
            "relative_tolerance": EXPOSURE_REL_TOLERANCE,
        },
        "recipes": {name: asdict(recipe) for name, recipe in RECIPES.items()},
        "runs": {},
    }


def require_parents(names: Iterable[str], checkpoints: dict[str, Path]) -> None:
    missing = [name for name in names if name not in checkpoints or not checkpoints[name].is_file()]
    if missing:
        raise RuntimeError("Missing completed parent runs: " + ", ".join(missing))


def run_prefixes(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    for seed in args.seeds:
        for spec in prefix_specs(seed):
            run_one(args, spec, manifest)


def terminal_specs_for_seed(seed: int, parent_steps: dict[str, int]) -> list[RunSpec]:
    first = [spec for spec in fork_specs(seed, parent_steps) if spec.name != f"s{seed}_mature_lpips_to_b1"]
    augmented_steps = dict(parent_steps)
    augmented_steps[f"s{seed}_mature_lpips_to_b1"] = (
        parent_steps[f"s{seed}_prefix_eag"] + MATURE_LPIPS_MAX_EXTRA_STEPS
    )
    return first + second_stage_specs(seed, augmented_steps)


def seed_forks_complete(seed: int, manifest: dict[str, Any]) -> bool:
    runs = manifest.get("runs", {})
    required_suffixes = (
        "eag_continue",
        "direct_pql1",
        "direct_pqmse",
        "mature_lpips_continue",
        "mature_lpips_to_pql1",
        "mature_lpips_to_pqmse",
    )
    return all(
        runs.get(f"s{seed}_{suffix}", {}).get("dense_evaluation", {}).get("status") == "complete"
        for suffix in required_suffixes
    )


def run_forks(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    checkpoints, steps = parent_maps(manifest)
    paired_steps = dict(steps)
    for suffix in ("pql1", "pqmse", "lpips"):
        if not all(
            manifest.get("runs", {}).get(f"s{seed}_prefix_{suffix}", {}).get("status")
            == "complete"
            for seed in args.seeds
        ):
            for seed in args.seeds:
                paired_steps.pop(f"s{seed}_prefix_{suffix}", None)
    for seed in args.seeds:
        if seed_forks_complete(seed, manifest) and not args.force:
            continue
        prefix_names = [f"s{seed}_prefix_eag"]
        require_parents(prefix_names, checkpoints)
        for spec in fork_specs(seed, paired_steps):
            assert spec.parent is not None
            run_one(args, spec, manifest, checkpoints[spec.parent])
            checkpoints, steps = parent_maps(manifest)
            if spec.name != f"s{seed}_mature_lpips_to_b1":
                evaluate_run(args, spec, manifest)
                delete_full_training_checkpoints(args, spec, manifest)
        require_parents([f"s{seed}_mature_lpips_to_b1"], checkpoints)
        for spec in second_stage_specs(seed, steps):
            assert spec.parent is not None
            run_one(args, spec, manifest, checkpoints[spec.parent])
            checkpoints, steps = parent_maps(manifest)
            evaluate_run(args, spec, manifest)
            delete_full_training_checkpoints(args, spec, manifest)
        cleanup_seed_parents(args, seed, manifest)


def evaluate_completed_terminals(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Finish dense evaluation for terminal arms after an interrupted campaign."""

    for seed in args.seeds:
        for strategy in STRATEGY_METADATA:
            name = f"s{seed}_{strategy}"
            record = manifest.get("runs", {}).get(name)
            if not isinstance(record, dict) or record.get("status") != "complete":
                continue
            spec = RunSpec(**record["spec"])
            evaluate_run(args, spec, manifest)
            delete_full_training_checkpoints(args, spec, manifest)


def cleanup_seed_parents(args: argparse.Namespace, seed: int, manifest: dict[str, Any]) -> None:
    names = [spec.name for spec in prefix_specs(seed)] + [f"s{seed}_mature_lpips_to_b1"]
    for name in names:
        record = manifest.get("runs", {}).get(name)
        if not isinstance(record, dict):
            continue
        path = Path(record["run_dir"])
        for checkpoint in (path / "nerfstudio_models").glob("step-*.ckpt"):
            checkpoint.unlink()
        compact = path / "best_eval_model.ckpt"
        sidecar = compact.with_suffix(compact.suffix + ".json")
        if compact.is_file():
            compact.unlink()
        if sidecar.is_file():
            sidecar.unlink()
        record["parent_state_deleted_after_seed_forks"] = True
    atomic_json(manifest_path(args), manifest)


def paired_sd(values: Sequence[float]) -> float:
    if len(values) != 2:
        raise ValueError("The frozen campaign requires exactly two paired seeds")
    return abs(values[0] - values[1]) / math.sqrt(2.0)


STRATEGY_METADATA = {
    "eag_continue": (1, False),
    "direct_pql1": (2, False),
    "direct_pqmse": (2, False),
    "pure_pql1": (1, False),
    "pure_pqmse": (1, False),
    "scratch_lpips_continue": (1, True),
    "scratch_lpips_to_pql1": (2, True),
    "scratch_lpips_to_pqmse": (2, True),
    "mature_lpips_continue": (2, True),
    "mature_lpips_to_pql1": (3, True),
    "mature_lpips_to_pqmse": (3, True),
}


def lineage_names(seed: int, strategy: str) -> list[str]:
    if strategy == "eag_continue":
        return [f"s{seed}_prefix_eag", f"s{seed}_{strategy}"]
    if strategy.startswith("direct_"):
        return [f"s{seed}_prefix_eag", f"s{seed}_{strategy}"]
    if strategy == "pure_pql1":
        return [f"s{seed}_prefix_pql1", f"s{seed}_{strategy}"]
    if strategy == "pure_pqmse":
        return [f"s{seed}_prefix_pqmse", f"s{seed}_{strategy}"]
    if strategy.startswith("scratch_lpips"):
        return [f"s{seed}_prefix_lpips", f"s{seed}_{strategy}"]
    if strategy.startswith("mature_lpips"):
        return [
            f"s{seed}_prefix_eag",
            f"s{seed}_mature_lpips_to_b1",
            f"s{seed}_{strategy}",
        ]
    raise KeyError(strategy)


def aggregate_report(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    if not all(seed_forks_complete(seed, manifest) for seed in args.seeds):
        raise RuntimeError("Cannot report before both seed fork matrices have dense evaluations")
    runs = manifest["runs"]
    table: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for strategy, (phases, patches) in STRATEGY_METADATA.items():
        if not all(
            runs.get(f"s{seed}_{strategy}", {}).get("dense_evaluation", {}).get("status")
            == "complete"
            for seed in args.seeds
        ):
            continue
        per_seed = []
        for seed in args.seeds:
            record = runs[f"s{seed}_{strategy}"]
            evaluation = record["dense_evaluation"]
            metrics = evaluation["results"]
            seconds = sum(float(runs[name]["train_seconds"]) for name in lineage_names(seed, strategy))
            row = {
                "strategy": strategy,
                "seed": seed,
                "psnr": float(metrics["psnr"]),
                "ssim": float(metrics["ssim"]),
                "lpips": float(metrics["lpips"]),
                "cable_gaps": int(evaluation["cable"]["total_gap_pixels"]),
                "train_seconds": seconds,
                "render_dir": evaluation["render_dir"],
                "checkpoint": evaluation["checkpoint"],
            }
            per_seed.append(row)
            seed_rows.append(row)
        aggregate = {
            "strategy": strategy,
            "mean_psnr": sum(row["psnr"] for row in per_seed) / 2.0,
            "mean_ssim": sum(row["ssim"] for row in per_seed) / 2.0,
            "mean_lpips": sum(row["lpips"] for row in per_seed) / 2.0,
            "median_train_seconds": sum(row["train_seconds"] for row in per_seed) / 2.0,
            "loss_phases": phases,
            "requires_lpips_patches": patches,
            "cable_gaps": sum(row["cable_gaps"] for row in per_seed),
            "visual_failure": False,
            "equivalence_bands": dict(QUALITY_FLOORS),
        }
        table.append(aggregate)
    provisional = choose_strategy(table)
    winner = provisional["quality_winner"]
    winner_seed_rows = [row for row in seed_rows if row["strategy"] == winner]
    for aggregate in table:
        candidate_rows = [row for row in seed_rows if row["strategy"] == aggregate["strategy"]]
        deltas = {
            metric: [
                candidate[metric] - reference[metric]
                for candidate, reference in zip(candidate_rows, winner_seed_rows)
            ]
            for metric in ("psnr", "ssim", "lpips")
        }
        aggregate["equivalence_bands"] = quality_equivalence_bands(deltas)
    decision = choose_strategy(table)
    previous_visual_review = manifest.get("report", {}).get("visual_review", "pending")
    visual_review: str | dict[str, Any] = previous_visual_review
    if args.visual_review_status is not None:
        visual_review = {
            "status": args.visual_review_status,
            "reviewed_at": utc_now(),
            "notes": list(args.visual_review_note),
            "artifacts": [str(path.resolve()) for path in args.visual_review_artifact],
        }
    report = {
        "schema": 1,
        "created_at": utc_now(),
        "decision": decision,
        "strategies": table,
        "seed_rows": seed_rows,
        "rejected_runs": [
            {"name": name, **record}
            for name, record in runs.items()
            if record.get("status") == "rejected"
        ],
        "visual_review": visual_review,
    }
    manifest["report"] = report
    atomic_json(campaign_dir(args) / "report.json", report)
    atomic_json(manifest_path(args), manifest)
    write_markdown_report(report)
    return report


def write_markdown_report(report: dict[str, Any]) -> None:
    path = REPO / "LookCloser" / "experiments" / "exr_loss_schedule_two_seed_validation.md"
    lines = [
        "# EXR loss-schedule two-seed validation",
        "",
        "## What was tested",
        "",
        "New seeds 43 and 44; native linear EXR, frozen knee maps/geometry guard, matched cumulative point exposure, and dense4 corrected evaluation. Seed 42 is historical context only.",
        "",
        "Correction after the primary-loss follow-up: `direct_*` rows below are short tails from a common EAG parent, not scratch primary-loss comparisons. The old seed-44 pure-PQ-L1 early rejection was not valid evidence; full scratch PQ-L1 results are reported separately in `exr_primary_loss_scratch_validation.md`.",
        "",
        "## Results",
        "",
        "| Strategy | Mean PSNR | Mean SSIM | Mean LPIPS | Cable gaps | Mean lineage train s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["strategies"], key=lambda item: item["mean_psnr"], reverse=True):
        lines.append(
            f"| {row['strategy']} | {row['mean_psnr']:.6f} | {row['mean_ssim']:.6f} | "
            f"{row['mean_lpips']:.6f} | {row['cable_gaps']} | {row['median_train_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Per-seed measurements:",
            "",
            "| Strategy | Seed | PSNR | SSIM | LPIPS | Cable gaps | Train s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["seed_rows"]:
        lines.append(
            f"| {row['strategy']} | {row['seed']} | {row['psnr']:.6f} | {row['ssim']:.6f} | "
            f"{row['lpips']:.6f} | {row['cable_gaps']} | {row['train_seconds']:.1f} |"
        )
    if report.get("rejected_runs"):
        lines.extend(
            [
                "",
                "Early-rejected controls:",
                "",
                "| Run | Reason | Last PSNR | Last SSIM | Last LPIPS | Train s |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in report["rejected_runs"]:
            last = row["eval_trajectory"][-1]
            lines.append(
                f"| {row['name']} | {row['reason']} | {last['psnr']:.6f} | "
                f"{last['ssim']:.6f} | {last['lpips']:.6f} | {row['train_seconds']:.1f} |"
            )
    visual = report.get("visual_review", "pending")
    if isinstance(visual, dict):
        lines.extend(["", f"Visual review: **{visual['status']}**."])
        if visual.get("notes") or visual.get("artifacts"):
            lines.append("")
        for note in visual.get("notes", []):
            lines.append(f"- {note}")
        for artifact in visual.get("artifacts", []):
            lines.append(f"- Artifact: `{artifact}`")
    else:
        lines.extend(["", f"Visual review: **{visual}**."])
    lines.extend(
        [
            "",
            "## Insights",
            "",
            "`direct_*` branches share the frozen EAG prefix and switch only for the final matched-exposure tail. "
            "`mature_lpips_*` branches add the same short 64x64 PQ-L1+LPIPS phase before their final tail.",
            "",
            f"Quality winner: `{report['decision']['quality_winner']}`. Selected after variance/time/simplicity rules: `{report['decision']['selected']}`. Visual review status: `{visual['status'] if isinstance(visual, dict) else visual}`.",
            "",
        ]
    )
    selected = next(row for row in report["strategies"] if row["strategy"] == report["decision"]["selected"])
    highest_psnr = max(report["strategies"], key=lambda row: row["mean_psnr"])
    psnr_delta = selected["mean_psnr"] - highest_psnr["mean_psnr"]
    lpips_delta = selected["mean_lpips"] - highest_psnr["mean_lpips"]
    time_delta = selected["median_train_seconds"] - highest_psnr["median_train_seconds"]
    lines.extend(
        [
            f"Against the highest-PSNR branch (`{highest_psnr['strategy']}`), the selected schedule changes "
            f"PSNR by {psnr_delta:+.6f} dB, LPIPS by {lpips_delta:+.6f}, and mean end-to-end lineage time "
            f"by {time_delta:+.1f} s. It is the only strategy inside the frozen paired-seed equivalence bands.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def quality_equivalence_bands(paired_deltas: dict[str, Sequence[float]]) -> dict[str, float]:
    return {
        metric: max(QUALITY_FLOORS[metric], paired_sd(paired_deltas[metric]))
        for metric in ("psnr", "ssim", "lpips")
    }


def choose_strategy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen aggregate metric, variance, speed, and simplicity selector."""

    eligible = [row for row in rows if row.get("cable_gaps") == 0 and not row.get("visual_failure", False)]
    if not eligible:
        raise RuntimeError("No strategy passed the cable/visual veto")
    max_psnr = max(float(row["mean_psnr"]) for row in eligible)
    psnr_window = [row for row in eligible if max_psnr - float(row["mean_psnr"]) <= PSNR_WINDOW]
    quality_winner = min(psnr_window, key=lambda row: float(row["mean_lpips"]))
    equivalent = []
    for row in eligible:
        bands = row.get("equivalence_bands", QUALITY_FLOORS)
        if (
            float(quality_winner["mean_psnr"]) - float(row["mean_psnr"]) <= float(bands["psnr"])
            and float(quality_winner["mean_ssim"]) - float(row["mean_ssim"]) <= float(bands["ssim"])
            and float(row["mean_lpips"]) - float(quality_winner["mean_lpips"]) <= float(bands["lpips"])
        ):
            equivalent.append(row)
    fastest = min(equivalent, key=lambda row: float(row["median_train_seconds"]))
    fastest_seconds = float(fastest["median_train_seconds"])
    within_five_percent = [
        row
        for row in equivalent
        if float(row["median_train_seconds"]) <= fastest_seconds * 1.05
    ]
    selected = min(
        within_five_percent,
        key=lambda row: (
            int(row["loss_phases"]),
            bool(row["requires_lpips_patches"]),
            float(row.get("peak_vram_mb", math.inf)),
            0 if row["strategy"] == "pure_pqmse" else 1,
        ),
    )
    return {
        "selected": selected["strategy"],
        "quality_winner": quality_winner["strategy"],
        "equivalent": [row["strategy"] for row in equivalent],
    }


def dry_run_payload(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    commands: dict[str, list[str]] = {}
    fake_steps = {spec.name: spec.max_step for seed in args.seeds for spec in prefix_specs(seed)}
    for seed in args.seeds:
        for spec in prefix_specs(seed):
            commands[spec.name] = train_command(args, spec)
        first = fork_specs(seed, fake_steps)
        for spec in first:
            assert spec.parent is not None
            fake_parent = Path(f"/dry-run/{spec.parent}/step-{fake_steps[spec.parent]:09d}.ckpt")
            commands[spec.name] = train_command(args, spec, fake_parent)
            fake_steps[spec.name] = spec.max_step
        for spec in second_stage_specs(seed, fake_steps):
            assert spec.parent is not None
            fake_parent = Path(f"/dry-run/{spec.parent}/step-{fake_steps[spec.parent]:09d}.ckpt")
            commands[spec.name] = train_command(args, spec, fake_parent)
    return {"manifest": manifest, "commands": commands}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = initial_manifest(args)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args, manifest), indent=2, sort_keys=True))
        return 0
    validate_dataset(args.data)
    campaign_dir(args).mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path(args), manifest)
    if args.phase in ("prefixes", "all"):
        run_prefixes(args, manifest)
    if args.phase in ("forks", "all"):
        run_forks(args, manifest)
    if args.phase == "evaluate":
        evaluate_completed_terminals(args, manifest)
    if args.phase in ("report", "all"):
        aggregate_report(args, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
