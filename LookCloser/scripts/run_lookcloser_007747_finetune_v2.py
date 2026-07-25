#!/usr/bin/env python3
"""Fail-closed 007740 -> 007747 hash23 fine-tuning campaign.

This runner is intentionally single-frame and recipe-specific.  It keeps the
legacy temporal controller as forensic evidence, loads only trainable fields
across frames, preserves complete state only inside 007747, and treats visual
review as a mandatory success/plateau gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOKCLOSER_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

LEADER_ROOT = Path(
    "/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/"
    "lookcloser/20260715_005006"
)
LEADER_CHECKPOINT = LEADER_ROOT / "nerfstudio_models" / "step-000091128.ckpt"
LEADER_CONFIG = LEADER_ROOT / "config.yml"
LEADER_RENDERS = LEADER_ROOT / "renders_candidate_step-000091128"
TARGET_DATASET = Path("/home/brans/temporal_perframe_stride7_45f/007747")
TARGET_MAPS = TARGET_DATASET / "lookcloser_frequencies"
DATASET_REVISION = TARGET_DATASET / "dataset_revision_422.json"
SCRATCH_RENDERS = Path(
    "/home/brans/lookcloser_007747_from_scratch_runs/evaluations/"
    "007747_fromscratch_E8_fw02/step-000197444/renders"
)
DEFAULT_OUTPUT = Path(
    "/home/brans/lookcloser_007747_finetune_v2_runs/hash23_lr_scheduler_seed42_v2"
)
DEFAULT_VENV = Path("/home/brans/repos/nerfstudio/.venv")
DEFAULT_TCNN_OVERLAY = Path("/home/brans/deps/tcnn_2e757_py310")

EXPECTED_LEADER_SHA256 = "3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930"
EXPECTED_CONFIG_SHA256 = "a8c017c96a19a42fde3d43492b8253d970408b71c24cd47bcc449fed5fd0e5fb"
EXPECTED_REVISION_SHA256 = "5983bc94168ded04ec6b8fe10ec01f0703417ba903115a01ced4d2b280e996e0"
EXPECTED_TRANSFORMS_SHA256 = "022f8748a1a039861a754e68ab3ef830beeb3e5dd94ccb00457a630d28f64aa1"
EXPECTED_TCNN_BINDING_SHA256 = "f2163346afd103c27e78b9f56f8d82b6eeb3317c1ce11caf57d45f0216aece36"
EXPECTED_PYTHON = "3.10.20"
EXPECTED_TORCH = "2.7.1+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_GPU = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"

INTERVAL = 15_188
INITIAL_FINAL_STEP = 60_752
LR_FINAL = 1e-4
PSNR_THRESHOLD = 29.840143
SSIM_THRESHOLD = 0.669203
LPIPS_THRESHOLD = 0.219455
PSNR_TIE_DB = 0.07
INITIAL_FREE_BYTES = 180 * 1024**3
RUNNING_FREE_FLOOR_BYTES = 100 * 1024**3
VRAM_PER_JOB_MIB = 20 * 1024
VRAM_RESERVE_MIB = 20 * 1024
QUALITY_EXIT = 2
INFRASTRUCTURE_EXIT = 3

WAVE_A = (
    ("L075-H200", 0.0075, 200_000),
    ("L100-H200", 0.0100, 200_000),
    ("L150-H200", 0.0150, 200_000),
)

EXTENDED_SCHEDULER_WAVE = (
    ("R-L125-H400", 0.0125, 400_000),
    ("R-L150-H300", 0.0150, 300_000),
    ("R-L150-H400", 0.0150, 400_000),
)

CAMPAIGN_PROFILES = ("leader_base", "extended_scheduler")

ALLOWED_CONFIG_DIFFS = {
    "checkpoint_load_mode",
    "checkpoint_load_parameter_hash_audit",
    "experiment_name",
    "load_checkpoint",
    "load_optimizers",
    "load_scheduler",
    "max_num_iterations",
    "optimizers.fields.optimizer.lr",
    "optimizers.fields.scheduler.max_steps",
    "output_dir",
    "pipeline.datamanager.dataparser.data",
    "timestamp",
}


class InfrastructureError(RuntimeError):
    """The campaign cannot safely interpret or reproduce the requested run."""


class QualityStop(RuntimeError):
    """A complete fail-closed quality gate intentionally stopped the campaign."""


class FinalQualityFailure(RuntimeError):
    """The authoritative trajectory plateaued without reaching the leader."""


@dataclass(frozen=True)
class Arm:
    arm_id: str
    lr_init: float
    scheduler_max_steps: int
    phase: Literal["wave_a", "wave_b", "authoritative"]


@dataclass(frozen=True)
class Segment:
    segment_id: str
    arm: Arm
    run_dir: Path
    target_step: int
    load_mode: Literal["model_parameters_only", "resume"]
    parent_checkpoint: Path


@dataclass(frozen=True)
class Boundary:
    arm_id: str
    local_step: int
    psnr: float
    ssim: float
    lpips: float
    checkpoint: Path
    eval_json: Path
    protocol_json: Optional[Path]
    eval_completed_wall_time_ns: int

    @property
    def numeric_pass(self) -> bool:
        return (
            self.psnr >= PSNR_THRESHOLD
            and self.ssim >= SSIM_THRESHOLD
            and self.lpips <= LPIPS_THRESHOLD
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--tcnn-overlay", type=Path, default=DEFAULT_TCNN_OVERLAY)
    parser.add_argument("--visual-decisions", type=Path)
    parser.add_argument("--max-parallel", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--campaign-profile",
        choices=CAMPAIGN_PROFILES,
        default="leader_base",
        help=(
            "leader_base runs the original v2 LR/horizon screen; "
            "extended_scheduler runs the evidence-driven long-horizon rescue screen"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-mode", choices=("baseline", "train"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except ValueError as error:
        raise InfrastructureError(f"Cannot parse checkpoint step: {path}") from error


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "nerfstudio_models" / f"step-{step:09d}.ckpt"


def expected_learning_rate(lr_init: float, scheduler_max_steps: int, step: int) -> float:
    if lr_init <= 0 or scheduler_max_steps <= 0 or step < 0:
        raise ValueError("LR, scheduler horizon, and step must be positive/non-negative")
    t = min(step / scheduler_max_steps, 1.0)
    return math.exp(math.log(lr_init) * (1.0 - t) + math.log(LR_FINAL) * t)


def run_environment(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{args.venv / 'bin'}:/usr/local/cuda-12.6/bin:{env.get('PATH', '')}"
    python_paths = [str(args.tcnn_overlay), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_paths)
    env["CUDA_HOME"] = "/usr/local/cuda-12.6"
    env["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
    env["TORCH_EXTENSIONS_DIR"] = "/home/brans/.cache/torch_extensions_lookcloser"
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def command_output(command: Sequence[str], *, env: Mapping[str, str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, env=dict(env), text=True).strip()


def git_preflight() -> Dict[str, Any]:
    branch = command_output(["git", "branch", "--show-current"], env=os.environ)
    commit = command_output(["git", "rev-parse", "HEAD"], env=os.environ)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    if branch != "main":
        raise InfrastructureError(f"Fine-tuning v2 requires branch main, got {branch!r}")
    if status.strip():
        raise InfrastructureError("Fine-tuning v2 requires a clean committed main worktree")
    tracked = (
        REPO_ROOT / "nerfstudio" / "engine" / "trainer.py",
        REPO_ROOT / "nerfstudio" / "engine" / "optimizers.py",
        REPO_ROOT / "nerfstudio" / "engine" / "schedulers.py",
        REPO_ROOT / "nerfstudio" / "configs" / "method_configs.py",
        REPO_ROOT / "nerfstudio" / "data" / "datamanagers" / "base_datamanager.py",
        REPO_ROOT / "nerfstudio" / "data" / "dataparsers" / "nerfstudio_dataparser.py",
        REPO_ROOT / "nerfstudio" / "fields" / "lookcloser_field.py",
        REPO_ROOT / "nerfstudio" / "model_components" / "lookcloser_grid.py",
        REPO_ROOT / "nerfstudio" / "models" / "lookcloser.py",
        REPO_ROOT / "nerfstudio" / "pipelines" / "lookcloser_pipeline.py",
        REPO_ROOT / "nerfstudio" / "scripts" / "eval.py",
        REPO_ROOT / "nerfstudio" / "scripts" / "train.py",
        REPO_ROOT / "nerfstudio" / "lookcloser_pixel_sampler.py",
        SCRIPT_PATH,
        SCRIPT_PATH.with_name("static_target_roi_protocol.py"),
        SCRIPT_PATH.with_name("detect_structural_artifacts.py"),
    )
    hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in tracked}
    fingerprint = hashlib.sha256(
        json.dumps({"commit": commit, "source": hashes}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "branch": branch,
        "commit": commit,
        "source_sha256": hashes,
        "source_fingerprint": fingerprint,
    }


def runtime_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    python = args.venv / "bin" / "python"
    if not python.is_file():
        raise InfrastructureError(f"Required Python is missing: {python}")
    code = (
        "import json,platform,torch,tinycudann.modules;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),"
        "'binding':tinycudann.modules._C.__file__}))"
    )
    runtime = json.loads(command_output([str(python), "-c", code], env=run_environment(args)))
    expected = {
        "python": EXPECTED_PYTHON,
        "torch": EXPECTED_TORCH,
        "cuda": EXPECTED_TORCH_CUDA,
        "gpu": EXPECTED_GPU,
    }
    mismatch = {
        key: {"actual": runtime.get(key), "expected": value}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if mismatch:
        raise InfrastructureError(f"Canonical runtime mismatch: {mismatch}")
    binding_hash = sha256_file(Path(runtime["binding"]))
    if binding_hash != EXPECTED_TCNN_BINDING_SHA256:
        raise InfrastructureError(f"TCNN binding SHA mismatch: {binding_hash}")
    runtime["binding_sha256"] = binding_hash
    return runtime


def _verify_manifest_files(
    directory: Path,
    expected: Mapping[str, str],
    *,
    label: str,
) -> Dict[str, Any]:
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    expected_names = set(expected)
    if actual_names != expected_names:
        raise InfrastructureError(
            f"{label} file set mismatch: missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    mismatches = {}
    for name, expected_hash in expected.items():
        actual_hash = sha256_file(directory / name)
        if actual_hash != expected_hash:
            mismatches[name] = {"actual": actual_hash, "expected": expected_hash}
    if mismatches:
        raise InfrastructureError(f"{label} SHA-256 mismatch: {mismatches}")
    return {"directory": str(directory), "count": len(expected), "files": dict(expected)}


def dataset_preflight() -> Dict[str, Any]:
    if not DATASET_REVISION.is_file():
        raise InfrastructureError(f"Required dataset revision is absent: {DATASET_REVISION}")
    revision_hash = sha256_file(DATASET_REVISION)
    if revision_hash != EXPECTED_REVISION_SHA256:
        raise InfrastructureError(f"dataset_revision_422.json SHA mismatch: {revision_hash}")
    if TARGET_MAPS.resolve(strict=True) != (TARGET_DATASET / "lookcloser_frequencies").resolve(strict=True):
        raise InfrastructureError("Target map directory does not resolve to the canonical standard maps")
    forbidden = (
        TARGET_DATASET / "lookcloser_frequencies_chroma422",
        Path("/home/brans/007747_4_4_4"),
    )
    if any(TARGET_MAPS.resolve().is_relative_to(path.resolve()) for path in forbidden):
        raise InfrastructureError("Canonical map directory resolves into a forbidden location")
    if any("_probe" in part for part in TARGET_MAPS.parts):
        raise InfrastructureError("Probe map directories are forbidden")

    revision = json.loads(DATASET_REVISION.read_text(encoding="utf-8"))
    if revision.get("frame") != "007747":
        raise InfrastructureError("Dataset revision is not bound to frame 007747")
    jpeg = revision.get("jpeg", {})
    maps = revision.get("frequency_maps", {})
    if jpeg.get("directory") != "images" or maps.get("directory") != "lookcloser_frequencies":
        raise InfrastructureError("Dataset revision names non-canonical image/map directories")
    jpeg_result = _verify_manifest_files(
        TARGET_DATASET / "images", jpeg.get("files", {}), label="JPEG"
    )
    map_result = _verify_manifest_files(TARGET_MAPS, maps.get("files", {}), label="frequency map")
    if jpeg_result["count"] != 69 or map_result["count"] != 132:
        raise InfrastructureError("Expected exactly 69 JPEGs and 66 PT+JSON map pairs")

    transforms = TARGET_DATASET / "transforms.json"
    transforms_hash = sha256_file(transforms)
    if transforms_hash != EXPECTED_TRANSFORMS_SHA256:
        raise InfrastructureError(f"Target transforms SHA mismatch: {transforms_hash}")
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    names = [Path(row["file_path"]).name for row in payload.get("frames", [])]
    train = [name for name in names if "_train_" in name]
    evaluate = [name for name in names if "_eval_" in name]
    if (len(train), len(evaluate)) != (66, 3):
        raise InfrastructureError(f"Target split is {len(train)}+{len(evaluate)}, expected 66+3")
    return {
        "dataset": str(TARGET_DATASET),
        "revision": str(DATASET_REVISION),
        "revision_sha256": revision_hash,
        "transforms_sha256": transforms_hash,
        "train_images": 66,
        "eval_images": 3,
        "jpeg": jpeg_result,
        "frequency_maps": map_result,
    }


def disk_guard(
    output_dir: Path,
    *,
    initial: bool = False,
    forecast_bytes: int = 0,
) -> Dict[str, int]:
    if forecast_bytes < 0:
        raise ValueError("forecast_bytes must be non-negative")
    anchor = output_dir
    while not anchor.exists():
        anchor = anchor.parent
    usage = shutil.disk_usage(anchor)
    floor = INITIAL_FREE_BYTES if initial else RUNNING_FREE_FLOOR_BYTES
    if usage.free - forecast_bytes < floor:
        raise InfrastructureError(
            f"Storage guard: {usage.free / 1024**3:.1f} GiB free minus "
            f"{forecast_bytes / 1024**3:.1f} GiB forecast is below "
            f"{floor / 1024**3:.0f} GiB"
        )
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "forecast": forecast_bytes,
        "floor": floor,
        "projected_free": usage.free - forecast_bytes,
    }


def full_preflight(args: argparse.Namespace, *, initial_storage: bool = True) -> Dict[str, Any]:
    leader_hash = sha256_file(LEADER_CHECKPOINT)
    config_hash = sha256_file(LEADER_CONFIG)
    if leader_hash != EXPECTED_LEADER_SHA256:
        raise InfrastructureError(f"Canonical leader checkpoint SHA mismatch: {leader_hash}")
    if config_hash != EXPECTED_CONFIG_SHA256:
        raise InfrastructureError(f"Canonical leader config SHA mismatch: {config_hash}")
    reference_renders = {}
    for label, render_dir in (
        ("leader_007740", LEADER_RENDERS),
        ("accepted_scratch_007747", SCRATCH_RENDERS),
    ):
        renders = sorted(render_dir.glob("eval_img_*.png"))
        if len(renders) != 3:
            raise InfrastructureError(f"Reference render directory is incomplete: {render_dir}")
        reference_renders[label] = {
            "directory": str(render_dir),
            "sha256": {path.name: sha256_file(path) for path in renders},
        }
    return {
        "git": git_preflight(),
        "runtime": runtime_preflight(args),
        "leader": {
            "checkpoint": str(LEADER_CHECKPOINT),
            "checkpoint_sha256": leader_hash,
            "config": str(LEADER_CONFIG),
            "config_sha256": config_hash,
            "checkpoint_step": checkpoint_step(LEADER_CHECKPOINT),
        },
        "reference_renders": reference_renders,
        "dataset": dataset_preflight(),
        "storage": disk_guard(args.output_dir, initial=initial_storage),
    }


def _materialize_effective_runtime(config: Any) -> None:
    """Make post-leader config additions explicit and disabled on both diff sides."""

    config.checkpoint_load_mode = getattr(config, "checkpoint_load_mode", "resume")
    config.resume_fields_lr_override = None
    config.resume_reset_occupancy_grid = False
    config.resume_reset_frequency_grid = False
    config.checkpoint_load_parameter_hash_audit = False
    config.fused_adam_switch_step = None
    config.replay_eval_trajectory = False
    config.pipeline.datamanager.cache_train_rays = False
    config.pipeline.datamanager.cpu_fas_prefetch = False
    config.pipeline.independent_rng_streams = False
    config.pipeline.target_num_samples_per_batch = 0
    config.pipeline.model.tcnn_network_jit = False
    config.optimizers["fields"]["optimizer"].fused = False


def _normalized(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    if hasattr(value, "__dict__"):
        return _normalized(vars(value))
    return repr(value)


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def config_diff(reference: Any, candidate: Any) -> Dict[str, Dict[str, Any]]:
    left = _flatten(_normalized(reference))
    right = _flatten(_normalized(candidate))
    keys = set(left) | set(right)
    return {
        key: {"leader": left.get(key), "target": right.get(key)}
        for key in sorted(keys)
        if left.get(key) != right.get(key)
    }


def assert_frozen_recipe(config: Any, arm: Arm) -> None:
    sampler = config.pipeline.datamanager.pixel_sampler
    model = config.pipeline.model
    optimizer = config.optimizers["fields"]["optimizer"]
    scheduler = config.optimizers["fields"]["scheduler"]
    checks = {
        "seed": config.machine.seed == 42,
        "mixed_precision": config.mixed_precision is True,
        "use_grad_scaler": config.use_grad_scaler is False,
        "grad_scaler_init": config.grad_scaler_init_scale == 65_536.0,
        "grad_scaler_growth": config.grad_scaler_growth_interval == 2_000,
        "save_all_checkpoints": config.save_only_latest_checkpoint is False,
        "save_cadence": config.steps_per_save == INTERVAL,
        "eval_batch_cadence": config.steps_per_eval_batch == INTERVAL,
        "eval_image_cadence": config.steps_per_eval_image == INTERVAL,
        "eval_all_cadence": config.steps_per_eval_all_images == INTERVAL,
        "train_batch": config.pipeline.datamanager.train_num_rays_per_batch == 4096,
        "eval_batch": config.pipeline.datamanager.eval_num_rays_per_batch == 4096,
        "ray_cache_off": config.pipeline.datamanager.cache_train_rays is False,
        "cpu_prefetch_off": config.pipeline.datamanager.cpu_fas_prefetch is False,
        "fas_enabled": sampler.enable_fas is True,
        "fas_strength": math.isclose(sampler.fas_strength, 1.0),
        "fas_map_dir": sampler.frequency_map_dir == "lookcloser_frequencies",
        "pipeline_map_dir": config.pipeline.frequency_map_dir == "lookcloser_frequencies",
        "hash23": model.log2_hashmap_size == 23,
        "hash_levels": model.num_frequency_levels == 16,
        "hash_features": model.hash_features_per_level == 2,
        "min_res": math.isclose(model.min_res, 16.0),
        "max_res": math.isclose(model.max_res, 8192.0),
        "max_res_base": math.isclose(model.max_res_base, 2048.0),
        "stable_occupancy": model.stable_occupancy_reduction is True,
        "feature_reweighting": model.enable_feature_reweighting is True
        and math.isclose(model.feature_reweighting_strength, 0.3),
        "adaptive_warmup": model.adaptive_warmup_steps == 4096,
        "occupancy_warmup": model.occupancy_warmup_steps == 4096,
        "occupancy_binary_warmup": model.occupancy_binary_warmup_steps == 4096,
        "fixed_traversal": model.ray_sampling_mode == "adaptive",
        "tcnn_jit_off": model.tcnn_network_jit is False,
        "independent_rng_off": config.pipeline.independent_rng_streams is False,
        "fused_adam_off": optimizer.fused is False and config.fused_adam_switch_step is None,
        "lr": math.isclose(optimizer.lr, arm.lr_init, rel_tol=0.0, abs_tol=1e-15),
        "adam_eps": math.isclose(optimizer.eps, 1e-15, rel_tol=0.0, abs_tol=1e-30),
        "weight_decay": optimizer.weight_decay == 0,
        "lr_final": math.isclose(scheduler.lr_final, LR_FINAL, rel_tol=0.0, abs_tol=1e-15),
        "scheduler_horizon": scheduler.max_steps == arm.scheduler_max_steps,
        "scheduler_warmup": scheduler.warmup_steps == 0,
        "scheduler_ramp": scheduler.ramp == "cosine",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise InfrastructureError(f"Frozen recipe assertions failed: {failed}")


def arm_run_dir(output_dir: Path, phase: str, arm_id: str) -> Path:
    return output_dir / phase / arm_id / "lookcloser" / "run"


def configured_segment(
    args: argparse.Namespace,
    segment: Segment,
) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
    leader = yaml.load(LEADER_CONFIG.read_text(encoding="utf-8"), Loader=yaml.Loader)
    _materialize_effective_runtime(leader)
    config = copy.deepcopy(leader)
    config.output_dir = segment.run_dir.parents[2]
    config.experiment_name = segment.run_dir.parents[1].name
    config.timestamp = segment.run_dir.name
    config.data = None
    config.pipeline.datamanager.data = None
    config.pipeline.datamanager.dataparser.data = TARGET_DATASET
    config.max_num_iterations = segment.target_step + 1
    config.load_dir = None
    config.load_step = None
    config.load_config = None
    config.load_checkpoint = segment.parent_checkpoint
    config.checkpoint_load_mode = segment.load_mode
    config.load_optimizers = segment.load_mode == "resume"
    config.load_scheduler = segment.load_mode == "resume"
    config.resume_fields_lr_override = None
    config.resume_reset_occupancy_grid = False
    config.resume_reset_frequency_grid = False
    config.checkpoint_load_parameter_hash_audit = segment.load_mode == "model_parameters_only"
    config.optimizers["fields"]["optimizer"].lr = segment.arm.lr_init
    scheduler = config.optimizers["fields"]["scheduler"]
    scheduler.lr_final = LR_FINAL
    scheduler.max_steps = segment.arm.scheduler_max_steps
    scheduler.warmup_steps = 0
    scheduler.ramp = "cosine"
    _materialize_effective_runtime(config)
    config.checkpoint_load_parameter_hash_audit = segment.load_mode == "model_parameters_only"
    assert_frozen_recipe(config, segment.arm)
    differences = config_diff(leader, config)
    unexpected = sorted(set(differences) - ALLOWED_CONFIG_DIFFS)
    if unexpected:
        raise InfrastructureError(f"Target config changed outside the leader whitelist: {unexpected}")
    return config, differences


def write_segment_config(
    args: argparse.Namespace,
    segment: Segment,
) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    config, differences = configured_segment(args, segment)
    config_path = (
        args.output_dir
        / "configs"
        / f"{segment.segment_id}_to_step-{segment.target_step:09d}.yml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    segment.run_dir.mkdir(parents=True, exist_ok=True)
    (segment.run_dir / "config.yml").write_text(yaml.dump(config), encoding="utf-8")
    atomic_json(config_path.with_suffix(".diff.json"), differences)
    return config_path, differences


def _initialize_fas_buckets(trainer: Any) -> Dict[str, Any]:
    sampler = trainer.pipeline.datamanager.train_pixel_sampler
    if not getattr(sampler, "is_initialized", False):
        dataset = getattr(sampler, "dataset", None)
        if dataset is None:
            raise InfrastructureError("Train pixel sampler has no target dataset for FAS buckets")
        sampler._initialize_buckets(dataset)
    counts = [int(sampler.buckets[level].shape[0]) for level in range(16)]
    if len(counts) != 16 or sum(counts) <= 0:
        raise InfrastructureError(f"Invalid target FAS buckets: {counts}")
    return {
        "initialized": bool(sampler.is_initialized),
        "sample_count": int(sampler.sample_count),
        "bucket_counts": counts,
        "nonempty_buckets": sum(count > 0 for count in counts),
    }


def _startup_audit(trainer: Any, *, expected_mode: str) -> Dict[str, Any]:
    load_audit = dict(trainer.checkpoint_load_audit)
    optimizer = trainer.optimizers.optimizers["fields"]
    scheduler = trainer.optimizers.schedulers["fields"]
    pipeline = trainer.pipeline
    model = pipeline.model
    sampler = pipeline.datamanager.train_pixel_sampler
    expected_scaler_state = {
        "scale": float(trainer.config.grad_scaler_init_scale),
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": int(trainer.config.grad_scaler_growth_interval),
        "_growth_tracker": 0,
    }
    audit = {
        "mode": expected_mode,
        "checkpoint_load_audit": load_audit,
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "scaler_state": trainer.grad_scaler.state_dict(),
        "occupancy_nonzero": int(model.occupancy_grid.occs.count_nonzero().item()),
        "occupancy_binary_true": int(model.occupancy_grid.binaries.sum().item()),
        "occupancy_binary_numel": int(model.occupancy_grid.binaries.numel()),
        "frequency_grid_nonzero": int(model.freq_grid.grid.count_nonzero().item()),
        "fas_sample_count": int(pipeline.fas_sample_count_state.item()),
        "pixel_sampler_count": int(sampler.sample_count),
        "cumulative_point_samples": int(pipeline.cumulative_point_samples.item()),
    }
    if expected_mode == "model_parameters_only":
        required = {
            "local_step_zero": load_audit.get("local_start_step") == 0,
            "optimizer_fresh": audit["optimizer_state_entries"] == 0,
            "optimizer_lr_fresh": math.isclose(
                audit["optimizer_lr"],
                float(trainer.config.optimizers["fields"]["optimizer"].lr),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "optimizer_not_loaded": load_audit.get("optimizer_loaded") is False,
            "scheduler_not_loaded": load_audit.get("scheduler_loaded") is False,
            "scheduler_fresh": audit["scheduler_last_epoch"] == 0,
            "scaler_not_loaded": load_audit.get("scaler_loaded") is False,
            "scaler_fresh": audit["scaler_state"] == expected_scaler_state,
            "rng_not_loaded": load_audit.get("rng_loaded") is False,
            "buffers_not_loaded": load_audit.get("pipeline_buffers_loaded") is False,
            "occupancy_zero": audit["occupancy_nonzero"] == 0,
            "occupancy_binary_fresh": audit["occupancy_binary_true"]
            == load_audit.get("fresh_state_assertions", {}).get(
                "occupancy_binary_constructor_true_count"
            ),
            "frequency_zero": audit["frequency_grid_nonzero"] == 0,
            "fas_zero": audit["fas_sample_count"] == 0 and audit["pixel_sampler_count"] == 0,
            "points_zero": audit["cumulative_point_samples"] == 0,
            "parameter_hashes": bool(load_audit.get("source_parameter_sha256"))
            and load_audit.get("source_parameter_sha256")
            == load_audit.get("copied_parameter_sha256"),
            "trainer_fresh_assertions": all(
                value is True
                for name, value in load_audit.get("fresh_state_assertions", {}).items()
                if name.endswith("_zero")
            ),
        }
    else:
        required = {
            "optimizer_loaded": load_audit.get("optimizer_loaded") is True,
            "scheduler_loaded": load_audit.get("scheduler_loaded") is True,
            "scaler_loaded": load_audit.get("scaler_loaded") is True,
            "buffers_loaded": load_audit.get("pipeline_buffers_loaded") is True,
            "no_occupancy_reset": load_audit.get("occupancy_reset") is None,
            "no_frequency_reset": load_audit.get("frequency_reset") is None,
        }
    audit["required"] = required
    if not all(required.values()):
        raise InfrastructureError(f"Worker startup audit failed: {required}")
    return audit


def _seed_worker(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def _install_training_eval_instrumentation(trainer: Any) -> None:
    run_dir = trainer.base_dir
    original_all = trainer.pipeline.get_average_eval_image_metrics
    original_eval_iteration = trainer.eval_iteration

    def measured_all(
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ) -> Dict[str, float]:
        if step is None or output_path is not None:
            return original_all(step=step, output_path=output_path, get_std=get_std)
        eval_dir = run_dir / "evaluations" / f"step-{step:09d}"
        render_dir = eval_dir / "renders"
        started_ns = time.time_ns()
        started_monotonic = time.monotonic()
        metrics = original_all(step=step, output_path=render_dir, get_std=False)
        completed_ns = time.time_ns()
        payload = {
            "schema_version": 1,
            "local_step": int(step),
            "started_wall_time_ns": started_ns,
            "completed_wall_time_ns": completed_ns,
            "full_eval_seconds": time.monotonic() - started_monotonic,
            "render_dir": str(render_dir),
            "results": {
                name: float(metrics[name])
                for name in ("psnr", "ssim", "lpips")
            },
        }
        atomic_json(eval_dir / "eval.json", payload)
        return metrics

    def measured_eval_iteration(step: int) -> None:
        started = time.monotonic()
        original_eval_iteration(step)
        elapsed = time.monotonic() - started
        if step > 0 and step % INTERVAL == 0:
            eval_path = run_dir / "evaluations" / f"step-{step:09d}" / "eval.json"
            if not eval_path.is_file():
                raise InfrastructureError(f"Scheduled full eval did not write {eval_path}")
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            payload["scheduled_eval_total_seconds"] = elapsed
            atomic_json(eval_path, payload)
            append_jsonl(
                run_dir / "eval_boundary_timings.jsonl",
                {
                    "local_step": step,
                    "scheduled_eval_total_seconds": elapsed,
                    "full_eval_seconds": payload["full_eval_seconds"],
                    "completed_wall_time_ns": payload["completed_wall_time_ns"],
                },
            )

    trainer.pipeline.get_average_eval_image_metrics = measured_all
    trainer.eval_iteration = measured_eval_iteration


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_config is None or args.worker_result is None or args.worker_mode is None:
        raise InfrastructureError("Worker mode requires --worker-config and --worker-result")
    config = yaml.load(args.worker_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    _seed_worker(config.machine.seed)
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup(test_mode="test" if args.worker_mode == "baseline" else "val")
    startup_audit = _startup_audit(trainer, expected_mode=config.checkpoint_load_mode)
    fas_audit = _initialize_fas_buckets(trainer)
    startup_audit["fas_buckets"] = fas_audit
    atomic_json(trainer.base_dir / "startup_audit.json", startup_audit)

    if args.worker_mode == "baseline":
        eval_dir = trainer.base_dir / "evaluations" / "preupdate-step-000000000"
        render_dir = eval_dir / "renders"
        started_ns = time.time_ns()
        started = time.monotonic()
        trainer.pipeline.eval()
        metrics = trainer.pipeline.get_average_eval_image_metrics(
            step=0, output_path=render_dir, get_std=True
        )
        payload = {
            "schema_version": 1,
            "local_step": 0,
            "preupdate": True,
            "started_wall_time_ns": started_ns,
            "completed_wall_time_ns": time.time_ns(),
            "full_eval_seconds": time.monotonic() - started,
            "render_dir": str(render_dir),
            "results": {name: float(metrics[name]) for name in ("psnr", "ssim", "lpips")},
            "startup_audit": startup_audit,
        }
        atomic_json(eval_dir / "eval.json", payload)
        atomic_json(args.worker_result, payload)
        return 0

    _install_training_eval_instrumentation(trainer)
    trainer.train()
    checkpoints = sorted(trainer.checkpoint_dir.glob("step-*.ckpt"))
    target_step = int(config.max_num_iterations) - 1
    target_checkpoint = checkpoint_path(trainer.base_dir, target_step)
    if not target_checkpoint.is_file():
        raise InfrastructureError(f"Worker did not write target checkpoint: {target_checkpoint}")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "run_dir": str(trainer.base_dir),
        "startup_audit": startup_audit,
        "checkpoints": [
            {
                "path": str(path),
                "step": checkpoint_step(path),
                "size_bytes": path.stat().st_size,
            }
            for path in checkpoints
        ],
        "target_checkpoint": str(target_checkpoint),
        "target_checkpoint_sha256": sha256_file(target_checkpoint),
    }
    atomic_json(args.worker_result, payload)
    return 0


class CampaignStore:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.lock = threading.RLock()
        if path.exists():
            if not resume:
                raise InfrastructureError(f"Campaign exists; use --resume: {path}")
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "created_at": utc_now(),
                "status": "initialized",
                "segments": {},
            }
            self.flush()

    def flush(self) -> None:
        with self.lock:
            self.data["updated_at"] = utc_now()
            atomic_json(self.path, self.data)


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _new_checkpoint_count(segment: Segment) -> int:
    if segment.load_mode == "model_parameters_only":
        return segment.target_step // INTERVAL
    return 1


def finalize_segment_record(
    store: CampaignStore,
    segment: Segment,
    record: Dict[str, Any],
    target: Path,
    result_path: Path,
) -> Dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    target_hash = sha256_file(target)
    if result.get("target_checkpoint_sha256") != target_hash:
        raise InfrastructureError(f"Worker result does not bind target checkpoint: {target}")
    eval_seconds = 0.0
    parent_step = (
        checkpoint_step(segment.parent_checkpoint)
        if segment.load_mode == "resume"
        else -1
    )
    for eval_path in segment.run_dir.glob("evaluations/step-*/eval.json"):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        local_step = int(payload["local_step"])
        if parent_step < local_step <= segment.target_step:
            eval_seconds += float(payload.get("scheduled_eval_total_seconds", 0))
    if record.get("trainer_wall_seconds") is None:
        started_ns = int(record["started_wall_time_ns"])
        record["trainer_wall_seconds"] = max(
            (result_path.stat().st_mtime_ns - started_ns) / 1e9, 0.0
        )
        record["trainer_wall_recovered_from_mtime"] = True
    record.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "checkpoint": str(target),
            "checkpoint_sha256": target_hash,
            "scheduled_eval_seconds_total": eval_seconds,
            "trainer_non_eval_seconds": max(
                float(record["trainer_wall_seconds"]) - eval_seconds, 0.0
            ),
            "worker_result": result,
        }
    )
    with store.lock:
        store.data["segments"][segment.segment_id] = record
        store.flush()
    return record


def run_segment(
    args: argparse.Namespace,
    store: CampaignStore,
    segment: Segment,
) -> Dict[str, Any]:
    with store.lock:
        existing = store.data["segments"].get(segment.segment_id)
    expected = asdict(segment)
    expected["run_dir"] = str(segment.run_dir)
    expected["parent_checkpoint"] = str(segment.parent_checkpoint)
    expected["arm"] = asdict(segment.arm)
    target = checkpoint_path(segment.run_dir, segment.target_step)
    result_path = (
        args.output_dir
        / "worker_results"
        / f"{segment.segment_id}_to_step-{segment.target_step:09d}.json"
    )
    if isinstance(existing, Mapping):
        mismatch = {
            key: {"stored": existing.get(key), "requested": value}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatch:
            raise InfrastructureError(f"Segment ID collision: {segment.segment_id}: {mismatch}")
        if existing.get("status") == "complete":
            if not target.is_file() or sha256_file(target) != existing.get("checkpoint_sha256"):
                raise InfrastructureError(f"Completed segment checkpoint changed: {target}")
            return dict(existing)
        if existing.get("status") == "running" and pid_alive(existing.get("pid")):
            raise InfrastructureError(
                f"Segment {segment.segment_id} is still active as PID {existing.get('pid')}"
            )
        if target.is_file() and result_path.is_file():
            return finalize_segment_record(
                store, segment, dict(existing), target, result_path
            )
        completed_parent_step = (
            checkpoint_step(segment.parent_checkpoint)
            if segment.load_mode == "resume"
            else -1
        )
        partial = [
            path
            for path in segment.run_dir.glob("nerfstudio_models/step-*.ckpt")
            if checkpoint_step(path) > completed_parent_step
        ]
        if partial:
            raise InfrastructureError(
                f"Segment {segment.segment_id} has incomplete checkpoint artifacts; "
                "refusing to overwrite them"
            )
    elif (
        segment.load_mode == "model_parameters_only"
        and segment.run_dir.exists()
        and any(segment.run_dir.iterdir())
    ):
        raise InfrastructureError(
            f"Untracked run directory collision: {segment.run_dir}"
        )
    if result_path.is_file() and not target.is_file():
        raise InfrastructureError(
            "Worker result exists without its target checkpoint; refusing to overwrite "
            f"{result_path}"
        )

    forecast = _new_checkpoint_count(segment) * LEADER_CHECKPOINT.stat().st_size
    disk_guard(args.output_dir, forecast_bytes=forecast)
    config_path, differences = write_segment_config(args, segment)
    command = [
        str(args.venv / "bin" / "python"),
        str(SCRIPT_PATH),
        "--worker-mode",
        "train",
        "--worker-config",
        str(config_path),
        "--worker-result",
        str(result_path),
    ]
    record = {
        **expected,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_diff": differences,
        "command": command,
        "status": "preparing",
        "attempt": int(existing.get("attempt", 0)) + 1 if isinstance(existing, Mapping) else 1,
    }
    with store.lock:
        store.data["segments"][segment.segment_id] = record
        store.flush()
    log_path = segment.run_dir / (
        f"train_to_step-{segment.target_step:09d}_attempt-{record['attempt']:02d}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        started = time.monotonic()
        started_wall_time_ns = time.time_ns()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        record.update(
            {
                "status": "running",
                "started_at": utc_now(),
                "started_wall_time_ns": started_wall_time_ns,
                "pid": process.pid,
            }
        )
        with store.lock:
            store.data["segments"][segment.segment_id] = record
            store.flush()
        returncode = process.wait()
    record["trainer_wall_seconds"] = time.monotonic() - started
    record["returncode"] = returncode
    if returncode != 0 or not result_path.is_file() or not target.is_file():
        record["status"] = "infrastructure_error"
        with store.lock:
            store.data["segments"][segment.segment_id] = record
            store.flush()
        raise InfrastructureError(f"Training segment failed; see {log_path}")
    return finalize_segment_record(store, segment, record, target, result_path)


def available_parallelism(args: argparse.Namespace, jobs: int) -> Tuple[int, Dict[str, Any]]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        env=run_environment(args),
    )
    devices = []
    for line in output.splitlines():
        index, name, free, utilization = [part.strip() for part in line.split(",", 3)]
        devices.append(
            {
                "index": int(index),
                "name": name,
                "free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    best = max(devices, key=lambda row: row["free_mib"])
    requested = min(args.max_parallel, jobs)
    memory_safe = best["free_mib"] >= requested * VRAM_PER_JOB_MIB + VRAM_RESERVE_MIB
    utilization_safe = best["utilization_percent"] < 98
    parallel = requested if memory_safe and utilization_safe else 1
    return parallel, {
        "requested": requested,
        "selected": parallel,
        "memory_safe": memory_safe,
        "utilization_safe": utilization_safe,
        "devices": devices,
    }


def run_segments(
    args: argparse.Namespace,
    store: CampaignStore,
    segments: Sequence[Segment],
) -> None:
    pending = [
        segment
        for segment in segments
        if store.data["segments"].get(segment.segment_id, {}).get("status") != "complete"
    ]
    if not pending:
        return
    forecast = sum(_new_checkpoint_count(segment) for segment in pending) * LEADER_CHECKPOINT.stat().st_size
    disk_guard(args.output_dir, forecast_bytes=forecast)
    parallel, audit = available_parallelism(args, len(pending))
    store.data.setdefault("parallelism", []).append({"at": utc_now(), **audit})
    store.flush()
    if parallel == 1:
        for segment in pending:
            run_segment(args, store, segment)
        return
    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="lookcloser-v2") as executor:
        futures = {executor.submit(run_segment, args, store, segment): segment for segment in pending}
        for future in as_completed(futures):
            future.result()


def visual_decisions(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for key, raw in payload.items():
        if isinstance(raw, str):
            raw = {"verdict": raw}
        if not isinstance(raw, Mapping):
            raise InfrastructureError(f"Invalid visual decision for {key}: {raw!r}")
        verdict = str(raw.get("verdict", "pending"))
        change = str(raw.get("change_from_previous", "not_applicable"))
        if verdict not in {"pending", "pass", "fail"}:
            raise InfrastructureError(f"Invalid visual verdict for {key}: {verdict}")
        if change not in {"not_applicable", "improved", "no_improvement", "regressed"}:
            raise InfrastructureError(f"Invalid visual change for {key}: {change}")
        result[key] = {
            "verdict": verdict,
            "change_from_previous": change,
            "note": str(raw.get("note", "")),
        }
    return result


def _protocol_decision(
    decisions: Mapping[str, Mapping[str, str]],
    arm_id: str,
    step: int,
) -> Dict[str, str]:
    return dict(
        decisions.get(
            f"{arm_id}:{step}",
            {"verdict": "pending", "change_from_previous": "not_applicable", "note": ""},
        )
    )


def build_boundary_protocol(
    *,
    arm_id: str,
    step: int,
    render_dir: Path,
    output_dir: Path,
    decisions: Mapping[str, Mapping[str, str]],
) -> Path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    from static_target_roi_protocol import build_protocol

    decision = _protocol_decision(decisions, arm_id, step)
    protocol_args = SimpleNamespace(
        frame="007747",
        dataset=TARGET_DATASET,
        render_dir=render_dir,
        out_dir=output_dir,
        leader_render_dir=LEADER_RENDERS,
        scratch_render_dir=SCRATCH_RENDERS,
        visual_verdict=decision["verdict"],
        visual_note=decision["note"],
        visual_change=decision["change_from_previous"],
    )
    build_protocol(protocol_args)
    return output_dir / "static_target_roi_protocol.json"


def discover_boundaries(
    arm_id: str,
    run_dir: Path,
    decisions: Mapping[str, Mapping[str, str]],
) -> List[Boundary]:
    boundaries = []
    for eval_json in sorted(run_dir.glob("evaluations/step-*/eval.json")):
        payload = json.loads(eval_json.read_text(encoding="utf-8"))
        step = int(payload["local_step"])
        checkpoint = checkpoint_path(run_dir, step)
        if not checkpoint.is_file():
            raise InfrastructureError(f"Evaluated boundary lacks checkpoint: {checkpoint}")
        protocol_dir = eval_json.parent / "priority_roi"
        protocol = build_boundary_protocol(
            arm_id=arm_id,
            step=step,
            render_dir=Path(payload["render_dir"]),
            output_dir=protocol_dir,
            decisions=decisions,
        )
        metrics = payload["results"]
        boundaries.append(
            Boundary(
                arm_id=arm_id,
                local_step=step,
                psnr=float(metrics["psnr"]),
                ssim=float(metrics["ssim"]),
                lpips=float(metrics["lpips"]),
                checkpoint=checkpoint,
                eval_json=eval_json,
                protocol_json=protocol,
                eval_completed_wall_time_ns=int(payload["completed_wall_time_ns"]),
            )
        )
    return boundaries


def protocol_payload(boundary: Boundary) -> Mapping[str, Any]:
    if boundary.protocol_json is None or not boundary.protocol_json.is_file():
        raise InfrastructureError(f"Missing visual protocol: {boundary.arm_id} {boundary.local_step}")
    return json.loads(boundary.protocol_json.read_text(encoding="utf-8"))


def visual_verdict(boundary: Boundary) -> str:
    return str(protocol_payload(boundary)["visual_gate"]["verdict"])


def visual_change(boundary: Boundary) -> str:
    return str(protocol_payload(boundary)["visual_gate"]["change_from_previous"])


def visual_pass(boundary: Boundary) -> bool:
    protocol = protocol_payload(boundary)
    return (
        protocol["visual_gate"]["verdict"] == "pass"
        and int(protocol["full_view_serious_count"]) == 0
        and protocol["roi"]["artifact"]["serious"] is False
    )


def require_reviewed(boundaries: Iterable[Boundary]) -> None:
    pending = [
        f"{row.arm_id}:{row.local_step}"
        for row in boundaries
        if visual_verdict(row) == "pending"
    ]
    if pending:
        raise QualityStop(
            "Visual decisions are pending. Review native contact sheets and add decisions for: "
            + ", ".join(pending)
        )


def select_boundary(boundaries: Sequence[Boundary]) -> Boundary:
    if not boundaries:
        raise QualityStop("No visual-pass checkpoints are available for selection")
    maximum = max(row.psnr for row in boundaries)
    tied = [row for row in boundaries if row.psnr >= maximum - PSNR_TIE_DB]
    return min(tied, key=lambda row: (row.lpips, -row.psnr, row.local_step, row.arm_id))


def select_wave_a_lr(boundaries: Sequence[Boundary]) -> float:
    matched = [
        row for row in boundaries if row.local_step == INITIAL_FINAL_STEP and visual_pass(row)
    ]
    selected = select_boundary(matched)
    return next(lr for arm_id, lr, _ in WAVE_A if arm_id == selected.arm_id)


def select_authoritative_arm(
    arms: Sequence[Arm],
    boundaries: Sequence[Boundary],
) -> Tuple[Arm, Optional[Boundary]]:
    passing = [row for row in boundaries if row.numeric_pass and visual_pass(row)]
    first_pass = None
    if passing:
        earliest_step = min(row.local_step for row in passing)
        first_pass = select_boundary([row for row in passing if row.local_step == earliest_step])
        arm_id = first_pass.arm_id
    else:
        selected = select_boundary([row for row in boundaries if visual_pass(row)])
        arm_id = selected.arm_id
    arm = next(row for row in arms if row.arm_id == arm_id)
    return Arm(
        arm_id=f"authoritative-{arm.arm_id}",
        lr_init=arm.lr_init,
        scheduler_max_steps=arm.scheduler_max_steps,
        phase="authoritative",
    ), first_pass


def select_extended_authoritative_arm(
    arms: Sequence[Arm],
    boundaries: Sequence[Boundary],
) -> Tuple[Arm, Optional[Boundary]]:
    """Select the long-horizon rescue arm for fastest credible LPIPS convergence.

    A complete leader pass still wins at the earliest evaluated boundary.  When
    no arm has passed yet, use the final discovery boundary and minimize LPIPS
    only among visually accepted arms that already clear PSNR and SSIM.  This
    keeps every gate intact while avoiding the PSNR-first fallback that selected
    the now-confirmed H200 LPIPS plateau.
    """

    passing = [row for row in boundaries if row.numeric_pass and visual_pass(row)]
    first_pass = None
    if passing:
        earliest_step = min(row.local_step for row in passing)
        first_pass = select_boundary([row for row in passing if row.local_step == earliest_step])
        selected = first_pass
    else:
        final_rows = [
            row
            for row in boundaries
            if row.local_step == INITIAL_FINAL_STEP
            and visual_pass(row)
            and row.psnr >= PSNR_THRESHOLD
            and row.ssim >= SSIM_THRESHOLD
        ]
        if not final_rows:
            raise FinalQualityFailure(
                "Extended scheduler screen produced no visually accepted final boundary "
                "that clears the PSNR and SSIM leader gates"
            )
        selected = min(
            final_rows,
            key=lambda row: (row.lpips, -row.psnr, -row.ssim, row.arm_id),
        )
    arm = next(row for row in arms if row.arm_id == selected.arm_id)
    return Arm(
        arm_id=f"authoritative-{arm.arm_id}",
        lr_init=arm.lr_init,
        scheduler_max_steps=arm.scheduler_max_steps,
        phase="authoritative",
    ), first_pass


def _interval_is_plateau(previous: Boundary, current: Boundary) -> bool:
    return (
        current.local_step - previous.local_step == INTERVAL
        and current.psnr - previous.psnr < 0.03
        and current.ssim - previous.ssim < 0.001
        and previous.lpips - current.lpips < 0.003
        and visual_change(current) in {"no_improvement", "regressed"}
    )


def plateau_confirmed(boundaries: Sequence[Boundary]) -> bool:
    ordered = sorted(boundaries, key=lambda row: row.local_step)
    if len(ordered) < 3:
        return False
    last = ordered[-3:]
    return _interval_is_plateau(last[0], last[1]) and _interval_is_plateau(last[1], last[2])


def baseline_segment(args: argparse.Namespace) -> Segment:
    arm = Arm("baseline-preupdate", 0.01, 200_000, "authoritative")
    return Segment(
        segment_id="baseline-preupdate",
        arm=arm,
        run_dir=arm_run_dir(args.output_dir, "baseline", arm.arm_id),
        target_step=0,
        load_mode="model_parameters_only",
        parent_checkpoint=LEADER_CHECKPOINT,
    )


def run_baseline(
    args: argparse.Namespace,
    store: CampaignStore,
    decisions: Mapping[str, Mapping[str, str]],
) -> None:
    existing = store.data.get("baseline")
    if isinstance(existing, Mapping) and existing.get("status") == "complete":
        return
    segment = baseline_segment(args)
    config_path, differences = write_segment_config(args, segment)
    result_path = args.output_dir / "worker_results" / "baseline-preupdate.json"
    command = [
        str(args.venv / "bin" / "python"),
        str(SCRIPT_PATH),
        "--worker-mode",
        "baseline",
        "--worker-config",
        str(config_path),
        "--worker-result",
        str(result_path),
    ]
    record = {
        "status": "running",
        "started_at": utc_now(),
        "command": command,
        "config": str(config_path),
        "config_diff": differences,
        "excluded_from_time_to_leader": True,
    }
    store.data["baseline"] = record
    store.flush()
    log_path = segment.run_dir / "baseline_stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not result_path.is_file():
        record["status"] = "infrastructure_error"
        store.flush()
        raise InfrastructureError(f"No-update baseline failed; see {log_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = build_boundary_protocol(
        arm_id="baseline-preupdate",
        step=0,
        render_dir=Path(result["render_dir"]),
        output_dir=Path(result["render_dir"]).parent / "priority_roi",
        decisions=decisions,
    )
    record.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "result": result,
            "protocol": str(protocol),
        }
    )
    store.data["baseline"] = record
    store.flush()


def wave_a_arms(profile: str = "leader_base") -> List[Arm]:
    if profile == "leader_base":
        recipes = WAVE_A
    elif profile == "extended_scheduler":
        recipes = EXTENDED_SCHEDULER_WAVE
    else:
        raise ValueError(f"Unknown campaign profile: {profile}")
    return [Arm(arm_id, lr, horizon, "wave_a") for arm_id, lr, horizon in recipes]


def initial_segment(args: argparse.Namespace, arm: Arm) -> Segment:
    return Segment(
        segment_id=f"{arm.phase}-{arm.arm_id}",
        arm=arm,
        run_dir=arm_run_dir(args.output_dir, arm.phase, arm.arm_id),
        target_step=INITIAL_FINAL_STEP,
        load_mode="model_parameters_only",
        parent_checkpoint=LEADER_CHECKPOINT,
    )


def reviewed_boundaries_for_arms(
    arms: Sequence[Arm],
    args: argparse.Namespace,
    decisions: Mapping[str, Mapping[str, str]],
) -> List[Boundary]:
    result = []
    for arm in arms:
        result.extend(
            discover_boundaries(
                arm.arm_id,
                arm_run_dir(args.output_dir, arm.phase, arm.arm_id),
                decisions,
            )
        )
    require_reviewed(result)
    return result


def authoritative_segment(
    args: argparse.Namespace,
    arm: Arm,
    *,
    target_step: int,
    parent: Optional[Path],
) -> Segment:
    run_dir = arm_run_dir(args.output_dir, "authoritative", arm.arm_id)
    return Segment(
        segment_id=f"{arm.arm_id}-to-{target_step}",
        arm=arm,
        run_dir=run_dir,
        target_step=target_step,
        load_mode="resume" if parent is not None else "model_parameters_only",
        parent_checkpoint=parent or LEADER_CHECKPOINT,
    )


def write_campaign_report(store: CampaignStore, summary: Mapping[str, Any]) -> None:
    report = store.path.parent / "report.md"
    first = summary.get("first_leader_pass")
    selected = summary["plateau_selected"]
    visual = summary["visual_selected"]
    lines = [
        "# 007740 → 007747 fine-tuning v2",
        "",
        "## What was tested",
        "",
        (
            "Direct hash23 model-parameters-only transfer with standard 007747 maps, "
            "FR0.3/FAS1.0, a staged LR/scheduler screen, and an authoritative solo replay."
        ),
        "",
        "## Results",
        "",
        "| Result | Step | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|",
    ]
    if first:
        lines.append(
            f"| First leader pass | {first['local_step']} | {first['psnr']:.6f} | "
            f"{first['ssim']:.6f} | {first['lpips']:.6f} |"
        )
    lines.extend(
        [
            (
                f"| Plateau selector | {selected['local_step']} | {selected['psnr']:.6f} | "
                f"{selected['ssim']:.6f} | {selected['lpips']:.6f} |"
            ),
            (
                f"| Visual selector | {visual['local_step']} | {visual['psnr']:.6f} | "
                f"{visual['ssim']:.6f} | {visual['lpips']:.6f} |"
            ),
            "",
            f"`time_to_leader_seconds`: `{summary.get('time_to_leader_seconds')}`",
            "",
            "Full training/evaluation timings, hashes, renders, and crop protocols are in `campaign.json`.",
            "",
            "## Insights",
            "",
            "The report intentionally excludes evaluation loss.",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def boundary_dict(row: Boundary) -> Dict[str, Any]:
    return {
        **asdict(row),
        "checkpoint": str(row.checkpoint),
        "eval_json": str(row.eval_json),
        "protocol_json": str(row.protocol_json) if row.protocol_json else None,
        "numeric_pass": row.numeric_pass,
        "visual_pass": visual_pass(row),
    }


def fresh_confirm_boundary(
    args: argparse.Namespace,
    arm: Arm,
    boundary: Boundary,
) -> Dict[str, Any]:
    run_dir = arm_run_dir(args.output_dir, "authoritative", arm.arm_id)
    config = yaml.load((run_dir / "config.yml").read_text(encoding="utf-8"), Loader=yaml.Loader)
    config.load_dir = boundary.checkpoint.parent
    config.load_step = boundary.local_step
    config.load_checkpoint = boundary.checkpoint
    config.pipeline.model.eval_num_rays_per_chunk = 2048
    confirmation_dir = (
        args.output_dir / "final_confirmation" / f"step-{boundary.local_step:09d}"
    )
    confirmation_dir.mkdir(parents=True, exist_ok=True)
    config_path = confirmation_dir / "eval_config.yml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    result_path = confirmation_dir / "eval.json"
    render_dir = confirmation_dir / "renders"
    command = [
        str(args.venv / "bin" / "ns-eval"),
        "--load-config",
        str(config_path),
        "--output-path",
        str(result_path),
        "--render-output-path",
        str(render_dir),
    ]
    started = time.monotonic()
    with (confirmation_dir / "eval_stdout.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0 or not result_path.is_file():
        raise InfrastructureError(
            f"Fresh final evaluation failed for {boundary.checkpoint}; "
            f"see {confirmation_dir / 'eval_stdout.log'}"
        )
    renders = sorted(render_dir.glob("eval_img_*.png"))
    if len(renders) != 3:
        raise InfrastructureError(f"Fresh confirmation wrote {len(renders)} renders, expected 3")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = {name: float(result["results"][name]) for name in ("psnr", "ssim", "lpips")}
    scheduled = {"psnr": boundary.psnr, "ssim": boundary.ssim, "lpips": boundary.lpips}
    drift = {name: metrics[name] - scheduled[name] for name in metrics}
    tolerances = {"psnr": 1e-4, "ssim": 1e-5, "lpips": 1e-4}
    if any(abs(drift[name]) > tolerances[name] for name in drift):
        raise InfrastructureError(f"Fresh final metrics drift from scheduled eval: {drift}")
    decision = {
        f"{arm.arm_id}:{boundary.local_step}": {
            "verdict": visual_verdict(boundary),
            "change_from_previous": visual_change(boundary),
            "note": "Fresh confirmation of plateau selection.",
        }
    }
    protocol = build_boundary_protocol(
        arm_id=arm.arm_id,
        step=boundary.local_step,
        render_dir=render_dir,
        output_dir=confirmation_dir / "priority_roi",
        decisions=decision,
    )
    return {
        "checkpoint": str(boundary.checkpoint),
        "checkpoint_sha256": sha256_file(boundary.checkpoint),
        "eval_seconds": elapsed,
        "metrics": metrics,
        "metric_drift": drift,
        "renders": str(render_dir),
        "protocol": str(protocol),
    }


def finalize_campaign(
    args: argparse.Namespace,
    store: CampaignStore,
    arm: Arm,
    boundaries: Sequence[Boundary],
) -> None:
    formal = select_boundary(boundaries)
    visual = select_boundary([row for row in boundaries if visual_pass(row)])
    passes = [row for row in boundaries if row.numeric_pass and visual_pass(row)]
    first = min(passes, key=lambda row: row.local_step) if passes else None
    authoritative_start = int(store.data["authoritative_started_wall_time_ns"])
    time_to_leader = (
        (first.eval_completed_wall_time_ns - authoritative_start) / 1e9 if first else None
    )
    confirmations = {
        f"step-{formal.local_step:09d}": fresh_confirm_boundary(args, arm, formal)
    }
    if visual.local_step != formal.local_step:
        confirmations[f"step-{visual.local_step:09d}"] = fresh_confirm_boundary(
            args, arm, visual
        )
    summary = {
        "arm": asdict(arm),
        "first_leader_pass": boundary_dict(first) if first else None,
        "time_to_leader_seconds": time_to_leader,
        "plateau_selected": boundary_dict(formal),
        "visual_selected": boundary_dict(visual),
        "fresh_confirmations": confirmations,
        "boundaries": [boundary_dict(row) for row in boundaries],
    }
    atomic_json(args.output_dir / "summary.json", summary)
    store.data["summary"] = summary
    store.data["status"] = "complete" if first is not None else "failed_to_reach_leader"
    store.flush()
    write_campaign_report(store, summary)
    if first is None:
        raise FinalQualityFailure(
            "Authoritative run reached plateau without passing all leader gates"
        )


def deterministic_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = []
    for arm in wave_a_arms(args.campaign_profile):
        segment = initial_segment(args, arm)
        _, differences = configured_segment(args, segment)
        rows.append(
            {
                "arm": asdict(arm),
                "segment": {
                    **asdict(segment),
                    "run_dir": str(segment.run_dir),
                    "parent_checkpoint": str(segment.parent_checkpoint),
                    "arm": asdict(segment.arm),
                },
                "config_diff": differences,
            }
        )
    return {
        "schema_version": 1,
        "output_dir": str(args.output_dir),
        "campaign_profile": args.campaign_profile,
        "wave_a": rows,
        "wave_b_horizons": (
            [100_000, 150_000] if args.campaign_profile == "leader_base" else []
        ),
        "discovery_selection_policy": (
            "leader pass first, otherwise final-boundary minimum LPIPS among "
            "visual PSNR/SSIM passes"
            if args.campaign_profile == "extended_scheduler"
            else "leader pass first, otherwise exact PSNR-0.07dB/LPIPS selector"
        ),
        "authoritative_policy": "solo replay from original hash23 leader",
    }


def run_campaign(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(json.dumps(deterministic_dry_run(args), indent=2, sort_keys=True, default=str))
        return 0
    campaign_path = args.output_dir / "campaign.json"
    preflight = full_preflight(
        args,
        initial_storage=not (args.resume and campaign_path.is_file()),
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CampaignStore(campaign_path, resume=args.resume)
    previous_profile = store.data.get("campaign_profile")
    if previous_profile is not None and previous_profile != args.campaign_profile:
        raise InfrastructureError(
            f"Campaign profile changed on resume: {previous_profile} -> "
            f"{args.campaign_profile}"
        )
    store.data["campaign_profile"] = args.campaign_profile
    if "preflight" in store.data:
        previous_static = {
            key: value for key, value in store.data["preflight"].items() if key != "storage"
        }
        current_static = {key: value for key, value in preflight.items() if key != "storage"}
        if previous_static != current_static:
            raise InfrastructureError("Preflight provenance changed since campaign creation")
    store.data["preflight"] = preflight
    store.data.setdefault("storage_checks", []).append(
        {"at": utc_now(), **preflight["storage"]}
    )
    store.flush()
    decisions = visual_decisions(args.visual_decisions)
    store.data.setdefault("visual_review_snapshots", []).append(
        {
            "at": utc_now(),
            "source": str(args.visual_decisions) if args.visual_decisions else None,
            "source_sha256": (
                sha256_file(args.visual_decisions) if args.visual_decisions else None
            ),
            "decisions": decisions,
        }
    )
    store.flush()
    run_baseline(args, store, decisions)

    arms_a = wave_a_arms(args.campaign_profile)
    run_segments(args, store, [initial_segment(args, arm) for arm in arms_a])
    boundaries_a = reviewed_boundaries_for_arms(arms_a, args, decisions)
    if args.campaign_profile == "leader_base":
        selected_lr = select_wave_a_lr(boundaries_a)
        store.data["wave_a_selected_lr"] = selected_lr
        store.flush()
        arms_b = [
            Arm(f"L{selected_lr:g}-H100", selected_lr, 100_000, "wave_b"),
            Arm(f"L{selected_lr:g}-H150", selected_lr, 150_000, "wave_b"),
        ]
        run_segments(args, store, [initial_segment(args, arm) for arm in arms_b])
        boundaries_b = reviewed_boundaries_for_arms(arms_b, args, decisions)
        all_discovery_arms = [*arms_a, *arms_b]
        all_discovery_boundaries = [*boundaries_a, *boundaries_b]
        authoritative_arm, discovery_pass = select_authoritative_arm(
            all_discovery_arms, all_discovery_boundaries
        )
        selection_policy = "leader_base_psnr_window_lpips"
    else:
        arms_b = []
        boundaries_b = []
        all_discovery_arms = arms_a
        all_discovery_boundaries = boundaries_a
        authoritative_arm, discovery_pass = select_extended_authoritative_arm(
            all_discovery_arms, all_discovery_boundaries
        )
        selection_policy = "extended_final_psnr_ssim_pass_min_lpips"
    store.data["discovery"] = {
        "selected_authoritative_arm": asdict(authoritative_arm),
        "earliest_passing_discovery_boundary": (
            boundary_dict(discovery_pass) if discovery_pass else None
        ),
        "selection_policy": selection_policy,
    }
    store.flush()

    initial = authoritative_segment(
        args,
        authoritative_arm,
        target_step=INITIAL_FINAL_STEP,
        parent=None,
    )
    record = run_segment(args, store, initial)
    store.data.setdefault("authoritative_started_wall_time_ns", record["started_wall_time_ns"])
    store.flush()
    run_dir = arm_run_dir(args.output_dir, "authoritative", authoritative_arm.arm_id)

    while True:
        boundaries = discover_boundaries(
            authoritative_arm.arm_id,
            run_dir,
            decisions,
        )
        require_reviewed(boundaries)
        ordered = sorted(boundaries, key=lambda row: row.local_step)
        if plateau_confirmed(ordered):
            finalize_campaign(args, store, authoritative_arm, ordered)
            return 0
        parent = ordered[-1].checkpoint
        next_step = ordered[-1].local_step + INTERVAL
        segment = authoritative_segment(
            args,
            authoritative_arm,
            target_step=next_step,
            parent=parent,
        )
        run_segment(args, store, segment)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.worker_mode is not None:
            return worker_main(args)
        return run_campaign(args)
    except QualityStop as error:
        if not args.worker_mode:
            campaign = args.output_dir / "campaign.json"
            if campaign.is_file():
                payload = json.loads(campaign.read_text(encoding="utf-8"))
                payload["status"] = "awaiting_visual_review"
                payload["quality_stop"] = str(error)
                atomic_json(campaign, payload)
        print(f"QUALITY_STOP: {error}", file=sys.stderr)
        return QUALITY_EXIT
    except FinalQualityFailure as error:
        print(f"QUALITY_FAILURE: {error}", file=sys.stderr)
        return QUALITY_EXIT
    except (InfrastructureError, FileNotFoundError, ValueError) as error:
        print(f"INFRASTRUCTURE_ERROR: {error}", file=sys.stderr)
        return INFRASTRUCTURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
