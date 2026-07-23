#!/usr/bin/env python3
"""Resumable, fail-closed temporal LookCloser fine-tuning campaign controller.

The legacy temporal-transfer runner remains forensic evidence only.  This controller
uses local zero-based steps for every cross-frame model-only transfer, preserves full
state only within a frame, and never forwards a checkpoint that has not completed the
fresh three-view/ROI/visual gate.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOKCLOSER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/brans/temporal_perframe_stride7_45f")
DEFAULT_OUTPUT = Path("/home/brans/lookcloser_temporal_finetune_runs")
DEFAULT_CAMPAIGN = DEFAULT_OUTPUT / "campaigns" / "temporal_stride7_seed42" / "campaign.json"
DEFAULT_REPORT = LOOKCLOSER_ROOT / "experiments" / "temporal_lookcloser_finetuning.md"
DEFAULT_VENV = Path("/home/brans/repos/nerfstudio/.venv")
DEFAULT_TCNN_OVERLAY = Path("/home/brans/deps/tcnn_2e757_py310")
DEFAULT_LEADER = Path(
    "/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/"
    "lookcloser/20260715_005006"
)
DEFAULT_LEADER_CHECKPOINT = DEFAULT_LEADER / "nerfstudio_models" / "step-000091128.ckpt"
DEFAULT_LEADER_CONFIG = DEFAULT_LEADER / "config.yml"

EXPECTED_LEADER_SHA256 = "3ba4472630d6332f60c58bd03a09a27894bca915139f9eee81b004ebf144a930"
EXPECTED_CONFIG_SHA256 = "a8c017c96a19a42fde3d43492b8253d970408b71c24cd47bcc449fed5fd0e5fb"
EXPECTED_TRANSFORMS_SHA256 = "022f8748a1a039861a754e68ab3ef830beeb3e5dd94ccb00457a630d28f64aa1"
EXPECTED_TCNN_BINDING_SHA256 = "f2163346afd103c27e78b9f56f8d82b6eeb3317c1ce11caf57d45f0216aece36"
EXPECTED_PYTHON = "3.10.20"
EXPECTED_TORCH = "2.7.1+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_GPU = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"

FRAME_NAMES = tuple(f"{frame:06d}" for frame in range(7_740, 8_049, 7))
CHAIN_FRAMES = FRAME_NAMES[1:]
LR_CANDIDATES = (5e-4, 1e-3, 2e-3)
TRAVERSAL_WARMUP_CANDIDATES = (4_096, 8_192)
INTERVAL = 15_188
SCREEN_FINAL_STEP = 60_752
SCREEN_MAX_ITERATIONS = SCREEN_FINAL_STEP + 1
MIN_DISK_FREE_BYTES = 100 * 1024**3
VRAM_PER_JOB_MIB = 20 * 1024
VRAM_RESERVE_MIB = 20 * 1024
PSNR_TIE_DB = 0.07
LEADER_PSNR_TOLERANCE_DB = 0.20
LEADER_SSIM_TOLERANCE = 0.010
LEADER_LPIPS_TOLERANCE = 0.015
QUALITY_EXIT = 2
INFRASTRUCTURE_EXIT = 3
CRITICAL_ROIS = {
    "thin_pipe_eval1",
    "tangled_cable_eval2",
    "hand_eval0",
    "chain_eval2",
    "fingers_eval2",
}
REPORT_COLUMNS = (
    "Frame",
    "Parent",
    "LR",
    "Selected local step",
    "PSNR",
    "SSIM",
    "LPIPS",
    "Gate",
    "Frame points",
    "Temporal points",
)


class InfrastructureError(RuntimeError):
    """The campaign cannot safely interpret a training/evaluation result."""


class QualityStop(RuntimeError):
    """A complete quality or ambiguity gate intentionally stopped the chain."""


@dataclass(frozen=True)
class Metrics:
    local_step: int
    psnr: float
    ssim: float
    lpips: float
    point_samples: Optional[int] = None
    checkpoint: Optional[Path] = None
    run_id: Optional[str] = None


@dataclass(frozen=True)
class BoundaryEvidence:
    metrics: Metrics
    critical_roi_lpips: Mapping[str, float]
    artifact_score: float


@dataclass(frozen=True)
class GateDecision:
    outcome: Literal["pass", "ambiguous", "fail"]
    reasons: Tuple[str, ...]
    critical_roi_regressions: Mapping[str, float]

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    frame: str
    seed: int
    lr: float
    phase: str
    feature_reweighting: float
    fas_strength: float
    load_mode: Literal["resume", "model_parameters_only"]
    parent_checkpoint: Optional[Path]
    target_local_step: int
    inherited_global_step: int
    lr_override: Optional[float] = None
    contended: bool = False
    from_scratch: bool = False
    scheduler_policy: Literal["constant", "leader"] = "constant"
    traversal_warmup_steps: int = 4096

    @property
    def local_updates_completed(self) -> int:
        return self.target_local_step + 1

    @property
    def effective_global_step(self) -> int:
        return self.inherited_global_step + self.target_local_step


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--tcnn-overlay", type=Path, default=DEFAULT_TCNN_OVERLAY)
    parser.add_argument("--leader-checkpoint", type=Path, default=DEFAULT_LEADER_CHECKPOINT)
    parser.add_argument("--leader-config", type=Path, default=DEFAULT_LEADER_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr-candidates",
        type=lambda value: _parse_candidate_tuple(value, float, "LR"),
        default=LR_CANDIDATES,
        help="Comma-separated constant fields LRs for the matched 007747 screen.",
    )
    parser.add_argument(
        "--traversal-warmup-candidates",
        type=lambda value: _parse_candidate_tuple(value, int, "traversal warmup"),
        default=TRAVERSAL_WARMUP_CANDIDATES,
        help=(
            "Comma-separated local update counts during which adaptive traversal stays fixed "
            "while fresh occupancy is rebuilt."
        ),
    )
    parser.add_argument("--max-parallel", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--start-frame", choices=CHAIN_FRAMES, default=CHAIN_FRAMES[0])
    parser.add_argument("--end-frame", choices=CHAIN_FRAMES, default=CHAIN_FRAMES[-1])
    parser.add_argument("--max-phase-a-intervals", type=int, default=10)
    parser.add_argument("--max-tail-intervals", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--visual-decisions", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a one-update 007747 model-only checkpoint/reset smoke instead of the campaign.",
    )
    return parser.parse_args(argv)


def _parse_candidate_tuple(
    value: str,
    converter: Any,
    label: str,
) -> Tuple[Any, ...]:
    try:
        parsed = tuple(converter(item.strip()) for item in value.split(",") if item.strip())
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Invalid {label} candidates: {value!r}") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{label} candidates must be positive and non-empty")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"{label} candidates must be unique")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
    if match is None:
        raise InfrastructureError(f"Cannot parse local checkpoint step: {path}")
    return int(match.group(1))


def checkpoint_path(run_dir: Path, local_step: int) -> Path:
    return run_dir / "nerfstudio_models" / f"step-{local_step:09d}.ckpt"


def select_metrics(candidates: Sequence[Metrics], tie_db: float = PSNR_TIE_DB) -> Metrics:
    """Paper-independent selector: max PSNR, inclusive 0.07 dB tie, then min LPIPS."""

    if not candidates:
        raise InfrastructureError("Checkpoint selector received no candidates")
    if tie_db < 0:
        raise ValueError("tie_db must be non-negative")
    for row in candidates:
        if not all(math.isfinite(value) for value in (row.psnr, row.ssim, row.lpips)):
            raise InfrastructureError(f"Non-finite checkpoint metric at local step {row.local_step}")
    maximum = max(row.psnr for row in candidates)
    tied = [row for row in candidates if maximum - row.psnr <= tie_db + 1e-12]
    return min(tied, key=lambda row: (row.lpips, -row.psnr, row.local_step))


def _interval_is_plateau(previous: BoundaryEvidence, current: BoundaryEvidence) -> bool:
    metric_plateau = (
        current.metrics.psnr - previous.metrics.psnr < 0.03
        and current.metrics.ssim - previous.metrics.ssim < 0.001
        and previous.metrics.lpips - current.metrics.lpips < 0.003
    )
    if not metric_plateau:
        return False
    common = CRITICAL_ROIS & set(previous.critical_roi_lpips) & set(current.critical_roi_lpips)
    if common != CRITICAL_ROIS:
        return False
    if any(
        previous.critical_roi_lpips[name] - current.critical_roi_lpips[name] >= 0.003
        for name in common
    ):
        return False
    # A decrease is an artifact-score improvement, so it blocks plateau.
    return current.artifact_score >= previous.artifact_score


def plateau_confirmed(evidence: Sequence[BoundaryEvidence], interval: int = INTERVAL) -> bool:
    """Require the last two complete, consecutive intervals to satisfy every condition."""

    if len(evidence) < 3:
        return False
    last = list(evidence[-3:])
    if any(
        right.metrics.local_step - left.metrics.local_step != interval
        for left, right in zip(last, last[1:])
    ):
        return False
    return _interval_is_plateau(last[0], last[1]) and _interval_is_plateau(last[1], last[2])


def protocol_roi_lpips(protocol: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in protocol.get("rois", []):
        metrics = row.get("metrics") if isinstance(row, Mapping) else None
        if isinstance(metrics, Mapping) and metrics.get("lpips") is not None:
            result[str(row["name"])] = float(metrics["lpips"])
    return result


def protocol_artifact_score(protocol: Mapping[str, Any]) -> float:
    full = [float(row.get("artifact_score", 0.0)) for row in protocol.get("full_views", [])]
    roi = [
        float(row.get("artifact", {}).get("artifact_score", 0.0))
        for row in protocol.get("rois", [])
        if isinstance(row, Mapping)
    ]
    return max(full + roi, default=float("inf"))


def quality_gate(
    protocol: Mapping[str, Any],
    *,
    previous_protocol: Optional[Mapping[str, Any]],
    visual_pass: Optional[bool],
) -> GateDecision:
    """Close promotion on missing evidence, artifacts, tracking ambiguity, or ROI regression."""

    fail: List[str] = []
    ambiguous: List[str] = []
    if protocol.get("status") != "complete":
        fail.append("ROI protocol is incomplete")
    views = protocol.get("full_views")
    rois = protocol.get("rois")
    if not isinstance(views, list) or len(views) != 3:
        fail.append("fresh eval does not contain exactly three views")
    if not isinstance(rois, list):
        fail.append("ROI list is missing")
        rois = []
    names = {str(row.get("name")) for row in rois if isinstance(row, Mapping)}
    missing = sorted(CRITICAL_ROIS - names)
    if missing:
        fail.append(f"critical ROIs are missing: {missing}")
    if int(protocol.get("full_view_serious_count", -1)) != 0:
        fail.append("one or more serious full-view artifacts")
    if int(protocol.get("roi_serious_count", -1)) != 0:
        fail.append("one or more serious ROI artifacts")
    tracking = protocol.get("tracking")
    if not isinstance(tracking, Mapping):
        fail.append("tracking evidence is missing")
    elif bool(tracking.get("ambiguous", True)):
        ambiguous.append("ROI tracking confidence is low")

    regressions: Dict[str, float] = {}
    if previous_protocol is not None:
        previous_lpips = protocol_roi_lpips(previous_protocol)
        current_lpips = protocol_roi_lpips(protocol)
        for name in sorted(CRITICAL_ROIS):
            if name not in previous_lpips or name not in current_lpips:
                fail.append(f"cannot compare critical ROI LPIPS for {name}")
                continue
            regression = current_lpips[name] - previous_lpips[name]
            regressions[name] = regression
            if regression > 0.02:
                fail.append(f"critical ROI LPIPS regression >0.02 for {name}: {regression:.6f}")
            elif regression > 0.01:
                ambiguous.append(f"critical ROI LPIPS regression 0.01-0.02 for {name}: {regression:.6f}")
    if visual_pass is None:
        ambiguous.append("manual contact-sheet visual gate is pending")
    elif visual_pass is False:
        fail.append("manual contact-sheet visual gate failed")

    if fail:
        return GateDecision("fail", tuple(fail + ambiguous), regressions)
    if ambiguous:
        return GateDecision("ambiguous", tuple(ambiguous), regressions)
    return GateDecision("pass", (), regressions)


def scratch_parity_gate(transfer: Metrics, scratch: Metrics, critical_roi_gap: float, artifacts: bool) -> GateDecision:
    reasons = []
    if transfer.psnr < scratch.psnr - 0.20:
        reasons.append("transfer PSNR is >0.20 dB below scratch")
    if transfer.ssim < scratch.ssim - 0.010:
        reasons.append("transfer SSIM is >0.010 below scratch")
    if transfer.lpips > scratch.lpips + 0.015:
        reasons.append("transfer LPIPS is >0.015 above scratch")
    if critical_roi_gap > 0.020:
        reasons.append("critical ROI LPIPS gap is >0.020")
    if artifacts:
        reasons.append("transfer or scratch contains a serious artifact")
    return GateDecision("fail" if reasons else "pass", tuple(reasons), {})


def leader_metric_gate(candidate: Metrics, leader: Metrics) -> GateDecision:
    """Require every promoted frame to remain within the declared leader envelope."""

    reasons = []
    if candidate.psnr < leader.psnr - LEADER_PSNR_TOLERANCE_DB:
        reasons.append("candidate PSNR is >0.20 dB below the canonical leader")
    if candidate.ssim < leader.ssim - LEADER_SSIM_TOLERANCE:
        reasons.append("candidate SSIM is >0.010 below the canonical leader")
    if candidate.lpips > leader.lpips + LEADER_LPIPS_TOLERANCE:
        reasons.append("candidate LPIPS is >0.015 above the canonical leader")
    return GateDecision("fail" if reasons else "pass", tuple(reasons), {})


def combine_gates(*decisions: GateDecision) -> GateDecision:
    """Combine independent fail-closed gates without losing ROI diagnostics."""

    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    regressions: Dict[str, float] = {}
    for decision in decisions:
        regressions.update(decision.critical_roi_regressions)
    if any(decision.outcome == "fail" for decision in decisions):
        outcome: Literal["pass", "ambiguous", "fail"] = "fail"
    elif any(decision.outcome == "ambiguous" for decision in decisions):
        outcome = "ambiguous"
    else:
        outcome = "pass"
    return GateDecision(outcome, reasons, regressions)


def require_accepted_parent(frames: Mapping[str, Any], frame: str) -> Mapping[str, Any]:
    parent = frames.get(frame)
    if (
        not isinstance(parent, Mapping)
        or parent.get("status") != "accepted"
        or parent.get("promotion_complete") is not True
    ):
        raise QualityStop(f"Rejected/missing parent {frame} cannot be forwarded")
    checkpoint = parent.get("selected_checkpoint")
    if not checkpoint or not Path(str(checkpoint)).is_file():
        raise InfrastructureError(f"Accepted parent {frame} lost its selected checkpoint")
    return parent


def seed_repeat_required(winner: Metrics, runner_up: Metrics) -> bool:
    return (
        abs(winner.psnr - runner_up.psnr) <= 0.06
        and abs(winner.ssim - runner_up.ssim) <= 0.01
        and abs(winner.lpips - runner_up.lpips) <= 0.005
    )


class CampaignStore:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.lock = threading.Lock()
        if path.exists():
            if not resume:
                raise InfrastructureError(f"Campaign already exists; pass --resume: {path}")
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "schema_version": 1,
                "created_at": utc_now(),
                "status": "initialized",
                "runs": {},
                "frames": {},
                "accepted_frames": [],
            }
            self.flush()

    def flush(self) -> None:
        with self.lock:
            self.data["updated_at"] = utc_now()
            atomic_json(self.path, self.data)


def run_environment(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    venv_bin = args.venv / "bin"
    env["PATH"] = f"{venv_bin}:/usr/local/cuda-12.6/bin:{env.get('PATH', '')}"
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
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if branch != "main":
        raise InfrastructureError(f"Temporal campaign requires branch main, got {branch!r}")
    if status.strip():
        raise InfrastructureError("Temporal campaign requires an initially clean worktree")
    tracked = (
        REPO_ROOT / "nerfstudio" / "engine" / "trainer.py",
        REPO_ROOT / "nerfstudio" / "engine" / "optimizers.py",
        REPO_ROOT / "nerfstudio" / "models" / "lookcloser.py",
        REPO_ROOT / "nerfstudio" / "pipelines" / "lookcloser_pipeline.py",
        REPO_ROOT / "nerfstudio" / "lookcloser_pixel_sampler.py",
        REPO_ROOT / "nerfstudio" / "configs" / "method_configs.py",
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("temporal_roi_protocol.py"),
        Path(__file__).resolve().with_name("detect_structural_artifacts.py"),
    )
    source_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in tracked}
    payload = json.dumps({"commit": commit, "source": source_hashes}, sort_keys=True).encode()
    return {
        "branch": branch,
        "commit": commit,
        "source_sha256": source_hashes,
        "source_fingerprint": hashlib.sha256(payload).hexdigest(),
    }


def dataset_preflight(data_root: Path) -> Dict[str, Any]:
    actual = tuple(sorted(path.name for path in data_root.iterdir() if path.is_dir()))
    if actual != FRAME_NAMES:
        raise InfrastructureError(f"Expected exactly the 45 stride-7 datasets, got {actual}")
    result: Dict[str, Any] = {}
    for frame in FRAME_NAMES:
        dataset = data_root / frame
        transforms = dataset / "transforms.json"
        if sha256_file(transforms) != EXPECTED_TRANSFORMS_SHA256:
            raise InfrastructureError(f"transforms SHA mismatch for {frame}")
        payload = json.loads(transforms.read_text(encoding="utf-8"))
        paths = [Path(row["file_path"]).name for row in payload.get("frames", [])]
        train = [name for name in paths if "_train_" in name]
        evaluate = [name for name in paths if "_eval_" in name]
        if len(train) != 66 or len(evaluate) != 3:
            raise InfrastructureError(f"Dataset {frame} split is {len(train)}+{len(evaluate)}, expected 66+3")
        maps = list((dataset / "lookcloser_frequencies").glob("*.pt"))
        if len(maps) != 66:
            raise InfrastructureError(f"Dataset {frame} has {len(maps)} frequency maps, expected 66")
        result[frame] = {
            "transforms_sha256": EXPECTED_TRANSFORMS_SHA256,
            "train_images": 66,
            "eval_images": 3,
            "frequency_maps": 66,
        }
    return result


def runtime_preflight(args: argparse.Namespace, env: Mapping[str, str]) -> Dict[str, Any]:
    python = args.venv / "bin" / "python"
    if not python.is_file():
        raise InfrastructureError(f"Required venv Python is missing: {python}")
    code = (
        "import json,platform,torch,tinycudann.modules;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),"
        "'binding':tinycudann.modules._C.__file__}))"
    )
    runtime = json.loads(command_output([str(python), "-c", code], env=env))
    expected = {
        "python": EXPECTED_PYTHON,
        "torch": EXPECTED_TORCH,
        "cuda": EXPECTED_TORCH_CUDA,
        "gpu": EXPECTED_GPU,
    }
    mismatch = {key: (runtime.get(key), value) for key, value in expected.items() if runtime.get(key) != value}
    if mismatch:
        raise InfrastructureError(f"Canonical runtime mismatch: {mismatch}")
    binding = Path(runtime["binding"])
    binding_hash = sha256_file(binding)
    if binding_hash != EXPECTED_TCNN_BINDING_SHA256:
        raise InfrastructureError(f"TCNN binding SHA mismatch: {binding_hash}")
    runtime["binding_sha256"] = binding_hash
    return runtime


def disk_guard(output_dir: Path, forecast_bytes: int = 0) -> Dict[str, int]:
    anchor = output_dir if output_dir.exists() else output_dir.parent
    anchor.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(anchor)
    if forecast_bytes < 0:
        raise ValueError("forecast_bytes must be non-negative")
    if usage.free - forecast_bytes < MIN_DISK_FREE_BYTES:
        raise InfrastructureError(
            f"Refusing next frame: {usage.free / 1024**3:.1f} GiB free minus "
            f"{forecast_bytes / 1024**3:.1f} GiB forecast would leave less than 100 GiB"
        )
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "forecast": forecast_bytes,
        "projected_free": usage.free - forecast_bytes,
    }


def frame_storage_forecast(
    args: argparse.Namespace,
    *,
    include_control: bool,
    completed_phase_checkpoints: int = 0,
) -> int:
    if completed_phase_checkpoints < 0:
        raise ValueError("completed_phase_checkpoints must be non-negative")
    checkpoint_bytes = args.leader_checkpoint.stat().st_size
    phase_checkpoints = max(
        args.max_phase_a_intervals + args.max_tail_intervals + 2 - completed_phase_checkpoints,
        0,
    )
    control_checkpoints = 9 if include_control else 0
    render_and_protocol_reserve = 8 * 1024**3
    return checkpoint_bytes * (phase_checkpoints + control_checkpoints) + render_and_protocol_reserve


def completed_boundaries(runs: Mapping[str, Any], run_ids: Iterable[str]) -> int:
    """Count already materialized eval/save boundaries without hashing large checkpoints."""

    completed = 0
    for run_id in run_ids:
        run = runs.get(run_id)
        if not isinstance(run, Mapping) or run.get("status") != "complete":
            continue
        steps = {
            int(row["local_step"])
            for row in run.get("scheduled_metrics", [])
            if isinstance(row, Mapping) and row.get("local_step") is not None
        }
        completed += len(steps)
    return completed


def vram_guard(jobs: int, env: Mapping[str, str]) -> Dict[str, Any]:
    required = VRAM_PER_JOB_MIB * jobs + VRAM_RESERVE_MIB
    output = command_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"], env=env
    )
    devices = []
    for line in output.splitlines():
        index, name, free = [part.strip() for part in line.split(",", 2)]
        devices.append({"index": int(index), "name": name, "free_mib": int(free)})
    best = max((row["free_mib"] for row in devices), default=0)
    if best < required:
        raise InfrastructureError(f"LR screen needs {required} MiB free VRAM on one GPU, found {best} MiB")
    return {"jobs": jobs, "required_mib": required, "devices": devices}


def full_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    if sha256_file(args.leader_checkpoint) != EXPECTED_LEADER_SHA256:
        raise InfrastructureError("Canonical leader checkpoint SHA-256 mismatch")
    if sha256_file(args.leader_config) != EXPECTED_CONFIG_SHA256:
        raise InfrastructureError("Canonical leader config SHA-256 mismatch")
    env = run_environment(args)
    return {
        "git": git_preflight(),
        "leader": {
            "checkpoint": str(args.leader_checkpoint),
            "checkpoint_sha256": EXPECTED_LEADER_SHA256,
            "raw_parent_step": checkpoint_step(args.leader_checkpoint),
            "exposure": "legacy_estimate; canonical checkpoint has no exact cumulative counter",
            "config": str(args.leader_config),
            "config_sha256": EXPECTED_CONFIG_SHA256,
        },
        "runtime": runtime_preflight(args, env),
        "datasets": dataset_preflight(args.data_root),
        "disk": disk_guard(args.output_dir),
    }


def _set_dataset(config: Any, dataset: Path) -> None:
    config.data = None
    config.pipeline.datamanager.data = None
    config.pipeline.datamanager.dataparser.data = dataset


def _freeze_recipe(config: Any, spec: RunSpec) -> None:
    """Apply the reviewed B4096 leader recipe and explicitly disable speed variants."""

    config.machine.seed = spec.seed
    config.mixed_precision = True
    config.save_only_latest_checkpoint = False
    config.steps_per_eval_batch = INTERVAL
    config.steps_per_eval_image = INTERVAL
    config.steps_per_eval_all_images = INTERVAL
    config.steps_per_save = INTERVAL
    config.max_num_iterations = spec.target_local_step + 1
    config.logging.csv_writer.enable = True
    config.logging.csv_writer.write_interval = INTERVAL
    config.logging.csv_writer.improvement_tolerance = 0.0
    config.logging.local_writer.enable = False
    config.logging.profiler = "none"
    config.viewer.quit_on_train_completion = True
    config.pipeline.datamanager.train_num_rays_per_batch = 4096
    config.pipeline.datamanager.eval_num_rays_per_batch = 4096
    config.pipeline.datamanager.cache_train_rays = False
    config.pipeline.datamanager.cpu_fas_prefetch = False
    sampler = config.pipeline.datamanager.pixel_sampler
    sampler.enable_fas = True
    sampler.fas_strength = spec.fas_strength
    sampler.sampling_ramp_start = 1.0
    sampler.sampling_ramp_end = 3.0
    sampler.fas_warmup_steps = 0
    sampler.fas_ramp_steps = 0
    model = config.pipeline.model
    model.log2_hashmap_size = 23
    model.max_res = 8192.0
    model.reconstruction_loss_type = "charbonnier"
    model.distortion_loss_mult = 0.01
    model.depth_loss_mult = 0.001
    model.depth_loss_steps = 5000
    model.enable_adaptive_ray_marching = True
    model.ray_sampling_mode = "adaptive"
    model.max_steps_per_ray = 1024
    model.adaptive_coarse_step_size = 0.00625
    if spec.traversal_warmup_steps < 0:
        raise ValueError("traversal_warmup_steps must be non-negative")
    model.adaptive_warmup_steps = spec.traversal_warmup_steps
    model.occupancy_warmup_steps = spec.traversal_warmup_steps
    model.occupancy_binary_warmup_steps = spec.traversal_warmup_steps
    model.stable_occupancy_reduction = True
    model.feature_reweighting_strength = spec.feature_reweighting
    model.corrected_arm_allocator = False
    model.tcnn_network_jit = False
    config.pipeline.independent_rng_streams = False
    config.pipeline.target_num_samples_per_batch = 0
    config.fused_adam_switch_step = None
    config.replay_eval_trajectory = False


def configured_run(args: argparse.Namespace, spec: RunSpec) -> Tuple[Any, Path, Path]:
    config = yaml.load(args.leader_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    run_dir = args.output_dir / spec.frame / "lookcloser" / spec.run_id
    config.output_dir = args.output_dir
    config.experiment_name = spec.frame
    config.timestamp = spec.run_id
    _set_dataset(config, args.data_root / spec.frame)
    _freeze_recipe(config, spec)

    config.load_dir = None
    config.load_step = None
    config.load_config = None
    config.load_checkpoint = spec.parent_checkpoint
    config.checkpoint_load_mode = spec.load_mode
    config.load_optimizers = True
    config.load_scheduler = True
    config.resume_fields_lr_override = spec.lr_override
    config.checkpoint_load_parameter_hash_audit = spec.phase == "gpu_smoke"
    config.optimizers["fields"]["optimizer"].lr = spec.lr
    scheduler = config.optimizers["fields"]["scheduler"]
    if spec.scheduler_policy == "constant":
        scheduler.lr_final = spec.lr
        scheduler.warmup_steps = 0
        scheduler.max_steps = max(200_000, spec.target_local_step + 1)
    elif spec.scheduler_policy != "leader":
        raise ValueError(f"Unknown scheduler policy: {spec.scheduler_policy}")

    input_config = args.campaign.parent / "configs" / f"{spec.run_id}.yml"
    return config, input_config, run_dir


def training_command(args: argparse.Namespace, input_config: Path) -> List[str]:
    code = (
        "import yaml;from pathlib import Path;from nerfstudio.scripts.train import main;"
        f"cfg=yaml.load(Path({str(input_config)!r}).read_text(),Loader=yaml.Loader);main(cfg)"
    )
    return [str(args.venv / "bin" / "python"), "-c", code]


def lr_screen_specs(args: argparse.Namespace) -> List[RunSpec]:
    parent_step = checkpoint_step(args.leader_checkpoint)
    return [
        RunSpec(
            run_id=(
                f"007747_lr{lr:.0e}_warm{warmup}_seed{args.seed}_"
                f"phase_a_s{SCREEN_FINAL_STEP}"
            ),
            frame="007747",
            seed=args.seed,
            lr=lr,
            phase="phase_a_lr_screen",
            feature_reweighting=1.0,
            fas_strength=1.0,
            load_mode="model_parameters_only",
            parent_checkpoint=args.leader_checkpoint,
            target_local_step=SCREEN_FINAL_STEP,
            inherited_global_step=parent_step,
            contended=args.max_parallel > 1,
            traversal_warmup_steps=warmup,
        )
        for warmup in args.traversal_warmup_candidates
        for lr in args.lr_candidates
    ]


def deterministic_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = []
    for spec in lr_screen_specs(args):
        _, input_config, run_dir = configured_run(args, spec)
        rows.append(
            {
                "run": asdict(spec) | {"parent_checkpoint": str(spec.parent_checkpoint)},
                "input_config": str(input_config),
                "run_dir": str(run_dir),
                "command": training_command(args, input_config),
            }
        )
    return {"schema_version": 1, "lr_screen": rows}


def read_metrics(path: Path, *, run_id: Optional[str] = None) -> List[Metrics]:
    if not path.is_file():
        return []
    result = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("eval_all_psnr"):
                continue
            required = ("eval_all_psnr", "eval_all_ssim", "eval_all_lpips")
            if any(not row.get(key) for key in required):
                raise InfrastructureError(f"Incomplete all-view metrics row in {path}: {row}")
            result.append(
                Metrics(
                    local_step=int(row["step"]),
                    psnr=float(row["eval_all_psnr"]),
                    ssim=float(row["eval_all_ssim"]),
                    lpips=float(row["eval_all_lpips"]),
                    point_samples=(
                        int(float(row["cumulative_point_samples"]))
                        if row.get("cumulative_point_samples")
                        else None
                    ),
                    run_id=run_id,
                )
            )
    return result


def _print_new_metrics(prefix: str, metrics_path: Path, printed_steps: set[int]) -> None:
    for row in read_metrics(metrics_path):
        if row.local_step in printed_steps:
            continue
        printed_steps.add(row.local_step)
        print(
            f"{prefix} step={row.local_step} psnr={row.psnr:.6f} "
            f"ssim={row.ssim:.6f} lpips={row.lpips:.6f}",
            flush=True,
        )


def checkpoint_cumulative_points(path: Path) -> Optional[int]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    value = checkpoint.get("pipeline", {}).get("cumulative_point_samples")
    if value is None:
        return None
    return int(value.item())


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def finalize_run_record(
    store: CampaignStore,
    spec: RunSpec,
    record: Dict[str, Any],
    run_dir: Path,
    target_checkpoint: Path,
) -> Dict[str, Any]:
    metrics = read_metrics(run_dir / "metrics_compact.csv", run_id=spec.run_id)
    boundary = next((row for row in metrics if row.local_step == spec.target_local_step), None)
    if spec.target_local_step != 0 and boundary is None and not spec.phase.startswith("gpu_smoke"):
        raise InfrastructureError(f"Run {spec.run_id} has no matched eval boundary {spec.target_local_step}")
    record.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "checkpoint": str(target_checkpoint),
            "checkpoint_sha256": sha256_file(target_checkpoint),
            "local_step": spec.target_local_step,
            "local_updates_completed": spec.local_updates_completed,
            "effective_global_step": spec.effective_global_step,
            "frame_cumulative_point_samples": checkpoint_cumulative_points(target_checkpoint),
            "scheduled_metrics": [asdict(row) | {"checkpoint": None} for row in metrics],
        }
    )
    store.data["runs"][spec.run_id] = record
    store.flush()
    return record


def existing_run_spec_mismatches(
    existing: Mapping[str, Any], spec: RunSpec
) -> Dict[str, Dict[str, Any]]:
    expected = asdict(spec)
    expected["parent_checkpoint"] = (
        str(spec.parent_checkpoint) if spec.parent_checkpoint is not None else None
    )
    return {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }


def validate_existing_run_spec(existing: Mapping[str, Any], spec: RunSpec) -> None:
    """Fail closed when a reused run ID describes a different experiment."""

    mismatches = existing_run_spec_mismatches(existing, spec)
    if mismatches:
        raise InfrastructureError(
            f"Run ID collision for {spec.run_id}; stored RunSpec differs: {mismatches}"
        )


def run_training(args: argparse.Namespace, store: CampaignStore, spec: RunSpec) -> Dict[str, Any]:
    existing = store.data["runs"].get(spec.run_id)
    _, input_config, run_dir = configured_run(args, spec)
    target_checkpoint = checkpoint_path(run_dir, spec.target_local_step)
    if isinstance(existing, Mapping):
        validate_existing_run_spec(existing, spec)
    if isinstance(existing, Mapping) and existing.get("status") == "complete":
        if not target_checkpoint.is_file() or sha256_file(target_checkpoint) != existing.get("checkpoint_sha256"):
            raise InfrastructureError(f"Completed manifest run changed or lost its checkpoint: {spec.run_id}")
        print(f"resume run={spec.run_id} status=complete", flush=True)
        return dict(existing)
    if isinstance(existing, Mapping):
        if existing.get("status") == "running" and pid_alive(existing.get("pid")):
            raise InfrastructureError(
                f"Run {spec.run_id} is still active as PID {existing.get('pid')}; do not launch a duplicate"
            )
        if target_checkpoint.is_file():
            print(f"resume run={spec.run_id} status=finalizing_existing_checkpoint", flush=True)
            return finalize_run_record(store, spec, dict(existing), run_dir, target_checkpoint)

    disk_guard(args.output_dir)
    config, input_config, run_dir = configured_run(args, spec)
    input_config.parent.mkdir(parents=True, exist_ok=True)
    input_config.write_text(yaml.dump(config), encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    command = training_command(args, input_config)
    record: Dict[str, Any] = {
        **asdict(spec),
        "parent_checkpoint": str(spec.parent_checkpoint) if spec.parent_checkpoint else None,
        "parent_sha256": sha256_file(spec.parent_checkpoint) if spec.parent_checkpoint else None,
        "run_dir": str(run_dir),
        "input_config": str(input_config),
        "command": command,
        "status": "running",
        "started_at": utc_now(),
        "wall_time_label": "contended" if spec.contended else "solo",
        "optimizer_policy": {
            "cross_frame": "fresh Adam/scheduler/scaler/RNG",
            "same_frame": "full resume",
            "lr_override": spec.lr_override,
        },
        "reset_assertions": {
            "local_step_zero": spec.load_mode == "model_parameters_only",
            "occupancy_occs_zero": spec.load_mode == "model_parameters_only",
            "frequency_grid_zero": spec.load_mode == "model_parameters_only",
            "fas_sample_count_zero": spec.load_mode == "model_parameters_only",
            "frame_cumulative_points_zero": spec.load_mode == "model_parameters_only",
            "fixed_traversal_updates": (
                spec.traversal_warmup_steps if spec.load_mode == "model_parameters_only" else None
            ),
        },
    }
    store.data["runs"][spec.run_id] = record
    store.flush()

    start = time.monotonic()
    train_log = run_dir / "train_stdout.log"
    metrics_path = run_dir / "metrics_compact.csv"
    printed_steps: set[int] = set()
    with train_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        record["pid"] = process.pid
        store.flush()
        while process.poll() is None:
            _print_new_metrics(spec.run_id, metrics_path, printed_steps)
            time.sleep(max(args.poll_seconds, 0.1))
        returncode = int(process.returncode)
    _print_new_metrics(spec.run_id, metrics_path, printed_steps)
    record["returncode"] = returncode
    record["train_seconds"] = time.monotonic() - start
    if returncode != 0:
        record["status"] = "infrastructure_error"
        store.flush()
        raise InfrastructureError(f"Training run {spec.run_id} exited {returncode}; see {train_log}")
    if not target_checkpoint.is_file():
        record["status"] = "infrastructure_error"
        store.flush()
        raise InfrastructureError(f"Run {spec.run_id} did not write {target_checkpoint}")
    return finalize_run_record(store, spec, record, run_dir, target_checkpoint)


def run_lr_screen(args: argparse.Namespace, store: CampaignStore) -> Tuple[RunSpec, List[Metrics]]:
    specs = lr_screen_specs(args)
    vram_guard(min(args.max_parallel, len(specs)), run_environment(args))
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel, thread_name_prefix="temporal-lr") as executor:
        futures = {executor.submit(run_training, args, store, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            results[spec.run_id] = future.result()
    matched = []
    for spec in specs:
        run = results[spec.run_id]
        rows = read_metrics(Path(run["run_dir"]) / "metrics_compact.csv", run_id=spec.run_id)
        row = next((candidate for candidate in rows if candidate.local_step == SCREEN_FINAL_STEP), None)
        if row is None:
            raise InfrastructureError(f"LR candidate {spec.lr} lacks matched boundary {SCREEN_FINAL_STEP}")
        matched.append(
            Metrics(**{**asdict(row), "checkpoint": Path(run["checkpoint"])})
        )
    selected = select_metrics(matched)
    winner = next(spec for spec in specs if spec.run_id == selected.run_id)
    store.data["lr_screen"] = {
        "matched_local_step": SCREEN_FINAL_STEP,
        "candidates": [
            asdict(row)
            | {
                "checkpoint": str(row.checkpoint),
                "lr": spec.lr,
                "traversal_warmup_steps": spec.traversal_warmup_steps,
            }
            for row, spec in zip(matched, specs)
        ],
        "winner_run_id": winner.run_id,
        "winner_lr": winner.lr,
        "winner_traversal_warmup_steps": winner.traversal_warmup_steps,
        "repeat_required": seed_repeat_required(
            selected,
            select_metrics([row for row in matched if row.run_id != selected.run_id]),
        ),
    }
    store.flush()
    return winner, matched


def eval_config_for_checkpoint(run_config: Path, checkpoint: Path, output: Path) -> Path:
    config = yaml.load(run_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
    config.load_dir = checkpoint.parent
    config.load_step = checkpoint_step(checkpoint)
    config.load_checkpoint = checkpoint
    config.pipeline.model.eval_num_rays_per_chunk = 2048
    config.pipeline.datamanager.cache_train_rays = False
    config.pipeline.datamanager.cpu_fas_prefetch = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.dump(config), encoding="utf-8")
    return output


def fresh_eval_and_protocol(
    args: argparse.Namespace,
    *,
    frame: str,
    run: Mapping[str, Any],
    checkpoint: Path,
    previous_dataset: Optional[Path],
    previous_protocol: Optional[Path],
) -> Dict[str, Any]:
    evaluation_dir = Path(run["run_dir"]) / "evaluations" / checkpoint.stem
    result_path = evaluation_dir / "eval.json"
    render_dir = evaluation_dir / "renders"
    protocol_path = evaluation_dir / "roi" / "temporal_roi_protocol.json"
    if result_path.is_file() and protocol_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (
            Path(result.get("checkpoint", "")).resolve() == checkpoint.resolve()
            and protocol.get("status") == "complete"
        ):
            return {"eval": result, "protocol": protocol, "protocol_path": str(protocol_path)}

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    eval_config = eval_config_for_checkpoint(
        Path(run["run_dir"]) / "config.yml", checkpoint, evaluation_dir / "eval_config.yml"
    )
    ns_eval = args.venv / "bin" / "ns-eval"
    command = [
        str(ns_eval),
        "--load-config",
        str(eval_config),
        "--output-path",
        str(result_path),
        "--render-output-path",
        str(render_dir),
    ]
    with (evaluation_dir / "eval_stdout.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=run_environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not result_path.is_file():
        raise InfrastructureError(
            f"Fresh eval failed for {frame} {checkpoint}; see {evaluation_dir / 'eval_stdout.log'}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if Path(result.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise InfrastructureError("Fresh eval JSON names a different checkpoint")
    renders = sorted(render_dir.glob("eval_img_*.png"))
    if len(renders) != 3:
        raise InfrastructureError(f"Fresh eval wrote {len(renders)} renders, expected exactly three")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from temporal_roi_protocol import build_protocol

    protocol_args = SimpleNamespace(
        frame=frame,
        dataset=args.data_root / frame,
        render_dir=render_dir,
        out_dir=evaluation_dir / "roi",
        previous_dataset=previous_dataset,
        previous_protocol=previous_protocol,
        tracking_confidence_min=0.60,
        thumbnail_width=256,
    )
    protocol = build_protocol(protocol_args)
    print(
        f"eval frame={frame} step={checkpoint_step(checkpoint)} "
        f"psnr={float(result['results']['psnr']):.6f} "
        f"ssim={float(result['results']['ssim']):.6f} "
        f"lpips={float(result['results']['lpips']):.6f} "
        f"artifacts={protocol['full_view_serious_count']}/{protocol['roi_serious_count']}",
        flush=True,
    )
    return {"eval": result, "protocol": protocol, "protocol_path": str(protocol_path)}


def ensure_leader_baseline(args: argparse.Namespace, store: CampaignStore) -> Mapping[str, Any]:
    """Fresh-evaluate the immutable 007740 parent for the first ROI-regression gate."""

    existing = store.data.get("leader_baseline")
    if isinstance(existing, Mapping):
        protocol_path = Path(str(existing.get("protocol", "")))
        if (
            protocol_path.is_file()
            and existing.get("checkpoint_sha256") == EXPECTED_LEADER_SHA256
            and isinstance(existing.get("metrics"), Mapping)
        ):
            return existing
    run_dir = args.output_dir / "007740" / "lookcloser" / "canonical_parent_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.leader_config, run_dir / "config.yml")
    evaluated = fresh_eval_and_protocol(
        args,
        frame="007740",
        run={"run_dir": str(run_dir)},
        checkpoint=args.leader_checkpoint,
        previous_dataset=None,
        previous_protocol=None,
    )
    baseline = {
        "checkpoint": str(args.leader_checkpoint),
        "checkpoint_sha256": EXPECTED_LEADER_SHA256,
        "eval": evaluated["eval"],
        "metrics": {
            name: float(evaluated["eval"]["results"][name])
            for name in ("psnr", "ssim", "lpips")
        },
        "protocol": evaluated["protocol_path"],
        "exposure": "legacy_estimate",
    }
    store.data["leader_baseline"] = baseline
    store.flush()
    return baseline


def previous_protocol_path(store: CampaignStore, frame: str) -> Optional[Path]:
    if frame == "007747":
        baseline = store.data.get("leader_baseline")
        if not isinstance(baseline, Mapping) or not baseline.get("protocol"):
            return None
        return Path(str(baseline["protocol"]))
    previous_name = f"{int(frame) - 7:06d}"
    previous = store.data["frames"].get(previous_name)
    if isinstance(previous, Mapping) and previous.get("selected_protocol"):
        return Path(str(previous["selected_protocol"]))
    return None


def _metrics_with_checkpoints(store: CampaignStore, run_ids: Iterable[str]) -> List[Metrics]:
    result: List[Metrics] = []
    for run_id in run_ids:
        run = store.data["runs"][run_id]
        model_dir = Path(run["run_dir"]) / "nerfstudio_models"
        for row in read_metrics(Path(run["run_dir"]) / "metrics_compact.csv", run_id=run_id):
            checkpoint = checkpoint_path(Path(run["run_dir"]), row.local_step)
            if checkpoint.is_file():
                result.append(Metrics(**{**asdict(row), "checkpoint": checkpoint}))
    # Same local boundary can appear as a parent/final save in adjacent run dirs.
    unique: Dict[Tuple[int, str], Metrics] = {}
    for row in result:
        assert row.checkpoint is not None
        unique[(row.local_step, sha256_file(row.checkpoint))] = row
    return sorted(unique.values(), key=lambda row: row.local_step)


def _boundary_evidence(
    args: argparse.Namespace,
    store: CampaignStore,
    frame: str,
    candidates: Sequence[Metrics],
    previous_dataset: Optional[Path],
    previous_protocol: Optional[Path],
) -> List[BoundaryEvidence]:
    evidence = []
    for metrics in candidates:
        assert metrics.run_id is not None and metrics.checkpoint is not None
        run = store.data["runs"][metrics.run_id]
        evaluated = fresh_eval_and_protocol(
            args,
            frame=frame,
            run=run,
            checkpoint=metrics.checkpoint,
            previous_dataset=previous_dataset,
            previous_protocol=previous_protocol,
        )
        protocol = evaluated["protocol"]
        evidence.append(
            BoundaryEvidence(metrics, protocol_roi_lpips(protocol), protocol_artifact_score(protocol))
        )
    return evidence


def train_frame_recipe(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    parent_checkpoint: Path,
    parent_effective_step: int,
    lr: float,
    seed: int,
    prefix: str,
    initial_run_id: Optional[str] = None,
    traversal_warmup_steps: int = 4096,
) -> Tuple[Metrics, List[str]]:
    run_ids: List[str] = []
    if initial_run_id is None:
        spec = RunSpec(
            run_id=f"{prefix}_{frame}_seed{seed}_phase_a_s{SCREEN_FINAL_STEP}",
            frame=frame,
            seed=seed,
            lr=lr,
            phase="phase_a",
            feature_reweighting=1.0,
            fas_strength=1.0,
            load_mode="model_parameters_only",
            parent_checkpoint=parent_checkpoint,
            target_local_step=SCREEN_FINAL_STEP,
            inherited_global_step=parent_effective_step,
            traversal_warmup_steps=traversal_warmup_steps,
        )
        run_training(args, store, spec)
        run_ids.append(spec.run_id)
    else:
        run_ids.append(initial_run_id)

    phase_a_intervals = SCREEN_FINAL_STEP // INTERVAL
    previous_dataset = args.data_root / f"{int(frame) - 7:06d}" if frame != "007747" else None
    previous_protocol = None if frame == "007747" else previous_protocol_path(store, frame)
    while True:
        candidates = _metrics_with_checkpoints(store, run_ids)
        phase_a = [row for row in candidates if row.local_step <= max(item.local_step for item in candidates)]
        evidence = _boundary_evidence(
            args, store, frame, phase_a[-3:], previous_dataset, previous_protocol
        )
        if phase_a[-1].local_step >= SCREEN_FINAL_STEP and plateau_confirmed(evidence):
            # Plateau decides when Phase A stops; it does not decide which checkpoint
            # is allowed to seed the tail.  Use the canonical PSNR/LPIPS selector
            # across Phase A so a late plateau drift cannot poison the FR0.3 tail.
            phase_a_parent = select_metrics(phase_a)
            break
        phase_a_intervals += 1
        if phase_a_intervals > args.max_phase_a_intervals:
            raise QualityStop(f"Frame {frame} did not confirm phase-A plateau")
        parent = phase_a[-1]
        assert parent.checkpoint is not None
        target = parent.local_step + INTERVAL
        spec = RunSpec(
            run_id=f"{prefix}_{frame}_seed{seed}_phase_a_s{target}",
            frame=frame,
            seed=seed,
            lr=lr,
            phase="phase_a_extension",
            feature_reweighting=1.0,
            fas_strength=1.0,
            load_mode="resume",
            parent_checkpoint=parent.checkpoint,
            target_local_step=target,
            inherited_global_step=parent_effective_step,
            traversal_warmup_steps=traversal_warmup_steps,
        )
        run_training(args, store, spec)
        run_ids.append(spec.run_id)

    assert phase_a_parent.checkpoint is not None
    tail_lr = lr / 4.0
    tail_target = phase_a_parent.local_step + 2 * INTERVAL
    tail_spec = RunSpec(
        run_id=f"{prefix}_{frame}_seed{seed}_tail_s{tail_target}",
        frame=frame,
        seed=seed,
        lr=tail_lr,
        phase="tail",
        feature_reweighting=0.3,
        fas_strength=1.0,
        load_mode="resume",
        parent_checkpoint=phase_a_parent.checkpoint,
        target_local_step=tail_target,
        inherited_global_step=parent_effective_step,
        lr_override=tail_lr,
        traversal_warmup_steps=traversal_warmup_steps,
    )
    run_training(args, store, tail_spec)
    run_ids.append(tail_spec.run_id)
    tail_intervals = 2
    while True:
        candidates = _metrics_with_checkpoints(store, run_ids)
        tail = [row for row in candidates if row.local_step >= phase_a_parent.local_step]
        evidence = _boundary_evidence(args, store, frame, tail[-3:], previous_dataset, previous_protocol)
        if tail_intervals >= 2 and plateau_confirmed(evidence):
            break
        tail_intervals += 1
        if tail_intervals > args.max_tail_intervals:
            raise QualityStop(f"Frame {frame} did not confirm tail plateau")
        parent = tail[-1]
        assert parent.checkpoint is not None
        target = parent.local_step + INTERVAL
        spec = RunSpec(
            run_id=(
                f"{prefix}_{frame}_seed{seed}_"
                f"tail_from{phase_a_parent.local_step}_s{target}"
            ),
            frame=frame,
            seed=seed,
            lr=tail_lr,
            phase="tail_extension",
            feature_reweighting=0.3,
            fas_strength=1.0,
            load_mode="resume",
            parent_checkpoint=parent.checkpoint,
            target_local_step=target,
            inherited_global_step=parent_effective_step,
            traversal_warmup_steps=traversal_warmup_steps,
        )
        # Campaigns created before origin-qualified IDs may already contain the
        # exact deterministic extension. Reuse it only when every RunSpec field
        # matches; a same-target run from another parent gets the qualified ID.
        legacy_spec = replace(
            spec,
            run_id=f"{prefix}_{frame}_seed{seed}_tail_s{target}",
        )
        legacy = store.data["runs"].get(legacy_spec.run_id)
        if isinstance(legacy, Mapping) and not existing_run_spec_mismatches(legacy, legacy_spec):
            spec = legacy_spec
        run_training(args, store, spec)
        run_ids.append(spec.run_id)

    selected = select_metrics(_metrics_with_checkpoints(store, run_ids))
    return selected, run_ids


def visual_decisions(path: Optional[Path]) -> Dict[str, bool]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for key, value in payload.items():
        if value not in ("pass", "fail", True, False):
            raise InfrastructureError(f"Invalid visual decision for {key}: {value!r}")
        result[str(key)] = value in ("pass", True)
    return result


def finalize_frame(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    selected: Metrics,
    run_ids: Sequence[str],
    parent_checkpoint: Path,
    parent_effective_step: int,
    lr: float,
    traversal_warmup_steps: int,
    decision_key: str,
) -> GateDecision:
    assert selected.run_id is not None and selected.checkpoint is not None
    previous_name = f"{int(frame) - 7:06d}"
    previous_record = store.data["frames"].get(previous_name)
    prior_protocol_path = previous_protocol_path(store, frame)
    evaluated = fresh_eval_and_protocol(
        args,
        frame=frame,
        run=store.data["runs"][selected.run_id],
        checkpoint=selected.checkpoint,
        previous_dataset=(args.data_root / previous_name if frame != "007747" else None),
        previous_protocol=(None if frame == "007747" else prior_protocol_path),
    )
    protocol = evaluated["protocol"]
    previous_protocol = (
        json.loads(prior_protocol_path.read_text(encoding="utf-8"))
        if prior_protocol_path is not None
        else None
    )
    protocol_gate = quality_gate(
        protocol,
        previous_protocol=previous_protocol,
        visual_pass=visual_decisions(args.visual_decisions).get(decision_key),
    )
    fresh_results = evaluated["eval"].get("results", {})
    candidate_fresh = Metrics(
        selected.local_step,
        float(fresh_results["psnr"]),
        float(fresh_results["ssim"]),
        float(fresh_results["lpips"]),
    )
    leader_record = store.data.get("leader_baseline")
    if not isinstance(leader_record, Mapping) or not isinstance(leader_record.get("metrics"), Mapping):
        raise InfrastructureError("Canonical leader baseline metrics are missing")
    leader_values = leader_record["metrics"]
    leader_fresh = Metrics(
        checkpoint_step(args.leader_checkpoint),
        float(leader_values["psnr"]),
        float(leader_values["ssim"]),
        float(leader_values["lpips"]),
    )
    numeric_gate = leader_metric_gate(candidate_fresh, leader_fresh)
    gate = combine_gates(protocol_gate, numeric_gate)
    checkpoint_hash = sha256_file(selected.checkpoint)
    frame_points = checkpoint_cumulative_points(selected.checkpoint)
    prior_temporal_points = int(previous_record.get("temporal_cumulative_point_samples", 0)) if previous_record else 0
    tracking_margin = float(protocol.get("tracking", {}).get("minimum_confidence", 0.0))
    gate_small_margin = (
        tracking_margin < 0.70
        or any(value > 0.008 for value in gate.critical_roi_regressions.values())
        or protocol_artifact_score(protocol) > 0.0
    )
    frame_record = {
        "frame": frame,
        "status": "accepted" if gate.passed else "rejected",
        "promotion_complete": False,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_sha256": sha256_file(parent_checkpoint),
        "raw_parent_step": checkpoint_step(parent_checkpoint),
        "inherited_global_step": parent_effective_step,
        "selected_checkpoint": str(selected.checkpoint),
        "selected_sha256": checkpoint_hash,
        "selected_local_step": selected.local_step,
        "local_updates_completed": selected.local_step + 1,
        "effective_global_step": parent_effective_step + selected.local_step,
        "lr": lr,
        "traversal_warmup_steps": traversal_warmup_steps,
        "fas": 1.0,
        "phase_a_fr": 1.0,
        "tail_fr": 0.3,
        "metrics": {"psnr": selected.psnr, "ssim": selected.ssim, "lpips": selected.lpips},
        "fresh_metrics": {
            "psnr": candidate_fresh.psnr,
            "ssim": candidate_fresh.ssim,
            "lpips": candidate_fresh.lpips,
        },
        "leader_metric_gate": {
            "leader": {
                "psnr": leader_fresh.psnr,
                "ssim": leader_fresh.ssim,
                "lpips": leader_fresh.lpips,
            },
            "tolerance": {
                "psnr_db": LEADER_PSNR_TOLERANCE_DB,
                "ssim": LEADER_SSIM_TOLERANCE,
                "lpips": LEADER_LPIPS_TOLERANCE,
            },
            "decision": asdict(numeric_gate),
        },
        "frame_cumulative_point_samples": frame_points,
        "temporal_cumulative_point_samples": prior_temporal_points + int(frame_points or 0),
        "runs": list(run_ids),
        "selected_eval": evaluated["eval"],
        "selected_protocol": evaluated["protocol_path"],
        "gate": asdict(gate),
        "gate_small_margin": gate_small_margin,
        "visual_decision_key": decision_key,
    }
    store.data["frames"][frame] = frame_record
    if gate.passed and frame not in store.data["accepted_frames"]:
        store.data["accepted_frames"].append(frame)
    store.flush()
    return gate


def evaluate_control_candidate(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    selected: Metrics,
    decision_key: str,
) -> Tuple[Dict[str, Any], GateDecision]:
    """Fresh-evaluate a control without replacing the transfer frame record."""

    assert selected.run_id is not None and selected.checkpoint is not None
    previous_name = f"{int(frame) - 7:06d}"
    prior_protocol_path = previous_protocol_path(store, frame)
    evaluated = fresh_eval_and_protocol(
        args,
        frame=frame,
        run=store.data["runs"][selected.run_id],
        checkpoint=selected.checkpoint,
        previous_dataset=(args.data_root / previous_name if frame != "007747" else None),
        previous_protocol=(None if frame == "007747" else prior_protocol_path),
    )
    previous_protocol = (
        json.loads(prior_protocol_path.read_text(encoding="utf-8"))
        if prior_protocol_path is not None
        else None
    )
    gate = quality_gate(
        evaluated["protocol"],
        previous_protocol=previous_protocol,
        visual_pass=visual_decisions(args.visual_decisions).get(decision_key),
    )
    return evaluated, gate


def run_scratch_control(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    transfer: Metrics,
) -> GateDecision:
    """Run the faithful seed42 FR1@75940 -> full-resume FR0.3@106316 control solo."""

    phase_a = RunSpec(
        run_id=f"scratch_{frame}_seed42_phase_a_s75940",
        frame=frame,
        seed=42,
        lr=0.01,
        phase="scratch_phase_a",
        feature_reweighting=1.0,
        fas_strength=1.0,
        load_mode="resume",
        parent_checkpoint=None,
        target_local_step=75_940,
        inherited_global_step=0,
        from_scratch=True,
        scheduler_policy="leader",
    )
    first = run_training(args, store, phase_a)
    phase_b = RunSpec(
        run_id=f"scratch_{frame}_seed42_tail_s106316",
        frame=frame,
        seed=42,
        lr=0.01,
        phase="scratch_tail",
        feature_reweighting=0.3,
        fas_strength=1.0,
        load_mode="resume",
        parent_checkpoint=Path(first["checkpoint"]),
        target_local_step=106_316,
        inherited_global_step=0,
        scheduler_policy="leader",
    )
    run_training(args, store, phase_b)
    scratch = select_metrics(_metrics_with_checkpoints(store, (phase_a.run_id, phase_b.run_id)))
    evaluated, quality = evaluate_control_candidate(
        args,
        store,
        frame=frame,
        selected=scratch,
        decision_key=f"{frame}_scratch",
    )
    transfer_protocol = json.loads(
        Path(store.data["frames"][frame]["selected_protocol"]).read_text(encoding="utf-8")
    )
    transfer_lpips = protocol_roi_lpips(transfer_protocol)
    scratch_lpips = protocol_roi_lpips(evaluated["protocol"])
    missing = CRITICAL_ROIS - (set(transfer_lpips) & set(scratch_lpips))
    if missing:
        raise InfrastructureError(f"Scratch parity lacks critical ROIs: {sorted(missing)}")
    critical_gap = max(transfer_lpips[name] - scratch_lpips[name] for name in CRITICAL_ROIS)
    parity = scratch_parity_gate(
        transfer,
        scratch,
        critical_gap,
        bool(
            transfer_protocol.get("full_view_serious_count")
            or transfer_protocol.get("roi_serious_count")
            or evaluated["protocol"].get("full_view_serious_count")
            or evaluated["protocol"].get("roi_serious_count")
        ),
    )
    reasons = tuple(quality.reasons) + tuple(parity.reasons)
    outcome: Literal["pass", "ambiguous", "fail"]
    if quality.outcome == "fail" or parity.outcome == "fail":
        outcome = "fail"
    elif quality.outcome == "ambiguous":
        outcome = "ambiguous"
    else:
        outcome = "pass"
    decision = GateDecision(outcome, reasons, quality.critical_roi_regressions)
    store.data["frames"][frame]["scratch_control"] = {
        "status": "pass" if decision.passed else "rejected",
        "selected_checkpoint": str(scratch.checkpoint),
        "selected_sha256": sha256_file(scratch.checkpoint),  # type: ignore[arg-type]
        "metrics": {"psnr": scratch.psnr, "ssim": scratch.ssim, "lpips": scratch.lpips},
        "critical_roi_lpips_gap": critical_gap,
        "gate": asdict(decision),
        "protocol": evaluated["protocol_path"],
        "wall_time_label": "solo",
    }
    store.flush()
    return decision


def run_seed43_repeat(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    parent_checkpoint: Path,
    parent_effective_step: int,
    lr: float,
    traversal_warmup_steps: int,
    transfer: Metrics,
) -> GateDecision:
    selected, run_ids = train_frame_recipe(
        args,
        store,
        frame="007747",
        parent_checkpoint=parent_checkpoint,
        parent_effective_step=parent_effective_step,
        lr=lr,
        seed=43,
        prefix="repeat",
        traversal_warmup_steps=traversal_warmup_steps,
    )
    evaluated, quality = evaluate_control_candidate(
        args,
        store,
        frame="007747",
        selected=selected,
        decision_key="007747_seed43",
    )
    deltas = {
        "psnr": abs(selected.psnr - transfer.psnr),
        "ssim": abs(selected.ssim - transfer.ssim),
        "lpips": abs(selected.lpips - transfer.lpips),
    }
    envelope_failures = [
        name for name, limit in {"psnr": 0.06, "ssim": 0.01, "lpips": 0.005}.items() if deltas[name] > limit
    ]
    reasons = list(quality.reasons)
    if envelope_failures:
        reasons.append(f"seed43 exceeds repeat envelope: {envelope_failures}")
    outcome: Literal["pass", "ambiguous", "fail"] = quality.outcome
    if envelope_failures or quality.outcome == "fail":
        outcome = "fail"
    decision = GateDecision(outcome, tuple(reasons), quality.critical_roi_regressions)
    store.data["frames"]["007747"]["seed43_repeat"] = {
        "status": "pass" if decision.passed else "seed_sensitive",
        "runs": run_ids,
        "selected_checkpoint": str(selected.checkpoint),
        "selected_sha256": sha256_file(selected.checkpoint),  # type: ignore[arg-type]
        "metrics": {"psnr": selected.psnr, "ssim": selected.ssim, "lpips": selected.lpips},
        "deltas": deltas,
        "gate": asdict(decision),
        "protocol": evaluated["protocol_path"],
    }
    store.flush()
    return decision


def create_diagnostic_package(
    args: argparse.Namespace,
    store: CampaignStore,
    *,
    frame: str,
    parent_checkpoint: Path,
    winner_lr: float,
    winner_traversal_warmup_steps: int,
    reason: GateDecision,
) -> Path:
    """Freeze the required isolated recovery matrix from the last good parent."""

    package = {
        "frame": frame,
        "created_at": utc_now(),
        "last_good_parent": str(parent_checkpoint),
        "last_good_parent_sha256": sha256_file(parent_checkpoint),
        "failed_gate": asdict(reason),
        "winner_traversal_warmup_steps": winner_traversal_warmup_steps,
        "treatments": [
            {"name": "lr_x0.5", "lr": winner_lr * 0.5, "isolated": True},
            {"name": "lr_x2", "lr": winner_lr * 2.0, "isolated": True},
            {
                "name": "alternate_traversal_warmup",
                "adaptive_warmup_steps": (
                    8_192 if winner_traversal_warmup_steps == 4_096 else 4_096
                ),
                "isolated": True,
            },
            {"name": "extra_fr1_interval", "updates": INTERVAL, "feature_reweighting": 1.0, "isolated": True},
            {
                "name": "extra_tail_if_trajectory_rising",
                "updates": INTERVAL,
                "condition": "only when PSNR/SSIM/LPIPS or critical ROI trajectory is still improving",
                "isolated": True,
            },
        ],
        "resume_rule": (
            "Do not forward this frame; resume the main campaign only after one isolated treatment "
            "passes the full gate."
        ),
    }
    path = args.campaign.parent / "diagnostics" / frame / "diagnostic_package.json"
    atomic_json(path, package)
    store.data["frames"].setdefault(frame, {})["diagnostic_package"] = str(path)
    store.flush()
    return path


def block_frame(store: CampaignStore, frame: str, decision: GateDecision) -> None:
    record = store.data["frames"][frame]
    record["status"] = "rejected"
    record["control_gate"] = asdict(decision)
    store.data["accepted_frames"] = [name for name in store.data["accepted_frames"] if name != frame]
    store.flush()


def write_report(path: Path, store: CampaignStore) -> None:
    rows = []
    rejected = []
    for frame in CHAIN_FRAMES:
        record = store.data["frames"].get(frame)
        if not isinstance(record, Mapping):
            continue
        metrics = record.get("metrics", {})
        rows.append(
            "| "
            + " | ".join(
                (
                    frame,
                    Path(str(record.get("parent_checkpoint", ""))).name,
                    f"{float(record.get('lr', 0)):.3g}",
                    str(record.get("selected_local_step", "")),
                    f"{float(metrics.get('psnr', float('nan'))):.6f}",
                    f"{float(metrics.get('ssim', float('nan'))):.6f}",
                    f"{float(metrics.get('lpips', float('nan'))):.6f}",
                    str(record.get("gate", {}).get("outcome", "")),
                    str(record.get("frame_cumulative_point_samples", "")),
                    str(record.get("temporal_cumulative_point_samples", "")),
                )
            )
            + " |"
        )
        if record.get("status") != "accepted":
            rejected.append(f"- `{frame}`: {record.get('gate', {}).get('reasons', [])}")
    text = [
        "# Temporal LookCloser fine-tuning",
        "",
        "## What was tested",
        "",
        (
            "Sequential model-only cross-frame transfer with fresh local optimizer/grids, FR1 plateau, "
            "and a full-resume FR0.3/LR÷4 tail."
        ),
        "",
        "## Results",
        "",
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "|" + "|".join("---" for _ in REPORT_COLUMNS) + "|",
        *rows,
        "",
        "Only PSNR, SSIM, and LPIPS are reported; training/eval loss is intentionally omitted.",
        "",
        "## Insights",
        "",
        (
            "- The campaign is fail-closed: only a complete fresh three-view eval and ROI/artifact/visual "
            "pass may become a parent."
        ),
        (
            "- The canonical leader exposure is labeled a legacy estimate because its accepted checkpoint "
            "has no exact cumulative counter."
        ),
        "",
        "## Rejected/failed frames",
        "",
        *(rejected or ["- None recorded."]),
        "",
        "## Next steps",
        "",
        "- Resume from the last accepted checkpoint after resolving any recorded infrastructure or quality stop.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def smoke_test(args: argparse.Namespace, store: CampaignStore) -> None:
    spec = RunSpec(
        run_id="smoke_007747_model_only_step0_v2",
        frame="007747",
        seed=args.seed,
        lr=LR_CANDIDATES[0],
        phase="gpu_smoke",
        feature_reweighting=1.0,
        fas_strength=1.0,
        load_mode="model_parameters_only",
        parent_checkpoint=args.leader_checkpoint,
        target_local_step=0,
        inherited_global_step=checkpoint_step(args.leader_checkpoint),
    )
    run = run_training(args, store, spec)
    checkpoint = Path(run["checkpoint"])
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    pipeline = state["pipeline"]
    optimizer_states = state["optimizers"]["fields"]["state"].values()
    optimizer_steps = [int(value["step"].item()) for value in optimizer_states if "step" in value]
    load_audit = state.get("checkpoint_load_audit", {})
    source_hashes = load_audit.get("source_parameter_sha256")
    copied_hashes = load_audit.get("copied_parameter_sha256")
    required = {
        "step": int(state["step"]) == 0,
        "fresh_optimizer": max(optimizer_steps, default=0) <= 1,
        "fas_count": int(pipeline["fas_sample_count_state"].item()) == 1,
        "frame_points_positive": int(pipeline["cumulative_point_samples"].item()) > 0,
        "frequency_grid_zero": int(torch.count_nonzero(pipeline["_model.freq_grid.grid"]).item()) == 0,
        "transferred_parameter_hashes": bool(source_hashes) and source_hashes == copied_hashes,
        "preupdate_reset_assertions": all(
            value is True
            for name, value in load_audit.get("fresh_state_assertions", {}).items()
            if name.endswith("_zero")
        ),
    }
    # Occupancy callback runs before local update zero, proving the fresh grid was
    # updated at step0; the fixed marcher itself is proven by the 4096-step config.
    required["occupancy_updated_at_step0"] = bool(torch.any(pipeline["_model.occupancy_grid.occs"] > 0))
    if not all(required.values()):
        raise InfrastructureError(f"GPU smoke reset assertions failed: {required}")
    resume_spec = RunSpec(
        run_id="smoke_007747_full_resume_step1_v2",
        frame="007747",
        seed=args.seed,
        lr=LR_CANDIDATES[0] / 4.0,
        phase="gpu_smoke_full_resume",
        feature_reweighting=0.3,
        fas_strength=1.0,
        load_mode="resume",
        parent_checkpoint=checkpoint,
        target_local_step=1,
        inherited_global_step=checkpoint_step(args.leader_checkpoint),
        lr_override=LR_CANDIDATES[0] / 4.0,
    )
    resumed_run = run_training(args, store, resume_spec)
    resumed_checkpoint = Path(resumed_run["checkpoint"])
    resumed = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    resume_audit = resumed.get("checkpoint_load_audit", {})
    resume_pipeline = resumed["pipeline"]
    resume_lr = float(resumed["optimizers"]["fields"]["param_groups"][0]["lr"])
    resume_required = {
        "step_continues": int(resumed["step"]) == 1,
        "pipeline_buffers_loaded": resume_audit.get("pipeline_buffers_loaded") is True,
        "optimizer_loaded": resume_audit.get("optimizer_loaded") is True,
        "scheduler_loaded": resume_audit.get("scheduler_loaded") is True,
        "scaler_loaded": resume_audit.get("scaler_loaded") is True,
        "rng_loaded": resume_audit.get("rng_loaded") is True,
        "lr_override": math.isclose(resume_lr, LR_CANDIDATES[0] / 4.0, rel_tol=0.0, abs_tol=1e-12),
        "occupancy_preserved": torch.equal(
            pipeline["_model.occupancy_grid.occs"], resume_pipeline["_model.occupancy_grid.occs"]
        ),
        "frequency_grid_preserved": torch.equal(
            pipeline["_model.freq_grid.grid"], resume_pipeline["_model.freq_grid.grid"]
        ),
        "fas_continues": int(resume_pipeline["fas_sample_count_state"].item()) == 2,
        "point_exposure_continues": int(resume_pipeline["cumulative_point_samples"].item())
        > int(pipeline["cumulative_point_samples"].item()),
    }
    if not all(resume_required.values()):
        raise InfrastructureError(f"GPU smoke full-resume assertions failed: {resume_required}")
    store.data["smoke_test"] = {
        "status": "pass",
        "model_only_assertions": required,
        "full_resume_assertions": resume_required,
        "model_only_run": spec.run_id,
        "full_resume_run": resume_spec.run_id,
    }
    store.flush()
    print(f"smoke status=pass model_only={required} full_resume={resume_required}", flush=True)


def campaign(args: argparse.Namespace, store: CampaignStore) -> None:
    store.data["status"] = "running"
    store.flush()
    ensure_leader_baseline(args, store)
    if "lr_screen" not in store.data:
        screen_specs = lr_screen_specs(args)
        checkpoints_per_candidate = SCREEN_FINAL_STEP // INTERVAL
        disk_guard(
            args.output_dir,
            forecast_bytes=(
                args.leader_checkpoint.stat().st_size
                * len(screen_specs)
                * checkpoints_per_candidate
                + 4 * 1024**3
            ),
        )
        winner_spec, matched = run_lr_screen(args, store)
    else:
        winner_lr = float(store.data["lr_screen"]["winner_lr"])
        winner_warmup = int(store.data["lr_screen"]["winner_traversal_warmup_steps"])
        winner_spec = next(
            spec
            for spec in lr_screen_specs(args)
            if spec.lr == winner_lr and spec.traversal_warmup_steps == winner_warmup
        )
        matched = [
            Metrics(
                local_step=int(row["local_step"]),
                psnr=float(row["psnr"]),
                ssim=float(row["ssim"]),
                lpips=float(row["lpips"]),
                point_samples=row.get("point_samples"),
                checkpoint=Path(row["checkpoint"]),
                run_id=row.get("run_id"),
            )
            for row in store.data["lr_screen"]["candidates"]
        ]
    winner_lr = winner_spec.lr
    winner_warmup = winner_spec.traversal_warmup_steps
    parent = args.leader_checkpoint
    inherited = checkpoint_step(parent)

    selected_frames = [frame for frame in CHAIN_FRAMES if args.start_frame <= frame <= args.end_frame]
    for frame in selected_frames:
        reusable_phase_run_ids = [
            run_id
            for run_id, run in store.data["runs"].items()
            if (
                run_id == winner_spec.run_id
                or (
                    isinstance(run, Mapping)
                    and run.get("frame") == frame
                    and str(run.get("phase", "")).startswith("phase_a")
                    and run_id.startswith(f"transfer_{frame}_")
                )
            )
        ]
        disk_guard(
            args.output_dir,
            frame_storage_forecast(
                args,
                include_control=frame in {"007747", "007838"},
                completed_phase_checkpoints=completed_boundaries(
                    store.data["runs"], reusable_phase_run_ids
                ),
            ),
        )
        prior = store.data["frames"].get(frame)
        if (
            isinstance(prior, Mapping)
            and prior.get("status") == "accepted"
            and prior.get("promotion_complete") is True
        ):
            parent = Path(prior["selected_checkpoint"])
            inherited = int(prior["effective_global_step"])
            print(f"resume frame={frame} status=accepted", flush=True)
            continue
        if frame != "007747":
            previous_name = f"{int(frame) - 7:06d}"
            previous = require_accepted_parent(store.data["frames"], previous_name)
            parent = Path(previous["selected_checkpoint"])
            inherited = int(previous["effective_global_step"])
        initial = winner_spec.run_id if frame == "007747" else None
        selected, run_ids = train_frame_recipe(
            args,
            store,
            frame=frame,
            parent_checkpoint=parent,
            parent_effective_step=inherited,
            lr=winner_lr,
            seed=args.seed,
            prefix="transfer",
            initial_run_id=initial,
            traversal_warmup_steps=winner_warmup,
        )
        gate = finalize_frame(
            args,
            store,
            frame=frame,
            selected=selected,
            run_ids=run_ids,
            parent_checkpoint=parent,
            parent_effective_step=inherited,
            lr=winner_lr,
            traversal_warmup_steps=winner_warmup,
            decision_key=frame,
        )
        write_report(args.report, store)
        if not gate.passed:
            create_diagnostic_package(
                args,
                store,
                frame=frame,
                parent_checkpoint=parent,
                winner_lr=winner_lr,
                winner_traversal_warmup_steps=winner_warmup,
                reason=gate,
            )
            store.data["status"] = f"stopped_{gate.outcome}"
            store.flush()
            raise QualityStop(f"Frame {frame} gate {gate.outcome}: {gate.reasons}")

        if frame == "007747" and (
            bool(store.data["lr_screen"].get("repeat_required"))
            or bool(store.data["frames"][frame].get("gate_small_margin"))
        ):
            repeat_gate = run_seed43_repeat(
                args,
                store,
                parent_checkpoint=parent,
                parent_effective_step=inherited,
                lr=winner_lr,
                traversal_warmup_steps=winner_warmup,
                transfer=selected,
            )
            if not repeat_gate.passed:
                block_frame(store, frame, repeat_gate)
                create_diagnostic_package(
                    args,
                    store,
                    frame=frame,
                    parent_checkpoint=parent,
                    winner_lr=winner_lr,
                    winner_traversal_warmup_steps=winner_warmup,
                    reason=repeat_gate,
                )
                raise QualityStop(f"Frame {frame} seed sensitivity: {repeat_gate.reasons}")

        if frame in {"007747", "007838"}:
            scratch_gate = run_scratch_control(args, store, frame=frame, transfer=selected)
            if not scratch_gate.passed:
                block_frame(store, frame, scratch_gate)
                create_diagnostic_package(
                    args,
                    store,
                    frame=frame,
                    parent_checkpoint=parent,
                    winner_lr=winner_lr,
                    winner_traversal_warmup_steps=winner_warmup,
                    reason=scratch_gate,
                )
                raise QualityStop(f"Frame {frame} scratch parity failed: {scratch_gate.reasons}")

        store.data["frames"][frame]["promotion_complete"] = True
        store.flush()
        parent = selected.checkpoint  # type: ignore[assignment]
        inherited += selected.local_step

    store.data["status"] = "complete"
    store.data["completed_at"] = utc_now()
    store.flush()
    write_report(args.report, store)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.dry_run:
            print(json.dumps(deterministic_dry_run(args), indent=2, sort_keys=True))
            return 0
        preflight = full_preflight(args)
        print(
            f"preflight branch={preflight['git']['branch']} commit={preflight['git']['commit'][:12]} "
            f"datasets={len(preflight['datasets'])} checkpoint_sha=ok config_sha=ok",
            flush=True,
        )
        if args.preflight_only:
            return 0
        store = CampaignStore(args.campaign, resume=args.resume)
        store.data["preflight"] = preflight
        store.data["output_dir"] = str(args.output_dir)
        store.data["report"] = str(args.report)
        store.flush()
        if args.smoke_test:
            smoke_test(args, store)
            return 0
        campaign(args, store)
        return 0
    except QualityStop as exc:
        print(f"quality_stop={exc}", file=sys.stderr, flush=True)
        return QUALITY_EXIT
    except Exception as exc:  # noqa: BLE001 - controller maps all unknown failures to infrastructure.
        print(f"infrastructure_error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return INFRASTRUCTURE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
