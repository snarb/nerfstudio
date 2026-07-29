#!/usr/bin/env python3
"""Run and evaluate a fail-closed from-scratch LookCloser campaign on one frame."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import yaml
from PIL import Image

import run_lookcloser_quiet as quiet


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_VENV = REPO / ".venv"
DEFAULT_DATA = Path("/home/brans/temporal_perframe_stride7_45f/007747")
DEFAULT_OUTPUT = Path("/home/brans/lookcloser_007747_from_scratch_runs")
DEFAULT_LEADER_CHECKPOINT = Path(
    "/home/brans/lookcloser_leader_repro_runs/leader_stableocc_S1_seed42_A_fw03/"
    "lookcloser/20260715_005006/nerfstudio_models/step-000091128.ckpt"
)
DEFAULT_LEADER_CONFIG = DEFAULT_LEADER_CHECKPOINT.parents[1] / "config.yml"
DEFAULT_LEADER_RENDER_DIR = Path(
    "/home/brans/lookcloser_temporal_finetune_runs/007740/lookcloser/canonical_parent_baseline/"
    "evaluations/step-000091128/renders"
)
DEFAULT_TCNN_OVERLAY = Path("/home/brans/deps/tcnn_2e757_py310")
DEFAULT_TORCH_EXTENSIONS = Path("/home/brans/.cache/torch_extensions_lookcloser")
ROI_PROTOCOL = SCRIPTS / "static_target_roi_protocol.py"

CHECKPOINT_INTERVAL = 15_188
STAGE_A_STEP = 75_940
STAGE_B_STEP = 106_316
DEFAULT_TAIL_INTERVALS = 1
DEFAULT_BUDGET_STEP = STAGE_B_STEP + DEFAULT_TAIL_INTERVALS * CHECKPOINT_INTERVAL
LEADER_METRICS = {"psnr": 29.840143, "ssim": 0.669203, "lpips": 0.219455}
PLATEAU_THRESHOLDS = {"psnr": 0.03, "ssim": 0.001, "lpips": 0.003}
PSNR_TIE_DB = 0.07
ALLOWED_DIRTY_PREFIXES = (
    "LookCloser/scripts/run_static_target_from_scratch.py",
    "LookCloser/scripts/static_target_roi_protocol.py",
    "LookCloser/scripts/build_chroma_normalized_frequency_maps.py",
    "LookCloser/tests/test_static_target_from_scratch.py",
    "LookCloser/architecture.md",
    "LookCloser/experiments/static_007747_from_scratch.md",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument(
        "--frame",
        default=None,
        help="Frame identifier; defaults to the dataset directory name.",
    )
    parser.add_argument(
        "--expected-branch",
        default="main",
        help="Fail unless the controller is running from this git branch.",
    )
    parser.add_argument("--variant", choices=("canonical", "fas075", "hash24", "custom"), default="canonical")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--frequency-map-dir", default="lookcloser_frequencies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fas-strength", type=float, default=None)
    parser.add_argument("--log2-hashmap-size", type=int, default=None)
    parser.add_argument("--max-res", type=float, default=8192.0)
    parser.add_argument("--adaptive-coarse-step-size", type=float, default=0.00625)
    parser.add_argument("--max-steps-per-ray", type=int, default=1024)
    parser.add_argument("--stage-b-feature-reweighting", type=float, default=0.3)
    parser.add_argument(
        "--tail-intervals",
        type=int,
        default=None,
        help=(
            "Explicit post-Stage-B intervals to train. When omitted, train only "
            f"until the reviewed default quality budget at step {DEFAULT_BUDGET_STEP}; "
            "use 0 for diagnostics or an explicit positive value for plateau research."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--record-visual-verdict", choices=("pass", "fail"), default=None)
    parser.add_argument("--visual-step", type=int, default=None)
    parser.add_argument("--visual-note", default="")
    parser.add_argument(
        "--record-interval-visual",
        choices=("improved", "no_improvement"),
        default=None,
        help="Record whether moving-detail crops improved across one consecutive eval interval.",
    )
    parser.add_argument("--interval-from-step", type=int, default=None)
    parser.add_argument("--interval-to-step", type=int, default=None)
    parser.add_argument("--interval-note", default="")
    parser.add_argument("--leader-checkpoint", type=Path, default=DEFAULT_LEADER_CHECKPOINT)
    parser.add_argument("--leader-config", type=Path, default=DEFAULT_LEADER_CONFIG)
    parser.add_argument("--leader-render-dir", type=Path, default=DEFAULT_LEADER_RENDER_DIR)
    parser.add_argument("--tcnn-overlay", type=Path, default=DEFAULT_TCNN_OVERLAY)
    args = parser.parse_args(argv)
    args.frame = args.data.resolve().name if args.frame is None else args.frame
    if args.frame != args.data.resolve().name:
        parser.error("--frame must match the dataset directory name")
    args.default_budget = args.tail_intervals is None
    if args.default_budget:
        args.tail_intervals = DEFAULT_TAIL_INTERVALS
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.campaign_name):
        parser.error("--campaign-name must be filesystem-safe")
    if args.tail_intervals < 0:
        parser.error("--tail-intervals cannot be negative")
    # Seed is an explicit recipe coordinate: controlled seed sweeps must remain
    # possible once the architecture/data recipe has been frozen. Resume
    # validation below prevents a campaign from silently changing its seed.
    if not (0 <= args.seed <= 2**32 - 1):
        parser.error("--seed must be in [0, 2^32 - 1]")
    defaults = {
        "canonical": (1.0, 23),
        "fas075": (0.75, 23),
        "hash24": (0.75, 24),
        "custom": (1.0, 23),
    }
    default_fas, default_hash = defaults[args.variant]
    args.fas_strength = default_fas if args.fas_strength is None else args.fas_strength
    args.log2_hashmap_size = default_hash if args.log2_hashmap_size is None else args.log2_hashmap_size
    if not (0.0 <= args.fas_strength <= 1.0):
        parser.error("--fas-strength must be in [0, 1]")
    if args.log2_hashmap_size not in (23, 24):
        parser.error("This campaign supports reviewed hash sizes 23 or 24")
    if not (0.0 <= args.stage_b_feature_reweighting <= 1.0):
        parser.error("--stage-b-feature-reweighting must be in [0, 1]")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def feature_reweighting_tag(strength: float) -> str:
    text = f"{strength:.6f}".rstrip("0").rstrip(".")
    return "fw" + text.replace(".", "")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def campaign_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "campaigns" / args.campaign_name


def manifest_path(args: argparse.Namespace) -> Path:
    return campaign_dir(args) / "campaign.json"


def run_path(args: argparse.Namespace, experiment_name: str, timestamp: str) -> Path:
    return args.output_dir / experiment_name / "lookcloser" / timestamp


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path}")
    return int(match.group(1))


def latest_completed_tail(manifest: Dict[str, Any]) -> Path | None:
    """Return the highest completed post-stage-B checkpoint, if one exists."""
    completed: List[Tuple[int, Path]] = []
    for stage in manifest.get("stages", {}).values():
        target = int(stage.get("target_step", -1))
        checkpoint = Path(stage.get("checkpoint", ""))
        if stage.get("status") == "complete" and target > STAGE_B_STEP and checkpoint.is_file():
            completed.append((target, checkpoint))
    return max(completed, default=(0, None), key=lambda item: item[0])[1]


def tail_intervals_to_run(args: argparse.Namespace, latest: Path) -> int:
    """Resolve the absolute default budget without extending it again on resume."""
    if not args.default_budget:
        return int(args.tail_intervals)
    latest_step = checkpoint_step(latest)
    if latest_step >= DEFAULT_BUDGET_STEP:
        return 0
    remaining = DEFAULT_BUDGET_STEP - latest_step
    if remaining % CHECKPOINT_INTERVAL:
        raise RuntimeError(
            f"Default budget {DEFAULT_BUDGET_STEP} is not aligned from step {latest_step}"
        )
    return remaining // CHECKPOINT_INTERVAL


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def validate_worktree(expected_branch: str) -> Dict[str, Any]:
    branch = git_output("branch", "--show-current")
    if branch != expected_branch:
        raise RuntimeError(
            f"Training must start from branch {expected_branch!r}, found {branch!r}"
        )
    # Preserve porcelain's leading status column (for example `` M path``).
    status_text = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
    status = [line for line in status_text.splitlines() if line]
    unexpected = []
    for line in status:
        path = line[3:].split(" -> ")[-1]
        if path not in ALLOWED_DIRTY_PREFIXES:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(f"Unexpected dirty worktree paths: {unexpected}")
    return {"branch": branch, "commit": git_output("rev-parse", "HEAD"), "status": status}


def load_frequency_map(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    except TypeError:
        return torch.load(path, map_location="cpu").float()


def frequency_map_audit(data: Path, frequency_map_dir: str) -> Dict[str, Any]:
    directory = Path(frequency_map_dir)
    if not directory.is_absolute():
        directory = data / directory
    paths = sorted(directory.glob("*.pt"))
    metadata_paths = sorted(directory.glob("*.json"))
    train_images = sorted((data / "images").glob("frame_train_*.jpg"))
    expected = {path.stem for path in train_images}
    actual = {path.stem for path in paths}
    if len(train_images) != 66 or len(paths) != 66 or len(metadata_paths) != 66 or actual != expected:
        raise RuntimeError(
            f"Frequency map binding failed: train={len(train_images)} pt={len(paths)} "
            f"json={len(metadata_paths)} missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    histogram = torch.zeros(16, dtype=torch.int64)
    means = []
    digest = hashlib.sha256()
    for path in paths:
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tensor = load_frequency_map(path)
        if list(tensor.shape) != [135, 240] or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid frequency map tensor: {path} {tuple(tensor.shape)}")
        if metadata.get("image") != f"{path.stem}.jpg":
            raise RuntimeError(f"Frequency metadata image binding mismatch: {metadata_path}")
        min_res, max_res = float(metadata["min_res"]), float(metadata["max_res"])
        n_levels = int(metadata["n_levels"])
        scale = math.exp((math.log(max_res) - math.log(min_res)) / max(n_levels - 1, 1))
        levels = (torch.log(tensor / min_res) / math.log(scale)).round().clamp(0, n_levels - 1).long()
        histogram += torch.bincount(levels.flatten(), minlength=16)
        means.append(float(tensor.mean().item()))
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
        digest.update(sha256_file(metadata_path).encode())
    total = int(histogram.sum().item())
    fractions = [int(value) / total for value in histogram]
    return {
        "path": str(directory),
        "map_count": len(paths),
        "metadata_count": len(metadata_paths),
        "shape": [135, 240],
        "histogram": histogram.tolist(),
        "fractions": fractions,
        "mean_scalar_resolution": sum(means) / len(means),
        "fraction_levels_14_15": fractions[14] + fractions[15],
        "manifest_sha256": digest.hexdigest(),
    }


def jpeg_audit(data: Path) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {}
    sizes = []
    for path in sorted((data / "images").glob("*.jpg")):
        with Image.open(path) as image:
            quantization = tuple(tuple(table) for _, table in sorted((image.quantization or {}).items()))
            layer = tuple(tuple(row) for row in getattr(image, "layer", ()))
        key = hashlib.sha256(repr((quantization, layer)).encode()).hexdigest()
        profile = profiles.setdefault(key, {"count": 0, "quantization_tables": len(quantization), "layer": layer})
        profile["count"] += 1
        sizes.append(path.stat().st_size)
    return {
        "count": len(sizes),
        "profiles": profiles,
        "size_min": min(sizes),
        "size_mean": sum(sizes) / len(sizes),
        "size_max": max(sizes),
    }


def dataset_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    data = args.data.resolve()
    train = sorted((data / "images").glob("frame_train_*.jpg"))
    evaluation = sorted((data / "images").glob("frame_eval_*.jpg"))
    transforms = data / "transforms.json"
    if len(train) != 66 or len(evaluation) != 3 or not transforms.is_file():
        raise RuntimeError(f"Expected 66 train + 3 eval and transforms.json in {data}")
    for path in (args.leader_checkpoint, args.leader_config, ROI_PROTOCOL):
        if not path.is_file():
            raise FileNotFoundError(path)
    leader_renders = sorted(args.leader_render_dir.glob("eval_img_*.png"))
    if len(leader_renders) != 3:
        raise RuntimeError(f"Expected three canonical leader renders in {args.leader_render_dir}")
    return {
        "worktree": validate_worktree(args.expected_branch),
        "data": str(data),
        "train_images": len(train),
        "eval_images": len(evaluation),
        "transforms_sha256": sha256_file(transforms),
        "frequency_maps": frequency_map_audit(data, args.frequency_map_dir),
        "jpeg": jpeg_audit(data),
        "leader": {
            "checkpoint": str(args.leader_checkpoint),
            "checkpoint_sha256": sha256_file(args.leader_checkpoint),
            "config": str(args.leader_config),
            "config_sha256": sha256_file(args.leader_config),
            "render_dir": str(args.leader_render_dir),
            "metrics": LEADER_METRICS,
            "usage": "reference_only_never_loaded",
        },
        "sources": {
            "controller": sha256_file(Path(__file__)),
            "quiet_runner": sha256_file(SCRIPTS / "run_lookcloser_quiet.py"),
            "roi_protocol": sha256_file(ROI_PROTOCOL),
            "chroma_frequency_builder": sha256_file(
                SCRIPTS / "build_chroma_normalized_frequency_maps.py"
            ),
        },
    }


def runtime_environment(args: argparse.Namespace) -> Dict[str, str]:
    if not (args.venv / "bin" / "python").is_file():
        raise FileNotFoundError(args.venv / "bin" / "python")
    environment = os.environ.copy()
    python_paths = [str(args.tcnn_overlay), str(REPO)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = ":".join(python_paths)
    environment["CUDA_HOME"] = "/usr/local/cuda-12.6"
    environment["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
    environment["TORCH_EXTENSIONS_DIR"] = str(DEFAULT_TORCH_EXTENSIONS)
    environment["PATH"] = f"{args.venv / 'bin'}:{environment['CUDA_HOME']}/bin:{environment.get('PATH', '')}"
    return environment


def common_runner_args(
    args: argparse.Namespace,
    experiment_name: str,
    timestamp: str,
    target_step: int,
    feature_reweighting_strength: float,
) -> List[str]:
    return [
        str(args.venv / "bin" / "python"),
        str(SCRIPTS / "run_lookcloser_quiet.py"),
        "--data", str(args.data),
        "--output-dir", str(args.output_dir),
        "--experiment-name", experiment_name,
        "--timestamp", timestamp,
        "--seed", str(args.seed),
        "--scene-scale", "1.5",
        "--scale-factor", "1.0",
        "--frequency-map-dir", args.frequency_map_dir,
        "--max-res", str(args.max_res),
        "--log2-hashmap-size", str(args.log2_hashmap_size),
        "--ray-sampling-mode", "adaptive",
        "--max-steps-per-ray", str(args.max_steps_per_ray),
        "--adaptive-coarse-step-size", str(args.adaptive_coarse_step_size),
        "--adaptive-warmup-steps", "4096",
        "--reconstruction-loss-type", "charbonnier",
        "--distortion-loss-mult", "0.01",
        "--depth-loss-mult", "0.001",
        "--grid-resolution", "128",
        "--background-color", "black",
        "--occupancy-warmup-steps", "4096",
        "--occupancy-binary-warmup-steps", "4096",
        "--train-num-rays-per-batch", "4096",
        "--fas-strength", str(args.fas_strength),
        "--feature-reweighting-strength", str(feature_reweighting_strength),
        "--fields-lr", "0.01",
        "--fields-lr-final", "0.0001",
        "--fields-scheduler-max-steps", "200000",
        "--step-interval", str(CHECKPOINT_INTERVAL),
        "--save-interval", str(CHECKPOINT_INTERVAL),
        "--max-num-iterations", str(target_step + 1),
        "--no-stop-on-no-improve",
        "--eval-checkpoint", "latest",
        "--keep-all-checkpoints",
        "--no-update-summary",
        "--no-render-final",
        "--no-artifact-score",
        "--no-artifact-roi-score",
        "--poll-seconds", str(args.poll_seconds),
        "--stable-occupancy-reduction",
    ]


def stage_command(
    args: argparse.Namespace,
    experiment_name: str,
    timestamp: str,
    target_step: int,
    feature_reweighting_strength: float,
    load_checkpoint: Path | None,
) -> List[str]:
    command = common_runner_args(
        args, experiment_name, timestamp, target_step, feature_reweighting_strength
    )
    if load_checkpoint is not None:
        command.extend(["--load-checkpoint", str(load_checkpoint)])
    return command


def run_compact(command: List[str], *, environment: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def read_trajectory(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    trajectory = []
    for row in rows:
        if not row.get("eval_all_psnr"):
            continue
        trajectory.append(
            {
                "step": int(row["step"]),
                "psnr": float(row["eval_all_psnr"]),
                "ssim": float(row["eval_all_ssim"]),
                "lpips": float(row["eval_all_lpips"]),
            }
        )
    return trajectory


def validate_stage_config(run_directory: Path, load_checkpoint: Path | None) -> None:
    config_path = run_directory / "config.yml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.Loader)
    if load_checkpoint is None:
        if config.load_checkpoint is not None or config.load_dir is not None or config.load_config is not None:
            raise RuntimeError("Stage A is not from scratch")
    else:
        if Path(config.load_checkpoint).resolve() != load_checkpoint.resolve():
            raise RuntimeError(f"Continuation checkpoint mismatch: {config.load_checkpoint} != {load_checkpoint}")


def run_stage(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    key: str,
    experiment_name: str,
    target_step: int,
    feature_reweighting_strength: float,
    load_checkpoint: Path | None,
    environment: Dict[str, str],
) -> Path:
    timestamp = str(manifest["timestamp"])
    directory = run_path(args, experiment_name, timestamp)
    expected = directory / "nerfstudio_models" / f"step-{target_step:09d}.ckpt"
    existing = manifest.get("stages", {}).get(key)
    if existing and existing.get("status") == "complete" and expected.is_file():
        validate_stage_config(directory, load_checkpoint)
        print(f"stage={key} status=resume-skip checkpoint={expected}", flush=True)
        return expected
    command = stage_command(
        args,
        experiment_name,
        timestamp,
        target_step,
        feature_reweighting_strength,
        load_checkpoint,
    )
    print(f"stage={key} target_step={target_step} command={' '.join(command)}", flush=True)
    returncode = run_compact(command, environment=environment, log_path=campaign_dir(args) / f"{key}.log")
    if returncode != 0 or not expected.is_file():
        raise RuntimeError(f"Stage {key} failed rc={returncode}; expected {expected}")
    validate_stage_config(directory, load_checkpoint)
    record = {
        "status": "complete",
        "experiment_name": experiment_name,
        "run_dir": str(directory),
        "target_step": target_step,
        "feature_reweighting_strength": feature_reweighting_strength,
        "load_checkpoint": str(load_checkpoint) if load_checkpoint else None,
        "checkpoint": str(expected),
        "checkpoint_sha256": sha256_file(expected),
        "trajectory": read_trajectory(directory / "metrics_compact.csv"),
        "completed_at": utc_now(),
    }
    manifest.setdefault("stages", {})[key] = record
    atomic_json(manifest_path(args), manifest)
    return expected


def fresh_eval(
    args: argparse.Namespace,
    checkpoint: Path,
    source_run_dir: Path,
    environment: Dict[str, str],
) -> Dict[str, Any]:
    step = checkpoint_step(checkpoint)
    evaluation_dir = campaign_dir(args) / "evaluations" / f"step-{step:09d}"
    render_dir = evaluation_dir / "renders"
    eval_json = evaluation_dir / "eval.json"
    roi_dir = evaluation_dir / "roi"
    protocol_json = roi_dir / "static_target_roi_protocol.json"
    if eval_json.is_file() and protocol_json.is_file() and len(list(render_dir.glob("eval_img_*.png"))) == 3:
        evaluation = json.loads(eval_json.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_json.read_text(encoding="utf-8"))
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_run_dir": str(source_run_dir),
            "step": step,
            "metrics": evaluation["results"],
            "eval_json": str(eval_json),
            "render_dir": str(render_dir),
            "roi_protocol": str(protocol_json),
            "roi": protocol["roi"],
            "full_view_serious_count": protocol["full_view_serious_count"],
            "visual_gate": protocol["visual_gate"],
        }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    config = quiet.eval_config_for_step(
        source_run_dir / "config.yml", checkpoint, 2048, cache_train_rays=False, filename_tag="target"
    )
    ns_eval = args.venv / "bin" / "ns-eval"
    command = [
        str(ns_eval),
        "--load-config", str(config),
        "--output-path", str(eval_json),
        "--render-output-path", str(render_dir),
    ]
    print(f"fresh-eval step={step} command={' '.join(command)}", flush=True)
    with (evaluation_dir / "eval_stdout.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    evaluation = json.loads(eval_json.read_text(encoding="utf-8"))
    roi_command = [
        str(args.venv / "bin" / "python"),
        str(ROI_PROTOCOL),
        "--frame", args.frame,
        "--dataset", str(args.data),
        "--render-dir", str(render_dir),
        "--out-dir", str(roi_dir),
        "--leader-render-dir", str(args.leader_render_dir),
    ]
    roi_result = subprocess.run(
        roi_command,
        cwd=REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (evaluation_dir / "roi_stdout.log").write_text(roi_result.stdout, encoding="utf-8")
    print(roi_result.stdout, end="", flush=True)
    if roi_result.returncode != 0 or not protocol_json.is_file():
        raise RuntimeError(f"ROI protocol failed for step {step}: {roi_result.stdout}")
    protocol = json.loads(protocol_json.read_text(encoding="utf-8"))
    metrics = evaluation["results"]
    print(
        f"candidate step={step} psnr={metrics['psnr']:.6f} ssim={metrics['ssim']:.6f} "
        f"lpips={metrics['lpips']:.6f} roi_lpips={protocol['roi']['metrics']['lpips']:.6f}",
        flush=True,
    )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_run_dir": str(source_run_dir),
        "step": step,
        "metrics": metrics,
        "eval_json": str(eval_json),
        "render_dir": str(render_dir),
        "roi_protocol": str(protocol_json),
        "roi": protocol["roi"],
        "full_view_serious_count": protocol["full_view_serious_count"],
        "visual_gate": protocol["visual_gate"],
    }


def all_checkpoint_sources(manifest: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    sources: Dict[int, Tuple[Path, Path]] = {}
    for stage in manifest.get("stages", {}).values():
        if stage.get("status") != "complete":
            continue
        directory = Path(stage["run_dir"])
        for checkpoint in sorted((directory / "nerfstudio_models").glob("step-*.ckpt")):
            sources[checkpoint_step(checkpoint)] = (checkpoint, directory)
    return [sources[step] for step in sorted(sources)]


def select_checkpoint(candidates: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(candidates)
    if not rows:
        raise RuntimeError("Cannot select from an empty candidate list")
    maximum = max(float(row["metrics"]["psnr"]) for row in rows)
    tied = [row for row in rows if maximum - float(row["metrics"]["psnr"]) <= PSNR_TIE_DB]
    return min(tied, key=lambda row: (float(row["metrics"]["lpips"]), -float(row["metrics"]["psnr"])))


def numeric_pass(candidate: Dict[str, Any]) -> bool:
    metrics = candidate["metrics"]
    return (
        float(metrics["psnr"]) >= LEADER_METRICS["psnr"]
        and float(metrics["ssim"]) >= LEADER_METRICS["ssim"]
        and float(metrics["lpips"]) <= LEADER_METRICS["lpips"]
    )


def interval_plateau(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    old, new = previous["metrics"], current["metrics"]
    delta = {
        "psnr": float(new["psnr"]) - float(old["psnr"]),
        "ssim": float(new["ssim"]) - float(old["ssim"]),
        "lpips_improvement": float(old["lpips"]) - float(new["lpips"]),
    }
    numeric = (
        delta["psnr"] < PLATEAU_THRESHOLDS["psnr"]
        and delta["ssim"] < PLATEAU_THRESHOLDS["ssim"]
        and delta["lpips_improvement"] < PLATEAU_THRESHOLDS["lpips"]
    )
    return {"from_step": previous["step"], "to_step": current["step"], "delta": delta, "numeric": numeric}


def plateau_summary(
    candidates: Iterable[Dict[str, Any]],
    visual_intervals: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    rows = sorted(candidates, key=lambda row: int(row["step"]))
    intervals = [interval_plateau(left, right) for left, right in zip(rows, rows[1:])]
    visual_intervals = visual_intervals or {}
    for interval in intervals:
        key = f"{interval['from_step']}-{interval['to_step']}"
        interval["visual"] = visual_intervals.get(key, {"verdict": "pending", "note": ""})
    streak = 0
    for interval in intervals:
        streak = streak + 1 if interval["numeric"] else 0
    trailing = intervals[-2:]
    visual_plateau = len(trailing) == 2 and all(
        interval["numeric"] and interval["visual"].get("verdict") == "no_improvement"
        for interval in trailing
    )
    return {
        "thresholds": PLATEAU_THRESHOLDS,
        "intervals": intervals,
        "trailing_numeric_plateau_intervals": streak,
        "visual_confirmation_required": streak >= 2,
        "confirmed": streak >= 2 and visual_plateau,
    }


def refresh_selection(manifest: Dict[str, Any]) -> None:
    candidates = list(manifest.get("candidates", {}).values())
    selected = select_checkpoint(candidates)
    selected["numeric_pass"] = numeric_pass(selected)
    selected["artifact_pass"] = (
        int(selected.get("full_view_serious_count", 1)) == 0
        and not bool(selected.get("roi", {}).get("artifact", {}).get("serious", True))
    )
    selected["accepted"] = (
        selected["numeric_pass"]
        and selected["artifact_pass"]
        and selected.get("visual_gate", {}).get("verdict") == "pass"
    )
    manifest["selected"] = selected
    manifest["plateau"] = plateau_summary(candidates, manifest.get("visual_intervals"))
    manifest["status"] = "accepted" if selected["accepted"] else "evaluated"
    manifest["updated_at"] = utc_now()


def evaluate_all(
    args: argparse.Namespace,
    manifest: Dict[str, Any],
    environment: Dict[str, str],
) -> None:
    for checkpoint, directory in all_checkpoint_sources(manifest):
        step = checkpoint_step(checkpoint)
        key = str(step)
        existing = manifest.setdefault("candidates", {}).get(key)
        if existing and Path(existing.get("roi_protocol", "")).is_file():
            continue
        manifest["candidates"][key] = fresh_eval(args, checkpoint, directory, environment)
        atomic_json(manifest_path(args), manifest)
    refresh_selection(manifest)
    atomic_json(manifest_path(args), manifest)


def record_visual_verdict(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    if not manifest.get("candidates"):
        raise RuntimeError("No evaluated candidate exists for visual review")
    step = args.visual_step
    if step is None:
        step = int(select_checkpoint(manifest["candidates"].values())["step"])
    candidate = manifest["candidates"].get(str(step))
    if candidate is None:
        raise RuntimeError(f"No candidate for --visual-step {step}")
    protocol_path = Path(candidate["roi_protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["visual_gate"] = {**protocol["visual_gate"], "verdict": args.record_visual_verdict, "note": args.visual_note}
    atomic_json(protocol_path, protocol)
    candidate["visual_gate"] = protocol["visual_gate"]
    refresh_selection(manifest)
    atomic_json(manifest_path(args), manifest)


def record_interval_visual(args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    if args.interval_from_step is None or args.interval_to_step is None:
        raise RuntimeError("Interval visual review requires --interval-from-step and --interval-to-step")
    if args.interval_to_step - args.interval_from_step != CHECKPOINT_INTERVAL:
        raise RuntimeError("Visual plateau reviews must cover exactly one 15188-update interval")
    candidates = manifest.get("candidates", {})
    if str(args.interval_from_step) not in candidates or str(args.interval_to_step) not in candidates:
        raise RuntimeError("Both interval endpoints must have fresh evaluated candidates")
    key = f"{args.interval_from_step}-{args.interval_to_step}"
    manifest.setdefault("visual_intervals", {})[key] = {
        "verdict": args.record_interval_visual,
        "note": args.interval_note,
        "recorded_at": utc_now(),
    }
    refresh_selection(manifest)
    atomic_json(manifest_path(args), manifest)


def initialize_manifest(args: argparse.Namespace, preflight: Dict[str, Any]) -> Dict[str, Any]:
    path = manifest_path(args)
    if path.exists():
        if not (
            args.resume
            or args.evaluate_only
            or args.record_visual_verdict
            or args.record_interval_visual
        ):
            raise RuntimeError(f"Campaign already exists; pass --resume: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "variant": args.variant,
            "frame": args.frame,
            "expected_branch": args.expected_branch,
            "data": str(args.data),
            "seed": args.seed,
            "fas_strength": args.fas_strength,
            "log2_hashmap_size": args.log2_hashmap_size,
            "max_res": args.max_res,
            "adaptive_coarse_step_size": args.adaptive_coarse_step_size,
            "max_steps_per_ray": args.max_steps_per_ray,
            "frequency_map_dir": args.frequency_map_dir,
            "stage_b_feature_reweighting": args.stage_b_feature_reweighting,
        }
        actual = {key: manifest.get("recipe", {}).get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"Resume recipe mismatch: {actual} != {expected}")
        # Controller/protocol fixes made between resumptions are provenance, not a
        # recipe change. Keep every validated source fingerprint in the manifest.
        if preflight.get("sources") != manifest.get("preflight", {}).get("sources"):
            manifest.setdefault("resume_preflights", []).append(
                {"recorded_at": utc_now(), "preflight": preflight}
            )
            atomic_json(path, manifest)
        return manifest
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest = {
        "schema_version": 1,
        "campaign_name": args.campaign_name,
        "created_at": utc_now(),
        "timestamp": timestamp,
        "status": "initialized",
        "recipe": {
            "variant": args.variant,
            "frame": args.frame,
            "expected_branch": args.expected_branch,
            "data": str(args.data),
            "seed": args.seed,
            "fas_strength": args.fas_strength,
            "log2_hashmap_size": args.log2_hashmap_size,
            "max_res": args.max_res,
            "adaptive_coarse_step_size": args.adaptive_coarse_step_size,
            "max_steps_per_ray": args.max_steps_per_ray,
            "frequency_map_dir": args.frequency_map_dir,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "stage_a_step": STAGE_A_STEP,
            "stage_b_step": STAGE_B_STEP,
            "default_budget_step": DEFAULT_BUDGET_STEP,
            "stage_a_feature_reweighting": 1.0,
            "stage_b_feature_reweighting": args.stage_b_feature_reweighting,
            "leader_checkpoint_loaded": False,
        },
        "leader_gates": LEADER_METRICS,
        "preflight": preflight,
        "stages": {},
        "candidates": {},
    }
    atomic_json(path, manifest)
    return manifest


def dry_run(args: argparse.Namespace, preflight: Dict[str, Any]) -> int:
    timestamp = "DRY_RUN"
    stage_a_name = f"{args.campaign_name}_A"
    stage_a_checkpoint = run_path(args, stage_a_name, timestamp) / "nerfstudio_models" / f"step-{STAGE_A_STEP:09d}.ckpt"
    stage_b_strength = args.stage_b_feature_reweighting
    stage_b_name = f"{args.campaign_name}_A_{feature_reweighting_tag(stage_b_strength)}"
    stage_b_checkpoint = (
        run_path(args, stage_b_name, timestamp)
        / "nerfstudio_models"
        / f"step-{STAGE_B_STEP:09d}.ckpt"
    )
    commands = {
        "stage_a": stage_command(args, stage_a_name, timestamp, STAGE_A_STEP, 1.0, None),
        "stage_b": stage_command(
            args, stage_b_name, timestamp, STAGE_B_STEP, stage_b_strength, stage_a_checkpoint
        ),
    }
    latest = stage_b_checkpoint
    for _ in range(tail_intervals_to_run(args, latest)):
        target = checkpoint_step(latest) + CHECKPOINT_INTERVAL
        key = f"tail_{target}"
        experiment = f"{args.campaign_name}_tail_s{target}"
        commands[key] = stage_command(
            args,
            experiment,
            timestamp,
            target,
            stage_b_strength,
            latest,
        )
        latest = (
            run_path(args, experiment, timestamp)
            / "nerfstudio_models"
            / f"step-{target:09d}.ckpt"
        )
    print(json.dumps({"preflight": preflight, "commands": commands}, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preflight = dataset_preflight(args)
    if args.dry_run:
        return dry_run(args, preflight)
    environment = runtime_environment(args)
    manifest = initialize_manifest(args, preflight)
    if args.record_visual_verdict is not None:
        record_visual_verdict(args, manifest)
        return 0 if manifest["status"] == "accepted" else 2
    if args.record_interval_visual is not None:
        record_interval_visual(args, manifest)
        return 0
    if not args.evaluate_only:
        stage_a_name = f"{args.campaign_name}_A"
        stage_a = run_stage(
            args, manifest, "stage_a", stage_a_name, STAGE_A_STEP, 1.0, None, environment
        )
        evaluate_all(args, manifest, environment)
        stage_b_strength = args.stage_b_feature_reweighting
        stage_b_name = f"{args.campaign_name}_A_{feature_reweighting_tag(stage_b_strength)}"
        latest = run_stage(
            args, manifest, "stage_b", stage_b_name, STAGE_B_STEP, stage_b_strength, stage_a, environment
        )
        evaluate_all(args, manifest, environment)
        latest = latest_completed_tail(manifest) or latest
        for _ in range(tail_intervals_to_run(args, latest)):
            target = checkpoint_step(latest) + CHECKPOINT_INTERVAL
            key = f"tail_{target}"
            experiment = f"{args.campaign_name}_tail_s{target}"
            latest = run_stage(args, manifest, key, experiment, target, stage_b_strength, latest, environment)
            evaluate_all(args, manifest, environment)
    else:
        evaluate_all(args, manifest, environment)
    selected = manifest["selected"]
    print(
        f"selected step={selected['step']} psnr={selected['metrics']['psnr']:.6f} "
        f"ssim={selected['metrics']['ssim']:.6f} lpips={selected['metrics']['lpips']:.6f} "
        f"numeric_pass={selected['numeric_pass']} visual={selected['visual_gate']['verdict']} "
        f"manifest={manifest_path(args)}",
        flush=True,
    )
    return 0 if selected["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
