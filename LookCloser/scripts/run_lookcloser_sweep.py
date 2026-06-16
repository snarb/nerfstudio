#!/usr/bin/env python3
"""Three-seed LookCloser frequency-grid sweep orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "LookCloser" / "scripts" / "run_lookcloser_quiet.py"
DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_OUTPUT = REPO_ROOT / "LookCloser" / "repro_runs" / "lookcloser_runs"
DEFAULT_EXPERIMENT = "007740_hd_aabb4_multicamera_eval3_ns_lookcloser_frequency_grid"
DEFAULT_REPORT = REPO_ROOT / "LookCloser" / "experiments" / "lookcloser_frequency_grid_optimization.md"
SEEDS = (42, 43, 44)
MAX_NUM_ITERATIONS = 60752


@dataclass(frozen=True)
class Candidate:
    stage: str
    param: str
    value: Any
    config: Dict[str, Any]

    @property
    def label(self) -> str:
        return value_label(self.value)


@dataclass
class RunRecord:
    candidate: Candidate
    seed: int
    timestamp: str
    run_dir: Path
    wrapper_log: Path
    status: str = "pending"
    returncode: Optional[int] = None
    eval_json: Optional[Path] = None
    render_dir: Optional[Path] = None
    checkpoint: Optional[str] = None
    selected_step: Optional[int] = None
    eval_loss: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    train_seconds: Optional[float] = None
    eval_seconds: Optional[float] = None
    artifact_seconds: Optional[float] = None
    total_seconds: Optional[float] = None
    artifact_score: Optional[float] = None
    artifact_count: Optional[int] = None
    error: Optional[str] = None


@dataclass
class CandidateStats:
    candidate: Candidate
    runs: List[RunRecord]
    mean_psnr: float
    max_psnr: float
    mean_ssim: float
    max_ssim: float
    mean_lpips: float
    min_lpips: float
    mean_eval_loss: float
    min_eval_loss: float
    max_eval_loss: float
    mean_train_seconds: float
    min_train_seconds: float
    mean_eval_seconds: Optional[float]
    mean_artifact_seconds: Optional[float]
    mean_total_seconds: Optional[float]
    mean_artifact_score: Optional[float]
    min_artifact_score: Optional[float]
    rank: Optional[int] = None
    carried_forward: bool = False


@dataclass
class ActiveJob:
    record: RunRecord
    proc: subprocess.Popen[Any]
    log_handle: Any
    started_at: float = field(default_factory=time.time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stage", choices=("baseline", "schedule", "grid", "update", "fallback", "all"), default="all")
    parser.add_argument("--allow-missing-frequency-maps", action="store_true")
    parser.add_argument("--base-grid-resolution", type=int, default=None)
    parser.add_argument("--base-grid-update-interval", type=int, default=None)
    parser.add_argument("--base-grid-update-batch-size", type=int, default=None)
    parser.add_argument("--base-fallback-frequency-level", type=float, default=None)
    parser.add_argument("--base-fixed-num-samples-per-ray", type=int, default=None)
    parser.add_argument("--base-fas-strength", type=float, default=None)
    parser.add_argument("--base-fas-warmup-steps", type=int, default=None)
    parser.add_argument("--base-fas-ramp-steps", type=int, default=None)
    parser.add_argument("--base-occupancy-occ-thre", type=float, default=None)
    parser.add_argument("--base-occupancy-ema-decay", type=float, default=None)
    parser.add_argument("--base-occupancy-warmup-steps", type=int, default=None)
    parser.add_argument("--base-occupancy-update-interval", type=int, default=None)
    parser.add_argument("--base-occupancy-thre-clamp-mult", type=float, default=None)
    parser.add_argument("--base-occupancy-dilation-radius", type=int, default=None)
    parser.add_argument("--artifact-render-names", default=None)
    parser.add_argument("--artifact-crop-top", type=int, default=0)
    parser.add_argument("--artifact-crop-bottom", type=int, default=0)
    parser.add_argument("--artifact-crop-left", type=int, default=0)
    parser.add_argument("--artifact-crop-right", type=int, default=0)
    parser.add_argument("--update-intervals", default="512,1024,2048")
    parser.add_argument("--update-batch-sizes", default="2048,4096,8192")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def value_label(value: Any) -> str:
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        text = f"{value:.10g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def run_dir(args: argparse.Namespace, timestamp: str) -> Path:
    return args.output_dir / args.experiment_name / "lookcloser" / timestamp


def timestamp_for(candidate: Candidate, seed: int) -> str:
    return f"{candidate.stage}_{candidate.param}_{candidate.label}_seed{seed}"


def compact_rows(metrics_path: Path) -> List[Dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_eval_loss(metrics_path: Path, selected_step: Optional[int]) -> Optional[float]:
    rows = [row for row in compact_rows(metrics_path) if row.get("eval_loss")]
    if not rows:
        return None
    if selected_step is not None:
        for row in rows:
            if int(float(row["step"])) == selected_step:
                return float(row["eval_loss"])
    return min(float(row["eval_loss"]) for row in rows)


def find_eval_json(path: Path) -> Optional[Path]:
    evals = sorted(path.glob("eval_best_step-*.json"))
    if not evals:
        evals = sorted(path.glob("eval_latest_step-*.json"))
    if not evals:
        evals = sorted(path.glob("eval_*.json"))
    return evals[-1] if evals else None


def parse_record(record: RunRecord) -> RunRecord:
    eval_json = find_eval_json(record.run_dir)
    if eval_json is None:
        record.status = "missing_eval"
        record.error = f"No eval JSON in {record.run_dir}"
        return record
    try:
        summary = read_run_summary(record.run_dir)
        data = json.loads(eval_json.read_text(encoding="utf-8"))
        results = data["results"]
        record.eval_json = eval_json
        record.checkpoint = data.get("checkpoint")
        if record.checkpoint:
            stem = Path(record.checkpoint).stem
            if stem.startswith("step-"):
                record.selected_step = int(stem.split("-")[-1])
        record.render_dir = infer_render_dir(record.run_dir, eval_json)
        record.eval_loss = best_eval_loss(record.run_dir / "metrics_compact.csv", record.selected_step)
        missing = [name for name in ("psnr", "ssim", "lpips") if name not in results]
        if missing:
            record.status = "missing_metrics"
            record.error = f"Eval JSON missing metrics: {', '.join(missing)}"
            return record
        record.psnr = float(results["psnr"])
        record.ssim = float(results["ssim"])
        record.lpips = float(results["lpips"])
        record.train_seconds = read_train_seconds(record)
        if record.train_seconds is None:
            record.status = "missing_metrics"
            record.error = "Missing train_seconds in run_summary.json and wrapper log"
            return record
        eval_data = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
        artifact = summary.get("artifact") if isinstance(summary.get("artifact"), dict) else eval_data.get("artifact", {})
        record.eval_seconds = optional_float(summary.get("eval_seconds") or eval_data.get("eval_seconds"))
        record.artifact_seconds = optional_float(summary.get("artifact_seconds") or artifact.get("artifact_seconds"))
        record.total_seconds = optional_float(summary.get("total_seconds"))
        record.artifact_score = optional_float(artifact.get("artifact_score"))
        if artifact.get("artifact_count") is not None:
            record.artifact_count = int(artifact["artifact_count"])
        record.status = "complete"
    except Exception as exc:  # noqa: BLE001
        record.status = "parse_error"
        record.error = str(exc)
    return record


def read_run_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def read_train_seconds(record: RunRecord) -> Optional[float]:
    summary = read_run_summary(record.run_dir)
    if summary.get("train_seconds") is not None:
        return float(summary["train_seconds"])
    if record.wrapper_log.exists():
        match = re.search(r"^train_seconds=([0-9.]+)$", record.wrapper_log.read_text(encoding="utf-8", errors="ignore"), re.MULTILINE)
        if match:
            return float(match.group(1))
    return None


def infer_render_dir(path: Path, eval_json: Path) -> Optional[Path]:
    suffix = eval_json.name
    if suffix.startswith("eval_"):
        suffix = suffix[len("eval_") :]
    if suffix.endswith(".json"):
        suffix = suffix[: -len(".json")]
    render_dir = path / f"renders_{suffix}"
    return render_dir if render_dir.exists() else None


def completed_record(args: argparse.Namespace, candidate: Candidate, seed: int) -> RunRecord:
    timestamp = timestamp_for(candidate, seed)
    path = run_dir(args, timestamp)
    record = RunRecord(
        candidate=candidate,
        seed=seed,
        timestamp=timestamp,
        run_dir=path,
        wrapper_log=path / "sweep_wrapper_stdout.log",
    )
    if find_eval_json(path) is not None:
        return parse_record(record)
    if path.exists() and any(path.iterdir()):
        archived = path.with_name(f"{path.name}_incomplete_{int(time.time())}")
        shutil.move(str(path), str(archived))
        print(f"archived incomplete run {path.name} -> {archived.name}", flush=True)
    return record


def runner_command(args: argparse.Namespace, record: RunRecord) -> List[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--data",
        str(args.data),
        "--output-dir",
        str(args.output_dir),
        "--experiment-name",
        args.experiment_name,
        "--timestamp",
        record.timestamp,
        "--seed",
        str(record.seed),
        "--max-num-iterations",
        str(MAX_NUM_ITERATIONS),
        "--eval-checkpoint",
        "best",
        "--eval-num-rays-per-chunk",
        str(args.eval_num_rays_per_chunk),
        "--poll-seconds",
        str(args.poll_seconds),
        "--no-update-summary",
    ]
    add_config_args(cmd, record.candidate.config)
    if args.allow_missing_frequency_maps:
        cmd.append("--allow-missing-frequency-maps")
    if args.artifact_render_names:
        cmd.extend(["--artifact-render-names", args.artifact_render_names])
    if args.artifact_crop_top:
        cmd.extend(["--artifact-crop-top", str(args.artifact_crop_top)])
    if args.artifact_crop_bottom:
        cmd.extend(["--artifact-crop-bottom", str(args.artifact_crop_bottom)])
    if args.artifact_crop_left:
        cmd.extend(["--artifact-crop-left", str(args.artifact_crop_left)])
    if args.artifact_crop_right:
        cmd.extend(["--artifact-crop-right", str(args.artifact_crop_right)])
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def add_config_args(cmd: List[str], config: Dict[str, Any]) -> None:
    mapping = {
        "scene_scale": "--scene-scale",
        "scale_factor": "--scale-factor",
        "center_method": "--center-method",
        "orientation_method": "--orientation-method",
        "train_num_rays_per_batch": "--train-num-rays-per-batch",
        "background_color": "--background-color",
        "grid_resolution": "--grid-resolution",
        "num_frequency_levels": "--num-frequency-levels",
        "min_res": "--min-res",
        "max_res": "--max-res",
        "max_res_base": "--max-res-base",
        "fallback_frequency_level": "--fallback-frequency-level",
        "grid_update_interval": "--grid-update-interval",
        "grid_update_batch_size": "--grid-update-batch-size",
        "sampling_ramp_start": "--sampling-ramp-start",
        "sampling_ramp_end": "--sampling-ramp-end",
        "fas_strength": "--fas-strength",
        "fas_warmup_steps": "--fas-warmup-steps",
        "fas_ramp_steps": "--fas-ramp-steps",
        "fas_decay_start_steps": "--fas-decay-start-steps",
        "fas_decay_steps": "--fas-decay-steps",
        "fixed_num_samples_per_ray": "--fixed-num-samples-per-ray",
        "max_steps_per_ray": "--max-steps-per-ray",
        "adaptive_warmup_steps": "--adaptive-warmup-steps",
        "adaptive_fixed_fallback_samples_per_ray": "--adaptive-fixed-fallback-samples-per-ray",
        "adaptive_coarse_step_size": "--adaptive-coarse-step-size",
        "adaptive_max_step_size": "--adaptive-max-step-size",
        "alpha_thre": "--alpha-thre",
        "occupancy_occ_thre": "--occupancy-occ-thre",
        "occupancy_ema_decay": "--occupancy-ema-decay",
        "occupancy_warmup_steps": "--occupancy-warmup-steps",
        "occupancy_update_interval": "--occupancy-update-interval",
        "occupancy_update_step_size": "--occupancy-update-step-size",
        "occupancy_thre_clamp_mult": "--occupancy-thre-clamp-mult",
        "occupancy_dilation_radius": "--occupancy-dilation-radius",
    }
    for key, flag in mapping.items():
        value = config.get(key)
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    if not config.get("enable_frequency_grid", True):
        cmd.append("--disable-frequency-grid")
    if not config.get("enable_feature_reweighting", True):
        cmd.append("--disable-feature-reweighting")
    if not config.get("enable_adaptive_ray_marching", True):
        cmd.append("--disable-adaptive-ray-marching")
    if not config.get("enable_fas", True):
        cmd.append("--disable-fas")


def start_job(args: argparse.Namespace, record: RunRecord) -> ActiveJob:
    record.run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = record.wrapper_log.open("a", encoding="utf-8")
    cmd = runner_command(args, record)
    log_handle.write(f"\n# sweep command: {' '.join(cmd)}\n")
    log_handle.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    record.status = "running"
    print(f"started {record.timestamp}", flush=True)
    return ActiveJob(record=record, proc=proc, log_handle=log_handle)


def is_oom(record: RunRecord) -> bool:
    needles = ("out of memory", "out_of_memory", "cuda_error", "cuda error", "cublas", "allocation")
    for path in (record.wrapper_log, record.run_dir / "train_stdout.log", record.run_dir / "eval_stdout.log"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(needle in text for needle in needles):
            return True
    return False


def archive_failed_run(record: RunRecord) -> None:
    if not record.run_dir.exists():
        return
    archived = record.run_dir.with_name(f"{record.run_dir.name}_failed_{int(time.time())}")
    shutil.move(str(record.run_dir), str(archived))
    record.error = f"Archived failed run at {archived}"


def run_records(args: argparse.Namespace, records: List[RunRecord], max_parallel: int) -> tuple[List[RunRecord], int]:
    pending = [record for record in records if record.status == "pending"]
    complete = [record for record in records if record.status == "complete"]
    active: List[ActiveJob] = []
    current_parallel = max(1, max_parallel)
    last_report = 0.0
    while pending or active:
        while pending and len(active) < current_parallel:
            active.append(start_job(args, pending.pop(0)))
        time.sleep(args.poll_seconds)
        still_active: List[ActiveJob] = []
        for job in active:
            rc = job.proc.poll()
            if rc is None:
                still_active.append(job)
                continue
            job.log_handle.close()
            job.record.returncode = rc
            parsed = parse_record(job.record)
            if parsed.status == "complete" and rc == 0:
                complete.append(parsed)
                print(format_run_result(parsed), flush=True)
                continue
            parsed.status = "failed"
            parsed.error = parsed.error or f"runner returned {rc}"
            if is_oom(parsed) and current_parallel > 1:
                current_parallel -= 1
                print(f"OOM detected in {parsed.timestamp}; reducing parallelism to {current_parallel}", flush=True)
                archive_failed_run(parsed)
                pending.insert(0, completed_record(args, parsed.candidate, parsed.seed))
            else:
                complete.append(parsed)
                print(f"failed {parsed.timestamp}: {parsed.error}", flush=True)
        active = still_active
        now = time.time()
        if now - last_report >= max(args.poll_seconds, 60.0):
            print_progress(active, pending, complete)
            last_report = now
    return complete, current_parallel


def run_candidate_groups(args: argparse.Namespace, candidates: List[Candidate], max_parallel: int) -> tuple[List[RunRecord], int]:
    all_runs: List[RunRecord] = []
    current_parallel = max_parallel
    for candidate in candidates:
        print(
            f"candidate {candidate.stage}/{candidate.param}={candidate.value}: "
            f"running seeds {', '.join(str(seed) for seed in SEEDS)}",
            flush=True,
        )
        records = stage_records(args, [candidate])
        runs, current_parallel = run_records(args, records, min(current_parallel, len(SEEDS)))
        all_runs.extend(runs)
    return all_runs, current_parallel


def print_progress(active: Sequence[ActiveJob], pending: Sequence[RunRecord], complete: Sequence[RunRecord]) -> None:
    running = []
    for job in active:
        step = latest_step(job.record.run_dir / "metrics_compact.csv")
        running.append(f"{job.record.timestamp}@{step or 'starting'}")
    print(
        f"progress complete={len(complete)} running={len(active)} pending={len(pending)}"
        + (f" | {'; '.join(running)}" if running else ""),
        flush=True,
    )


def latest_step(metrics_path: Path) -> Optional[str]:
    rows = compact_rows(metrics_path)
    return rows[-1]["step"] if rows else None


def format_run_result(record: RunRecord) -> str:
    return (
        f"complete {record.timestamp} "
        f"step={record.selected_step} "
        f"loss={fmt(record.eval_loss)} "
        f"psnr={fmt(record.psnr)} "
        f"ssim={fmt(record.ssim)} "
        f"lpips={fmt(record.lpips)} "
        f"artifact={fmt(record.artifact_score)} "
        f"train_s={fmt(record.train_seconds)} "
        f"total_s={fmt(record.total_seconds)}"
    )


def fmt(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.6f}"


def candidate_stats(candidate: Candidate, runs: List[RunRecord]) -> CandidateStats:
    good = [run for run in runs if run.status == "complete"]
    if len(good) != len(SEEDS):
        raise RuntimeError(f"{candidate.stage}/{candidate.param}={candidate.value} has incomplete runs")
    return CandidateStats(
        candidate=candidate,
        runs=good,
        mean_psnr=mean(required(run.psnr) for run in good),
        max_psnr=max(required(run.psnr) for run in good),
        mean_ssim=mean(required(run.ssim) for run in good),
        max_ssim=max(required(run.ssim) for run in good),
        mean_lpips=mean(required(run.lpips) for run in good),
        min_lpips=min(required(run.lpips) for run in good),
        mean_eval_loss=mean(required(run.eval_loss) for run in good),
        min_eval_loss=min(required(run.eval_loss) for run in good),
        max_eval_loss=max(required(run.eval_loss) for run in good),
        mean_train_seconds=mean(required(run.train_seconds) for run in good),
        min_train_seconds=min(required(run.train_seconds) for run in good),
        mean_eval_seconds=optional_mean(run.eval_seconds for run in good),
        mean_artifact_seconds=optional_mean(run.artifact_seconds for run in good),
        mean_total_seconds=optional_mean(run.total_seconds for run in good),
        mean_artifact_score=optional_mean(run.artifact_score for run in good),
        min_artifact_score=optional_min(run.artifact_score for run in good),
    )


def required(value: Optional[float]) -> float:
    if value is None:
        raise RuntimeError("missing metric")
    return value


def optional_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def optional_min(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def rank_value(value: Optional[float], missing: float) -> float:
    return value if value is not None else missing


def rank_candidates(stats: List[CandidateStats]) -> List[CandidateStats]:
    ssim_rank = rank_map(stats, "mean_ssim", reverse=True)
    lpips_rank = rank_map(stats, "mean_lpips", reverse=False)
    psnr_rank = rank_map(stats, "mean_psnr", reverse=True)
    artifact_available = all(item.mean_artifact_score is not None for item in stats)
    ranked = sorted(
        stats,
        key=lambda item: (
            rank_value(item.mean_artifact_score, math.inf) if artifact_available else 0.0,
            ssim_rank[id(item)],
            lpips_rank[id(item)],
            psnr_rank[id(item)],
            item.mean_eval_loss,
            rank_value(item.mean_total_seconds, item.mean_train_seconds),
        ),
    )
    for idx, item in enumerate(ranked, start=1):
        item.rank = idx
    ranked[0].carried_forward = True
    return ranked


def rank_map(stats: List[CandidateStats], attr: str, reverse: bool) -> Dict[int, int]:
    ordered = sorted(stats, key=lambda item: getattr(item, attr), reverse=reverse)
    return {id(item): idx for idx, item in enumerate(ordered, start=1)}


def base_config() -> Dict[str, Any]:
    return {
        "scene_scale": 2.0,
        "scale_factor": 1.15,
        "center_method": "focus",
        "orientation_method": "up",
        "train_num_rays_per_batch": 1024,
        "background_color": "black",
        "grid_resolution": 128,
        "num_frequency_levels": 16,
        "min_res": 16.0,
        "max_res": None,
        "max_res_base": 2048.0,
        "fallback_frequency_level": 0.0,
        "grid_update_interval": 1024,
        "grid_update_batch_size": 2048,
        "fixed_num_samples_per_ray": 512,
        "enable_frequency_grid": True,
        "enable_feature_reweighting": True,
        "enable_adaptive_ray_marching": False,
        "enable_fas": True,
        "sampling_ramp_start": 1.0,
        "sampling_ramp_end": 3.0,
        "fas_strength": 1.0,
        "fas_warmup_steps": 0,
        "fas_ramp_steps": 0,
        "occupancy_occ_thre": 1e-2,
        "occupancy_ema_decay": 0.95,
        "occupancy_warmup_steps": 256,
        "occupancy_update_interval": 16,
        "occupancy_update_step_size": None,
        "occupancy_thre_clamp_mult": 1.0,
        "occupancy_dilation_radius": 0,
    }


def apply_base_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(config)
    overrides = {
        "grid_resolution": args.base_grid_resolution,
        "grid_update_interval": args.base_grid_update_interval,
        "grid_update_batch_size": args.base_grid_update_batch_size,
        "fallback_frequency_level": args.base_fallback_frequency_level,
        "fixed_num_samples_per_ray": args.base_fixed_num_samples_per_ray,
        "fas_strength": args.base_fas_strength,
        "fas_warmup_steps": args.base_fas_warmup_steps,
        "fas_ramp_steps": args.base_fas_ramp_steps,
        "occupancy_occ_thre": args.base_occupancy_occ_thre,
        "occupancy_ema_decay": args.base_occupancy_ema_decay,
        "occupancy_warmup_steps": args.base_occupancy_warmup_steps,
        "occupancy_update_interval": args.base_occupancy_update_interval,
        "occupancy_thre_clamp_mult": args.base_occupancy_thre_clamp_mult,
        "occupancy_dilation_radius": args.base_occupancy_dilation_radius,
    }
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def with_value(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    updated = dict(config)
    updated[key] = value
    return updated


def make_candidate(stage: str, param: str, value: Any, config: Dict[str, Any]) -> Candidate:
    return Candidate(stage=stage, param=param, value=value, config=dict(config))


def stage_records(args: argparse.Namespace, candidates: List[Candidate]) -> List[RunRecord]:
    records: List[RunRecord] = []
    for candidate in candidates:
        for seed in SEEDS:
            records.append(completed_record(args, candidate, seed))
    return records


def summarize_stage(
    args: argparse.Namespace,
    all_stats: List[CandidateStats],
    candidates: List[Candidate],
    stage_runs: List[RunRecord],
) -> CandidateStats:
    by_key: Dict[tuple[str, str, str], List[RunRecord]] = {}
    for run in stage_runs:
        key = (run.candidate.stage, run.candidate.param, run.candidate.label)
        by_key.setdefault(key, []).append(run)
    stats = [
        candidate_stats(candidate, by_key[(candidate.stage, candidate.param, candidate.label)])
        for candidate in candidates
    ]
    ranked = rank_candidates(stats)
    all_stats.extend(ranked)
    write_report(args.report_path, args.data, all_stats)
    print_stage_leaderboard(ranked)
    return ranked[0]


def print_stage_leaderboard(stats: Sequence[CandidateStats]) -> None:
    print("leaderboard", flush=True)
    for item in stats:
        print(
            f"  #{item.rank} {item.candidate.stage}/{item.candidate.param}={item.candidate.value} "
            f"mean_artifact={fmt(item.mean_artifact_score)} "
            f"mean_ssim={item.mean_ssim:.6f} mean_lpips={item.mean_lpips:.6f} "
            f"mean_psnr={item.mean_psnr:.6f} mean_loss={item.mean_eval_loss:.8f} "
            f"mean_train_s={item.mean_train_seconds:.3f} mean_total_s={fmt(item.mean_total_seconds)}",
            flush=True,
        )


def write_report(path: Path, data: Path, stats: Sequence[CandidateStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LookCloser Frequency Grid Optimization",
        "",
        "## What was tested",
        "",
        f"- Dataset: `{data}`",
        f"- Seeds per candidate: `{', '.join(str(seed) for seed in SEEDS)}`",
        f"- Max iterations per run: `{MAX_NUM_ITERATIONS}`",
        "- Checkpoint protocol: final eval on the best in-training eval-loss checkpoint.",
        "- Selection order when artifact scores are available for all candidates: mean artifact score, mean SSIM, mean LPIPS, mean PSNR, mean eval loss, mean total time.",
        "- Selection order without artifact scores: mean SSIM, mean LPIPS, mean PSNR, mean eval loss, mean total/train time.",
        "",
    ]
    if stats:
        winner = next(item for item in reversed(stats) if item.carried_forward)
        best_by_ssim = max(stats, key=lambda item: item.max_ssim)
        best_by_lpips = min(stats, key=lambda item: item.min_lpips)
        best_by_psnr = max(stats, key=lambda item: item.max_psnr)
        best_by_loss = min(stats, key=lambda item: item.min_eval_loss)
        lines.extend(
            [
                "## Results",
                "",
                f"- Current best carried config: `{json.dumps(winner.candidate.config, sort_keys=True)}`",
                f"- Mean metrics: artifact score `{fmt(winner.mean_artifact_score)}`, "
                f"SSIM `{winner.mean_ssim:.6f}`, LPIPS `{winner.mean_lpips:.6f}`, "
                f"PSNR `{winner.mean_psnr:.6f}`, eval loss `{winner.mean_eval_loss:.8f}`, "
                f"training time `{winner.mean_train_seconds:.3f}s`, total time `{fmt(winner.mean_total_seconds)}` seconds",
                f"- Best SSIM candidate: `{best_by_ssim.candidate.stage}/{best_by_ssim.candidate.param}={best_by_ssim.candidate.value}` "
                f"with max SSIM `{best_by_ssim.max_ssim:.6f}`",
                f"- Best LPIPS candidate: `{best_by_lpips.candidate.stage}/{best_by_lpips.candidate.param}={best_by_lpips.candidate.value}` "
                f"with min LPIPS `{best_by_lpips.min_lpips:.6f}`",
                f"- Best PSNR candidate: `{best_by_psnr.candidate.stage}/{best_by_psnr.candidate.param}={best_by_psnr.candidate.value}` "
                f"with max PSNR `{best_by_psnr.max_psnr:.6f}`",
                f"- Best eval-loss candidate: `{best_by_loss.candidate.stage}/{best_by_loss.candidate.param}={best_by_loss.candidate.value}` "
                f"with min eval loss `{best_by_loss.min_eval_loss:.8f}`",
                f"- Best single run render directory: `{best_single_run(winner).render_dir}`",
                "",
                "| Stage | Param | Value | Rank | Carried | Mean Artifact | Min Artifact | Mean SSIM | Max SSIM | Mean LPIPS | Min LPIPS | Mean PSNR | Max PSNR | Mean Eval Loss | Min Eval Loss | Mean Train s | Mean Eval s | Mean Artifact s | Mean Total s | Config |",
                "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in stats:
            lines.append(
                f"| {item.candidate.stage} | {item.candidate.param} | `{item.candidate.value}` | "
                f"{item.rank or ''} | {'yes' if item.carried_forward else 'no'} | "
                f"{fmt(item.mean_artifact_score)} | {fmt(item.min_artifact_score)} | "
                f"{item.mean_ssim:.6f} | {item.max_ssim:.6f} | "
                f"{item.mean_lpips:.6f} | {item.min_lpips:.6f} | "
                f"{item.mean_psnr:.6f} | {item.max_psnr:.6f} | "
                f"{item.mean_eval_loss:.8f} | {item.min_eval_loss:.8f} | "
                f"{item.mean_train_seconds:.3f} | {fmt(item.mean_eval_seconds)} | "
                f"{fmt(item.mean_artifact_seconds)} | {fmt(item.mean_total_seconds)} | "
                f"`{json.dumps(item.candidate.config, sort_keys=True)}` |"
            )
        lines.extend(["", "## Per-run results", ""])
        lines.extend(
            [
                "| Timestamp | Stage | Param | Value | Seed | Checkpoint | Eval Loss | PSNR | SSIM | LPIPS | Artifact | Train s | Eval s | Artifact s | Total s | Eval JSON | Renders |",
                "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for item in stats:
            for run in sorted(item.runs, key=lambda rec: rec.seed):
                lines.append(
                    f"| {run.timestamp} | {item.candidate.stage} | {item.candidate.param} | `{item.candidate.value}` | "
                    f"{run.seed} | `{run.checkpoint}` | {fmt(run.eval_loss)} | {fmt(run.psnr)} | "
                    f"{fmt(run.ssim)} | {fmt(run.lpips)} | {fmt(run.artifact_score)} | "
                    f"{fmt(run.train_seconds)} | {fmt(run.eval_seconds)} | {fmt(run.artifact_seconds)} | "
                    f"{fmt(run.total_seconds)} | "
                    f"`{run.eval_json}` | `{run.render_dir}` |"
                )
        lines.extend(["", "## Insights", "", "- Pending visual inspection and follow-up fix experiments."])
    else:
        lines.extend(["## Results", "", "- No completed runs yet."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_single_run(stats: CandidateStats) -> RunRecord:
    if all(run.artifact_score is not None for run in stats.runs):
        return min(
            stats.runs,
            key=lambda run: (
                required(run.artifact_score),
                -required(run.ssim),
                required(run.lpips),
                -required(run.psnr),
            ),
        )
    return max(stats.runs, key=lambda run: (required(run.ssim), -required(run.lpips), required(run.psnr)))


def maybe_run_stage(args: argparse.Namespace, requested: str, stage: str) -> bool:
    return requested in ("all", stage)


def dry_run_candidates(args: argparse.Namespace, stage: str, current: Dict[str, Any]) -> List[Candidate]:
    candidates: List[Candidate] = []
    if stage in ("all", "baseline"):
        candidates.append(make_candidate("control", "current", "baseline", current))
    if stage in ("all", "schedule"):
        candidates.extend(
            make_candidate("stage1", "max_res_base", value, with_value(current, "max_res_base", value))
            for value in (1024.0, 2048.0, 4096.0)
        )
        candidates.extend(
            make_candidate("stage1", "num_frequency_levels", value, with_value(current, "num_frequency_levels", value))
            for value in (12, 16)
        )
    if stage in ("all", "grid"):
        candidates.extend(
            make_candidate("stage2", "grid_resolution", value, with_value(current, "grid_resolution", value))
            for value in (64, 128, 192, 256)
        )
    if stage in ("all", "update"):
        for param, values in (
            ("grid_update_interval", parse_int_list(args.update_intervals)),
            ("grid_update_batch_size", parse_int_list(args.update_batch_sizes)),
        ):
            candidates.extend(make_candidate("stage3", param, value, with_value(current, param, value)) for value in values)
    if stage in ("all", "fallback"):
        candidates.extend(
            make_candidate("stage4", "fallback_frequency_level", value, with_value(current, "fallback_frequency_level", value))
            for value in (0.0, 7.5, 15.0)
        )
    return candidates


def print_dry_run_commands(args: argparse.Namespace, current: Dict[str, Any]) -> None:
    for candidate in dry_run_candidates(args, args.stage, current):
        for seed in SEEDS:
            record = RunRecord(
                candidate=candidate,
                seed=seed,
                timestamp=timestamp_for(candidate, seed),
                run_dir=run_dir(args, timestamp_for(candidate, seed)),
                wrapper_log=run_dir(args, timestamp_for(candidate, seed)) / "sweep_wrapper_stdout.log",
            )
            print("dry_run " + " ".join(runner_command(args, record)), flush=True)


def main() -> int:
    args = parse_args()
    max_parallel = max(1, args.max_parallel)
    all_stats: List[CandidateStats] = []
    current = apply_base_overrides(base_config(), args)
    print(f"report={args.report_path}", flush=True)
    print(f"max_parallel={max_parallel}", flush=True)
    if args.dry_run:
        print_dry_run_commands(args, current)
        return 0

    if maybe_run_stage(args, args.stage, "baseline"):
        control = make_candidate("control", "current", "baseline", current)
        control_runs, max_parallel = run_candidate_groups(args, [control], max_parallel)
        best = summarize_stage(args, all_stats, [control], control_runs)
        current = dict(best.candidate.config)

    if maybe_run_stage(args, args.stage, "schedule"):
        schedule_candidates = [
            make_candidate("stage1", "max_res_base", value, with_value(current, "max_res_base", value))
            for value in (1024.0, 2048.0, 4096.0)
        ]
        schedule_runs, max_parallel = run_candidate_groups(args, schedule_candidates, max_parallel)
        best = summarize_stage(args, all_stats, schedule_candidates, schedule_runs)
        current = dict(best.candidate.config)
        level_candidates = [
            make_candidate("stage1", "num_frequency_levels", value, with_value(current, "num_frequency_levels", value))
            for value in (12, 16)
        ]
        level_runs, max_parallel = run_candidate_groups(args, level_candidates, max_parallel)
        best = summarize_stage(args, all_stats, level_candidates, level_runs)
        current = dict(best.candidate.config)

    if maybe_run_stage(args, args.stage, "grid"):
        candidates = [
            make_candidate("stage2", "grid_resolution", value, with_value(current, "grid_resolution", value))
            for value in (64, 128, 192, 256)
        ]
        runs, max_parallel = run_candidate_groups(args, candidates, max_parallel)
        best = summarize_stage(args, all_stats, candidates, runs)
        current = dict(best.candidate.config)

    if maybe_run_stage(args, args.stage, "update"):
        for param, values in (
            ("grid_update_interval", parse_int_list(args.update_intervals)),
            ("grid_update_batch_size", parse_int_list(args.update_batch_sizes)),
        ):
            candidates = [make_candidate("stage3", param, value, with_value(current, param, value)) for value in values]
            runs, max_parallel = run_candidate_groups(args, candidates, max_parallel)
            best = summarize_stage(args, all_stats, candidates, runs)
            current = dict(best.candidate.config)

    if maybe_run_stage(args, args.stage, "fallback"):
        candidates = [
            make_candidate("stage4", "fallback_frequency_level", value, with_value(current, "fallback_frequency_level", value))
            for value in (0.0, 7.5, 15.0)
        ]
        runs, max_parallel = run_candidate_groups(args, candidates, max_parallel)
        summarize_stage(args, all_stats, candidates, runs)

    write_report(args.report_path, args.data, all_stats)
    print(f"final_report={args.report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
