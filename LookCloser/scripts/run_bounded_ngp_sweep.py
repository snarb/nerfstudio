#!/usr/bin/env python3
"""Staged bounded Instant-NGP hyperparameter sweep orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "LookCloser" / "scripts" / "run_bounded_ngp_quiet.py"
DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_OUTPUT = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_runs")
DEFAULT_EXPERIMENT = "007740_hd_aabb4_multicamera_eval3_ns_focus_scene15"
DEFAULT_REPORT = REPO_ROOT / "LookCloser" / "experiments" / "bounded_ngp_param_sweep.md"
SEEDS = (42, 43, 44)
MAX_NUM_ITERATIONS = 80000


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
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=2048)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--force-stage3", action="store_true")
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
    return args.output_dir / args.experiment_name / "instant-ngp-bounded" / timestamp


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
        record.psnr = float(results["psnr"])
        record.ssim = float(results["ssim"])
        record.lpips = float(results["lpips"])
        record.status = "complete"
    except Exception as exc:  # noqa: BLE001 - preserve the parse failure in the report.
        record.status = "parse_error"
        record.error = str(exc)
    return record


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
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def add_config_args(cmd: List[str], config: Dict[str, Any]) -> None:
    mapping = {
        "scene_scale": "--scene-scale",
        "center_method": "--center-method",
        "orientation_method": "--orientation-method",
        "scale_factor": "--scale-factor",
        "near_plane": "--near-plane",
        "far_plane": "--far-plane",
        "render_step_size_mult": "--render-step-size-mult",
        "alpha_thre": "--alpha-thre",
        "cone_angle": "--cone-angle",
        "background_color": "--background-color",
        "train_num_rays_per_batch": "--train-num-rays-per-batch",
    }
    for key, flag in mapping.items():
        value = config.get(key)
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    if config.get("use_gradient_scaling"):
        cmd.append("--use-gradient-scaling")


def start_job(args: argparse.Namespace, record: RunRecord) -> ActiveJob:
    record.run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = record.wrapper_log.open("a", encoding="utf-8")
    cmd = runner_command(args, record)
    log_handle.write(f"\n# sweep command: {' '.join(cmd)}\n")
    log_handle.flush()
    if args.dry_run:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    else:
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
                print(
                    f"OOM detected in {parsed.timestamp}; reducing parallelism to {current_parallel} and retrying",
                    flush=True,
                )
                archive_failed_run(parsed)
                retry = completed_record(args, parsed.candidate, parsed.seed)
                pending.insert(0, retry)
            else:
                complete.append(parsed)
                print(f"failed {parsed.timestamp}: {parsed.error}", flush=True)
        active = still_active

        now = time.time()
        if now - last_report >= max(args.poll_seconds, 60.0):
            print_progress(active, pending, complete)
            last_report = now

    return complete, current_parallel


def run_candidate_group(args: argparse.Namespace, candidate: Candidate, max_parallel: int) -> tuple[List[RunRecord], int]:
    records = stage_records(args, [candidate])
    return run_records(args, records, min(max_parallel, len(SEEDS)))


def run_candidate_groups(
    args: argparse.Namespace, candidates: List[Candidate], max_parallel: int
) -> tuple[List[RunRecord], int]:
    all_runs: List[RunRecord] = []
    current_parallel = max_parallel
    for candidate in candidates:
        print(
            f"candidate {candidate.stage}/{candidate.param}={candidate.value}: "
            f"running seeds {', '.join(str(seed) for seed in SEEDS)} in parallel",
            flush=True,
        )
        runs, current_parallel = run_candidate_group(args, candidate, current_parallel)
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
        f"lpips={fmt(record.lpips)}"
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
    )


def required(value: Optional[float]) -> float:
    if value is None:
        raise RuntimeError("missing metric")
    return value


def rank_candidates(stats: List[CandidateStats]) -> List[CandidateStats]:
    psnr_rank = rank_map(stats, "mean_psnr", reverse=True)
    ssim_rank = rank_map(stats, "mean_ssim", reverse=True)
    ranked = sorted(
        stats,
        key=lambda item: (
            psnr_rank[id(item)] + ssim_rank[id(item)],
            -item.max_psnr,
            item.mean_lpips,
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
        "scene_scale": 1.5,
        "center_method": "focus",
        "orientation_method": "up",
        "scale_factor": None,
        "near_plane": 0.01,
        "far_plane": 1000.0,
        "render_step_size_mult": None,
        "alpha_thre": 0.0,
        "cone_angle": 0.0,
        "background_color": "black",
        "use_gradient_scaling": False,
        "train_num_rays_per_batch": 8192,
    }


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
    write_report(args.report_path, all_stats)
    print_stage_leaderboard(ranked)
    return ranked[0]


def print_stage_leaderboard(stats: Sequence[CandidateStats]) -> None:
    print("leaderboard", flush=True)
    for item in stats:
        print(
            f"  #{item.rank} {item.candidate.stage}/{item.candidate.param}={item.candidate.value} "
            f"mean_psnr={item.mean_psnr:.6f} mean_ssim={item.mean_ssim:.6f} "
            f"mean_lpips={item.mean_lpips:.6f} mean_loss={item.mean_eval_loss:.8f}",
            flush=True,
        )


def has_clear_winner(control: CandidateStats, current: CandidateStats) -> bool:
    return current.mean_psnr > control.max_psnr and current.mean_ssim > control.max_ssim


def write_report(path: Path, stats: Sequence[CandidateStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bounded Instant-NGP Parameter Sweep",
        "",
        f"Dataset: `{DEFAULT_DATA}`",
        "",
        f"Max iterations per run: `{MAX_NUM_ITERATIONS}`. Checkpoint protocol: best eval-loss checkpoint.",
        "",
    ]
    if stats:
        winner = next(item for item in reversed(stats) if item.carried_forward)
        lines.extend(
            [
                "## Current Recommendation",
                "",
                f"- Best tested config: `{json.dumps(winner.candidate.config, sort_keys=True)}`",
                f"- Mean metrics: PSNR `{winner.mean_psnr:.6f}`, SSIM `{winner.mean_ssim:.6f}`, "
                f"LPIPS `{winner.mean_lpips:.6f}`, eval loss `{winner.mean_eval_loss:.8f}`",
                f"- Best single run: `{best_single_run(winner).timestamp}`",
                f"- Render directory: `{best_single_run(winner).render_dir}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Candidate Results",
            "",
            "| Stage | Param | Value | Rank | Carried | Mean PSNR | Max PSNR | Mean SSIM | Max SSIM | Mean LPIPS | Min LPIPS | Mean Eval Loss | Min Eval Loss | Max Eval Loss | Config |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in stats:
        lines.append(
            f"| {item.candidate.stage} | {item.candidate.param} | `{item.candidate.value}` | "
            f"{item.rank or ''} | {'yes' if item.carried_forward else 'no'} | "
            f"{item.mean_psnr:.6f} | {item.max_psnr:.6f} | "
            f"{item.mean_ssim:.6f} | {item.max_ssim:.6f} | "
            f"{item.mean_lpips:.6f} | {item.min_lpips:.6f} | "
            f"{item.mean_eval_loss:.8f} | {item.min_eval_loss:.8f} | {item.max_eval_loss:.8f} | "
            f"`{json.dumps(item.candidate.config, sort_keys=True)}` |"
        )
    lines.extend(["", "## Per-Run Results", ""])
    lines.extend(
        [
            "| Timestamp | Stage | Param | Value | Seed | Checkpoint | Eval Loss | PSNR | SSIM | LPIPS | Eval JSON | Renders |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in stats:
        for run in sorted(item.runs, key=lambda rec: rec.seed):
            lines.append(
                f"| {run.timestamp} | {item.candidate.stage} | {item.candidate.param} | `{item.candidate.value}` | "
                f"{run.seed} | `{run.checkpoint}` | {fmt(run.eval_loss)} | {fmt(run.psnr)} | "
                f"{fmt(run.ssim)} | {fmt(run.lpips)} | `{run.eval_json}` | `{run.render_dir}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_single_run(stats: CandidateStats) -> RunRecord:
    return max(stats.runs, key=lambda run: (required(run.psnr), required(run.ssim), -required(run.lpips)))


def main() -> int:
    args = parse_args()
    max_parallel = max(1, args.max_parallel)
    all_stats: List[CandidateStats] = []
    current = base_config()

    print(f"report={args.report_path}", flush=True)
    print(f"max_parallel={max_parallel}", flush=True)

    control = make_candidate("control", "scene_scale", 1.5, current)
    control_runs, max_parallel = run_candidate_groups(args, [control], max_parallel)
    control_best = summarize_stage(args, all_stats, [control], control_runs)

    scene_candidates = [
        make_candidate("stage1", "scene_scale", value, with_value(current, "scene_scale", value))
        for value in (1.0, 1.25, 1.5, 2.0, 2.5)
    ]
    scene_runs, max_parallel = run_candidate_groups(args, scene_candidates, max_parallel)
    scene_best = summarize_stage(args, all_stats, scene_candidates, scene_runs)
    current = dict(scene_best.candidate.config)

    final_stage2_best = scene_best
    for param, values in (
        ("render_step_size_mult", (0.5, 0.75, 1.0, 1.25)),
        ("near_plane", (0.005, 0.01, 0.02)),
        ("alpha_thre", (0.0, 0.0025, 0.005)),
        ("cone_angle", (0.0, 0.001, 0.00390625)),
    ):
        candidates = [
            make_candidate("stage2", param, value, with_value(current, param, value))
            for value in values
        ]
        runs, max_parallel = run_candidate_groups(args, candidates, max_parallel)
        best = summarize_stage(args, all_stats, candidates, runs)
        current = dict(best.candidate.config)
        final_stage2_best = best

    if args.force_stage3 or not has_clear_winner(control_best, final_stage2_best):
        for param, values in (
            ("center_method", ("focus", "poses")),
            ("orientation_method", ("up", "none")),
            ("scale_factor", (None, 0.85, 1.15)),
        ):
            candidates = [
                make_candidate("stage3", param, value, with_value(current, param, value))
                for value in values
            ]
            runs, max_parallel = run_candidate_groups(args, candidates, max_parallel)
            best = summarize_stage(args, all_stats, candidates, runs)
            current = dict(best.candidate.config)
    else:
        print("stage3 skipped: stage2 winner exceeds control max PSNR and max SSIM", flush=True)

    for param, values in (
        ("background_color", ("black", "random")),
        ("use_gradient_scaling", (False, True)),
        ("train_num_rays_per_batch", (8192, 12288, 16384)),
    ):
        candidates = [
            make_candidate("stage4", param, value, with_value(current, param, value))
            for value in values
        ]
        runs, max_parallel = run_candidate_groups(args, candidates, max_parallel)
        best = summarize_stage(args, all_stats, candidates, runs)
        current = dict(best.candidate.config)

    write_report(args.report_path, all_stats)
    print(f"final_report={args.report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
