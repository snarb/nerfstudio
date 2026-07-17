#!/usr/bin/env python3
"""Sequential static LookCloser transfer over per-frame temporal datasets.

This runner intentionally starts from the archived single-frame LookCloser
leader config and mutates only run-control fields, data path, checkpoint path,
and LR/scheduler settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/home/brans/temporal_perframe_stride7_45f")
DEFAULT_ARTIFACT = Path("/home/brans/lookcloser_temporal_artifacts/static_lookcloser_leader_007740")
DEFAULT_LEADER_CONFIG = DEFAULT_ARTIFACT / "config.yml"
DEFAULT_LEADER_CHECKPOINT = DEFAULT_ARTIFACT / "nerfstudio_models" / "step-000106316.ckpt"
DEFAULT_OUTPUT = Path("/home/brans/lookcloser_temporal_runs")
DEFAULT_REPORT = REPO_ROOT / "LookCloser" / "experiments" / "temporal_lookcloser_transfer.md"
DEFAULT_STEP_INTERVAL = 15188
DEFAULT_MAX_DELTA_STEPS = 50000
DEFAULT_TRANSFER_LR = 5e-4
DEFAULT_TRANSFER_SCHEDULER = "constant"
LEADER_PSNR = 29.617965698242188
LEADER_SSIM = 0.6684514880180359


@dataclass(frozen=True)
class Schedule:
    label: str
    lr: float
    kind: str
    lr_final: Optional[float] = None


@dataclass
class RunResult:
    frame: str
    label: str
    run_dir: Path
    checkpoint: Optional[Path]
    reason: str
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    eval_json: Optional[Path] = None
    render_dir: Optional[Path] = None
    train_seconds: Optional[float] = None
    status: str = "complete"
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--leader-config", type=Path, default=DEFAULT_LEADER_CONFIG)
    parser.add_argument("--leader-checkpoint", type=Path, default=DEFAULT_LEADER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--mode", choices=("sanity", "sweep", "chain", "all"), default="all")
    parser.add_argument("--skip-frame", default="007740")
    parser.add_argument("--sweep-frame", default="007747")
    parser.add_argument("--start-frame", default=None)
    parser.add_argument("--end-frame", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--chosen-lr", type=float, default=DEFAULT_TRANSFER_LR)
    parser.add_argument("--chosen-lr-final", type=float, default=None)
    parser.add_argument("--chosen-scheduler", choices=("exp", "constant"), default=DEFAULT_TRANSFER_SCHEDULER)
    parser.add_argument(
        "--lr-candidates",
        default="exp1e-3:0.001:exp:0.0001,exp5e-4:0.0005:exp:0.00005,exp2e-4:0.0002:exp:0.00002,exp1e-4:0.0001:exp:0.00001,const5e-4:0.0005:constant:",
        help="Comma-separated label:lr:kind:lr_final candidates.",
    )
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--max-delta-steps", type=int, default=DEFAULT_MAX_DELTA_STEPS)
    parser.add_argument("--step-interval", type=int, default=DEFAULT_STEP_INTERVAL)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--psnr-min-gain", type=float, default=0.03)
    parser.add_argument("--psnr-tie-db", type=float, default=0.05)
    parser.add_argument("--ssim-min-gain", type=float, default=0.001)
    parser.add_argument("--lpips-min-gain", type=float, default=0.003)
    parser.add_argument("--plateau-evals", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["PATH"] = f"/home/brans/repos/nerfstudio/.venv/bin:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/home/brans/.cache/torch_extensions_lookcloser")
    return env


def checkpoint_step(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def frames(data_root: Path, skip: str, start: Optional[str], end: Optional[str]) -> List[str]:
    names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    selected = [name for name in names if name != skip]
    if start is not None:
        selected = [name for name in selected if name >= start]
    if end is not None:
        selected = [name for name in selected if name <= end]
    return selected


def parse_schedules(text: str) -> List[Schedule]:
    schedules: List[Schedule] = []
    for raw in text.split(","):
        if not raw.strip():
            continue
        label, lr, kind, lr_final = raw.split(":", 3)
        schedules.append(
            Schedule(
                label=label,
                lr=float(lr),
                kind=kind,
                lr_final=float(lr_final) if lr_final else None,
            )
        )
    return schedules


def load_leader_config(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.Loader)


def set_nested_data_path(config, data_path: Path) -> None:
    config.data = None
    config.pipeline.datamanager.data = None
    config.pipeline.datamanager.dataparser.data = data_path


def configure_run(
    leader_config: Path,
    frame: str,
    data_path: Path,
    checkpoint: Path,
    output_dir: Path,
    experiment_name: str,
    timestamp: str,
    schedule: Schedule,
    max_delta_steps: int,
    step_interval: int,
):
    config = load_leader_config(leader_config)
    start_step = checkpoint_step(checkpoint)
    config.output_dir = output_dir
    config.experiment_name = experiment_name
    config.timestamp = timestamp
    config.load_checkpoint = checkpoint
    config.load_dir = None
    config.load_step = None
    config.load_config = None
    config.load_optimizers = False
    config.load_scheduler = False
    config.save_only_latest_checkpoint = False
    config.max_num_iterations = start_step + max_delta_steps
    config.steps_per_eval_batch = step_interval
    config.steps_per_eval_image = step_interval
    config.steps_per_eval_all_images = step_interval
    config.steps_per_save = step_interval
    config.logging.csv_writer.enable = True
    config.logging.csv_writer.write_interval = step_interval
    config.logging.csv_writer.improvement_tolerance = 0.0
    config.logging.local_writer.enable = False
    config.logging.profiler = "none"
    config.viewer.quit_on_train_completion = True
    set_nested_data_path(config, data_path)

    config.optimizers["fields"]["optimizer"].lr = schedule.lr
    if schedule.kind == "constant":
        config.optimizers["fields"]["scheduler"] = None
    elif schedule.kind == "exp":
        scheduler = config.optimizers["fields"]["scheduler"]
        scheduler.max_steps = max_delta_steps
        scheduler.warmup_steps = 0
        scheduler.lr_final = schedule.lr_final if schedule.lr_final is not None else schedule.lr / 10.0
    else:
        raise ValueError(f"Unknown scheduler kind: {schedule.kind}")
    return config


def run_dir(output_dir: Path, experiment_name: str, timestamp: str) -> Path:
    return output_dir / experiment_name / "lookcloser" / timestamp


def train_config_subprocess(config_path: Path) -> List[str]:
    code = (
        "import yaml; "
        "from pathlib import Path; "
        "from nerfstudio.scripts.train import main; "
        f"cfg=yaml.load(Path({str(config_path)!r}).read_text(), Loader=yaml.Loader); "
        "main(cfg)"
    )
    return [sys.executable, "-c", code]


def save_input_config(config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(config), encoding="utf-8")


def read_rows(metrics_path: Path) -> List[Dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eval_rows(metrics_path: Path) -> List[Dict[str, str]]:
    return [row for row in read_rows(metrics_path) if row.get("eval_all_psnr")]


def latest_step(metrics_path: Path) -> Optional[str]:
    rows = read_rows(metrics_path)
    return rows[-1]["step"] if rows else None


def row_metric(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def row_improves(row: Dict[str, str], best: Optional[Dict[str, str]], args: argparse.Namespace) -> bool:
    if best is None:
        return True
    psnr = row_metric(row, "eval_all_psnr")
    best_psnr = row_metric(best, "eval_all_psnr")
    if psnr is None or best_psnr is None:
        return False
    if psnr > best_psnr + args.psnr_min_gain:
        return True
    if psnr < best_psnr - args.psnr_tie_db:
        return False
    ssim = row_metric(row, "eval_all_ssim")
    best_ssim = row_metric(best, "eval_all_ssim")
    lpips = row_metric(row, "eval_all_lpips")
    best_lpips = row_metric(best, "eval_all_lpips")
    if ssim is not None and best_ssim is not None and ssim > best_ssim + args.ssim_min_gain:
        return True
    if lpips is not None and best_lpips is not None and lpips < best_lpips - args.lpips_min_gain:
        return True
    return False


def print_eval(prefix: str, row: Dict[str, str]) -> None:
    print(
        f"{prefix} eval "
        f"step={row.get('step')} "
        f"psnr={row.get('eval_all_psnr')} "
        f"ssim={row.get('eval_all_ssim')} "
        f"lpips={row.get('eval_all_lpips')} "
        f"status={row.get('status')}",
        flush=True,
    )


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def latest_checkpoint(model_dir: Path) -> Optional[Path]:
    checkpoints = sorted(model_dir.glob("step-*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def best_checkpoint(metrics_path: Path, model_dir: Path, args: argparse.Namespace) -> Tuple[Optional[Path], str]:
    rows = eval_rows(metrics_path)
    checkpoints = sorted(model_dir.glob("step-*.ckpt"))
    if not checkpoints:
        return None, "missing"
    if not rows:
        return checkpoints[-1], "latest_no_eval_rows"
    best: Optional[Dict[str, str]] = None
    for row in rows:
        if row_improves(row, best, args):
            best = row
    assert best is not None
    target = int(float(best["step"]))
    by_step = {checkpoint_step(path): path for path in checkpoints}
    if target in by_step:
        return by_step[target], f"best_metrics_step_{target}"
    earlier = [step for step in by_step if step <= target]
    if earlier:
        step = max(earlier)
        return by_step[step], f"nearest_saved_for_best_metrics_step_{target}"
    return checkpoints[-1], f"latest_no_checkpoint_for_best_metrics_step_{target}"


def eval_config_for_step(config_path: Path, checkpoint: Path) -> Path:
    step = checkpoint_step(checkpoint)
    text = config_path.read_text(encoding="utf-8")
    if re.search(r"^load_step:", text, flags=re.MULTILINE):
        text = re.sub(r"^load_step:.*$", f"load_step: {step}", text, count=1, flags=re.MULTILINE)
    else:
        text = text.replace("load_scheduler:", f"load_step: {step}\nload_scheduler:", 1)
    out = config_path.with_name(f"eval_config_step_{step:09d}.yml")
    out.write_text(text, encoding="utf-8")
    return out


def run_eval(run_path: Path, checkpoint: Path, label: str) -> Tuple[Optional[Path], Optional[Path], Optional[Dict[str, float]]]:
    config_path = run_path / "config.yml"
    eval_config = eval_config_for_step(config_path, checkpoint)
    output_json = run_path / f"eval_{label}_{checkpoint.stem}.json"
    render_dir = run_path / f"renders_{label}_{checkpoint.stem}"
    log_path = run_path / f"eval_{label}_stdout.log"
    cmd = [
        "ns-eval",
        "--load-config",
        str(eval_config),
        "--output-path",
        str(output_json),
        "--render-output-path",
        str(render_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=REPO_ROOT, env=run_env(), stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads(output_json.read_text(encoding="utf-8"))
    results = data.get("results", {})
    metrics = {
        "psnr": float(results["psnr"]),
        "ssim": float(results["ssim"]),
        "lpips": float(results["lpips"]) if "lpips" in results else float("nan"),
    }
    print(
        f"final label={label} checkpoint={checkpoint} "
        f"psnr={metrics['psnr']:.6f} ssim={metrics['ssim']:.6f} lpips={metrics['lpips']:.6f} "
        f"renders={render_dir}",
        flush=True,
    )
    return output_json, render_dir, metrics


def train_one(
    args: argparse.Namespace,
    frame: str,
    checkpoint: Path,
    schedule: Schedule,
    experiment_name: str,
    timestamp: str,
) -> RunResult:
    data_path = args.data_root / frame
    run_path = run_dir(args.output_dir, experiment_name, timestamp)
    input_config = run_path / "input_config.yml"
    metrics_path = run_path / "metrics_compact.csv"
    model_dir = run_path / "nerfstudio_models"
    log_path = run_path / "train_stdout.log"
    config = configure_run(
        args.leader_config,
        frame,
        data_path,
        checkpoint,
        args.output_dir,
        experiment_name,
        timestamp,
        schedule,
        args.max_delta_steps,
        args.step_interval,
    )
    save_input_config(config, input_config)
    print(f"train frame={frame} label={schedule.label} run_dir={run_path}", flush=True)
    print(f"init_checkpoint={checkpoint}", flush=True)
    print(f"input_config={input_config}", flush=True)
    if args.dry_run:
        return RunResult(frame, schedule.label, run_path, None, "dry_run", status="dry_run")

    start = time.monotonic()
    cmd = train_config_subprocess(input_config)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=run_env(), stdout=log, stderr=subprocess.STDOUT)
        seen = 0
        best: Optional[Dict[str, str]] = None
        no_improve = 0
        while proc.poll() is None:
            time.sleep(args.poll_seconds)
            step = latest_step(metrics_path)
            if step is not None:
                print(f"{frame}/{schedule.label} step={step}", flush=True)
            current = eval_rows(metrics_path)
            for row in current[seen:]:
                print_eval(f"{frame}/{schedule.label}", row)
                if row_improves(row, best, args):
                    best = row
                    no_improve = 0
                else:
                    no_improve += 1
            if len(current) > seen:
                seen = len(current)
                if no_improve >= args.plateau_evals:
                    print(f"{frame}/{schedule.label} stopping plateau no_improve={no_improve}", flush=True)
                    stop_process(proc)
                    break
    train_seconds = time.monotonic() - start
    if proc.returncode not in (0, -signal.SIGINT):
        return RunResult(
            frame,
            schedule.label,
            run_path,
            latest_checkpoint(model_dir),
            "train_failed",
            train_seconds=train_seconds,
            status="failed",
            error=f"returncode={proc.returncode}; see {log_path}",
        )

    selected, reason = best_checkpoint(metrics_path, model_dir, args)
    result = RunResult(frame, schedule.label, run_path, selected, reason, train_seconds=train_seconds)
    if selected is not None:
        try:
            eval_json, render_dir, metrics = run_eval(run_path, selected, "selected")
            result.eval_json = eval_json
            result.render_dir = render_dir
            result.psnr = metrics["psnr"]
            result.ssim = metrics["ssim"]
            result.lpips = metrics["lpips"]
        except Exception as exc:  # noqa: BLE001
            result.status = "eval_failed"
            result.error = str(exc)
    write_run_summary(result)
    return result


def prepare_eval_only_run(
    args: argparse.Namespace,
    frame: str,
    checkpoint: Path,
    experiment_name: str,
    timestamp: str,
    schedule: Schedule,
) -> Path:
    run_path = run_dir(args.output_dir, experiment_name, timestamp)
    config = configure_run(
        args.leader_config,
        frame,
        args.data_root / frame,
        checkpoint,
        args.output_dir,
        experiment_name,
        timestamp,
        schedule,
        args.max_delta_steps,
        args.step_interval,
    )
    run_path.mkdir(parents=True, exist_ok=True)
    model_dir = run_path / "nerfstudio_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    linked = model_dir / checkpoint.name
    if not linked.exists():
        linked.symlink_to(checkpoint)
    config_path = run_path / "config.yml"
    config.load_checkpoint = None
    config.load_dir = None
    config.load_step = None
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return run_path


def sanity(args: argparse.Namespace) -> RunResult:
    schedule = Schedule("leader_sanity", 0.001, "exp", 0.0001)
    run_path = prepare_eval_only_run(
        args,
        args.skip_frame,
        args.leader_checkpoint,
        "temporal_lookcloser_sanity",
        f"{args.skip_frame}_leader_local_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        schedule,
    )
    eval_json, render_dir, metrics = run_eval(run_path, args.leader_checkpoint, "local_data")
    result = RunResult(
        args.skip_frame,
        "leader_sanity",
        run_path,
        args.leader_checkpoint,
        "leader_local_eval",
        psnr=metrics["psnr"] if metrics else None,
        ssim=metrics["ssim"] if metrics else None,
        lpips=metrics["lpips"] if metrics else None,
        eval_json=eval_json,
        render_dir=render_dir,
    )
    write_run_summary(result)
    return result


def result_key(result: RunResult) -> Tuple[float, float, float, float]:
    psnr = result.psnr if result.psnr is not None else float("-inf")
    ssim = result.ssim if result.ssim is not None else float("-inf")
    lpips = result.lpips if result.lpips is not None else float("inf")
    speed = -(result.train_seconds or float("inf"))
    return (psnr, ssim, -lpips, speed)


def choose_sweep_winner(results: List[RunResult]) -> RunResult:
    complete = [r for r in results if r.status == "complete" and r.checkpoint is not None and r.psnr is not None]
    if not complete:
        raise RuntimeError("No complete LR sweep candidates.")
    best_psnr = max(r.psnr or float("-inf") for r in complete)
    tied = [r for r in complete if best_psnr - (r.psnr or float("-inf")) <= 0.05]
    return max(tied, key=result_key)


def sweep(args: argparse.Namespace) -> Tuple[RunResult, Schedule, List[RunResult]]:
    schedules = parse_schedules(args.lr_candidates)
    init = args.init_checkpoint or args.leader_checkpoint
    experiment = "temporal_lookcloser_lr_sweep_007747"
    print(f"sweep frame={args.sweep_frame} candidates={','.join(s.label for s in schedules)}", flush=True)
    results: List[RunResult] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = []
        for schedule in schedules:
            timestamp = f"{args.sweep_frame}_{schedule.label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            futures.append(
                executor.submit(train_one, args, args.sweep_frame, init, schedule, experiment, timestamp)
            )
            time.sleep(2.0)
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"sweep_done label={result.label} status={result.status} "
                f"psnr={result.psnr} ssim={result.ssim} lpips={result.lpips} checkpoint={result.checkpoint}",
                flush=True,
            )
    winner = choose_sweep_winner(results)
    schedule = next(s for s in schedules if s.label == winner.label)
    print(f"sweep_winner label={winner.label} checkpoint={winner.checkpoint}", flush=True)
    return winner, schedule, results


def chain(args: argparse.Namespace, init_checkpoint: Path, schedule: Schedule, already_done_first: Optional[RunResult]) -> List[RunResult]:
    names = frames(args.data_root, args.skip_frame, args.start_frame, args.end_frame)
    if already_done_first is not None and already_done_first.frame in names:
        start_index = names.index(already_done_first.frame) + 1
        results = [already_done_first]
    else:
        start_index = 0
        results = []
    previous = init_checkpoint
    if already_done_first is not None and already_done_first.checkpoint is not None:
        previous = already_done_first.checkpoint
    for frame in names[start_index:]:
        timestamp = f"{frame}_from_{checkpoint_step(previous):09d}_{schedule.label}"
        result = train_one(args, frame, previous, schedule, "temporal_lookcloser_transfer_chain", timestamp)
        results.append(result)
        if args.dry_run:
            continue
        append_report(args.report_path, [result], title=f"Frame {frame}")
        if result.status != "complete" or result.checkpoint is None:
            raise RuntimeError(f"Stopping chain at {frame}: {result.status} {result.error}")
        if result.psnr is not None and (result.psnr < LEADER_PSNR - 0.8):
            if result.ssim is None or result.ssim < LEADER_SSIM:
                raise RuntimeError(f"Stopping chain at {frame}: PSNR {result.psnr:.3f} is far below leader {LEADER_PSNR:.3f}.")
            print(
                f"quality_warning frame={frame} psnr={result.psnr:.3f} leader_psnr={LEADER_PSNR:.3f} "
                f"ssim={result.ssim:.3f} leader_ssim={LEADER_SSIM:.3f}; continuing because SSIM is not degraded.",
                flush=True,
            )
        if result.ssim is not None and result.ssim < LEADER_SSIM - 0.04:
            raise RuntimeError(f"Stopping chain at {frame}: SSIM {result.ssim:.3f} is far below leader {LEADER_SSIM:.3f}.")
        previous = result.checkpoint
    return results


def write_run_summary(result: RunResult) -> None:
    data = {
        "frame": result.frame,
        "label": result.label,
        "status": result.status,
        "error": result.error,
        "run_dir": str(result.run_dir),
        "checkpoint": str(result.checkpoint) if result.checkpoint else None,
        "reason": result.reason,
        "psnr": result.psnr,
        "ssim": result.ssim,
        "lpips": result.lpips,
        "eval_json": str(result.eval_json) if result.eval_json else None,
        "render_dir": str(result.render_dir) if result.render_dir else None,
        "train_seconds": result.train_seconds,
    }
    result.run_dir.mkdir(parents=True, exist_ok=True)
    (result.run_dir / "run_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_report(path: Path, results: List[RunResult], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Temporal Static LookCloser Transfer\n\n"
            "## What was tested\n"
            "Sequential static LookCloser checkpoint transfer over per-frame temporal datasets. "
            "Reports use PSNR/SSIM as primary metrics; loss is not reported.\n\n"
            "## Results\n\n"
            "| Section | Frame | Label | PSNR | SSIM | LPIPS | Checkpoint | Renders |\n"
            "|---|---|---|---:|---:|---:|---|---|\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(
                f"| {title} | {r.frame} | {r.label} | "
                f"{r.psnr if r.psnr is not None else ''} | "
                f"{r.ssim if r.ssim is not None else ''} | "
                f"{r.lpips if r.lpips is not None else ''} | "
                f"`{r.checkpoint or ''}` | `{r.render_dir or ''}` |\n"
            )


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.data_root, args.leader_config, args.leader_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.leader_checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"Leader checkpoint looks incomplete: {args.leader_checkpoint} ({args.leader_checkpoint.stat().st_size} bytes)")
    frame_names = frames(args.data_root, args.skip_frame, args.start_frame, args.end_frame)
    if not frame_names:
        raise RuntimeError("No frames selected.")
    for name in [args.skip_frame, args.sweep_frame, *frame_names[:1]]:
        data_path = args.data_root / name
        if data_path.exists():
            tf = data_path / "transforms.json"
            freqs = data_path / "lookcloser_frequencies"
            if not tf.exists() or len(list(freqs.glob("*.pt"))) != 66:
                raise RuntimeError(f"Bad frame data: {data_path}")


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    sanity_result: Optional[RunResult] = None
    sweep_winner: Optional[RunResult] = None
    chosen_schedule: Optional[Schedule] = None

    if args.mode in ("sanity", "all"):
        if args.dry_run:
            print("dry-run sanity", flush=True)
        else:
            sanity_result = sanity(args)
            append_report(args.report_path, [sanity_result], "Sanity")
            if sanity_result.psnr is not None and sanity_result.psnr < LEADER_PSNR - 0.5:
                raise RuntimeError(
                    f"Local 007740 sanity PSNR {sanity_result.psnr:.3f} is too far below leader {LEADER_PSNR:.3f}."
                )
        if args.mode == "sanity":
            return 0

    if args.mode in ("sweep", "all"):
        sweep_winner, chosen_schedule, sweep_results = sweep(args)
        append_report(args.report_path, sweep_results, "LR sweep")
        if args.mode == "sweep":
            return 0

    if args.mode in ("chain", "all"):
        if chosen_schedule is None:
            if args.chosen_lr is None:
                raise ValueError("--chosen-lr is required for --mode chain without sweep.")
            chosen_schedule = Schedule(
                f"{args.chosen_scheduler}{args.chosen_lr:g}".replace(".", "p"),
                args.chosen_lr,
                args.chosen_scheduler,
                args.chosen_lr_final if args.chosen_scheduler == "exp" else None,
            )
        init_checkpoint = args.init_checkpoint or args.leader_checkpoint
        already_done = sweep_winner if args.mode == "all" else None
        if already_done is not None and already_done.checkpoint is not None:
            init_checkpoint = already_done.checkpoint
        chain_results = chain(args, init_checkpoint, chosen_schedule, already_done)
        append_report(args.report_path, chain_results, "Chain summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
