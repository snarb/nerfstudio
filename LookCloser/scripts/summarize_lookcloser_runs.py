#!/usr/bin/env python3
"""Summarize manually launched LookCloser seed runs.

The quiet sweep runner already writes aggregate reports for runs it starts
itself. This helper covers guarded/manual experiment folders that contain one
subdirectory per seed with `run_summary.json` plus `metrics_compact.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


REQUIRED_METRICS = ("ssim", "lpips", "psnr", "eval_loss", "train_seconds")
OPTIONAL_METRICS = (
    "artifact_score",
    "serious_artifact_score",
    "artifact_roi_score",
    "artifact_roi_serious_score",
    "artifact_roi_serious_count",
    "stand_connector_score",
    "eval_seconds",
    "artifact_seconds",
    "total_seconds",
)
METRICS = REQUIRED_METRICS + OPTIONAL_METRICS


@dataclass(frozen=True)
class RunSummary:
    seed: int
    run_dir: Path
    selected_checkpoint: str
    selected_step: Optional[int]
    eval_loss: float
    psnr: float
    ssim: float
    lpips: float
    train_seconds: float
    artifact_score: Optional[float]
    serious_artifact_score: Optional[float]
    artifact_roi_score: Optional[float]
    artifact_roi_serious_score: Optional[float]
    artifact_roi_serious_count: Optional[float]
    stand_connector_score: Optional[float]
    eval_seconds: Optional[float]
    artifact_seconds: Optional[float]
    total_seconds: Optional[float]
    render_dir: str
    eval_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="Experiment directory containing seed run subdirectories.")
    parser.add_argument(
        "--pattern",
        default="**/run_summary.json",
        help="Glob pattern under experiment_dir for run_summary.json files.",
    )
    parser.add_argument("--expected-seeds", default="42,43,44", help="Comma-separated seeds required for complete output.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path for machine-readable summary.")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional path for Markdown summary.")
    return parser.parse_args()


def checkpoint_step(path_text: str) -> Optional[int]:
    match = re.search(r"step-(\d+)\.ckpt", path_text)
    return int(match.group(1)) if match else None


def compact_rows(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selected_eval_loss(run_dir: Path, selected_step: Optional[int]) -> float:
    rows = compact_rows(run_dir / "metrics_compact.csv")
    eval_rows = [row for row in rows if row.get("eval_loss")]
    if not eval_rows:
        raise RuntimeError(f"No eval_loss rows in {run_dir / 'metrics_compact.csv'}")
    if selected_step is not None:
        for row in eval_rows:
            if int(float(row["step"])) == selected_step:
                return float(row["eval_loss"])
    return min(float(row["eval_loss"]) for row in eval_rows)


def load_run(summary_path: Path) -> RunSummary:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    params = data.get("params") or {}
    eval_data = data.get("eval") or {}
    results = eval_data.get("results") or {}
    artifact = data.get("artifact") or eval_data.get("artifact") or {}
    roi = artifact.get("roi") if isinstance(artifact, dict) else {}
    if roi is None:
        roi = {}
    missing = [name for name in ("psnr", "ssim", "lpips") if name not in results]
    if missing:
        raise RuntimeError(f"{summary_path} missing eval results: {', '.join(missing)}")
    if data.get("train_seconds") is None:
        raise RuntimeError(f"{summary_path} missing train_seconds")
    selected_checkpoint = str(data.get("selected_checkpoint") or eval_data.get("checkpoint") or "")
    selected_step = checkpoint_step(selected_checkpoint)
    return RunSummary(
        seed=int(params["seed"]),
        run_dir=summary_path.parent,
        selected_checkpoint=selected_checkpoint,
        selected_step=selected_step,
        eval_loss=selected_eval_loss(summary_path.parent, selected_step),
        psnr=float(results["psnr"]),
        ssim=float(results["ssim"]),
        lpips=float(results["lpips"]),
        train_seconds=float(data["train_seconds"]),
        artifact_score=optional_float(artifact.get("artifact_score")),
        serious_artifact_score=optional_float(artifact.get("serious_artifact_score")),
        artifact_roi_score=optional_float(roi.get("roi_artifact_score")),
        artifact_roi_serious_score=optional_float(roi.get("roi_serious_artifact_score")),
        artifact_roi_serious_count=optional_float(roi.get("roi_serious_count")),
        stand_connector_score=optional_float(roi.get("stand_connector_score")),
        eval_seconds=optional_float(data.get("eval_seconds") or eval_data.get("eval_seconds")),
        artifact_seconds=optional_float(data.get("artifact_seconds") or artifact.get("artifact_seconds")),
        total_seconds=optional_float(data.get("total_seconds")),
        render_dir=str(eval_data.get("render_dir") or ""),
        eval_json=str(eval_data.get("eval_json") or ""),
    )


def optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def finite(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric: {value}")
    return value


def metric_value(run: RunSummary, metric: str) -> float:
    return finite(float(getattr(run, metric)))


def optional_metric_value(run: RunSummary, metric: str) -> Optional[float]:
    value = getattr(run, metric)
    if value is None:
        return None
    return finite(float(value))


def best_run(runs: Iterable[RunSummary], metric: str) -> RunSummary:
    reverse = metric in {"ssim", "psnr"}
    available = [run for run in runs if optional_metric_value(run, metric) is not None]
    if not available:
        raise RuntimeError(f"No values available for metric: {metric}")
    return sorted(available, key=lambda run: optional_metric_value(run, metric) or 0.0, reverse=reverse)[0]


def mean_optional(runs: list[RunSummary], metric: str) -> Optional[float]:
    values = [optional_metric_value(run, metric) for run in runs]
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def build_summary(runs: list[RunSummary]) -> dict:
    means = {metric: mean(metric_value(run, metric) for run in runs) for metric in REQUIRED_METRICS}
    means.update({metric: mean_optional(runs, metric) for metric in OPTIONAL_METRICS})
    best = {
        metric: {
            "seed": best_run(runs, metric).seed,
            "value": optional_metric_value(best_run(runs, metric), metric),
            "render_dir": best_run(runs, metric).render_dir,
        }
        for metric in METRICS
        if mean_optional(runs, metric) is not None
    }
    return {
        "num_runs": len(runs),
        "seeds": [run.seed for run in runs],
        "means": means,
        "best": best,
        "runs": [
            {
                "seed": run.seed,
                "selected_checkpoint": run.selected_checkpoint,
                "selected_step": run.selected_step,
                "eval_loss": run.eval_loss,
                "psnr": run.psnr,
                "ssim": run.ssim,
                "lpips": run.lpips,
                "train_seconds": run.train_seconds,
                "artifact_score": run.artifact_score,
                "serious_artifact_score": run.serious_artifact_score,
                "artifact_roi_score": run.artifact_roi_score,
                "artifact_roi_serious_score": run.artifact_roi_serious_score,
                "artifact_roi_serious_count": run.artifact_roi_serious_count,
                "stand_connector_score": run.stand_connector_score,
                "eval_seconds": run.eval_seconds,
                "artifact_seconds": run.artifact_seconds,
                "total_seconds": run.total_seconds,
                "render_dir": run.render_dir,
                "eval_json": run.eval_json,
            }
            for run in runs
        ],
    }


def markdown(summary: dict) -> str:
    lines = [
        "Per-run metrics:",
        "",
        "| Seed | Selected step | Eval loss | PSNR | SSIM | LPIPS | Artifact score | Serious artifact | ROI score | ROI serious score | ROI serious count | Stand connector | Train time | Eval time | Artifact time | Total time | Renders |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['seed']} | {run['selected_step'] or ''} | {run['eval_loss']:.8f} | "
            f"{run['psnr']:.6f} | {run['ssim']:.6f} | {run['lpips']:.6f} | "
            f"{fmt(run['artifact_score'])} | {fmt(run['serious_artifact_score'])} | "
            f"{fmt(run['artifact_roi_score'])} | {fmt(run['artifact_roi_serious_score'])} | "
            f"{fmt(run['artifact_roi_serious_count'])} | {fmt(run['stand_connector_score'])} | "
            f"{run['train_seconds']:.3f}s | "
            f"{fmt_seconds(run['eval_seconds'])} | {fmt_seconds(run['artifact_seconds'])} | "
            f"{fmt_seconds(run['total_seconds'])} | `{run['render_dir']}` |"
        )
    means = summary["means"]
    lines.extend(
        [
            "",
            "Mean metrics:",
            "",
            "| SSIM | LPIPS | PSNR | Eval loss | Artifact score | Serious artifact | ROI score | ROI serious score | ROI serious count | Stand connector | Train time | Eval time | Artifact time | Total time |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {means['ssim']:.6f} | {means['lpips']:.6f} | {means['psnr']:.6f} | "
                f"{means['eval_loss']:.8f} | {fmt(means['artifact_score'])} | "
                f"{fmt(means['serious_artifact_score'])} | {fmt(means['artifact_roi_score'])} | "
                f"{fmt(means['artifact_roi_serious_score'])} | {fmt(means['artifact_roi_serious_count'])} | "
                f"{fmt(means['stand_connector_score'])} | "
                f"{means['train_seconds']:.3f}s | {fmt_seconds(means['eval_seconds'])} | "
                f"{fmt_seconds(means['artifact_seconds'])} | {fmt_seconds(means['total_seconds'])} |"
            ),
            "",
            "Best single result by metric:",
            "",
            "| Metric | Best seed | Value | Render directory |",
            "|---|---:|---:|---|",
        ]
    )
    labels = {
        "ssim": "SSIM, higher better",
        "lpips": "LPIPS, lower better",
        "psnr": "PSNR, higher better",
        "eval_loss": "Eval loss, lower better",
        "artifact_score": "Artifact score, lower better",
        "serious_artifact_score": "Serious artifact score, lower better",
        "artifact_roi_score": "ROI artifact score, lower better",
        "artifact_roi_serious_score": "ROI serious artifact score, lower better",
        "artifact_roi_serious_count": "ROI serious count, lower better",
        "stand_connector_score": "Stand connector ROI, lower better",
        "train_seconds": "Train time, lower better",
        "eval_seconds": "Eval time, lower better",
        "artifact_seconds": "Artifact detector time, lower better",
        "total_seconds": "Total time, lower better",
    }
    for metric in (
        "artifact_score",
        "serious_artifact_score",
        "artifact_roi_score",
        "artifact_roi_serious_score",
        "artifact_roi_serious_count",
        "stand_connector_score",
        "ssim",
        "lpips",
        "psnr",
        "eval_loss",
        "train_seconds",
        "eval_seconds",
        "artifact_seconds",
        "total_seconds",
    ):
        item = summary["best"].get(metric)
        if item is None:
            continue
        suffix = "s" if metric.endswith("_seconds") else ""
        lines.append(f"| {labels[metric]} | {item['seed']} | {item['value']:.6f}{suffix} | `{item['render_dir']}` |")
    return "\n".join(lines) + "\n"


def fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def fmt_seconds(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def main() -> int:
    args = parse_args()
    expected = {int(seed.strip()) for seed in args.expected_seeds.split(",") if seed.strip()}
    summary_paths = sorted(args.experiment_dir.glob(args.pattern))
    runs = sorted((load_run(path) for path in summary_paths), key=lambda run: run.seed)
    seen = {run.seed for run in runs}
    missing = sorted(expected - seen)
    if missing:
        raise SystemExit(f"Missing expected seeds {missing}; found {sorted(seen)} under {args.experiment_dir}")
    summary = build_summary([run for run in runs if run.seed in expected])
    text = markdown(summary)
    print(text, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
