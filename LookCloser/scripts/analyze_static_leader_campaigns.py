#!/usr/bin/env python3
"""Summarize exact static-leader campaigns and quantify trajectory variance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARCHIVE = {
    15_188: (28.5960, 0.651726, 0.371653),
    30_376: (29.2098, 0.676160, 0.305969),
    45_564: (29.3952, 0.673553, 0.279821),
    60_752: (29.5279, 0.677030, 0.262007),
    75_940: (29.6217, 0.675272, 0.252857),
    91_128: (29.6920, 0.672744, 0.240396),
    106_316: (29.617966, 0.668451, 0.231120),
}
ARCHIVE_ADAPTIVE_POINTS = {
    15_188: 17_565_390_330,
    30_376: 50_671_038_830,
    45_564: 87_565_979_530,
    60_752: 126_587_891_630,
    75_940: 166_919_022_830,
    91_128: 208_139_736_330,
    106_316: 250_035_332_330,
}
# User-approved same-seed repeat ranges, relaxed on 2026-07-15 after the
# stable-occupancy and FP32 accumulation controls quantified Blackwell noise.
TOLERANCE = (0.06, 0.01, 0.005)
FINAL_GATE = {"psnr": 29.617964, "ssim": 0.668450, "lpips": 0.231135}
FIXED_WARMUP_STEPS = 4_096
FIXED_POINTS_PER_STEP = 4_096 * 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path, help="campaign.json files or directories")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def campaign_json(path: Path) -> Path:
    return path if path.name == "campaign.json" else path / "campaign.json"


def value(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    return float(raw) if raw else None


def read_metrics(
    path: Path,
) -> tuple[int | None, dict[int, dict[str, float]], int, dict[int, int]]:
    if not path.exists():
        return None, {}, 0, {}
    latest: int | None = None
    evaluations: dict[int, dict[str, float]] = {}
    sample_rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            latest = step if latest is None else max(latest, step)
            sample_count = value(row, "train_num_samples_per_batch")
            if sample_count is not None:
                sample_rows.append((step, sample_count))
            psnr = value(row, "eval_all_psnr")
            ssim = value(row, "eval_all_ssim")
            lpips = value(row, "eval_all_lpips")
            if psnr is not None and ssim is not None and lpips is not None:
                evaluations[step] = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    intervals = [b[0] - a[0] for a, b in zip(sample_rows, sample_rows[1:]) if b[0] > a[0]]
    logging_interval = int(statistics.median(intervals)) if intervals else 10
    adaptive_points = int(sum(count * logging_interval for _, count in sample_rows))
    points_at_eval = {
        step: int(sum(count * logging_interval for sample_step, count in sample_rows if sample_step <= step))
        for step in evaluations
    }
    return latest, evaluations, adaptive_points, points_at_eval


def load_campaign(path: Path) -> dict[str, Any]:
    manifest_path = campaign_json(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "campaign": manifest["campaign_name"],
        "manifest": str(manifest_path),
        "seed": manifest["seed"],
        "seed_policy": manifest.get("seed_policy", "unknown"),
        "status": manifest["status"],
        "evaluations": {},
        "adaptive_points_legacy": 0,
        "accepted_candidate": manifest.get("accepted_candidate"),
    }
    latest_steps = []
    stage_a_latest: int | None = None
    prior_stage_adaptive_points = 0
    for stage_name in ("stage_a", "stage_a_fw03"):
        run_path = Path(manifest[stage_name]["run_path"])
        latest, evaluations, points, points_at_eval = read_metrics(run_path / "metrics_compact.csv")
        if latest is not None:
            latest_steps.append(latest)
            if stage_name == "stage_a":
                stage_a_latest = latest
        for step, metrics in evaluations.items():
            cumulative_adaptive = prior_stage_adaptive_points + points_at_eval[step]
            result["evaluations"][str(step)] = {
                **metrics,
                "adaptive_points_legacy": cumulative_adaptive,
                "total_points_estimated": cumulative_adaptive + FIXED_WARMUP_STEPS * FIXED_POINTS_PER_STEP,
            }
        result["adaptive_points_legacy"] += points
        prior_stage_adaptive_points += points
    result["latest_step"] = max(latest_steps) if latest_steps else None
    completed_fixed_steps = (
        min(FIXED_WARMUP_STEPS, stage_a_latest + 1) if stage_a_latest is not None else 0
    )
    result["fixed_warmup_points"] = completed_fixed_steps * FIXED_POINTS_PER_STEP
    result["total_point_samples_estimated"] = (
        result["adaptive_points_legacy"] + result["fixed_warmup_points"]
    )
    return result


def summarize(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    common_steps = sorted(
        set.intersection(*(set(run["evaluations"]) for run in campaigns)) if campaigns else set(),
        key=int,
    )
    comparisons = []
    for step_key in common_steps:
        values = [run["evaluations"][step_key] for run in campaigns]
        ranges = {
            metric: max(v[metric] for v in values) - min(v[metric] for v in values)
            for metric in ("psnr", "ssim", "lpips")
        }
        std = {
            metric: statistics.pstdev(v[metric] for v in values) if len(values) > 1 else 0.0
            for metric in ("psnr", "ssim", "lpips")
        }
        step = int(step_key)
        archive = ARCHIVE.get(step)
        point_values = [v["adaptive_points_legacy"] for v in values]
        values_by_seed: dict[str, list[dict[str, float]]] = {}
        for run in campaigns:
            values_by_seed.setdefault(str(run["seed"]), []).append(run["evaluations"][step_key])
        same_seed_groups = []
        for seed, seed_values in sorted(values_by_seed.items(), key=lambda item: int(item[0])):
            if len(seed_values) < 2:
                continue
            seed_ranges = {
                metric: max(v[metric] for v in seed_values) - min(v[metric] for v in seed_values)
                for metric in ("psnr", "ssim", "lpips")
            }
            same_seed_groups.append(
                {
                    "seed": int(seed),
                    "count": len(seed_values),
                    "range": seed_ranges,
                    "std": {
                        metric: statistics.pstdev(v[metric] for v in seed_values)
                        for metric in ("psnr", "ssim", "lpips")
                    },
                    "tolerance_pass": seed_ranges["psnr"] <= TOLERANCE[0]
                    and seed_ranges["ssim"] <= TOLERANCE[1]
                    and seed_ranges["lpips"] <= TOLERANCE[2],
                }
            )
        comparisons.append(
            {
                "step": step,
                "values": values,
                "range": ranges,
                "std": std,
                "same_seed_groups": same_seed_groups,
                "same_seed_tolerance_pass": (
                    all(group["tolerance_pass"] for group in same_seed_groups)
                    if same_seed_groups
                    else None
                ),
                "archive": None
                if archive is None
                else {
                    "psnr": archive[0],
                    "ssim": archive[1],
                    "lpips": archive[2],
                    "adaptive_points_legacy": ARCHIVE_ADAPTIVE_POINTS[step],
                },
                "adaptive_points_legacy_min": min(point_values),
                "adaptive_points_legacy_max": max(point_values),
            }
        )
    random_runs = [run for run in campaigns if run["seed_policy"] == "random_recorded"]
    random_comparisons = []
    if len(random_runs) >= 2:
        random_common_steps = sorted(
            set.intersection(*(set(run["evaluations"]) for run in random_runs)),
            key=int,
        )
        for step_key in random_common_steps:
            values = [run["evaluations"][step_key] for run in random_runs]
            random_comparisons.append(
                {
                    "step": int(step_key),
                    "count": len(random_runs),
                    "seeds": [run["seed"] for run in random_runs],
                    "range": {
                        metric: max(value_[metric] for value_ in values)
                        - min(value_[metric] for value_ in values)
                        for metric in ("psnr", "ssim", "lpips")
                    },
                    "std": {
                        metric: statistics.pstdev(value_[metric] for value_ in values)
                        for metric in ("psnr", "ssim", "lpips")
                    },
                }
            )
    return {
        "campaigns": campaigns,
        "common_checkpoint_comparisons": comparisons,
        "random_seed_comparisons": random_comparisons,
    }


def fmt(value_: float | None, digits: int = 6) -> str:
    return "—" if value_ is None or not math.isfinite(value_) else f"{value_:.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Static leader campaign monitor",
        "",
        "Generated from campaign manifests and `metrics_compact.csv`; training-batch metrics are not quality evidence.",
        "",
        "| Campaign | Seed | Status | Latest step | Legacy ARM points | Total points incl. fixed warm-up |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for run in summary["campaigns"]:
        lines.append(
            f"| {run['campaign']} | {run['seed']} | {run['status']} | "
            f"{run['latest_step'] if run['latest_step'] is not None else '—'} | "
            f"{run['adaptive_points_legacy']:,} | {run['total_point_samples_estimated']:,} |"
        )
    lines += [
        "",
        "## Common full-eval checkpoints",
        "",
        "| Step | Current / archive ARM points | All-run PSNR range | All-run SSIM range | All-run LPIPS range | Repeated-seed range P/S/L | Same-seed tolerance | Archive PSNR / SSIM / LPIPS |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in summary["common_checkpoint_comparisons"]:
        archive = row["archive"]
        archive_text = "—" if archive is None else (
            f"{fmt(archive['psnr'], 4)} / {fmt(archive['ssim'])} / {fmt(archive['lpips'])}"
        )
        point_text = "—" if archive is None else (
            f"{row['adaptive_points_legacy_min'] / 1e9:.3f}–"
            f"{row['adaptive_points_legacy_max'] / 1e9:.3f} / "
            f"{archive['adaptive_points_legacy'] / 1e9:.3f} B"
        )
        repeated = "; ".join(
            f"seed{group['seed']} n={group['count']}: "
            f"{fmt(group['range']['psnr'])}/{fmt(group['range']['ssim'])}/{fmt(group['range']['lpips'])}"
            for group in row["same_seed_groups"]
        ) or "—"
        tolerance = row["same_seed_tolerance_pass"]
        lines.append(
            f"| {row['step']} | {point_text} | {fmt(row['range']['psnr'])} | {fmt(row['range']['ssim'])} | "
            f"{fmt(row['range']['lpips'])} | {repeated} | "
            f"{'pending' if tolerance is None else ('pass' if tolerance else 'FAIL')} | "
            f"{archive_text} |"
        )
    if not summary["common_checkpoint_comparisons"]:
        lines.append("| — | — | — | — | — | — | pending | — |")
    if summary["random_seed_comparisons"]:
        lines += [
            "",
            "## Random-recorded seed ensemble",
            "",
            "This table excludes explicit same-seed repeats, separating between-seed variation from CUDA repeat nondeterminism.",
            "",
            "| Step | n (seeds) | Range PSNR / SSIM / LPIPS | Population std PSNR / SSIM / LPIPS |",
            "|---:|---|---|---|",
        ]
        for row in summary["random_seed_comparisons"]:
            seeds = ",".join(str(seed) for seed in row["seeds"])
            lines.append(
                f"| {row['step']} | {row['count']} ({seeds}) | "
                f"{fmt(row['range']['psnr'])} / {fmt(row['range']['ssim'])} / {fmt(row['range']['lpips'])} | "
                f"{fmt(row['std']['psnr'])} / {fmt(row['std']['ssim'])} / {fmt(row['std']['lpips'])} |"
            )
    accepted = [run for run in summary["campaigns"] if run.get("accepted_candidate")]
    if accepted:
        lines += [
            "",
            "## Accepted scheduled candidates",
            "",
            "| Campaign | Checkpoint | PSNR | SSIM | LPIPS | Significant artifacts | Serious ROI | Numeric | Automatic | Detail reference |",
            "|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
        for run in accepted:
            candidate = run["accepted_candidate"]
            metrics = candidate["metrics"]
            artifacts = candidate.get("artifacts") or {}
            detail = candidate.get("detail") or {}
            detail_pass = (detail.get("reference_comparison") or {}).get("pass")
            lines.append(
                f"| {run['campaign']} | {Path(candidate['checkpoint']).name} | "
                f"{fmt(metrics['psnr'])} | {fmt(metrics['ssim'])} | {fmt(metrics['lpips'])} | "
                f"{artifacts.get('significant_artifact_count', '—')} | "
                f"{artifacts.get('roi_serious_count', '—')} | "
                f"{'pass' if candidate.get('numeric_pass') else 'FAIL'} | "
                f"{'pass' if candidate.get('automatic_pass') else 'FAIL'} | "
                f"{'—' if detail_pass is None else ('pass' if detail_pass else 'FAIL')} |"
            )
    lines += [
        "",
        "## Per-campaign trajectory",
        "",
        "| Campaign | Step | ARM points | PSNR (delta) | SSIM (delta) | LPIPS (delta) | Numeric gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    trajectory_rows = 0
    for run in summary["campaigns"]:
        for step_key, metrics in sorted(run["evaluations"].items(), key=lambda item: int(item[0])):
            trajectory_rows += 1
            step = int(step_key)
            archive = ARCHIVE.get(step)
            if archive:
                cells = [
                    f"{metrics['psnr']:.6f} ({metrics['psnr'] - archive[0]:+.6f})",
                    f"{metrics['ssim']:.6f} ({metrics['ssim'] - archive[1]:+.6f})",
                    f"{metrics['lpips']:.6f} ({metrics['lpips'] - archive[2]:+.6f})",
                ]
            else:
                cells = [fmt(metrics[name]) for name in ("psnr", "ssim", "lpips")]
            gate = "pass" if (
                metrics["psnr"] >= FINAL_GATE["psnr"]
                and metrics["ssim"] >= FINAL_GATE["ssim"]
                and metrics["lpips"] <= FINAL_GATE["lpips"]
            ) else "FAIL"
            lines.append(
                f"| {run['campaign']} | {step} | {metrics['adaptive_points_legacy'] / 1e9:.3f} B | "
                f"{cells[0]} | {cells[1]} | {cells[2]} | {gate} |"
            )
    if not trajectory_rows:
        lines.append("| — | — | — | — | — | — | pending |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary = summarize([load_campaign(path) for path in args.campaigns])
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered = markdown(summary)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
