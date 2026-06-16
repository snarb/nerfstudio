#!/usr/bin/env python3
"""Build a compact artifact-gate report for selected LookCloser runs.

The report combines global eval metrics, full-frame artifact scores, serious
full-frame scores, and curated ROI scores. It is meant for artifact-sensitive
confirmation runs where the old full-frame score and the meaningful structural
ROI gate need to be read together.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

from audit_artifact_components import (
    DEFAULT_STRUCTURAL_EXCLUDE,
    aggregate as aggregate_components,
    classify_regions,
    eval_idx_from_view,
    view_rois,
)
from detect_structural_artifacts import PRESETS, detect_defects, detector_kwargs_from_args, load_pair
from score_artifact_rois import selected_crops


DEFAULT_VIEW_NAMES = ("eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec LABEL=RUN_DIR. RUN_DIR should contain run_summary.json and rendered eval images.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--render-dir", action="append", default=None, help="Optional LABEL=RENDER_DIR override.")
    parser.add_argument("--eval-json", action="append", default=None, help="Optional LABEL=EVAL_JSON metrics override.")
    parser.add_argument("--view-name", action="append", default=None, help="Eval render filename to score. Repeatable.")
    parser.add_argument("--panels", type=int, default=2)
    parser.add_argument("--gt-panel", type=int, default=0)
    parser.add_argument("--cand-panel", type=int, default=1)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--crop-right", type=int, default=0)
    parser.add_argument("--drop-border-components", type=int, default=0)
    parser.add_argument("--all-rois", action="store_true", help="Use broad/debug ROI set instead of curated default.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="legacy")
    parser.add_argument("--ssim-severe", type=float, default=None)
    parser.add_argument("--area-serious", type=int, default=None)
    parser.add_argument("--area-box", type=int, default=None)
    parser.add_argument("--sev-min", type=float, default=None)
    parser.add_argument("--ssim-suspect", type=float, default=None)
    parser.add_argument("--area-suspect", type=int, default=None)
    return parser.parse_args()


def parse_mapping(values: Optional[Iterable[str]]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label:
            raise ValueError(f"Empty label in {value!r}")
        result[label] = Path(path)
    return result


def summary_render_dir(run_dir: Path) -> Optional[Path]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        eval_data = data.get("eval") or {}
        render_dir = eval_data.get("render_dir")
        if render_dir and Path(render_dir).exists():
            return Path(render_dir)
    renders = sorted(run_dir.glob("renders_best_step-*"))
    return renders[-1] if renders else None


def load_metrics(run_dir: Path, eval_json: Optional[Path] = None) -> Dict[str, Optional[float]]:
    if eval_json is not None:
        if not eval_json.exists():
            raise FileNotFoundError(eval_json)
        data = json.loads(eval_json.read_text(encoding="utf-8"))
        results = data.get("results") or {}
        return {
            "psnr": optional_float(results.get("psnr")),
            "ssim": optional_float(results.get("ssim")),
            "lpips": optional_float(results.get("lpips")),
        }
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {"psnr": None, "ssim": None, "lpips": None}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    results = ((data.get("eval") or {}).get("results") or {})
    return {
        "psnr": optional_float(results.get("psnr")),
        "ssim": optional_float(results.get("ssim")),
        "lpips": optional_float(results.get("lpips")),
    }


def optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def load_full_pair(image_path: Path, args: argparse.Namespace):
    load_args = SimpleNamespace(
        gt_file=None,
        cand_file=None,
        image=str(image_path),
        panels=args.panels,
        gt=args.gt_panel,
        cand=args.cand_panel,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        crop_left=args.crop_left,
        crop_right=args.crop_right,
    )
    return load_pair(load_args, args.gt_panel, args.cand_panel)


def crop_arrays(gt, cand, box: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, gt.shape[1]))
    x1 = max(x0, min(x1, gt.shape[1]))
    y0 = max(0, min(y0, gt.shape[0]))
    y1 = max(y0, min(y1, gt.shape[0]))
    return gt[y0:y1, x0:x1], cand[y0:y1, x0:x1]


def score_run(
    label: str,
    run_dir: Path,
    render_dir: Path,
    args: argparse.Namespace,
    eval_json: Optional[Path] = None,
) -> Dict[str, object]:
    views = []
    component_rows = []
    component_audit_enabled = not any(
        value != 0 for value in (args.crop_top, args.crop_bottom, args.crop_left, args.crop_right)
    )
    detector_kwargs = detector_kwargs_from_args(args)
    view_names = tuple(args.view_name or DEFAULT_VIEW_NAMES)
    for view_name in view_names:
        image_path = render_dir / view_name
        if not image_path.exists():
            views.append({"view": view_name, "status": "missing", "render_file": str(image_path)})
            continue
        gt, cand = load_full_pair(image_path, args)
        res = detect_defects(gt, cand, drop_border_components=args.drop_border_components, **detector_kwargs)
        if component_audit_enabled:
            eval_idx = eval_idx_from_view(view_name)
            if eval_idx is None:
                raise ValueError(f"Cannot infer eval index from {view_name}")
            component_rows.extend(
                classify_regions(
                    list(res["major"]) + list(res["minor"]),
                    rois=view_rois(eval_idx),
                    structural_excludes=set(DEFAULT_STRUCTURAL_EXCLUDE),
                    frame_area=int(gt.shape[0] * gt.shape[1]),
                    view_name=view_name,
                )
            )
        views.append(
            {
                "view": view_name,
                "status": "complete",
                "artifact_score": float(res["artifact_score"]),
                "serious_artifact_score": float(res["serious_artifact_score"]),
                "artifact_count": int(res["artifact_count"]),
                "largest_area": int(res["largest_area"]),
                "serious": bool(res["serious"]),
            }
        )

    roi_rows = []
    first_view_cache = {}
    for crop_name, eval_idx, box in selected_crops(None, all_rois=args.all_rois):
        view_name = f"eval_img_{eval_idx:04d}.png"
        image_path = render_dir / view_name
        if not image_path.exists():
            roi_rows.append({"crop": crop_name, "eval_idx": eval_idx, "status": "missing", "render_file": str(image_path)})
            continue
        if view_name not in first_view_cache:
            first_view_cache[view_name] = load_full_pair(image_path, args)
        gt, cand = first_view_cache[view_name]
        crop_gt, crop_cand = crop_arrays(gt, cand, box)
        res = detect_defects(crop_gt, crop_cand, drop_border_components=args.drop_border_components, **detector_kwargs)
        roi_rows.append(
            {
                "crop": crop_name,
                "eval_idx": eval_idx,
                "bbox_xyxy": list(box),
                "status": "complete",
                "artifact_score": float(res["artifact_score"]),
                "serious_artifact_score": float(res["serious_artifact_score"]),
                "artifact_count": int(res["artifact_count"]),
                "largest_area": int(res["largest_area"]),
                "serious": bool(res["serious"]),
            }
        )

    completed_views = [view for view in views if view.get("status") == "complete"]
    completed_rois = [row for row in roi_rows if row.get("status") == "complete"]
    metrics = load_metrics(run_dir, eval_json)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "render_dir": str(render_dir),
        "eval_json": str(eval_json) if eval_json is not None else None,
        "detector_preset": args.preset,
        "detector_kwargs": detector_kwargs,
        "metrics": metrics,
        "full_frame": {
            "artifact_score": max((float(view["artifact_score"]) for view in completed_views), default=None),
            "artifact_score_mean": mean([float(view["artifact_score"]) for view in completed_views]) if completed_views else None,
            "serious_artifact_score": max((float(view["serious_artifact_score"]) for view in completed_views), default=None),
            "serious_count": sum(1 for view in completed_views if view.get("serious")),
            "views": views,
        },
        "roi": {
            "artifact_score": max((float(row["artifact_score"]) for row in completed_rois), default=None),
            "serious_artifact_score": max((float(row["serious_artifact_score"]) for row in completed_rois), default=None),
            "serious_count": sum(1 for row in completed_rois if row.get("serious")),
            "stand_connector_score": next(
                (float(row["artifact_score"]) for row in completed_rois if row.get("crop") == "left_stand_connector_eval0"),
                None,
            ),
            "rows": roi_rows,
        },
        "component_audit": (
            {
                "status": "complete",
                "structural_exclude_crop": sorted(DEFAULT_STRUCTURAL_EXCLUDE),
                "aggregate": aggregate_components(component_rows),
            }
            if component_audit_enabled
            else {"status": "disabled_for_crop_margins", "aggregate": aggregate_components([])}
        ),
    }


def aggregate(runs: List[Dict[str, object]]) -> Dict[str, object]:
    def values(path: Tuple[str, ...]) -> List[float]:
        output = []
        for run in runs:
            value = run
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                output.append(float(value))
        return output

    paths = {
        "psnr": ("metrics", "psnr"),
        "ssim": ("metrics", "ssim"),
        "lpips": ("metrics", "lpips"),
        "full_artifact_score": ("full_frame", "artifact_score"),
        "full_serious_artifact_score": ("full_frame", "serious_artifact_score"),
        "roi_artifact_score": ("roi", "artifact_score"),
        "roi_serious_artifact_score": ("roi", "serious_artifact_score"),
        "stand_connector_score": ("roi", "stand_connector_score"),
        "structural_component_score": ("component_audit", "aggregate", "structural_roi_component_score"),
        "structural_serious_component_score": (
            "component_audit",
            "aggregate",
            "structural_roi_serious_component_score",
        ),
        "off_roi_component_score": ("component_audit", "aggregate", "off_roi_component_score"),
    }
    result = {}
    for name, path in paths.items():
        vals = values(path)
        result[name] = {
            "mean": mean(vals) if vals else None,
            "std": pstdev(vals) if len(vals) > 1 else 0.0 if vals else None,
            "max": max(vals) if vals else None,
            "min": min(vals) if vals else None,
        }
    result["roi_serious_count_total"] = sum(int(run["roi"]["serious_count"]) for run in runs)
    result["full_serious_count_total"] = sum(int(run["full_frame"]["serious_count"]) for run in runs)
    return result


def write_csv(path: Path, runs: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "psnr",
                "ssim",
                "lpips",
                "full_artifact_score",
                "full_serious_artifact_score",
                "full_serious_count",
                "roi_artifact_score",
                "roi_serious_artifact_score",
                "roi_serious_count",
                "stand_connector_score",
                "structural_component_score",
                "structural_serious_component_score",
                "off_roi_component_score",
                "render_dir",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run["label"],
                    "psnr": run["metrics"]["psnr"],
                    "ssim": run["metrics"]["ssim"],
                    "lpips": run["metrics"]["lpips"],
                    "full_artifact_score": run["full_frame"]["artifact_score"],
                    "full_serious_artifact_score": run["full_frame"]["serious_artifact_score"],
                    "full_serious_count": run["full_frame"]["serious_count"],
                    "roi_artifact_score": run["roi"]["artifact_score"],
                    "roi_serious_artifact_score": run["roi"]["serious_artifact_score"],
                    "roi_serious_count": run["roi"]["serious_count"],
                    "stand_connector_score": run["roi"]["stand_connector_score"],
                    "structural_component_score": run["component_audit"]["aggregate"]["structural_roi_component_score"],
                    "structural_serious_component_score": run["component_audit"]["aggregate"][
                        "structural_roi_serious_component_score"
                    ],
                    "off_roi_component_score": run["component_audit"]["aggregate"]["off_roi_component_score"],
                    "render_dir": run["render_dir"],
                }
            )


def write_markdown(path: Path, runs: List[Dict[str, object]], agg: Dict[str, object]) -> None:
    lines = [
        "# Artifact Gate Summary",
        "",
        "| Run | PSNR | SSIM | LPIPS | Full artifact | Full serious | Full serious views | ROI artifact | ROI serious | ROI serious count | Stand connector | Structural comp | Structural serious comp | Off-ROI comp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        comp = run["component_audit"]["aggregate"]
        lines.append(
            f"| {run['label']} | {fmt(run['metrics']['psnr'])} | {fmt(run['metrics']['ssim'])} | "
            f"{fmt(run['metrics']['lpips'])} | {fmt(run['full_frame']['artifact_score'])} | "
            f"{fmt(run['full_frame']['serious_artifact_score'])} | {run['full_frame']['serious_count']} | "
            f"{fmt(run['roi']['artifact_score'])} | {fmt(run['roi']['serious_artifact_score'])} | "
            f"{run['roi']['serious_count']} | {fmt(run['roi']['stand_connector_score'])} | "
            f"{fmt(comp['structural_roi_component_score'])} | "
            f"{fmt(comp['structural_roi_serious_component_score'])} | "
            f"{fmt(comp['off_roi_component_score'])} |"
        )
    lines.extend(
        [
            "",
            "Aggregate:",
            "",
            "```json",
            json.dumps(agg, indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_specs = parse_mapping(args.run)
    render_overrides = parse_mapping(args.render_dir)
    eval_overrides = parse_mapping(args.eval_json)
    runs = []
    for label, run_dir in run_specs.items():
        render_dir = render_overrides.get(label) or summary_render_dir(run_dir)
        if render_dir is None:
            raise FileNotFoundError(f"Could not determine render dir for {label}: {run_dir}")
        runs.append(score_run(label, run_dir, render_dir, args, eval_overrides.get(label)))
    agg = aggregate(runs)
    payload = {"runs": runs, "aggregate": agg}
    json_path = args.out_dir / "artifact_gate_summary.json"
    csv_path = args.out_dir / "artifact_gate_summary.csv"
    md_path = args.out_dir / "artifact_gate_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, runs)
    write_markdown(md_path, runs, agg)
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    print(
        "aggregate "
        f"roi_serious_total={agg['roi_serious_count_total']} "
        f"stand_connector_max={fmt(agg['stand_connector_score']['max'])} "
        f"full_artifact_mean={fmt(agg['full_artifact_score']['mean'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
