#!/usr/bin/env python3
"""Audit full-frame artifact components against named eval-view ROIs.

This does not replace detect_structural_artifacts.py. It explains where the
official full-frame score comes from by labeling each detected component with
the curated ROI boxes it overlaps, and by splitting the score into structural
ROI, all curated ROI, and off-ROI buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

from detect_structural_artifacts import AREA_SERIOUS, detect_defects, load_pair
from score_artifact_rois import DEFAULT_RUNNER_CROPS, selected_crops


DEFAULT_VIEW_NAMES = ("eval_img_0000.png", "eval_img_0001.png", "eval_img_0002.png")
DEFAULT_STRUCTURAL_EXCLUDE = {"floor_crack_eval0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--view-name", action="append", default=None)
    parser.add_argument("--panels", type=int, default=2)
    parser.add_argument("--gt-panel", type=int, default=0)
    parser.add_argument("--cand-panel", type=int, default=1)
    parser.add_argument("--drop-border-components", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true", help="Write missing-view rows instead of failing.")
    parser.add_argument(
        "--structural-exclude-crop",
        action="append",
        default=sorted(DEFAULT_STRUCTURAL_EXCLUDE),
        help="Curated ROI crop to exclude from the stricter structural bucket. Repeatable.",
    )
    return parser.parse_args()


def load_full_pair(image_path: Path, args: argparse.Namespace):
    load_args = SimpleNamespace(
        gt_file=None,
        cand_file=None,
        image=str(image_path),
        panels=args.panels,
        gt=args.gt_panel,
        cand=args.cand_panel,
        crop_top=0,
        crop_bottom=0,
        crop_left=0,
        crop_right=0,
    )
    return load_pair(load_args, args.gt_panel, args.cand_panel)


def eval_idx_from_view(view_name: str) -> Optional[int]:
    stem = Path(view_name).stem
    if not stem.startswith("eval_img_"):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 < x0 or y1 < y0:
        return 0
    return int((x1 - x0 + 1) * (y1 - y0 + 1))


def region_score(area: int, severity: float, frame_area: int) -> float:
    return 1000.0 * float(area) * float(severity) / float(frame_area)


def view_rois(eval_idx: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    return [
        (name, box)
        for name, crop_eval_idx, box in selected_crops(DEFAULT_RUNNER_CROPS, all_rois=False)
        if crop_eval_idx == eval_idx
    ]


def classify_regions(
    regions: Iterable[Tuple[int, int, int, int, int, float]],
    *,
    rois: List[Tuple[str, Tuple[int, int, int, int]]],
    structural_excludes: set[str],
    frame_area: int,
    view_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for area, x0, y0, x1, y1, severity in regions:
        region_box = (x0, y0, x1, y1)
        overlaps = []
        for roi_name, roi_box in rois:
            inter = intersection_area(region_box, roi_box)
            if inter <= 0:
                continue
            overlaps.append(
                {
                    "crop": roi_name,
                    "intersection_area": inter,
                    "component_overlap_frac": inter / float(area),
                }
            )
        overlap_names = [item["crop"] for item in overlaps]
        structural_names = [name for name in overlap_names if name not in structural_excludes]
        rows.append(
            {
                "view": view_name,
                "area": int(area),
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "mean_severity": float(severity),
                "score_contribution": region_score(area, severity, frame_area),
                "major": int(area) >= AREA_SERIOUS,
                "roi_overlap": bool(overlap_names),
                "structural_roi_overlap": bool(structural_names),
                "roi_names": overlap_names,
                "structural_roi_names": structural_names,
                "overlaps": overlaps,
            }
        )
    return rows


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    def score_where(key: str, value: bool) -> float:
        return sum(float(row["score_contribution"]) for row in rows if bool(row[key]) == value)

    def serious_where(key: str, value: bool) -> float:
        return sum(
            float(row["score_contribution"])
            for row in rows
            if bool(row[key]) == value and bool(row["major"])
        )

    return {
        "component_count": len(rows),
        "major_count": sum(1 for row in rows if row["major"]),
        "full_component_score": round(sum(float(row["score_contribution"]) for row in rows), 3),
        "full_component_serious_score": round(sum(float(row["score_contribution"]) for row in rows if row["major"]), 3),
        "curated_roi_component_score": round(score_where("roi_overlap", True), 3),
        "curated_roi_serious_component_score": round(serious_where("roi_overlap", True), 3),
        "structural_roi_component_score": round(score_where("structural_roi_overlap", True), 3),
        "structural_roi_serious_component_score": round(serious_where("structural_roi_overlap", True), 3),
        "off_roi_component_score": round(score_where("roi_overlap", False), 3),
        "off_roi_serious_component_score": round(serious_where("roi_overlap", False), 3),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "view",
                "area",
                "x0",
                "y0",
                "x1",
                "y1",
                "mean_severity",
                "score_contribution",
                "major",
                "roi_overlap",
                "structural_roi_overlap",
                "roi_names",
                "structural_roi_names",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["roi_names"] = ",".join(row["roi_names"])
            out["structural_roi_names"] = ",".join(row["structural_roi_names"])
            writer.writerow({key: out[key] for key in writer.fieldnames})


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Artifact Component Audit",
        "",
        f"Render dir: `{payload['render_dir']}`",
        "",
        "Aggregate:",
        "",
        "```json",
        json.dumps(payload["aggregate"], indent=2, sort_keys=True),
        "```",
        "",
        "| View | Full score | Full serious | Structural ROI score | Structural serious | Curated ROI score | Off-ROI score | Components | Major |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in payload["views"]:
        agg = view["aggregate"]
        lines.append(
            f"| {view['view']} | {agg['full_component_score']:.3f} | "
            f"{agg['full_component_serious_score']:.3f} | "
            f"{agg['structural_roi_component_score']:.3f} | "
            f"{agg['structural_roi_serious_component_score']:.3f} | "
            f"{agg['curated_roi_component_score']:.3f} | "
            f"{agg['off_roi_component_score']:.3f} | "
            f"{agg['component_count']} | {agg['major_count']} |"
        )
    lines.extend(["", "Top components:", "", "| View | Area | Score | Major | BBox | ROI names | Structural ROI names |", "|---|---:|---:|---|---|---|---|"])
    for row in sorted(payload["components"], key=lambda item: float(item["score_contribution"]), reverse=True)[:20]:
        bbox = f"({row['x0']},{row['y0']})-({row['x1']},{row['y1']})"
        lines.append(
            f"| {row['view']} | {row['area']} | {float(row['score_contribution']):.3f} | "
            f"{row['major']} | {bbox} | {', '.join(row['roi_names']) or '-'} | "
            f"{', '.join(row['structural_roi_names']) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    structural_excludes = set(args.structural_exclude_crop or [])
    view_payloads = []
    all_rows: List[Dict[str, object]] = []
    for view_name in args.view_name or DEFAULT_VIEW_NAMES:
        image_path = args.render_dir / view_name
        if not image_path.exists():
            if not args.allow_missing:
                raise FileNotFoundError(image_path)
            view_payloads.append({"view": view_name, "status": "missing", "render_file": str(image_path), "aggregate": aggregate([])})
            continue
        eval_idx = eval_idx_from_view(view_name)
        if eval_idx is None:
            raise ValueError(f"Cannot infer eval index from {view_name}")
        gt, cand = load_full_pair(image_path, args)
        res = detect_defects(gt, cand, drop_border_components=args.drop_border_components)
        frame_area = int(gt.shape[0] * gt.shape[1])
        rows = classify_regions(
            list(res["major"]) + list(res["minor"]),
            rois=view_rois(eval_idx),
            structural_excludes=structural_excludes,
            frame_area=frame_area,
            view_name=view_name,
        )
        all_rows.extend(rows)
        view_payloads.append(
            {
                "view": view_name,
                "status": "complete",
                "render_file": str(image_path),
                "detector_artifact_score": float(res["artifact_score"]),
                "detector_serious_artifact_score": float(res["serious_artifact_score"]),
                "aggregate": aggregate(rows),
            }
        )

    payload = {
        "label": args.label,
        "render_dir": str(args.render_dir),
        "structural_exclude_crop": sorted(structural_excludes),
        "aggregate": aggregate(all_rows),
        "views": view_payloads,
        "components": all_rows,
    }
    json_path = args.out_dir / "artifact_component_audit.json"
    csv_path = args.out_dir / "artifact_component_audit.csv"
    md_path = args.out_dir / "artifact_component_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, all_rows)
    write_markdown(md_path, payload)
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    print(
        "aggregate "
        f"full={payload['aggregate']['full_component_score']:.3f} "
        f"structural={payload['aggregate']['structural_roi_component_score']:.3f} "
        f"structural_serious={payload['aggregate']['structural_roi_serious_component_score']:.3f} "
        f"off_roi={payload['aggregate']['off_roi_component_score']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
