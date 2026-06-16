#!/usr/bin/env python3
"""Score structural artifacts inside named eval-view ROIs.

This is a diagnostic companion to detect_structural_artifacts.py. It keeps the
official full-frame score untouched, but lets us check whether residual
full-frame blobs are inside the small structures we care about or mostly on
floor/edge/equipment regions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from detect_structural_artifacts import (
    PRESETS,
    detect_defects,
    detector_kwargs_from_args,
    load_pair,
    save_boxes,
    save_heatmap,
    save_suspicion,
)


CROPS: List[Tuple[str, int, Tuple[int, int, int, int]]] = [
    ("left_stand_connector_eval0", 0, (320, 0, 617, 530)),
    ("left_stand_eval0", 0, (300, 0, 650, 650)),
    ("left_hand_background_eval0", 0, (300, 210, 560, 500)),
    ("left_hand_outlet_stand_eval0", 0, (300, 250, 500, 560)),
    ("floor_crack_eval0", 0, (1110, 715, 1410, 900)),
    ("fingers_right_eval1", 1, (860, 290, 1210, 590)),
    ("fingers_right_tight_eval1", 1, (1030, 430, 1210, 610)),
    ("stand_label_eval2", 2, (60, 450, 290, 900)),
    ("tangled_cable_eval2", 2, (0, 130, 300, 500)),
    ("fingers_center_eval2", 2, (690, 330, 980, 610)),
]

DEFAULT_RUNNER_CROPS = [
    "left_stand_connector_eval0",
    "left_stand_eval0",
    "left_hand_background_eval0",
    "left_hand_outlet_stand_eval0",
    "floor_crack_eval0",
    "fingers_right_tight_eval1",
    "stand_label_eval2",
    "tangled_cable_eval2",
    "fingers_center_eval2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True, help="Directory containing eval_img_000*.png.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--crop-name", action="append", default=None, help="Named ROI to score. Repeatable.")
    parser.add_argument("--all-rois", action="store_true", help="Score every ROI, including broad/debug crops.")
    parser.add_argument("--panels", type=int, default=2)
    parser.add_argument("--gt-panel", type=int, default=0)
    parser.add_argument("--cand-panel", type=int, default=1)
    parser.add_argument("--drop-border-components", type=int, default=0)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="legacy")
    parser.add_argument("--ssim-severe", type=float, default=None)
    parser.add_argument("--area-serious", type=int, default=None)
    parser.add_argument("--area-box", type=int, default=None)
    parser.add_argument("--sev-min", type=float, default=None)
    parser.add_argument("--ssim-suspect", type=float, default=None)
    parser.add_argument("--area-suspect", type=int, default=None)
    parser.add_argument("--write-images", action="store_true", help="Write per-ROI heatmap/box/suspicion images.")
    return parser.parse_args()


def selected_crops(names: Iterable[str] | None, *, all_rois: bool = False) -> List[Tuple[str, int, Tuple[int, int, int, int]]]:
    if all_rois:
        return CROPS
    if names is None:
        names = DEFAULT_RUNNER_CROPS
    wanted = set(names)
    crops = [crop for crop in CROPS if crop[0] in wanted]
    missing = wanted - {crop[0] for crop in crops}
    if missing:
        raise ValueError(f"Unknown crop name(s): {sorted(missing)}")
    return crops


def crop_pair(image_path: Path, eval_idx: int, box: Tuple[int, int, int, int], args: argparse.Namespace):
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
    gt, cand = load_pair(load_args, args.gt_panel, args.cand_panel)
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, gt.shape[1]))
    x1 = max(x0, min(x1, gt.shape[1]))
    y0 = max(0, min(y0, gt.shape[0]))
    y1 = max(y0, min(y1, gt.shape[0]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Empty crop for eval_idx={eval_idx}, box={box}, image_shape={gt.shape}")
    return gt[y0:y1, x0:x1], cand[y0:y1, x0:x1]


def jsonable_result(res: Dict[str, object]) -> Dict[str, object]:
    return {
        "serious": bool(res["serious"]),
        "artifact_score": float(res["artifact_score"]),
        "serious_artifact_score": float(res["serious_artifact_score"]),
        "artifact_count": int(res["artifact_count"]),
        "largest_area": int(res["largest_area"]),
        "major": [list(region) for region in res["major"]],
        "minor": [list(region) for region in res["minor"]],
        "suspect_regions": [list(region) for region in res["suspect_regions"]],
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    detector_kwargs = detector_kwargs_from_args(args)

    for name, eval_idx, box in selected_crops(args.crop_name, all_rois=args.all_rois):
        image_path = args.render_dir / f"eval_img_{eval_idx:04d}.png"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        gt, cand = crop_pair(image_path, eval_idx, box, args)
        res = detect_defects(gt, cand, drop_border_components=args.drop_border_components, **detector_kwargs)
        row = {
            "crop": name,
            "eval_idx": eval_idx,
            "bbox_xyxy": list(box),
            "width": int(cand.shape[1]),
            "height": int(cand.shape[0]),
            **jsonable_result(res),
        }
        rows.append(row)
        if args.write_images:
            prefix = args.out_dir / name
            save_heatmap(gt, res["error_map"], f"{prefix}_heatmap.png")
            save_boxes(cand, res, f"{prefix}_boxes.png", name)
            save_suspicion(cand, res, f"{prefix}_suspicion.png", name)

    csv_path = args.out_dir / "roi_artifact_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop",
                "eval_idx",
                "bbox_xyxy",
                "width",
                "height",
                "serious",
                "artifact_score",
                "serious_artifact_score",
                "artifact_count",
                "largest_area",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    json_path = args.out_dir / "roi_artifact_scores.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    max_score = max((float(row["artifact_score"]) for row in rows), default=0.0)
    serious_count = sum(1 for row in rows if row["serious"])
    print(f"roi_max_artifact_score={max_score:.3f} serious_rois={serious_count}/{len(rows)}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    for row in rows:
        print(
            f"{row['crop']} eval{row['eval_idx']} score={row['artifact_score']:.3f} "
            f"serious={row['serious']} count={row['artifact_count']} largest={row['largest_area']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
