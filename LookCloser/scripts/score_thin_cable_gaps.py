#!/usr/bin/env python3
"""Trace a declared thin dark cable in GT and measure missing prediction runs.

The broad edge-continuity metric is intentionally insufficient for this case:
large pipes and people dominate its average.  This detector uses only coarse
waypoints to define a cable corridor.  It snaps them to a multi-pixel black
ridge in the native-EXR GT, traces an ordered centerline, and then looks for
contiguous locations where the prediction loses that dark ridge while becoming
brighter than GT.  Outputs include exact masks and native-resolution crops.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter, minimum_filter
from skimage.graph import route_through_array

from nerfstudio.data.utils.data_utils import load_exr_image
from nerfstudio.utils.hdr import BT709_LUMA, calibrate_exr_paths, scene_linear_to_pq


DEFAULT_CABLES = (
    ("left_black_cable_eval0", 0, ((260, 200), (245, 300), (225, 400), (190, 500))),
    ("left_black_cable_eval1", 1, ((260, 250), (245, 350), (225, 450), (190, 550))),
    ("left_black_cable_eval2", 2, ((690, 250), (650, 400), (600, 550), (540, 700), (455, 835))),
)


@dataclass(frozen=True)
class DetectorConfig:
    ridge_width: int = 15
    snap_radius: int = 18
    corridor_radius: int = 35
    prediction_tolerance: int = 3
    gt_ridge_quantile: float = 0.25
    max_support_ratio: float = 0.70
    min_bright_error: float = 0.003
    close_length: int = 5
    min_gap_length: int = 10
    crop_margin: int = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cable",
        action="append",
        default=None,
        metavar="NAME:EVAL:X,Y;X,Y;...",
        help="Override the default cable corridors; repeatable.",
    )
    parser.add_argument("--min-gap-length", type=int, default=10)
    parser.add_argument("--crop-margin", type=int, default=120)
    args = parser.parse_args()
    if args.min_gap_length <= 0 or args.crop_margin < 0:
        parser.error("min-gap-length must be positive and crop-margin non-negative")
    return args


def parse_cables(values: Iterable[str] | None) -> list[tuple[str, int, tuple[tuple[int, int], ...]]]:
    if values is None:
        return list(DEFAULT_CABLES)
    output = []
    for value in values:
        header, separator, point_text = value.partition(":")
        eval_text, separator2, point_text = point_text.partition(":")
        if not header or not separator or not separator2:
            raise ValueError(f"Invalid cable {value!r}; expected NAME:EVAL:X,Y;X,Y;...")
        points = []
        for pair in point_text.split(";"):
            coordinates = pair.split(",")
            if len(coordinates) != 2:
                raise ValueError(f"Invalid cable point {pair!r}")
            points.append((int(coordinates[0]), int(coordinates[1])))
        if len(points) < 2:
            raise ValueError("A cable corridor requires at least two points")
        output.append((header, int(eval_text), tuple(points)))
    return output


def pq_luminance(path: Path, nits_per_scene_unit: float) -> np.ndarray:
    image = load_exr_image(path)[..., :3]
    pq = scene_linear_to_pq(
        torch.from_numpy(np.ascontiguousarray(image)), nits_per_scene_unit=nits_per_scene_unit
    ).numpy()
    return np.tensordot(pq, np.asarray(BT709_LUMA, dtype=np.float32), axes=([-1], [0])).astype(np.float32)


def dark_ridge(image: np.ndarray, width: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_BLACKHAT, kernel)


def snap_anchors(ridge: np.ndarray, anchors: tuple[tuple[int, int], ...], radius: int) -> list[tuple[int, int]]:
    height, width = ridge.shape
    snapped = []
    for x, y in anchors:
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Anchor {(x, y)} is outside image bounds {(width, height)}")
        local_y, local_x = np.unravel_index(np.argmax(ridge[y0:y1, x0:x1]), (y1 - y0, x1 - x0))
        snapped.append((int(x0 + local_x), int(y0 + local_y)))
    return snapped


def trace_centerline(
    ridge: np.ndarray, anchors: list[tuple[int, int]], corridor_radius: int
) -> list[tuple[int, int]]:
    """Trace an ordered dark-ridge path with a weak straight-corridor prior."""
    height, width = ridge.shape
    route: list[tuple[int, int]] = []
    for (xa, ya), (xb, yb) in zip(anchors[:-1], anchors[1:]):
        x0, x1 = max(0, min(xa, xb) - corridor_radius), min(width, max(xa, xb) + corridor_radius + 1)
        y0, y1 = max(0, min(ya, yb) - corridor_radius), min(height, max(ya, yb) + corridor_radius + 1)
        local_ridge = ridge[y0:y1, x0:x1]
        low, high = np.quantile(local_ridge, (0.50, 0.995))
        normalized = np.clip((local_ridge - low) / max(float(high - low), 1e-6), 0.0, 1.0)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        vx, vy = xb - xa, yb - ya
        denominator = max(vx * vx + vy * vy, 1)
        position = np.clip(((xx - xa) * vx + (yy - ya) * vy) / denominator, 0.0, 1.0)
        line_x, line_y = xa + position * vx, ya + position * vy
        line_distance = np.hypot(xx - line_x, yy - line_y)
        cost = 1.05 - normalized + 0.03 * (line_distance / float(corridor_radius)) ** 2

        local_path, _ = route_through_array(
            cost,
            (ya - y0, xa - x0),
            (yb - y0, xb - x0),
            fully_connected=True,
            geometric=True,
        )
        route.extend((int(x0 + x), int(y0 + y)) for y, x in local_path[:-1])
    route.append(anchors[-1])
    return route


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts_and_ends = []
    start = None
    for index, value in enumerate(np.concatenate((mask.astype(bool), np.asarray([False])))):
        if value and start is None:
            start = index
        elif not value and start is not None:
            starts_and_ends.append((start, index))
            start = None
    return starts_and_ends


def detect_gaps(
    gt: np.ndarray,
    prediction: np.ndarray,
    route: list[tuple[int, int]],
    config: DetectorConfig,
) -> tuple[np.ndarray, list[dict], dict[str, float]]:
    gt_ridge_image = dark_ridge(gt, config.ridge_width)
    pred_ridge_image = dark_ridge(prediction, config.ridge_width)
    diameter = 2 * config.prediction_tolerance + 1
    pred_ridge_near = maximum_filter(pred_ridge_image, size=diameter)
    pred_dark_near = minimum_filter(prediction, size=diameter)
    xs = np.asarray([point[0] for point in route], dtype=np.int32)
    ys = np.asarray([point[1] for point in route], dtype=np.int32)
    gt_response = gt_ridge_image[ys, xs]
    pred_response = pred_ridge_near[ys, xs]
    support_ratio = pred_response / (gt_response + 1e-6)
    bright_error = np.maximum(pred_dark_near[ys, xs] - gt[ys, xs], 0.0)
    ridge_cutoff = float(np.quantile(gt_response, config.gt_ridge_quantile))
    # ``>=`` matters for a uniformly dark synthetic or real cable: every
    # centerline sample may have exactly the same valid ridge response.
    strong_gt = (gt_response >= ridge_cutoff) & (gt_response > 1e-6)
    missing = strong_gt & (support_ratio < config.max_support_ratio) & (bright_error > config.min_bright_error)
    if config.close_length > 1:
        missing = cv2.morphologyEx(
            missing.astype(np.uint8)[None, :],
            cv2.MORPH_CLOSE,
            np.ones((1, config.close_length), dtype=np.uint8),
        )[0].astype(bool)

    gaps = []
    for start, end in contiguous_runs(missing):
        if end - start < config.min_gap_length:
            continue
        gap_x, gap_y = xs[start:end], ys[start:end]
        gaps.append(
            {
                "start_index": start,
                "end_index": end,
                "length_pixels": end - start,
                "start_xy": [int(gap_x[0]), int(gap_y[0])],
                "end_xy": [int(gap_x[-1]), int(gap_y[-1])],
                "bbox_xyxy": [int(gap_x.min()), int(gap_y.min()), int(gap_x.max() + 1), int(gap_y.max() + 1)],
                "mean_bright_error": float(bright_error[start:end].mean()),
                "mean_support_ratio": float(support_ratio[start:end].mean()),
            }
        )
    long_missing = np.zeros_like(missing)
    for gap in gaps:
        long_missing[gap["start_index"] : gap["end_index"]] = True
    total_gap_pixels = int(long_missing.sum())
    summary = {
        "path_length_pixels": len(route),
        "gap_count": len(gaps),
        "gap_pixels": total_gap_pixels,
        "gap_fraction": float(total_gap_pixels / max(len(route), 1)),
        "longest_gap_pixels": max((gap["length_pixels"] for gap in gaps), default=0),
        "longest_gap_fraction": float(max((gap["length_pixels"] for gap in gaps), default=0) / max(len(route), 1)),
    }
    return long_missing, gaps, summary


def masks_from_route(
    shape: tuple[int, int], route: list[tuple[int, int]], missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    path_mask = np.zeros(shape, dtype=np.uint8)
    gap_mask = np.zeros(shape, dtype=np.uint8)
    for index, (x, y) in enumerate(route):
        path_mask[y, x] = 255
        if missing[index]:
            gap_mask[y, x] = 255
    path_mask = cv2.dilate(path_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gap_mask = cv2.dilate(gap_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return path_mask, gap_mask


def display_pair(render_dir: Path, eval_idx: int) -> tuple[np.ndarray, np.ndarray]:
    pair = np.asarray(Image.open(render_dir / f"eval_img_{eval_idx:04d}.png").convert("RGB"))
    width = pair.shape[1] // 2
    return pair[:, :width].copy(), pair[:, width:].copy()


def crop_box(gaps: list[dict], shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    if not gaps:
        raise ValueError("No long cable gap detected; cannot produce a claimed hole crop")
    x0 = min(gap["bbox_xyxy"][0] for gap in gaps)
    y0 = min(gap["bbox_xyxy"][1] for gap in gaps)
    x1 = max(gap["bbox_xyxy"][2] for gap in gaps)
    y1 = max(gap["bbox_xyxy"][3] for gap in gaps)
    height, width = shape
    return max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin)


def save_outputs(
    output_dir: Path,
    name: str,
    gt_display: np.ndarray,
    pred_display: np.ndarray,
    path_mask: np.ndarray,
    gap_mask: np.ndarray,
    box: tuple[int, int, int, int],
    summary: dict[str, float],
) -> dict[str, str]:
    x0, y0, x1, y1 = box
    gt_crop = gt_display[y0:y1, x0:x1]
    pred_crop = pred_display[y0:y1, x0:x1]
    path_overlay = gt_crop.copy()
    path_overlay[path_mask[y0:y1, x0:x1] > 0] = (0, 255, 0)
    gap_overlay = pred_crop.copy()
    path_pixels = path_mask[y0:y1, x0:x1] > 0
    gap_pixels = gap_mask[y0:y1, x0:x1] > 0
    gap_overlay[path_pixels] = (0, 180, 255)
    gap_overlay[gap_pixels] = (255, 0, 0)

    arrays = {
        "gt_crop": gt_crop,
        "prediction_crop": pred_crop,
        "gt_cable_mask": path_overlay,
        "detected_gaps": gap_overlay,
    }
    paths = {}
    for label, array in arrays.items():
        path = output_dir / f"{name}_{label}.png"
        Image.fromarray(array).save(path)
        paths[label] = str(path)

    label_height = 34
    panel_width, panel_height = gt_crop.shape[1], gt_crop.shape[0]
    sheet = Image.new("RGB", (panel_width * 4, panel_height + label_height), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (label_text, array) in enumerate(arrays.items()):
        sheet.paste(Image.fromarray(array), (index * panel_width, label_height))
        draw.text((index * panel_width + 5, 7), label_text, fill="white")
    draw.text(
        (5, panel_height + 17),
        f"longest={int(summary['longest_gap_pixels'])}px gap_fraction={summary['gap_fraction']:.3f}",
        fill="white",
    )
    sheet_path = output_dir / f"{name}_review.png"
    sheet.save(sheet_path)
    paths["review"] = str(sheet_path)
    return paths


def main() -> int:
    args = parse_args()
    config = DetectorConfig(min_gap_length=args.min_gap_length, crop_margin=args.crop_margin)
    train_paths = sorted((args.data / "images").glob("frame_train_*.exr"))
    if not train_paths:
        raise FileNotFoundError(f"No training EXRs under {args.data / 'images'}")
    calibration = calibrate_exr_paths(train_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, eval_idx, requested_anchors in parse_cables(args.cable):
        gt_path = args.render_dir / f"eval_gt_{eval_idx:04d}.exr"
        pred_path = args.render_dir / f"eval_pred_{eval_idx:04d}.exr"
        gt = pq_luminance(gt_path, calibration.nits_per_scene_unit)
        prediction = pq_luminance(pred_path, calibration.nits_per_scene_unit)
        ridge = dark_ridge(gt, config.ridge_width)
        anchors = snap_anchors(ridge, requested_anchors, config.snap_radius)
        route = trace_centerline(ridge, anchors, config.corridor_radius)
        missing, gaps, summary = detect_gaps(gt, prediction, route, config)
        path_mask, gap_mask = masks_from_route(gt.shape, route, missing)
        box = crop_box(gaps, gt.shape, config.crop_margin)
        gt_display, pred_display = display_pair(args.render_dir, eval_idx)
        paths = save_outputs(
            args.output_dir, name, gt_display, pred_display, path_mask, gap_mask, box, summary
        )
        row = {
            "name": name,
            "eval_idx": eval_idx,
            "requested_anchors_xy": [list(point) for point in requested_anchors],
            "snapped_anchors_xy": [list(point) for point in anchors],
            "crop_bbox_xyxy": list(box),
            "gaps": gaps,
            **summary,
            "outputs": paths,
        }
        rows.append(row)
        print(
            f"cable={name} gaps={summary['gap_count']} longest={summary['longest_gap_pixels']} "
            f"gap_fraction={summary['gap_fraction']:.5f} crop={box}",
            flush=True,
        )

    aggregate = {
        "mean_gap_fraction": float(np.mean([row["gap_fraction"] for row in rows])),
        "max_longest_gap_pixels": int(max(row["longest_gap_pixels"] for row in rows)),
        "total_gap_pixels": int(sum(row["gap_pixels"] for row in rows)),
    }
    output = {
        "schema": 1,
        "render_dir": str(args.render_dir.resolve()),
        "nits_per_scene_unit": calibration.nits_per_scene_unit,
        "detector": config.__dict__,
        "aggregate": aggregate,
        "cables": rows,
    }
    output_path = args.output_dir / "thin_cable_gaps.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"aggregate={json.dumps(aggregate, sort_keys=True)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
