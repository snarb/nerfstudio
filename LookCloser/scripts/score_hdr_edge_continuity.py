#!/usr/bin/env python3
"""Measure thin-structure continuity in paired scene-linear EXR renders.

The metric deliberately complements PSNR/SSIM/LPIPS.  It extracts edges in a
shared PQ display domain, skeletonizes them, and measures whether ground-truth
edge pixels have predicted edge support within a small spatial tolerance.
Long unsupported skeleton components are reported separately so a thin cable
gap cannot disappear inside a good full-frame average.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label
from skimage.feature import canny
from skimage.morphology import skeletonize

from nerfstudio.data.utils.data_utils import load_exr_image
from nerfstudio.utils.hdr import BT709_LUMA, calibrate_exr_paths, scene_linear_to_pq


DEFAULT_ROIS = (
    ("tangled_cable_eval2", 2, (0, 130, 300, 500)),
    ("cable_loop_eval2", 2, (180, 220, 360, 560)),
    ("hanging_cable_eval0", 0, (650, 0, 850, 430)),
    ("hanging_cable_eval1", 1, (650, 0, 850, 500)),
    ("left_stand_cables_eval0", 0, (300, 0, 650, 650)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--low-quantile", type=float, default=0.70)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--long-gap-min-pixels", type=int, default=4)
    parser.add_argument(
        "--roi",
        action="append",
        default=None,
        metavar="NAME:EVAL:X0:Y0:X1:Y1",
        help="Override default ROIs; repeatable.",
    )
    args = parser.parse_args()
    if args.tolerance < 0 or args.sigma < 0:
        parser.error("tolerance and sigma must be non-negative")
    if not 0 <= args.low_quantile < args.high_quantile <= 1:
        parser.error("expected 0 <= low-quantile < high-quantile <= 1")
    if args.long_gap_min_pixels <= 0:
        parser.error("long-gap-min-pixels must be positive")
    return args


def parse_rois(values: Iterable[str] | None) -> list[tuple[str, int, tuple[int, int, int, int]]]:
    if values is None:
        return list(DEFAULT_ROIS)
    output = []
    for value in values:
        fields = value.split(":")
        if len(fields) != 6:
            raise ValueError(f"Invalid ROI {value!r}; expected NAME:EVAL:X0:Y0:X1:Y1")
        name, eval_text, *coords = fields
        box = tuple(int(item) for item in coords)
        if not name or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid ROI {value!r}")
        output.append((name, int(eval_text), box))
    return output


def paired_paths(render_dir: Path, eval_idx: int) -> tuple[Path, Path]:
    pred = render_dir / f"eval_pred_{eval_idx:04d}.exr"
    gt = render_dir / f"eval_gt_{eval_idx:04d}.exr"
    if not pred.is_file() or not gt.is_file():
        raise FileNotFoundError(f"Missing paired EXR for eval index {eval_idx} in {render_dir}")
    return pred, gt


def pq_luminance(image: np.ndarray, nits_per_scene_unit: float) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(image[..., :3]))
    pq = scene_linear_to_pq(tensor, nits_per_scene_unit=nits_per_scene_unit).numpy()
    return np.tensordot(pq, np.asarray(BT709_LUMA, dtype=np.float32), axes=([-1], [0])).astype(np.float32)


def shared_normalize(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = gt[np.isfinite(gt)]
    if finite.size == 0:
        raise ValueError("ROI contains no finite GT luminance")
    lo, hi = np.quantile(finite, (0.005, 0.995))
    scale = max(float(hi - lo), 1e-6)
    return np.clip((pred - lo) / scale, 0.0, 1.0), np.clip((gt - lo) / scale, 0.0, 1.0)


def edge_mask(image: np.ndarray, sigma: float, low_quantile: float, high_quantile: float) -> np.ndarray:
    return canny(
        image.astype(np.float32),
        sigma=float(sigma),
        low_threshold=float(low_quantile),
        high_threshold=float(high_quantile),
        use_quantiles=True,
    )


def component_pixels(mask: np.ndarray) -> list[int]:
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    return np.bincount(labels.reshape(-1), minlength=count + 1)[1:].astype(int).tolist()


def metrics(gt_edges: np.ndarray, pred_edges: np.ndarray, tolerance: float, long_gap_min_pixels: int) -> dict:
    gt_skeleton = skeletonize(gt_edges)
    pred_skeleton = skeletonize(pred_edges)
    gt_count = int(gt_skeleton.sum())
    pred_count = int(pred_skeleton.sum())
    if gt_count == 0 or pred_count == 0:
        raise ValueError(f"Edge extraction collapsed: gt={gt_count}, prediction={pred_count}")
    pred_distance = distance_transform_edt(~pred_skeleton)
    gt_distance = distance_transform_edt(~gt_skeleton)
    supported_gt = gt_skeleton & (pred_distance <= float(tolerance))
    supported_pred = pred_skeleton & (gt_distance <= float(tolerance))
    missing = gt_skeleton & ~supported_gt
    gap_sizes = component_pixels(missing)
    long_gap_pixels = sum(size for size in gap_sizes if size >= int(long_gap_min_pixels))
    recall = float(supported_gt.sum() / gt_count)
    precision = float(supported_pred.sum() / pred_count)
    return {
        "edge_recall": recall,
        "edge_precision": precision,
        "edge_f1": float(2.0 * recall * precision / max(recall + precision, 1e-12)),
        "missing_edge_fraction": float(missing.sum() / gt_count),
        "long_gap_fraction": float(long_gap_pixels / gt_count),
        "long_gap_count": int(sum(size >= int(long_gap_min_pixels) for size in gap_sizes)),
        "largest_gap_pixels": int(max(gap_sizes, default=0)),
        "gt_edge_pixels": gt_count,
        "prediction_edge_pixels": pred_count,
        "missing_mask": missing,
        "gt_skeleton": gt_skeleton,
        "pred_skeleton": pred_skeleton,
    }


def preview(image: np.ndarray) -> np.ndarray:
    value = np.clip(image, 0.0, 1.0)
    srgb = np.where(value <= 0.0031308, 12.92 * value, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)
    return np.uint8(np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5)


def save_sheet(path: Path, gt: np.ndarray, pred: np.ndarray, result: dict, name: str) -> None:
    gt_rgb = np.repeat(preview(gt)[..., None], 3, axis=-1)
    pred_rgb = np.repeat(preview(pred)[..., None], 3, axis=-1)
    overlay = pred_rgb.copy()
    overlay[result["gt_skeleton"]] = (0, 180, 0)
    overlay[result["missing_mask"]] = (255, 0, 0)
    edge_compare = np.zeros_like(pred_rgb)
    edge_compare[result["gt_skeleton"]] = (0, 180, 0)
    edge_compare[result["pred_skeleton"]] = (0, 130, 255)
    edge_compare[result["missing_mask"]] = (255, 0, 0)
    panels = [("GT PQ luma", gt_rgb), ("Prediction", pred_rgb), ("Missed GT edges", overlay), ("Edges", edge_compare)]
    label_height = 28
    canvas = Image.new("RGB", (gt_rgb.shape[1] * len(panels), gt_rgb.shape[0] + label_height), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label_text, array) in enumerate(panels):
        x = index * gt_rgb.shape[1]
        canvas.paste(Image.fromarray(array), (x, label_height))
        draw.text((x + 5, 7), label_text, fill="white")
    draw.text((5, gt_rgb.shape[0] + 10), name, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    train_paths = sorted((args.data / "images").glob("frame_train_*.exr"))
    if not train_paths:
        raise FileNotFoundError(f"No train EXRs in {args.data / 'images'}")
    calibration = calibrate_exr_paths(train_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, eval_idx, (x0, y0, x1, y1) in parse_rois(args.roi):
        pred_path, gt_path = paired_paths(args.render_dir, eval_idx)
        pred_rgb = load_exr_image(pred_path)[y0:y1, x0:x1, :3]
        gt_rgb = load_exr_image(gt_path)[y0:y1, x0:x1, :3]
        pred, gt = shared_normalize(
            pq_luminance(pred_rgb, calibration.nits_per_scene_unit),
            pq_luminance(gt_rgb, calibration.nits_per_scene_unit),
        )
        result = metrics(
            edge_mask(gt, args.sigma, args.low_quantile, args.high_quantile),
            edge_mask(pred, args.sigma, args.low_quantile, args.high_quantile),
            args.tolerance,
            args.long_gap_min_pixels,
        )
        save_sheet(args.output_dir / f"{name}.png", gt, pred, result, name)
        row = {
            "name": name,
            "eval_idx": eval_idx,
            "bbox_xyxy": [x0, y0, x1, y1],
            **{key: value for key, value in result.items() if not isinstance(value, np.ndarray)},
        }
        rows.append(row)
        print(
            f"roi={name} recall={row['edge_recall']:.5f} f1={row['edge_f1']:.5f} "
            f"long_gap_fraction={row['long_gap_fraction']:.5f} gaps={row['long_gap_count']}",
            flush=True,
        )
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("edge_recall", "edge_precision", "edge_f1", "missing_edge_fraction", "long_gap_fraction")
    }
    aggregate["long_gap_count"] = int(sum(row["long_gap_count"] for row in rows))
    output = {
        "schema": 1,
        "render_dir": str(args.render_dir.resolve()),
        "nits_per_scene_unit": calibration.nits_per_scene_unit,
        "parameters": {
            "tolerance": args.tolerance,
            "sigma": args.sigma,
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "long_gap_min_pixels": args.long_gap_min_pixels,
        },
        "aggregate": aggregate,
        "rois": rows,
    }
    output_path = args.output_dir / "edge_continuity.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"aggregate={json.dumps(aggregate, sort_keys=True)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
