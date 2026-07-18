#!/usr/bin/env python3
"""Build the fail-closed visual/ROI protocol for one temporal LookCloser frame."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from detect_structural_artifacts import PRESETS, detect_defects


Box = Tuple[int, int, int, int]

PERMANENT_ROIS: Tuple[Tuple[str, int, Box], ...] = (
    ("thin_pipe_eval1", 1, (250, 95, 1010, 310)),
    ("tangled_cable_eval2", 2, (0, 130, 300, 500)),
)
SEED_TRACKED_ROIS: Tuple[Tuple[str, int, Box], ...] = (
    ("hand_eval0", 0, (300, 210, 560, 500)),
    ("chain_eval2", 2, (0, 130, 300, 500)),
    ("fingers_eval2", 2, (980, 300, 1250, 550)),
)
REQUIRED_CATEGORIES = {"permanent", "tracked", "broad_motion", "possible_hole"}


@dataclass(frozen=True)
class TrackedBox:
    box: Box
    confidence: float
    valid_fraction: float
    median_forward_backward_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--previous-dataset", type=Path)
    parser.add_argument("--previous-protocol", type=Path)
    parser.add_argument("--tracking-confidence-min", type=float, default=0.60)
    parser.add_argument("--thumbnail-width", type=int, default=256)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def eval_gt_path(dataset: Path, eval_idx: int) -> Path:
    path = dataset / "images" / f"frame_eval_{eval_idx + 1:05d}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_render_pair(render_dir: Path, eval_idx: int) -> Tuple[np.ndarray, np.ndarray, Path]:
    path = render_dir / f"eval_img_{eval_idx:04d}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    image = load_rgb(path)
    if image.shape[1] % 2 != 0:
        raise RuntimeError(f"Expected GT|render pair with even width: {path} {image.shape}")
    panel_width = image.shape[1] // 2
    return image[:, :panel_width], image[:, panel_width:], path


def clamp_box(box: Sequence[int], width: int, height: int, *, minimum: int = 8) -> Box:
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    if x1 - x0 < minimum or y1 - y0 < minimum:
        raise ValueError(f"ROI became too small after clamping: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def expand_box_minimum(box: Sequence[int], width: int, height: int, minimum: int = 64) -> Box:
    """Expand a valid ROI around its center so LPIPS has enough spatial support."""

    x0, y0, x1, y1 = clamp_box(box, width, height)
    if x1 - x0 < minimum:
        center = (x0 + x1) / 2.0
        x0, x1 = int(round(center - minimum / 2)), int(round(center + minimum / 2))
    if y1 - y0 < minimum:
        center = (y0 + y1) / 2.0
        y0, y1 = int(round(center - minimum / 2)), int(round(center + minimum / 2))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return clamp_box((x0, y0, x1, y1), width, height, minimum=min(minimum, width, height))


def _feature_points(gray: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=240,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=7,
    )
    if points is not None and len(points) >= 12:
        return points.astype(np.float32)
    xs = np.linspace(x0 + 3, x1 - 4, 8, dtype=np.float32)
    ys = np.linspace(y0 + 3, y1 - 4, 6, dtype=np.float32)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)


def track_box(previous_rgb: np.ndarray, current_rgb: np.ndarray, box: Box) -> TrackedBox:
    """Track a seed box with pyramidal LK and a forward/backward confidence."""

    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    points = _feature_points(previous_gray, box)
    next_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status_forward is None:
        return TrackedBox(box, 0.0, 0.0, float("inf"))
    back_points, status_back, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        next_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if back_points is None or status_back is None:
        return TrackedBox(box, 0.0, 0.0, float("inf"))

    fb_error = np.linalg.norm(points[:, 0] - back_points[:, 0], axis=1)
    valid = (status_forward[:, 0] > 0) & (status_back[:, 0] > 0) & np.isfinite(fb_error) & (fb_error <= 2.5)
    valid_fraction = float(valid.mean())
    if int(valid.sum()) < 8:
        return TrackedBox(box, valid_fraction * 0.25, valid_fraction, float("inf"))
    displacement = next_points[valid, 0] - points[valid, 0]
    dx, dy = np.median(displacement, axis=0)
    median_error = float(np.median(fb_error[valid]))
    confidence = float(valid_fraction * math.exp(-median_error / 2.5))
    x0, y0, x1, y1 = box
    tracked = clamp_box(
        (x0 + dx, y0 + dy, x1 + dx, y1 + dy),
        current_rgb.shape[1],
        current_rgb.shape[0],
    )
    return TrackedBox(tracked, confidence, valid_fraction, median_error)


def exposure_compensated_difference(previous: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return absolute and darkening differences after robust per-channel exposure alignment."""

    previous_f = previous.astype(np.float32) / 255.0
    current_f = current.astype(np.float32) / 255.0
    compensated = np.empty_like(previous_f)
    for channel in range(3):
        source = previous_f[..., channel].reshape(-1)
        target = current_f[..., channel].reshape(-1)
        subset = (source > 0.03) & (source < 0.97) & (target > 0.03) & (target < 0.97)
        if int(subset.sum()) < 100:
            gain, bias = 1.0, 0.0
        else:
            design = np.stack((source[subset], np.ones(int(subset.sum()), dtype=np.float32)), axis=1)
            gain, bias = np.linalg.lstsq(design, target[subset], rcond=None)[0]
            gain = float(np.clip(gain, 0.5, 2.0))
            bias = float(np.clip(bias, -0.25, 0.25))
        compensated[..., channel] = np.clip(previous_f[..., channel] * gain + bias, 0.0, 1.0)
    absolute = np.mean(np.abs(current_f - compensated), axis=2)
    darkening = np.mean(np.maximum(compensated - current_f, 0.0), axis=2)
    return absolute, darkening


def _component_boxes(score: np.ndarray, *, quantile: float, minimum_area: int, limit: int) -> List[Box]:
    threshold = max(float(np.quantile(score, quantile)), 0.035)
    mask = (score >= threshold).astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Tuple[int, Box]] = []
    height, width = score.shape
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < minimum_area:
            continue
        margin = 16
        candidates.append(
            (area, clamp_box((x - margin, y - margin, x + w + margin, y + h + margin), width, height))
        )
    return [box for _, box in sorted(candidates, reverse=True)[:limit]]


def _fallback_peak_box(score: np.ndarray, size: int = 192) -> Box:
    height, width = score.shape
    y, x = np.unravel_index(int(np.argmax(cv2.GaussianBlur(score, (0, 0), 9))), score.shape)
    half = size // 2
    return clamp_box((x - half, y - half, x + half, y + half), width, height)


def motion_boxes(previous: np.ndarray, current: np.ndarray) -> Tuple[List[Box], List[Box]]:
    absolute, darkening = exposure_compensated_difference(previous, current)
    broad = _component_boxes(absolute, quantile=0.965, minimum_area=500, limit=2)
    holes = _component_boxes(darkening, quantile=0.985, minimum_area=180, limit=1)
    return broad or [_fallback_peak_box(absolute)], holes or [_fallback_peak_box(darkening, size=128)]


def _metric_tensors(gt: np.ndarray, render: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_tensor = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    render_tensor = (
        torch.from_numpy(np.ascontiguousarray(render)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    )
    return gt_tensor, render_tensor


def roi_metrics(
    gt: np.ndarray,
    render: np.ndarray,
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> Dict[str, float]:
    gt_tensor, render_tensor = _metric_tensors(gt, render)
    mse = float(torch.mean((gt_tensor - render_tensor) ** 2).item())
    psnr = float(-10.0 * math.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity_index_measure(render_tensor, gt_tensor, data_range=1.0).item())
    with torch.no_grad():
        lpips_value = float(lpips(render_tensor, gt_tensor).item())
    lpips.reset()
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_value}


def _artifact_result(gt: np.ndarray, render: np.ndarray, preset: str = "significant") -> Dict[str, Any]:
    result = detect_defects(gt, render, **PRESETS[preset])
    return {
        "serious": bool(result["serious"]),
        "artifact_score": float(result["artifact_score"]),
        "serious_artifact_score": float(result["serious_artifact_score"]),
        "artifact_count": int(result["artifact_count"]),
        "largest_area": int(result["largest_area"]),
    }


def _thumbnail(image: np.ndarray, width: int) -> Image.Image:
    pil = Image.fromarray(image)
    height = max(1, int(round(pil.height * width / pil.width)))
    return pil.resize((width, height), Image.Resampling.LANCZOS)


def write_contact_sheet(rows: Iterable[Tuple[str, np.ndarray, np.ndarray]], path: Path, width: int) -> None:
    prepared = []
    for name, gt, render in rows:
        residual = np.clip(np.abs(gt.astype(np.int16) - render.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        panels = [_thumbnail(image, width) for image in (gt, render, residual)]
        row_height = max(panel.height for panel in panels)
        prepared.append((name, panels, row_height))
    if not prepared:
        raise RuntimeError("Cannot write an empty temporal ROI contact sheet")
    label_height = 22
    canvas = Image.new("RGB", (width * 3, sum(row_height + label_height for _, _, row_height in prepared)), "black")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for name, panels, row_height in prepared:
        draw.text((4, y + 3), f"{name}: GT | render | residual x4", fill="white")
        y += label_height
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * width, y))
        y += row_height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _previous_tracked_boxes(path: Path | None) -> Dict[str, Box]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["name"]): tuple(int(value) for value in row["bbox_xyxy"])
        for row in payload.get("rois", [])
        if row.get("category") == "tracked"
    }


def build_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    if (args.previous_dataset is None) != (args.previous_protocol is None):
        raise ValueError("previous-dataset and previous-protocol must be provided together")
    if args.thumbnail_width <= 0:
        raise ValueError("thumbnail-width must be positive")

    pairs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    full_views = []
    for eval_idx in range(3):
        gt, render, render_path = load_render_pair(args.render_dir, eval_idx)
        pairs[eval_idx] = (gt, render)
        full_views.append(
            {
                "eval_idx": eval_idx,
                "render": str(render_path),
                **_artifact_result(gt, render),
            }
        )

    roi_specs: List[Dict[str, Any]] = [
        {"name": name, "eval_idx": eval_idx, "box": box, "category": "permanent", "confidence": 1.0}
        for name, eval_idx, box in PERMANENT_ROIS
    ]
    previous_boxes = _previous_tracked_boxes(args.previous_protocol)
    for name, eval_idx, seed_box in SEED_TRACKED_ROIS:
        if args.previous_dataset is None:
            tracked = TrackedBox(seed_box, 1.0, 1.0, 0.0)
            tracking_source = "007747_seed" if args.frame == "007747" else "fixed_reference_coordinates"
        else:
            prior = previous_boxes.get(name)
            if prior is None:
                raise RuntimeError(f"Previous protocol is missing tracked ROI {name}")
            tracked = track_box(
                load_rgb(eval_gt_path(args.previous_dataset, eval_idx)),
                load_rgb(eval_gt_path(args.dataset, eval_idx)),
                prior,
            )
            tracking_source = "pyramidal_lk_forward_backward"
        roi_specs.append(
            {
                "name": name,
                "eval_idx": eval_idx,
                "box": tracked.box,
                "category": "tracked",
                "confidence": tracked.confidence,
                "valid_fraction": tracked.valid_fraction,
                "median_forward_backward_error": tracked.median_forward_backward_error,
                "tracking_source": tracking_source,
            }
        )

    if args.previous_dataset is None:
        temporal_reference = args.dataset.parent / "007740"
        if not temporal_reference.is_dir():
            raise FileNotFoundError(temporal_reference)
    else:
        temporal_reference = args.previous_dataset
    for eval_idx in range(3):
        previous_gt = load_rgb(eval_gt_path(temporal_reference, eval_idx))
        current_gt = load_rgb(eval_gt_path(args.dataset, eval_idx))
        broad_boxes, hole_boxes = motion_boxes(previous_gt, current_gt)
        for index, box in enumerate(broad_boxes):
            roi_specs.append(
                {
                    "name": f"broad_motion_eval{eval_idx}_{index}",
                    "eval_idx": eval_idx,
                    "box": box,
                    "category": "broad_motion",
                    "confidence": 1.0,
                }
            )
        for index, box in enumerate(hole_boxes):
            roi_specs.append(
                {
                    "name": f"possible_hole_eval{eval_idx}_{index}",
                    "eval_idx": eval_idx,
                    "box": box,
                    "category": "possible_hole",
                    "confidence": 1.0,
                }
            )

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    rois = []
    contact_rows = []
    for spec in roi_specs:
        eval_idx = int(spec["eval_idx"])
        gt_full, render_full = pairs[eval_idx]
        box = expand_box_minimum(spec["box"], gt_full.shape[1], gt_full.shape[0])
        x0, y0, x1, y1 = box
        gt = gt_full[y0:y1, x0:x1]
        render = render_full[y0:y1, x0:x1]
        row = {
            **{key: value for key, value in spec.items() if key != "box"},
            "bbox_xyxy": list(box),
            "metrics": roi_metrics(gt, render, lpips),
            "artifact": _artifact_result(gt, render),
        }
        rois.append(row)
        contact_rows.append((str(spec["name"]), gt, render))

    categories = {str(row["category"]) for row in rois}
    tracking_confidences = [float(row["confidence"]) for row in rois if row["category"] == "tracked"]
    completeness_errors = []
    if len(full_views) != 3:
        completeness_errors.append("expected exactly three eval views")
    missing_categories = sorted(REQUIRED_CATEGORIES - categories)
    if missing_categories:
        completeness_errors.append(f"missing ROI categories: {missing_categories}")
    required_names = {name for name, _, _ in PERMANENT_ROIS + SEED_TRACKED_ROIS}
    missing_names = sorted(required_names - {str(row["name"]) for row in rois})
    if missing_names:
        completeness_errors.append(f"missing required ROIs: {missing_names}")

    contact_sheet = args.out_dir / "contact_sheet_gt_render_residual.jpg"
    write_contact_sheet(contact_rows, contact_sheet, args.thumbnail_width)
    protocol = {
        "schema_version": 1,
        "frame": args.frame,
        "dataset": str(args.dataset),
        "render_dir": str(args.render_dir),
        "temporal_difference_reference": str(temporal_reference),
        "status": "complete" if not completeness_errors else "incomplete",
        "completeness_errors": completeness_errors,
        "full_views": full_views,
        "rois": rois,
        "tracking": {
            "minimum_confidence": min(tracking_confidences, default=0.0),
            "required_confidence": float(args.tracking_confidence_min),
            "ambiguous": any(value < args.tracking_confidence_min for value in tracking_confidences),
        },
        "contact_sheet": str(contact_sheet),
        "full_view_serious_count": sum(bool(row["serious"]) for row in full_views),
        "roi_serious_count": sum(bool(row["artifact"]["serious"]) for row in rois),
    }
    atomic_json(args.out_dir / "temporal_roi_protocol.json", protocol)
    return protocol


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(args)
    print(
        f"frame={args.frame} status={protocol['status']} "
        f"full_serious={protocol['full_view_serious_count']} "
        f"roi_serious={protocol['roi_serious_count']} "
        f"tracking_confidence={protocol['tracking']['minimum_confidence']:.3f}",
        flush=True,
    )
    return 0 if protocol["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
