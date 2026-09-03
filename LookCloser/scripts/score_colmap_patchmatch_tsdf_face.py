#!/usr/bin/env python3
"""Score a manually defined held-out face ROI and write native-resolution review crops.

The polygon is drawn on held-out ground truth only.  It is a post-hoc metric
region: it is never exposed to PatchMatch, TSDF fusion, source selection, or
prediction construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


REVIEW_BOXES = {
    "face_ear_hair": (650, 400, 1150, 950),
    "ear_native": (820, 740, 1060, 950),
    "lipstick_lips_hand": (687, 540, 987, 800),
    "actor_overview": (0, 150, 1400, 1030),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_display_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None or encoded.ndim != 3 or encoded.shape[-1] not in (3, 4):
            raise ValueError(f"Failed to decode RGB OpenEXR: {path}")
        result = encoded[..., (2, 1, 0)].astype(np.float32, copy=False)
    else:
        with Image.open(path) as image:
            result = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if result.ndim != 3 or result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite RGB image: {path}")
    return np.clip(result, 0.0, 1.0)


def validate_polygon(points: object, width: int, height: int) -> list[tuple[int, int]]:
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Every polygon must contain at least three [x, y] vertices")
    result: list[tuple[int, int]] = []
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(value, int) for value in point)
        ):
            raise ValueError(f"Polygon vertex must be two integers: {point!r}")
        x, y = point
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"Polygon vertex outside {width}x{height}: {point!r}")
        result.append((x, y))
    return result


def load_manual_face_mask(path: Path, ground_truth: Path, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    height, width = shape
    if payload.get("selection_method") != "manual_polygon_on_heldout_gt_only":
        raise ValueError("Face ROI must declare selection_method=manual_polygon_on_heldout_gt_only")
    if payload.get("prediction_used_for_selection") is not False:
        raise ValueError("Face ROI must explicitly declare prediction_used_for_selection=false")
    ground_truth_hash = sha256(ground_truth)
    if payload.get("ground_truth_sha256") != ground_truth_hash:
        raise ValueError("Face ROI ground-truth hash does not match the scored image")
    raw_include = payload.get("include_polygons")
    raw_exclude = payload.get("exclude_polygons", [])
    if not isinstance(raw_include, list) or not raw_include:
        raise ValueError("Face ROI requires at least one include polygon")
    if not isinstance(raw_exclude, list):
        raise ValueError("exclude_polygons must be a list")
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in raw_include:
        draw.polygon(validate_polygon(polygon, width, height), fill=255)
    for polygon in raw_exclude:
        draw.polygon(validate_polygon(polygon, width, height), fill=0)
    mask = np.asarray(canvas, dtype=np.uint8) > 0
    if not mask.any():
        raise ValueError("Manual face ROI is empty")
    if float(mask.mean()) > 0.25:
        raise ValueError("Manual face ROI unexpectedly exceeds 25% of the full image")
    return mask, payload


def masked_display_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    lpips_model,
) -> dict[str, float | list[int]]:
    """Match the accepted masked protocol: exact-pixel PSNR and zero-outside SSIM/LPIPS."""

    if prediction.shape != ground_truth.shape or prediction.ndim != 3:
        raise ValueError("prediction and ground_truth must be matching CHW tensors")
    if mask.shape != prediction.shape[-2:] or not bool(mask.any()):
        raise ValueError("mask must be non-empty and match image height/width")
    yy, xx = torch.where(mask)
    x0, x1 = int(xx.min().item()), int(xx.max().item()) + 1
    y0, y1 = int(yy.min().item()), int(yy.max().item()) + 1
    selected_error = (prediction[:, mask] - ground_truth[:, mask]).square().mean()
    psnr = -10.0 * torch.log10(selected_error.clamp_min(1e-12))
    crop_mask = mask[y0:y1, x0:x1].unsqueeze(0)
    pred_crop = torch.where(crop_mask, prediction[:, y0:y1, x0:x1], 0.0).unsqueeze(0)
    gt_crop = torch.where(crop_mask, ground_truth[:, y0:y1, x0:x1], 0.0).unsqueeze(0)
    ssim = structural_similarity_index_measure(pred_crop, gt_crop, data_range=1.0)
    lpips = lpips_model(pred_crop, gt_crop)
    result: dict[str, float | list[int]] = {
        "face_psnr": float(psnr.item()),
        "face_ssim": float(ssim.item()),
        "face_lpips": float(lpips.item()),
        "face_pixel_fraction": float(mask.float().mean().item()),
        "face_bbox_xyxy": [x0, y0, x1, y1],
    }
    if not all(math.isfinite(float(result[key])) for key in ("face_psnr", "face_ssim", "face_lpips")):
        raise ValueError("Non-finite face metric")
    return result


def uint8_rgb(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def write_review_crops(output: Path, prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, str]:
    height, width = prediction.shape[:2]
    result: dict[str, str] = {}
    for name, (x0, y0, x1, y1) in REVIEW_BOXES.items():
        if x1 > width or y1 > height:
            raise ValueError(f"Review crop {name} exceeds {width}x{height}")
        pred_crop = uint8_rgb(prediction[y0:y1, x0:x1])
        gt_crop = uint8_rgb(ground_truth[y0:y1, x0:x1])
        if name == "actor_overview":
            target_width = 700
            target_height = round(pred_crop.shape[0] * target_width / pred_crop.shape[1])
            pred_crop = np.asarray(Image.fromarray(pred_crop).resize((target_width, target_height), Image.Resampling.LANCZOS))
            gt_crop = np.asarray(Image.fromarray(gt_crop).resize((target_width, target_height), Image.Resampling.LANCZOS))
        pair = np.concatenate((gt_crop, pred_crop), axis=1)
        path = output / f"{name}_gt_pred.png"
        Image.fromarray(pair, "RGB").save(path, compress_level=3)
        result[name] = str(path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--face-polygons", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    for name in ("prediction", "ground_truth", "face_polygons"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")
    if not (len(args.frame_id) == 6 and args.frame_id.isdigit()):
        parser.error("--frame-id must be six digits")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prediction = load_display_rgb(args.prediction)
    ground_truth = load_display_rgb(args.ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Prediction/GT shapes differ: {prediction.shape} vs {ground_truth.shape}")
    mask, polygon_payload = load_manual_face_mask(
        args.face_polygons, args.ground_truth, prediction.shape[:2]
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    pred_tensor = torch.from_numpy(prediction).permute(2, 0, 1).to(device)
    gt_tensor = torch.from_numpy(ground_truth).permute(2, 0, 1).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    with torch.inference_mode():
        metrics = masked_display_metrics(pred_tensor, gt_tensor, mask_tensor, lpips_model)

    args.output_dir.mkdir(parents=True)
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(args.output_dir / "face_mask.png", compress_level=3)
    overlay = uint8_rgb(ground_truth)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    overlay[boundary] = (255, 0, 255)
    Image.fromarray(overlay, "RGB").save(args.output_dir / "face_mask_overlay.png", compress_level=3)
    crops = write_review_crops(args.output_dir, prediction, ground_truth)
    receipt = {
        "schema_version": 1,
        "frame_id": args.frame_id,
        "metric_domain": "display-referred RGB clamped to [0, 1]",
        "protocol": {
            "roi": "manual_polygon_on_heldout_gt_only",
            "candidate_surface_mask": False,
            "prediction_used_for_roi_selection": False,
            "face_psnr_definition": "exact selected RGB pixels",
            "face_ssim_lpips_definition": "tight face bbox with both images zero outside face mask",
            "lpips_network": "alex",
            "lpips_normalize": True,
            "not_numerically_comparable_to_old_surface_mask_table": True,
        },
        "prediction": str(args.prediction),
        "prediction_sha256": sha256(args.prediction),
        "ground_truth": str(args.ground_truth),
        "ground_truth_sha256": sha256(args.ground_truth),
        "face_polygons": str(args.face_polygons),
        "face_polygons_sha256": sha256(args.face_polygons),
        "face_polygon_notes": polygon_payload.get("notes"),
        **metrics,
        "review_crops": crops,
    }
    atomic_json(args.output_dir / "metrics.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("frame_id", "face_psnr", "face_ssim", "face_lpips")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
