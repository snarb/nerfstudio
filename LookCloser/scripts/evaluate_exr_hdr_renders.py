#!/usr/bin/env python3
"""Evaluate paired linear-EXR renders and produce exposure-bracket review sheets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.data.utils.data_utils import load_exr_image
from nerfstudio.utils.hdr import calibrate_exr_paths, scene_linear_to_pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nits-per-scene-unit", type=float, default=None)
    parser.add_argument("--data", type=Path, default=None, help="Dataset used to reproduce auto calibration.")
    parser.add_argument("--black-nits", type=float, default=0.005)
    parser.add_argument("--peak-nits", type=float, default=10000.0)
    parser.add_argument("--preview-reference-nits", type=float, default=100.0)
    parser.add_argument("--preview-ev", type=float, nargs="+", default=(-2.0, 0.0, 2.0))
    parser.add_argument(
        "--lpips-max-edge",
        type=int,
        default=0,
        help="Optional LPIPS downscale; zero keeps authoritative full resolution.",
    )
    return parser.parse_args()


def paired_paths(render_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for pred in sorted(render_dir.glob("*_pred_*.exr")):
        gt = pred.with_name(pred.name.replace("_pred_", "_gt_", 1))
        if not gt.is_file():
            raise FileNotFoundError(f"Missing GT pair for {pred}: {gt}")
        pairs.append((pred, gt))
    if not pairs:
        raise FileNotFoundError(f"No *_pred_*.exr renders found in {render_dir}")
    return pairs


def chw(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).to(device)


def neutral_preview(image: np.ndarray, nits_per_unit: float, reference_nits: float, ev: float) -> np.ndarray:
    """Fixed, monotone display transform used for review only; exposure never affects metrics/training."""
    linear = np.maximum(image.astype(np.float32), 0.0)
    linear *= nits_per_unit * (2.0**ev) / reference_nits
    mapped = linear / (1.0 + linear)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def resize_for_sheet(image: np.ndarray, width: int = 480) -> np.ndarray:
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    return np.asarray(pil.resize((width, height), Image.Resampling.LANCZOS))


def save_review_sheet(
    path: Path,
    pred: np.ndarray,
    gt: np.ndarray,
    nits_per_unit: float,
    reference_nits: float,
    exposures: list[float],
) -> None:
    columns: list[tuple[str, np.ndarray]] = []
    for ev in exposures:
        columns.append((f"GT {ev:+g} EV", neutral_preview(gt, nits_per_unit, reference_nits, ev)))
        columns.append((f"Prediction {ev:+g} EV", neutral_preview(pred, nits_per_unit, reference_nits, ev)))
    error = np.abs(
        scene_linear_to_pq(
            torch.from_numpy(np.maximum(pred, 0.0)), nits_per_scene_unit=nits_per_unit
        ).numpy()
        - scene_linear_to_pq(
            torch.from_numpy(np.maximum(gt, 0.0)), nits_per_scene_unit=nits_per_unit
        ).numpy()
    )
    error = np.clip(error / max(float(np.quantile(error, 0.995)), 1e-6), 0.0, 1.0)
    columns.append(("PQ absolute error (q99.5)", np.repeat(error.mean(axis=-1, keepdims=True), 3, axis=-1)))
    thumbs = [(label, resize_for_sheet(image)) for label, image in columns]
    label_height = 28
    canvas = Image.new("RGB", (sum(x.shape[1] for _, x in thumbs), thumbs[0][1].shape[0] + label_height), "black")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, thumb in thumbs:
        canvas.paste(Image.fromarray(thumb), (x, label_height))
        draw.text((x + 6, 7), label, fill="white")
        x += thumb.shape[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    calibration = None
    if args.nits_per_scene_unit is None:
        if args.data is None:
            raise ValueError("Provide --nits-per-scene-unit or --data for deterministic calibration")
        train_paths = sorted((args.data / "images").glob("frame_train_*.exr"))
        if not train_paths:
            raise FileNotFoundError(f"No training EXRs in {args.data / 'images'}")
        calibration = calibrate_exr_paths(train_paths)
        args.nits_per_scene_unit = calibration.nits_per_scene_unit
    if args.nits_per_scene_unit <= 0 or args.black_nits < 0 or args.peak_nits <= 0:
        raise ValueError("Invalid PQ luminance parameters")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    rows = []
    for index, (pred_path, gt_path) in enumerate(paired_paths(args.render_dir)):
        pred = load_exr_image(pred_path)[..., :3]
        gt = load_exr_image(gt_path)[..., :3]
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {pred_path} {pred.shape} vs {gt_path} {gt.shape}")
        finite = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
        if not finite.all():
            pred = np.where(finite[..., None], pred, 0.0)
            gt = np.where(finite[..., None], gt, 0.0)
        pred_t, gt_t = chw(pred, device), chw(gt, device)
        with torch.inference_mode():
            pred_pq = scene_linear_to_pq(
                pred_t.clamp_min(0.0),
                nits_per_scene_unit=args.nits_per_scene_unit,
                black_nits=args.black_nits,
            ).clamp(0.0, 1.0)
            gt_pq = scene_linear_to_pq(
                gt_t.clamp_min(0.0),
                nits_per_scene_unit=args.nits_per_scene_unit,
                black_nits=args.black_nits,
            ).clamp(0.0, 1.0)
            mse = torch.mean((pred_pq - gt_pq) ** 2)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            ssim = structural_similarity_index_measure(pred_pq, gt_pq, data_range=1.0)
            lp_pred, lp_gt = pred_pq, gt_pq
            max_edge = max(pred.shape[:2])
            if args.lpips_max_edge > 0 and max_edge > args.lpips_max_edge:
                scale = args.lpips_max_edge / max_edge
                size = (max(32, round(pred.shape[0] * scale)), max(32, round(pred.shape[1] * scale)))
                lp_pred = torch.nn.functional.interpolate(pred_pq, size=size, mode="area")
                lp_gt = torch.nn.functional.interpolate(gt_pq, size=size, mode="area")
            lpips_value = lpips(lp_pred, lp_gt)
        row = {
            "index": index,
            "prediction": str(pred_path),
            "ground_truth": str(gt_path),
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips_value.item()),
            "nonfinite_pixel_fraction": float(1.0 - finite.mean()),
            "negative_prediction_channel_fraction": float((pred < 0).mean()),
            "prediction_above_peak_channel_fraction": float(
                (pred * args.nits_per_scene_unit > args.peak_nits).mean()
            ),
        }
        rows.append(row)
        save_review_sheet(
            args.output_dir / f"review_{index:04d}.jpg",
            pred,
            gt,
            args.nits_per_scene_unit,
            args.preview_reference_nits,
            list(args.preview_ev),
        )
        print(
            f"image={index} psnr={row['psnr']:.5f} ssim={row['ssim']:.6f} lpips={row['lpips']:.6f}",
            flush=True,
        )
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "psnr",
            "ssim",
            "lpips",
            "nonfinite_pixel_fraction",
            "negative_prediction_channel_fraction",
            "prediction_above_peak_channel_fraction",
        )
    }
    result = {
        "schema": 1,
        "metric_domain": "ST2084 PQ of non-negative scene-linear RGB",
        "nits_per_scene_unit": args.nits_per_scene_unit,
        "black_nits": args.black_nits,
        "peak_nits": args.peak_nits,
        "lpips_max_edge": args.lpips_max_edge,
        "calibration": calibration.as_metadata() if calibration is not None else None,
        "aggregate": aggregate,
        "images": rows,
    }
    output = args.output_dir / "hdr_metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"metrics={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
