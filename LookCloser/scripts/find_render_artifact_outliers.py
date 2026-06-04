#!/usr/bin/env python3
"""Find local render artifact outliers against GT and an optional baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--baseline-render-dir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-indices", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--min-std", type=float, default=4.0)
    return parser.parse_args()


def load_prediction(path: Path, gt_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    gt_w, gt_h = gt_size
    if image.width == gt_w * 2 and image.height == gt_h:
        image = image.crop((gt_w, 0, gt_w * 2, gt_h))
    elif image.size != gt_size:
        image = image.resize(gt_size)
    return image


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return -10.0 * math.log10(np.mean((pred / 255.0 - target / 255.0) ** 2) + 1e-12)


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    win_size = min(7, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
    if win_size < 3:
        return float("nan")
    return float(ssim(a, b, channel_axis=2, data_range=255, win_size=win_size))


def crop_array(image: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    return image[y : y + size, x : x + size]


def draw_sheet(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    crop_size = rows[0]["gt"].width
    label_h = 48
    columns = 3
    tile_w = crop_size * columns
    tile_h = crop_size + label_h
    sheet = Image.new("RGB", (tile_w, tile_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        y0 = row_idx * tile_h
        labels = [("gt", row["gt"]), ("baseline", row.get("baseline")), ("candidate", row["candidate"])]
        for col, (label, image) in enumerate(labels):
            x0 = col * crop_size
            if image is None:
                image = Image.new("RGB", (crop_size, crop_size), (30, 30, 30))
            sheet.paste(image, (x0, y0 + label_h))
            draw.text((x0 + 4, y0 + 4), label, fill=(0, 0, 0))
        summary = (
            f"eval={row['eval_idx']} xy=({row['x']},{row['y']}) "
            f"cand_ssim={row['candidate_ssim']:.4f} base_ssim={row['baseline_ssim']:.4f} "
            f"delta={row['ssim_delta']:.4f}"
        )
        draw.text((4, y0 + 22), summary, fill=(0, 0, 0))
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for eval_idx in args.eval_indices:
        gt_path = args.data / "images" / f"frame_eval_{eval_idx + 1:05d}.jpg"
        candidate_path = args.render_dir / f"eval_img_{eval_idx:04d}.png"
        if not gt_path.exists() or not candidate_path.exists():
            continue
        gt_image = Image.open(gt_path).convert("RGB")
        candidate_image = load_prediction(candidate_path, gt_image.size)
        baseline_image = None
        if args.baseline_render_dir is not None:
            baseline_path = args.baseline_render_dir / f"eval_img_{eval_idx:04d}.png"
            if baseline_path.exists():
                baseline_image = load_prediction(baseline_path, gt_image.size)

        gt = np.asarray(gt_image)
        candidate = np.asarray(candidate_image)
        baseline = np.asarray(baseline_image) if baseline_image is not None else None
        height, width = gt.shape[:2]
        size = args.crop_size
        for y in range(0, height - size + 1, args.stride):
            for x in range(0, width - size + 1, args.stride):
                gt_crop = crop_array(gt, x, y, size)
                if float(gt_crop.std()) < args.min_std:
                    continue
                cand_crop = crop_array(candidate, x, y, size)
                cand_ssim = safe_ssim(gt_crop, cand_crop)
                cand_psnr = psnr(cand_crop, gt_crop)
                if baseline is not None:
                    base_crop = crop_array(baseline, x, y, size)
                    base_ssim = safe_ssim(gt_crop, base_crop)
                    base_psnr = psnr(base_crop, gt_crop)
                else:
                    base_crop = None
                    base_ssim = float("nan")
                    base_psnr = float("nan")
                records.append(
                    {
                        "eval_idx": eval_idx,
                        "x": x,
                        "y": y,
                        "candidate_ssim": cand_ssim,
                        "candidate_psnr": cand_psnr,
                        "baseline_ssim": base_ssim,
                        "baseline_psnr": base_psnr,
                        "ssim_delta": cand_ssim - base_ssim if baseline is not None else float("nan"),
                        "psnr_delta": cand_psnr - base_psnr if baseline is not None else float("nan"),
                        "gt": Image.fromarray(gt_crop),
                        "candidate": Image.fromarray(cand_crop),
                        "baseline": Image.fromarray(base_crop) if base_crop is not None else None,
                    }
                )

    worst_candidate = sorted(records, key=lambda row: row["candidate_ssim"])[: args.top_k]
    if args.baseline_render_dir is not None:
        worst_delta = sorted(records, key=lambda row: row["ssim_delta"])[: args.top_k]
    else:
        worst_delta = []

    for name, rows in [("worst_candidate_ssim", worst_candidate), ("worst_vs_baseline_delta", worst_delta)]:
        csv_path = args.output_dir / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "eval_idx",
                    "x",
                    "y",
                    "candidate_ssim",
                    "candidate_psnr",
                    "baseline_ssim",
                    "baseline_psnr",
                    "ssim_delta",
                    "psnr_delta",
                ],
            )
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                output_row = {"rank": rank}
                for key in writer.fieldnames:
                    if key == "rank":
                        continue
                    value = row.get(key)
                    output_row[key] = f"{value:.6f}" if isinstance(value, float) else value
                writer.writerow(output_row)
        draw_sheet(rows, args.output_dir / f"{name}.png")

    print(f"output_dir={args.output_dir}")
    print(f"windows={len(records)}")
    if worst_candidate:
        row = worst_candidate[0]
        print(
            "worst_candidate="
            f"eval{row['eval_idx']} x={row['x']} y={row['y']} "
            f"ssim={row['candidate_ssim']:.6f} psnr={row['candidate_psnr']:.4f}"
        )
    if worst_delta:
        row = worst_delta[0]
        print(
            "worst_delta="
            f"eval{row['eval_idx']} x={row['x']} y={row['y']} "
            f"delta={row['ssim_delta']:.6f} cand={row['candidate_ssim']:.6f} base={row['baseline_ssim']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
