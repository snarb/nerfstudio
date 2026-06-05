#!/usr/bin/env python3
"""Render low-resolution LookCloser crop sheets for visual gates."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

from nerfstudio.utils.eval_utils import eval_setup


DEFAULT_DATA = Path("/fsx/oregon/tank_bkup/6A_4_EXR/nerfstudio_processed/007740_hd_aabb4_multicamera_eval3_ns")
DEFAULT_BASELINE = Path(
    "/home/ubuntu/repos/nerfstudio/LookCloser/repro_runs/local_nerfstudio_runs/"
    "007740_hd_aabb4_multicamera_eval3_ns_focus_scene15/instant-ngp-bounded/"
    "stage4_train_num_rays_per_batch_12288_seed44/renders_best_step-000030376"
)

CROPS = [
    ("left_stand_connector_eval0", 0, (320, 0, 617, 530)),
    ("left_stand_eval0", 0, (300, 0, 650, 650)),
    ("floor_crack_eval0", 0, (1110, 715, 1410, 900)),
    ("fingers_right_eval1", 1, (860, 290, 1210, 590)),
    ("stand_label_eval2", 2, (60, 450, 290, 900)),
    ("tangled_cable_eval2", 2, (0, 130, 300, 500)),
    ("fingers_center_eval2", 2, (690, 330, 980, 610)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline-renders", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--crop-name", default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=1024)
    return parser.parse_args()


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return -10 * math.log10(np.mean((pred / 255.0 - target / 255.0) ** 2) + 1e-12)


def checkpoint_step(checkpoint: Path) -> int:
    return int(checkpoint.stem.split("-")[-1])


def selected_checkpoint(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = summary.get("selected_checkpoint")
    return Path(selected) if selected else None


def eval_config_for_checkpoint(
    config: Path,
    checkpoint: Path,
    eval_num_rays_per_chunk: int,
) -> Path:
    step = checkpoint_step(checkpoint)
    eval_config = config.with_name(f"crop_gate_config_step_{step}.yml")
    text = config.read_text(encoding="utf-8")
    if re.search(r"^load_step:", text, flags=re.MULTILINE):
        text = re.sub(r"^load_step:.*$", f"load_step: {step}", count=1, string=text, flags=re.MULTILINE)
    else:
        text = text.replace("load_scheduler:", f"load_step: {step}\nload_scheduler:", 1)
    text = re.sub(
        r"^(\s*eval_num_rays_per_chunk:\s*).*$",
        rf"\g<1>{eval_num_rays_per_chunk}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    eval_config.write_text(text, encoding="utf-8")
    return eval_config


def main() -> int:
    args = parse_args()
    config = args.run_dir / "config.yml"
    output_dir = args.output_dir or (args.run_dir / f"crop_gate_stride{args.stride}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = selected_checkpoint(args.run_dir, args.checkpoint)
    if checkpoint is not None:
        config = eval_config_for_checkpoint(
            config,
            checkpoint,
            args.eval_num_rays_per_chunk,
        )

    _, pipeline, checkpoint, step = eval_setup(config, eval_num_rays_per_chunk=args.eval_num_rays_per_chunk)
    pipeline.eval()
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader

    rows = []
    sheets = []
    crops = [crop for crop in CROPS if args.crop_name is None or crop[0] == args.crop_name]
    if not crops:
        raise ValueError(f"Unknown crop name {args.crop_name!r}. Available: {[crop[0] for crop in CROPS]}")
    with torch.no_grad():
        for name, eval_idx, (x0, y0, x1, y1) in crops:
            camera, _ = dataloader.get_camera(eval_idx)
            ys = torch.arange(y0, y1, args.stride)
            xs = torch.arange(x0, x1, args.stride)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            # Nerfstudio camera coords are stored as (row/y, col/x).
            coords = torch.stack([yy, xx], dim=-1).float()
            ray_bundle = camera.generate_rays(camera_indices=0, coords=coords, keep_shape=True)

            start_time = time.time()
            pred = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)["rgb"].clamp(0, 1).cpu().numpy()
            seconds = time.time() - start_time
            pred_u8 = (pred * 255).astype(np.uint8)

            gt_path = args.data / "images" / f"frame_eval_{eval_idx + 1:05d}.jpg"
            gt = Image.open(gt_path).convert("RGB").crop((x0, y0, x1, y1))
            gt = gt.resize((pred_u8.shape[1], pred_u8.shape[0]))

            baseline_image = Image.open(args.baseline_renders / f"eval_img_{eval_idx:04d}.png").convert("RGB")
            baseline_width = baseline_image.width // 2
            baseline_crop = baseline_image.crop((baseline_width + x0, y0, baseline_width + x1, y1))
            baseline_crop = baseline_crop.resize((pred_u8.shape[1], pred_u8.shape[0]))

            candidate = Image.fromarray(pred_u8)
            gt_np = np.asarray(gt)
            cand_np = np.asarray(candidate)
            baseline_np = np.asarray(baseline_crop)
            row = {
                "crop": name,
                "rays": int(pred_u8.shape[0] * pred_u8.shape[1]),
                "seconds": seconds,
                "candidate_psnr": psnr(cand_np, gt_np),
                "candidate_ssim": ssim(gt_np, cand_np, channel_axis=2, data_range=255),
                "candidate_pixel_std": float(cand_np.std()),
                "baseline_psnr": psnr(baseline_np, gt_np),
                "baseline_ssim": ssim(gt_np, baseline_np, channel_axis=2, data_range=255),
            }
            rows.append(row)

            sheet = Image.new("RGB", (gt.width * 3, gt.height + 24), "white")
            draw = ImageDraw.Draw(sheet)
            for idx, (label, image) in enumerate([("gt", gt), ("instant_ngp", baseline_crop), ("candidate", candidate)]):
                sheet.paste(image, (idx * gt.width, 18))
                draw.text((idx * gt.width + 4, 2), label, fill=(0, 0, 0))
            crop_path = output_dir / f"{name}.png"
            sheet.save(crop_path)
            sheets.append(sheet)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8") as f:
        f.write(
            "crop,rays,seconds,candidate_psnr,candidate_ssim,candidate_pixel_std,"
            "baseline_psnr,baseline_ssim\n"
        )
        for row in rows:
            f.write(
                f"{row['crop']},{row['rays']},{row['seconds']:.3f},"
                f"{row['candidate_psnr']:.4f},{row['candidate_ssim']:.5f},"
                f"{row['candidate_pixel_std']:.4f},{row['baseline_psnr']:.4f},"
                f"{row['baseline_ssim']:.5f}\n"
            )

    if sheets:
        width = max(sheet.width for sheet in sheets)
        height = sum(sheet.height for sheet in sheets)
        combined = Image.new("RGB", (width, height), "white")
        y = 0
        for sheet in sheets:
            combined.paste(sheet, (0, y))
            y += sheet.height
        combined.save(output_dir / "all_crops.png")

    print(f"checkpoint={checkpoint}")
    print(f"step={step}")
    print(f"crop_dir={output_dir}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
