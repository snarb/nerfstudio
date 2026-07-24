#!/usr/bin/env python3
"""Score and archive the fixed 007747 contact-hands/chain visual protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from detect_structural_artifacts import PRESETS, detect_defects


Box = Tuple[int, int, int, int]
CONTACT_HANDS_CHAIN_BOX: Box = (700, 100, 1120, 480)
LEADER_ROI_METRICS = {
    "psnr": 29.73538011794537,
    "ssim": 0.7735831141471863,
    "lpips": 0.11203832924365997,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default="007747")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--leader-render-dir", type=Path, required=True)
    parser.add_argument(
        "--scratch-render-dir",
        type=Path,
        default=None,
        help="Optional accepted 007747 scratch renders for the native comparison sheet.",
    )
    parser.add_argument(
        "--visual-verdict",
        choices=("pending", "pass", "fail"),
        default="pending",
        help="Human/agent review of finger separation, chain continuity, and blur.",
    )
    parser.add_argument("--visual-note", default="")
    parser.add_argument(
        "--visual-change",
        choices=("not_applicable", "improved", "no_improvement", "regressed"),
        default="not_applicable",
        help="Manual comparison with the preceding checkpoint for plateau decisions.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def split_render_pair(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = load_rgb(path)
    if image.shape[1] % 2:
        raise RuntimeError(f"Expected an even-width GT|render image: {path} {image.shape}")
    width = image.shape[1] // 2
    return image[:, :width], image[:, width:]


def crop(image: np.ndarray, box: Box = CONTACT_HANDS_CHAIN_BOX) -> np.ndarray:
    x0, y0, x1, y1 = box
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"ROI {box} does not fit image {width}x{height}")
    return image[y0:y1, x0:x1]


def metric_tensors(gt: np.ndarray, render: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    for image in (gt, render):
        tensors.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0)
    return tensors[0], tensors[1]


def gradient_metrics(gt_tensor: torch.Tensor, render_tensor: torch.Tensor) -> Dict[str, float]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gradients = []
    for tensor in (gt_tensor, render_tensor):
        gray = tensor.mean(dim=1, keepdim=True)
        grad_x = torch.nn.functional.conv2d(gray, kernel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray, kernel_y, padding=1)
        gradients.append(torch.sqrt(grad_x.square() + grad_y.square() + 1e-12))
    return {
        "gradient_ratio": float((gradients[1].mean() / gradients[0].mean()).item()),
        "gradient_mae": float(torch.mean(torch.abs(gradients[1] - gradients[0])).item()),
    }


def roi_metrics(
    gt: np.ndarray,
    render: np.ndarray,
    lpips: LearnedPerceptualImagePatchSimilarity,
) -> Dict[str, float]:
    gt_tensor, render_tensor = metric_tensors(gt, render)
    mse = float(torch.mean((gt_tensor - render_tensor).square()).item())
    with torch.no_grad():
        lpips_value = float(lpips(render_tensor, gt_tensor).item())
    lpips.reset()
    return {
        "psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
        "ssim": float(
            structural_similarity_index_measure(render_tensor, gt_tensor, data_range=1.0).item()
        ),
        "lpips": lpips_value,
        **gradient_metrics(gt_tensor, render_tensor),
    }


def artifact_result(gt: np.ndarray, render: np.ndarray) -> Dict[str, Any]:
    result = detect_defects(gt, render, **PRESETS["significant"])
    return {
        "serious": bool(result["serious"]),
        "artifact_score": float(result["artifact_score"]),
        "serious_artifact_score": float(result["serious_artifact_score"]),
        "artifact_count": int(result["artifact_count"]),
        "largest_area": int(result["largest_area"]),
    }


def save_contact_sheet(
    leader_gt: np.ndarray,
    leader_render: np.ndarray,
    target_gt: np.ndarray,
    target_render: np.ndarray,
    path: Path,
    scratch_gt: np.ndarray | None = None,
    scratch_render: np.ndarray | None = None,
) -> None:
    if (scratch_gt is None) != (scratch_render is None):
        raise ValueError("scratch_gt and scratch_render must be provided together")
    panels = [leader_gt, leader_render]
    labels = ["leader 007740 GT", "leader 007740 render"]
    if scratch_gt is not None and scratch_render is not None:
        panels.extend([scratch_gt, scratch_render])
        labels.extend(["scratch 007747 GT", "scratch 007747 render"])
    panels.extend([target_gt, target_render])
    labels.extend(["target 007747 GT", "target 007747 render"])
    panel_height, panel_width = panels[0].shape[:2]
    if any(panel.shape[:2] != (panel_height, panel_width) for panel in panels):
        raise ValueError("Every contact-sheet panel must have identical native dimensions")
    label_height = 24
    gap = 8
    rows = len(panels) // 2
    canvas = Image.new(
        "RGB",
        (panel_width * 2 + gap * 3, (panel_height + label_height) * rows + gap * (rows + 1)),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        column, row = index % 2, index // 2
        x = gap + column * (panel_width + gap)
        y = gap + row * (panel_height + label_height + gap)
        draw.text((x + 3, y + 4), label, fill="white")
        canvas.paste(Image.fromarray(panel), (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    if args.frame != "007747":
        raise ValueError("The fixed contact-hands/chain protocol is defined for frame 007747")
    eval_path = args.dataset / "images" / "frame_eval_00001.jpg"
    dataset_gt = load_rgb(eval_path)
    full_views = []
    pairs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for eval_idx in range(3):
        render_path = args.render_dir / f"eval_img_{eval_idx:04d}.png"
        gt, render = split_render_pair(render_path)
        if eval_idx == 0 and not np.array_equal(gt, dataset_gt):
            # ns-eval writes the decoded GT panel; JPEG decode should be byte-identical.
            raise RuntimeError(f"eval_img_0000 GT panel is not bound to {eval_path}")
        pairs[eval_idx] = (gt, render)
        full_views.append({"eval_idx": eval_idx, "render": str(render_path), **artifact_result(gt, render)})

    leader_gt_full, leader_render_full = split_render_pair(args.leader_render_dir / "eval_img_0000.png")
    scratch_gt_full = None
    scratch_gt = None
    scratch_render = None
    if getattr(args, "scratch_render_dir", None) is not None:
        scratch_gt_full, scratch_render_full = split_render_pair(
            args.scratch_render_dir / "eval_img_0000.png"
        )
        scratch_gt = crop(scratch_gt_full)
        scratch_render = crop(scratch_render_full)
    target_gt = crop(pairs[0][0])
    target_render = crop(pairs[0][1])
    leader_gt = crop(leader_gt_full)
    leader_render = crop(leader_render_full)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    target_metrics = roi_metrics(target_gt, target_render, lpips)
    leader_metrics = roi_metrics(leader_gt, leader_render, lpips)
    scratch_metrics = (
        roi_metrics(scratch_gt, scratch_render, lpips)
        if scratch_gt is not None and scratch_render is not None
        else None
    )
    deltas = {
        "psnr": target_metrics["psnr"] - leader_metrics["psnr"],
        "ssim": target_metrics["ssim"] - leader_metrics["ssim"],
        "lpips": target_metrics["lpips"] - leader_metrics["lpips"],
    }
    crop_dir = args.out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for name, image in (
        ("leader_gt.png", leader_gt),
        ("leader_render.png", leader_render),
        *((("scratch_gt.png", scratch_gt), ("scratch_render.png", scratch_render)) if scratch_gt is not None else ()),
        ("target_gt.png", target_gt),
        ("target_render.png", target_render),
    ):
        assert image is not None
        Image.fromarray(image).save(crop_dir / name)
    sheet_suffix = "3x2" if scratch_gt is not None else "2x2"
    contact_sheet = args.out_dir / f"contact_hands_chain_{sheet_suffix}.png"
    save_contact_sheet(
        leader_gt,
        leader_render,
        target_gt,
        target_render,
        contact_sheet,
        scratch_gt=scratch_gt,
        scratch_render=scratch_render,
    )

    protocol = {
        "schema_version": 1,
        "status": "complete",
        "frame": args.frame,
        "dataset": str(args.dataset),
        "render_dir": str(args.render_dir),
        "leader_render_dir": str(args.leader_render_dir),
        "scratch_render_dir": (
            str(args.scratch_render_dir) if getattr(args, "scratch_render_dir", None) is not None else None
        ),
        "full_views": full_views,
        "full_view_serious_count": sum(bool(row["serious"]) for row in full_views),
        "roi": {
            "name": "contact_hands_chain_eval0",
            "eval_idx": 0,
            "bbox_xyxy": list(CONTACT_HANDS_CHAIN_BOX),
            "metrics": target_metrics,
            "artifact": artifact_result(target_gt, target_render),
            "leader_metrics": leader_metrics,
            "leader_frozen_metrics": LEADER_ROI_METRICS,
            "scratch_metrics": scratch_metrics,
            "scratch_gt_matches_target_revision": (
                bool(np.array_equal(scratch_gt_full, dataset_gt))
                if scratch_gt_full is not None
                else None
            ),
            "delta_to_leader": deltas,
        },
        "visual_gate": {
            "verdict": args.visual_verdict,
            "note": args.visual_note,
            "change_from_previous": getattr(args, "visual_change", "not_applicable"),
            "requirements": [
                "individual fingers remain visibly separated",
                "chain links remain sharp and continuous without gaps",
                "contact area is not visibly blurrier than the canonical leader",
            ],
        },
        "contact_sheet": str(contact_sheet),
    }
    atomic_json(args.out_dir / "static_target_roi_protocol.json", protocol)
    return protocol


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(args)
    metrics = protocol["roi"]["metrics"]
    print(
        f"frame={args.frame} psnr={metrics['psnr']:.6f} ssim={metrics['ssim']:.6f} "
        f"lpips={metrics['lpips']:.6f} full_serious={protocol['full_view_serious_count']} "
        f"visual={protocol['visual_gate']['verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
