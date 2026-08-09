#!/usr/bin/env python3
"""Train or apply the small PQ residual renderer used by the EXR Pareto leader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.data.utils.data_utils import load_exr_image
from nerfstudio.utils.hdr import hdr_display_preview, pq_to_scene_linear, scene_linear_to_pq


class ResidualBlock(nn.Module):
    """Two convolutions with a conservative residual update."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.2 * self.conv2(F.silu(self.conv1(value)))


class HDRResidualRenderer(nn.Module):
    """Predict a bounded PQ correction from primary and auxiliary renders."""

    def __init__(self, channels: int = 48, correction_limit: float = 0.04) -> None:
        super().__init__()
        self.channels = int(channels)
        self.correction_limit = float(correction_limit)
        self.head = nn.Conv2d(9, self.channels, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(self.channels, 1),
            ResidualBlock(self.channels, 2),
            ResidualBlock(self.channels, 4),
            ResidualBlock(self.channels, 2),
            ResidualBlock(self.channels, 1),
        )
        self.tail = nn.Conv2d(self.channels, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((primary, auxiliary, primary - auxiliary), dim=1)
        correction = self.correction_limit * torch.tanh(
            self.tail(self.blocks(F.silu(self.head(inputs))))
        )
        return (primary + correction).clamp(0.0, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pq(path: Path, nits_per_scene_unit: float) -> torch.Tensor:
    image = torch.from_numpy(np.ascontiguousarray(load_exr_image(path)[..., :3])).permute(2, 0, 1)
    return scene_linear_to_pq(
        image.clamp_min(0.0), nits_per_scene_unit=nits_per_scene_unit
    ).clamp(0.0, 1.0)


def save_exr(path: Path, rgb: torch.Tensor) -> None:
    array = rgb.detach().permute(1, 2, 0).cpu().numpy().astype(np.float32)
    if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save EXR {path}")


def load_training_rows(root: Path, nits_per_scene_unit: float) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        index = int(item["index"])
        rows.append(
            {
                **item,
                "primary": load_pq(root / f"p_{index:03d}.exr", nits_per_scene_unit)
                .half()
                .pin_memory(),
                "auxiliary": load_pq(root / f"g_{index:03d}.exr", nits_per_scene_unit)
                .half()
                .pin_memory(),
                "target": load_pq(root / f"gt_{index:03d}.exr", nits_per_scene_unit)
                .half()
                .pin_memory(),
            }
        )
        print(f"loaded={index}", flush=True)
    train = [row for row in rows if row["role"] == "train"]
    validation = [row for row in rows if row["role"] == "validation"]
    if not train or not validation:
        raise ValueError("manifest.json must contain both train and validation rows")
    return train, validation


def sample_patches(
    rows: list[dict], patch_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    batches: list[list[torch.Tensor]] = [[], [], []]
    for _ in range(batch_size):
        row = random.choice(rows)
        _, height, width = row["primary"].shape
        y = random.randrange(height - patch_size + 1)
        x = random.randrange(width - patch_size + 1)
        patches = [
            row[key][:, y : y + patch_size, x : x + patch_size]
            for key in ("primary", "auxiliary", "target")
        ]
        if random.random() < 0.5:
            patches = [patch.flip(-1) for patch in patches]
        if random.random() < 0.5:
            patches = [patch.flip(-2) for patch in patches]
        for destination, patch in zip(batches, patches):
            destination.append(patch)
    return tuple(
        torch.stack(batch).to(device=device, dtype=torch.float32, non_blocking=True)
        for batch in batches
    )


@torch.inference_mode()
def evaluate_validation(
    model: nn.Module, rows: list[dict], lpips: nn.Module, device: torch.device
) -> dict:
    values = []
    model.eval()
    for row in rows:
        primary, auxiliary, target = (
            row[key].unsqueeze(0).to(device=device, dtype=torch.float32)
            for key in ("primary", "auxiliary", "target")
        )
        prediction = model(primary, auxiliary)
        mse = F.mse_loss(prediction, target)
        values.append(
            {
                "index": int(row["index"]),
                "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
                "ssim": float(
                    structural_similarity_index_measure(prediction, target, data_range=1.0).item()
                ),
                "lpips": float(lpips(prediction, target).item()),
            }
        )
    return {
        key: float(np.mean([row[key] for row in values]))
        for key in ("psnr", "ssim", "lpips")
    } | {"images": values}


def train(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_rows, validation_rows = load_training_rows(args.pairs, args.nits_per_scene_unit)
    model = HDRResidualRenderer(args.channels, args.correction_limit).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    lpips.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.05
    )
    baseline = evaluate_validation(model, validation_rows, lpips, device)
    history = []
    print(f"step=0 validation={baseline}", flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        primary, auxiliary, target = sample_patches(
            train_rows, args.patch_size, args.batch_size, device
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(primary, auxiliary)
        mse = F.mse_loss(prediction, target)
        dssim = 1.0 - structural_similarity_index_measure(prediction, target, data_range=1.0)
        perceptual = lpips(prediction, target)
        loss = mse + args.dssim_weight * dssim + args.lpips_weight * perceptual
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0:
            print(
                f"step={step} loss={loss.item():.7f} mse={mse.item():.7f} "
                f"dssim={dssim.item():.6f} lpips={perceptual.item():.6f}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            validation = evaluate_validation(model, validation_rows, lpips, device)
            history.append({"step": step, "validation": validation})
            checkpoint = {
                "state_dict": model.state_dict(),
                "step": step,
                "architecture": {
                    "channels": args.channels,
                    "correction_limit": args.correction_limit,
                },
                "validation": validation,
                "baseline": baseline,
                "nits_per_scene_unit": args.nits_per_scene_unit,
            }
            torch.save(checkpoint, args.output_dir / f"step-{step:06d}.pt")
            (args.output_dir / "history.json").write_text(
                json.dumps({"baseline": baseline, "history": history}, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"step={step} validation={validation}", flush=True)
    return 0


@torch.inference_mode()
def apply(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", {})
    model = HDRResidualRenderer(
        channels=int(architecture.get("channels", 48)),
        correction_limit=float(architecture.get("correction_limit", 0.04)),
    ).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    for primary_path in sorted(args.primary_render_dir.glob("eval_pred_*.exr")):
        suffix = primary_path.name.removeprefix("eval_pred_")
        auxiliary_path = args.auxiliary_render_dir / primary_path.name
        target_path = args.primary_render_dir / f"eval_gt_{suffix}"
        primary = load_pq(primary_path, args.nits_per_scene_unit).unsqueeze(0).to(device)
        auxiliary = load_pq(auxiliary_path, args.nits_per_scene_unit).unsqueeze(0).to(device)
        residual_prediction = model(primary, auxiliary)
        prediction = primary + float(args.blend_beta) * (residual_prediction - primary)
        linear = pq_to_scene_linear(
            prediction[0].clamp(0.0, 1.0), nits_per_scene_unit=args.nits_per_scene_unit
        )
        target = torch.from_numpy(
            np.ascontiguousarray(load_exr_image(target_path)[..., :3])
        ).permute(2, 0, 1)
        save_exr(args.output_dir / primary_path.name, linear)
        save_exr(args.output_dir / f"eval_gt_{suffix}", target)
        if args.preview_exposure_ev is not None:
            pair = torch.cat(
                [
                    hdr_display_preview(target, exposure_ev=args.preview_exposure_ev),
                    hdr_display_preview(linear.cpu(), exposure_ev=args.preview_exposure_ev),
                ],
                dim=1,
            )
            preview = pair.permute(1, 2, 0).numpy()
            Image.fromarray((preview * 255.0 + 0.5).clip(0, 255).astype(np.uint8)).save(
                args.output_dir / f"eval_img_{suffix.replace('.exr', '.png')}"
            )
        print(f"rendered={suffix}", flush=True)
    metadata = {
        "schema": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "primary_render_dir": str(args.primary_render_dir.resolve()),
        "auxiliary_render_dir": str(args.auxiliary_render_dir.resolve()),
        "blend_beta": args.blend_beta,
        "domain": "ST2084 PQ",
        "nits_per_scene_unit": args.nits_per_scene_unit,
    }
    (args.output_dir / "residual_renderer.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--pairs", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--nits-per-scene-unit", type=float, required=True)
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument("--patch-size", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--dssim-weight", type=float, default=0.1)
    train_parser.add_argument("--lpips-weight", type=float, default=0.02)
    train_parser.add_argument("--channels", type=int, default=48)
    train_parser.add_argument("--correction-limit", type=float, default=0.04)
    train_parser.add_argument("--eval-every", type=int, default=250)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="cuda")
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--checkpoint", type=Path, required=True)
    apply_parser.add_argument("--primary-render-dir", type=Path, required=True)
    apply_parser.add_argument("--auxiliary-render-dir", type=Path, required=True)
    apply_parser.add_argument("--output-dir", type=Path, required=True)
    apply_parser.add_argument("--nits-per-scene-unit", type=float, required=True)
    apply_parser.add_argument("--blend-beta", type=float, default=1.0)
    apply_parser.add_argument("--preview-exposure-ev", type=float, default=None)
    apply_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "train" and min(args.steps, args.patch_size, args.batch_size) <= 0:
        parser.error("steps, patch-size and batch-size must be positive")
    if args.command == "apply" and not 0.0 <= args.blend_beta <= 1.0:
        parser.error("blend-beta must be in [0, 1]")
    if args.nits_per_scene_unit <= 0:
        parser.error("nits-per-scene-unit must be positive")
    return args


def main() -> int:
    args = parse_args()
    return train(args) if args.command == "train" else apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
