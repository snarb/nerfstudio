# nerfstudio/scripts/lookcloser_preprocess.py

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image, ImageDraw
from rich.progress import track
from typing_extensions import Literal

try:
    import tinycudann as tcnn
except ImportError:
    print("Error: tinycudann is not installed. Please install it to use LookCloser preprocessing.")
    sys.exit(1)

try:
    from pytorch_msssim import ssim as _pt_ssim  # type: ignore
except Exception:
    _pt_ssim = None

try:
    from torchvision.transforms import functional as TF
except Exception as e:
    raise ImportError("torchvision is required for image loading.") from e

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.utils.rich_utils import CONSOLE


# ============================================================
# SSIM
# ============================================================

def _ssim_fallback(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    SSIM fallback for tensors in [0, 1].

    Args:
        img1: (B, C, H, W)
        img2: (B, C, H, W)

    Returns:
        scalar if size_average=True, otherwise (B,)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )

    per_image = ssim_map.mean(dim=(1, 2, 3))
    return per_image.mean() if size_average else per_image


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    SSIM wrapper. Uses pytorch-msssim if available.
    """
    if _pt_ssim is not None:
        return _pt_ssim(
            img1,
            img2,
            data_range=1.0,
            size_average=size_average,
            win_size=window_size,
        )
    return _ssim_fallback(img1, img2, window_size=window_size, size_average=size_average)


# ============================================================
# 2D Instant-NGP model
# ============================================================

class InstantNGP2D(nn.Module):
    def __init__(
        self,
        n_levels: int = 16,
        n_features: int = 2,
        min_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
    ):
        super().__init__()

        self.n_levels = int(n_levels)
        self.n_features = int(n_features)
        self.min_res = int(min_res)
        self.max_res = int(max_res)

        self.b = float(np.exp((np.log(max_res) - np.log(min_res)) / (n_levels - 1)))

        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features,
                "log2_hashmap_size": int(log2_hashmap_size),
                "base_resolution": int(min_res),
                "per_level_scale": float(self.b),
            },
        )

        self.decoder = tcnn.Network(
            n_input_dims=self.n_levels * self.n_features,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def get_resolution_at_level(self, level_idx: int) -> float:
        level_idx = int(np.clip(level_idx, 0, self.n_levels - 1))
        return float(self.min_res * (self.b ** level_idx))

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoding(uv))

    def render_masked(self, uv: torch.Tensor, max_active_level: int) -> torch.Tensor:
        """
        Forward pass using only levels [0..max_active_level].
        This must be used during progressive training, not only during evaluation.
        """
        max_active_level = int(np.clip(max_active_level, 0, self.n_levels - 1))

        feats = self.encoding(uv)
        n = feats.shape[0]

        feats = feats.view(n, self.n_levels, self.n_features)
        mask = torch.zeros((self.n_levels,), device=uv.device, dtype=feats.dtype)
        mask[: max_active_level + 1] = 1.0

        feats = feats * mask.view(1, self.n_levels, 1)
        feats = feats.view(n, self.n_levels * self.n_features)

        return self.decoder(feats)


# ============================================================
# Image / render helpers
# ============================================================

def load_image_as_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """
    Returns image as (H, W, 3), float32, [0, 1].
    RGBA is composited on white.
    """
    pil = Image.open(path)
    img = TF.to_tensor(pil).permute(1, 2, 0).contiguous()

    if img.shape[-1] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        img = rgb * alpha + (1.0 - alpha)

    if img.shape[-1] != 3:
        img = img[..., :3]

    return img.to(device=device, dtype=torch.float32)


def random_crop_image(
    image: torch.Tensor,
    crop_size: int,
    seed: int = 0,
) -> torch.Tensor:
    if crop_size <= 0:
        return image

    h, w, _ = image.shape
    ch = min(crop_size, h)
    cw = min(crop_size, w)

    rng = np.random.RandomState(seed)
    y0 = int(rng.randint(0, max(1, h - ch + 1)))
    x0 = int(rng.randint(0, max(1, w - cw + 1)))

    return image[y0 : y0 + ch, x0 : x0 + cw, :].contiguous()


def to_uint8_np(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_tensor_image_hwc(image: torch.Tensor, path: Path) -> None:
    arr = image.detach().cpu().numpy()
    Image.fromarray(to_uint8_np(arr)).save(path)


def make_full_uv_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([xx / float(w), yy / float(h)], dim=-1)
    return uv.view(-1, 2)


def render_full_image(
    model: InstantNGP2D,
    h: int,
    w: int,
    level: int,
    chunk: int = 1 << 16,
) -> torch.Tensor:
    device = next(model.parameters()).device
    uv = make_full_uv_grid(h, w, device)

    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, uv.shape[0], chunk):
            outs.append(model.render_masked(uv[start : start + chunk], max_active_level=level))

    return torch.cat(outs, dim=0).view(h, w, 3)


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


# ============================================================
# Patch helpers
# ============================================================

def compute_patch_starts(length: int, patch_size: int, stride: int) -> List[int]:
    """
    Important: no tail patches by default.

    Reason:
    Downstream sampler currently assumes map_y * patch_size and map_x * patch_size.
    Adding tail patches would require storing true starts as metadata.
    """
    if length < patch_size:
        raise ValueError(f"Image side {length} is smaller than patch_size={patch_size}.")
    return list(range(0, length - patch_size + 1, stride))


def make_patch_uv(
    xs: torch.Tensor,
    ys: torch.Tensor,
    h: int,
    w: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Args:
        xs, ys: (B,) patch top-left pixel coordinates.

    Returns:
        uv: (B * P * P, 2)
    """
    device = xs.device
    p = patch_size

    local_x = torch.arange(p, device=device, dtype=torch.float32) + 0.5
    local_y = torch.arange(p, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(local_y, local_x, indexing="ij")

    x = xx.unsqueeze(0) + xs.float().view(-1, 1, 1)
    y = yy.unsqueeze(0) + ys.float().view(-1, 1, 1)

    uv = torch.stack([x / float(w), y / float(h)], dim=-1)
    return uv.view(-1, 2)


def extract_gt_patches(
    image: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Returns:
        patches: (B, 3, P, P)
    """
    patches = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        patch = image[y : y + patch_size, x : x + patch_size, :]
        patches.append(patch.permute(2, 0, 1))

    return torch.stack(patches, dim=0).contiguous()


# ============================================================
# Debug artifact helpers
# ============================================================

def parse_debug_levels(s: str, n_levels: int) -> List[int]:
    levels = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        levels.append(int(np.clip(int(x), 0, n_levels - 1)))

    levels = sorted(set(levels))
    if not levels:
        levels = [0, n_levels - 1]
    return levels


def colorize_freq_map(freq_map: torch.Tensor, min_res: float, max_res: float) -> Image.Image:
    """
    Simple blue-red heatmap without matplotlib dependency.
    """
    f = freq_map.detach().cpu().float().numpy()
    norm = (f - float(min_res)) / max(float(max_res - min_res), 1e-8)
    norm = np.clip(norm, 0.0, 1.0)

    r = norm
    g = 0.25 * (1.0 - np.abs(norm - 0.5) * 2.0)
    b = 1.0 - norm

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(to_uint8_np(rgb))


def upsample_heatmap_to_image(
    heatmap: Image.Image,
    image_h: int,
    image_w: int,
) -> Image.Image:
    return heatmap.resize((image_w, image_h), resample=Image.Resampling.NEAREST)


def save_freq_overlay(
    image: torch.Tensor,
    freq_map: torch.Tensor,
    out_dir: Path,
    min_res: float,
    max_res: float,
) -> None:
    h, w, _ = image.shape

    img_pil = Image.fromarray(to_uint8_np(image.detach().cpu().numpy()))
    heat_small = colorize_freq_map(freq_map, min_res=min_res, max_res=max_res)
    heat_big = upsample_heatmap_to_image(heat_small, h, w)

    heat_small.save(out_dir / "freq_heatmap_patch_grid.png")
    heat_big.save(out_dir / "freq_heatmap_fullres.png")

    overlay = Image.blend(img_pil.convert("RGB"), heat_big.convert("RGB"), alpha=0.45)
    overlay.save(out_dir / "freq_overlay.png")


def save_patch_mosaic(
    gt_patches: torch.Tensor,
    pred_by_level: Dict[int, torch.Tensor],
    ssim_by_level: Dict[int, torch.Tensor],
    out_path: Path,
) -> None:
    """
    Mosaic layout:
        row 0: GT patches
        row 1..N: predictions at selected levels
    """
    levels = sorted(pred_by_level.keys())

    gt_np = gt_patches.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred_np = {
        lvl: pred_by_level[lvl].detach().cpu().permute(0, 2, 3, 1).numpy()
        for lvl in levels
    }

    b, p, _, _ = gt_np.shape
    cols = int(np.ceil(np.sqrt(b)))
    rows = int(np.ceil(b / cols))

    label_w = 110
    panel_h = rows * p
    panel_w = cols * p
    total_h = panel_h * (1 + len(levels))
    total_w = label_w + panel_w

    canvas = Image.new("RGB", (total_w, total_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    def paste_panel(patches: np.ndarray, y_offset: int, label: str) -> None:
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.float32)

        for i in range(b):
            r = i // cols
            c = i % cols
            panel[r * p : (r + 1) * p, c * p : (c + 1) * p, :] = patches[i]

        panel_pil = Image.fromarray(to_uint8_np(panel))
        canvas.paste(panel_pil, (label_w, y_offset))
        draw.text((8, y_offset + 8), label, fill=(255, 255, 255))

    paste_panel(gt_np, 0, "GT")

    y = panel_h
    for lvl in levels:
        scores = ssim_by_level[lvl].detach().cpu().numpy()
        label = f"L{lvl}\nSSIM {scores.mean():.3f}"
        paste_panel(pred_np[lvl], y, label)
        y += panel_h

    canvas.save(out_path)


def save_stats_json(
    path: Path,
    image_name: str,
    image_shape: Tuple[int, int],
    freq_map: torch.Tensor,
    level_map: torch.Tensor,
    min_res: float,
    max_res: float,
    debug_ssim_by_level: Optional[Dict[int, torch.Tensor]] = None,
    full_recon_metrics: Optional[Dict[str, float]] = None,
) -> None:
    freq_cpu = freq_map.detach().cpu().float()
    level_cpu = level_map.detach().cpu().long()

    hist = {}
    for lvl in torch.unique(level_cpu).tolist():
        hist[str(int(lvl))] = int((level_cpu == int(lvl)).sum().item())

    flat = freq_cpu.flatten().numpy()

    stats = {
        "image": image_name,
        "image_h": int(image_shape[0]),
        "image_w": int(image_shape[1]),
        "freq_min": float(freq_cpu.min().item()),
        "freq_max": float(freq_cpu.max().item()),
        "freq_mean": float(freq_cpu.mean().item()),
        "freq_percentiles": {
            "p00": float(np.percentile(flat, 0)),
            "p05": float(np.percentile(flat, 5)),
            "p25": float(np.percentile(flat, 25)),
            "p50": float(np.percentile(flat, 50)),
            "p75": float(np.percentile(flat, 75)),
            "p95": float(np.percentile(flat, 95)),
            "p100": float(np.percentile(flat, 100)),
        },
        "level_histogram": hist,
        "min_res": float(min_res),
        "max_res": float(max_res),
    }

    if debug_ssim_by_level is not None:
        stats["debug_patch_ssim_mean_by_level"] = {
            str(k): float(v.detach().cpu().mean().item())
            for k, v in debug_ssim_by_level.items()
        }

    if full_recon_metrics is not None:
        stats["full_reconstruction"] = full_recon_metrics

    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


# ============================================================
# Core progressive preprocessing
# ============================================================

@dataclass
class DebugBundle:
    patch_gt: Optional[torch.Tensor]
    pred_by_level: Dict[int, torch.Tensor]
    ssim_by_level: Dict[int, torch.Tensor]


def train_progressive_and_estimate_frequency_map(
    image_tensor: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    ssim_threshold: float,
    patch_size: int,
    eval_patch_batch_size: int,
    n_levels: int,
    n_features: int,
    min_res: int,
    max_res: int,
    log2_hashmap_size: int,
    ssim_window_size: int,
    debug_levels: Optional[List[int]] = None,
    debug_patch_count: int = 24,
    debug_seed: int = 0,
) -> Tuple[InstantNGP2D, torch.Tensor, torch.Tensor, DebugBundle]:
    """
    Trains 2D NGP progressively and assigns each patch the first level
    where SSIM crosses threshold.

    Returns:
        model
        freq_map: float map of actual resolutions
        level_map: integer level map
        debug bundle
    """
    if image_tensor.device.type != "cuda":
        raise RuntimeError("LookCloser preprocessing requires CUDA because tinycudann is CUDA-only.")

    if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
        raise ValueError(f"Expected image_tensor=(H,W,3), got {tuple(image_tensor.shape)}")

    device = image_tensor.device
    h, w, _ = image_tensor.shape

    stride = patch_size
    y_starts = compute_patch_starts(h, patch_size, stride)
    x_starts = compute_patch_starts(w, patch_size, stride)

    h_steps = len(y_starts)
    w_steps = len(x_starts)

    model = InstantNGP2D(
        n_levels=n_levels,
        n_features=n_features,
        min_res=min_res,
        max_res=max_res,
        log2_hashmap_size=log2_hashmap_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-15)

    level_map = torch.full((h_steps, w_steps), -1, dtype=torch.int16, device=device)

    # Fixed debug patches for visual progressive check.
    pred_by_level: Dict[int, torch.Tensor] = {}
    ssim_by_level: Dict[int, torch.Tensor] = {}
    debug_patch_gt: Optional[torch.Tensor] = None

    debug_positions: List[Tuple[int, int]] = []
    if debug_levels:
        rng = np.random.RandomState(debug_seed)
        all_positions = [(iy, ix) for iy in range(h_steps) for ix in range(w_steps)]
        if len(all_positions) <= debug_patch_count:
            debug_positions = all_positions
        else:
            picked = rng.choice(len(all_positions), size=debug_patch_count, replace=False)
            debug_positions = [all_positions[int(i)] for i in picked]

    def render_debug_patches(level: int) -> None:
        nonlocal debug_patch_gt

        if not debug_positions:
            return

        xs = torch.tensor([x_starts[ix] for _, ix in debug_positions], device=device, dtype=torch.long)
        ys = torch.tensor([y_starts[iy] for iy, _ in debug_positions], device=device, dtype=torch.long)

        gt = extract_gt_patches(image_tensor, xs, ys, patch_size)
        uv = make_patch_uv(xs, ys, h, w, patch_size)

        model.eval()
        with torch.no_grad():
            pred_flat = model.render_masked(uv, max_active_level=level)
            b = len(debug_positions)
            pred = pred_flat.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
            scores = compute_ssim(gt, pred, window_size=ssim_window_size, size_average=False)

        debug_patch_gt = gt.detach().cpu()
        pred_by_level[level] = pred.detach().cpu()
        ssim_by_level[level] = scores.detach().cpu()
        model.train()

    def eval_and_assign(level: int) -> None:
        unresolved = (level_map < 0).nonzero(as_tuple=False)
        if unresolved.numel() == 0:
            return

        model.eval()
        with torch.no_grad():
            for start in range(0, unresolved.shape[0], eval_patch_batch_size):
                idxs = unresolved[start : start + eval_patch_batch_size]

                ys = torch.tensor([y_starts[int(iy)] for iy in idxs[:, 0].tolist()], device=device, dtype=torch.long)
                xs = torch.tensor([x_starts[int(ix)] for ix in idxs[:, 1].tolist()], device=device, dtype=torch.long)

                gt = extract_gt_patches(image_tensor, xs, ys, patch_size)
                uv = make_patch_uv(xs, ys, h, w, patch_size)

                pred_flat = model.render_masked(uv, max_active_level=level)
                b = idxs.shape[0]
                pred = pred_flat.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()

                scores = compute_ssim(gt, pred, window_size=ssim_window_size, size_average=False)
                ok = scores >= float(ssim_threshold)

                if ok.any():
                    ok_idxs = idxs[ok]
                    level_map[ok_idxs[:, 0], ok_idxs[:, 1]] = int(level)

        model.train()

    # distribute train steps across levels
    steps = int(steps)
    per_level = max(1, steps // n_levels)
    per_level_steps = [per_level] * n_levels
    per_level_steps[-1] += steps - per_level * n_levels

    global_step = 0

    model.train()
    for level in range(n_levels):
        for _ in range(per_level_steps[level]):
            global_step += 1

            iy = torch.randint(0, h, (batch_size,), device=device)
            ix = torch.randint(0, w, (batch_size,), device=device)

            target = image_tensor[iy, ix]
            uv = torch.stack(
                [
                    (ix.float() + 0.5) / float(w),
                    (iy.float() + 0.5) / float(h),
                ],
                dim=-1,
            )

            optimizer.zero_grad(set_to_none=True)
            pred = model.render_masked(uv, max_active_level=level)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

        # Assign unresolved patches at this level.
        eval_and_assign(level)

        if debug_levels and level in debug_levels:
            render_debug_patches(level)

        resolved_ratio = float((level_map >= 0).float().mean().item())
        CONSOLE.print(
            f"[lookcloser] level={level:02d}/{n_levels - 1}, "
            f"resolved={resolved_ratio * 100:.1f}%"
        )

        if (level_map >= 0).all():
            # Still render missing requested debug levels using current model if needed.
            if debug_levels:
                for lvl in debug_levels:
                    if lvl not in pred_by_level and lvl <= level:
                        render_debug_patches(lvl)
            break

    # Assign max level to unresolved patches.
    unresolved_mask = level_map < 0
    if unresolved_mask.any():
        level_map[unresolved_mask] = n_levels - 1

    # Make sure all requested debug levels exist.
    if debug_levels:
        for lvl in debug_levels:
            if lvl not in pred_by_level:
                render_debug_patches(lvl)

    freq_map = torch.empty((h_steps, w_steps), dtype=torch.float32, device=device)
    for lvl in range(n_levels):
        freq_map[level_map == lvl] = model.get_resolution_at_level(lvl)

    debug_bundle = DebugBundle(
        patch_gt=debug_patch_gt,
        pred_by_level=pred_by_level,
        ssim_by_level=ssim_by_level,
    )

    return model, freq_map, level_map, debug_bundle


# ============================================================
# Saving debug artifacts
# ============================================================

def save_debug_artifacts(
    image_tensor: torch.Tensor,
    image_name: str,
    model: InstantNGP2D,
    freq_map: torch.Tensor,
    level_map: torch.Tensor,
    debug_bundle: DebugBundle,
    out_dir: Path,
    min_res: int,
    max_res: int,
    render_full: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w, _ = image_tensor.shape

    save_tensor_image_hwc(image_tensor, out_dir / "gt.png")
    save_freq_overlay(
        image_tensor,
        freq_map,
        out_dir,
        min_res=float(min_res),
        max_res=float(max_res),
    )

    full_metrics = None

    if render_full:
        CONSOLE.print(f"[lookcloser-debug] Rendering full reconstruction for {image_name}...")
        recon = render_full_image(model, h, w, level=model.n_levels - 1)
        save_tensor_image_hwc(recon, out_dir / "recon_full.png")

        diff = torch.abs(recon - image_tensor)
        diff_vis = diff / (diff.max() + 1e-8)
        save_tensor_image_hwc(diff_vis, out_dir / "diff.png")

        mse = float(torch.mean((recon - image_tensor) ** 2).item())
        gt_nchw = image_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        rc_nchw = recon.permute(2, 0, 1).unsqueeze(0).contiguous()
        ssim = float(compute_ssim(gt_nchw, rc_nchw, size_average=True).item())

        full_metrics = {
            "mse": mse,
            "psnr": psnr_from_mse(mse),
            "ssim": ssim,
        }

    if debug_bundle.patch_gt is not None and debug_bundle.pred_by_level:
        save_patch_mosaic(
            debug_bundle.patch_gt,
            debug_bundle.pred_by_level,
            debug_bundle.ssim_by_level,
            out_dir / "patch_mosaic.png",
        )

    save_stats_json(
        out_dir / "stats.json",
        image_name=image_name,
        image_shape=(h, w),
        freq_map=freq_map,
        level_map=level_map,
        min_res=float(min_res),
        max_res=float(max_res),
        debug_ssim_by_level=debug_bundle.ssim_by_level,
        full_recon_metrics=full_metrics,
    )

    CONSOLE.print(f"[lookcloser-debug] Saved debug artifacts to: {out_dir}")


# ============================================================
# CLI config
# ============================================================

@dataclass
class LookCloserPreprocessConfig:
    """LookCloser 2D frequency preprocessing + debug visualization."""

    dataparser: AnnotatedDataParserUnion
    """Data parser config to load the dataset."""

    run_mode: Literal["preprocess", "debug-overfit"] = "preprocess"
    """
    preprocess:
        process train images and save frequency maps.
    debug-overfit:
        run only one image crop, save visual debug artifacts, do not save dataset .pt maps.
    """

    output_name: str = "lookcloser_frequencies"
    """Directory name for frequency .pt maps, inside dataset data dir."""

    steps_per_image: int = 3000
    """Total 2D NGP training steps per image."""

    train_batch_size: int = 1 << 14
    """Random pixels sampled per training step."""

    lr: float = 1e-2
    """Adam learning rate."""

    ssim_threshold: float = 0.95
    """Patch is assigned to the first level where SSIM >= threshold."""

    patch_size: int = 32
    """Patch size and stride."""

    eval_patch_batch_size: int = 64
    """How many patches to evaluate in one SSIM batch."""

    n_levels: int = 16
    n_features: int = 2
    min_res: int = 16
    max_res: int = 2048
    log2_hashmap_size: int = 19

    ssim_window_size: int = 11

    device: Literal["cuda"] = "cuda"
    """CUDA only; tinycudann is CUDA-only."""

    force_recompute: bool = False
    """Overwrite existing .pt frequency maps."""

    # Debug flags.
    debug_save: bool = False
    """Save visual debug artifacts during preprocess mode."""

    debug_dir_name: str = "lookcloser_debug"
    """Debug artifact directory inside dataset data dir."""

    debug_max_images: int = 1
    """How many images to save debug artifacts for in preprocess mode."""

    debug_levels: str = "0,2,4,8,12,15"
    """Comma-separated levels for patch_mosaic visualization."""

    debug_patch_count: int = 24
    """How many patches to show in patch_mosaic."""

    debug_seed: int = 0

    debug_render_full: bool = False
    """
    Save recon_full.png and diff.png.
    For full 4K images this is expensive. Recommended True only for debug-overfit.
    """

    debug_crop_size: int = 512
    """
    Used only in run_mode=debug-overfit.
    Crop size for quick overfit sanity-check. Use 0 to disable crop.
    """

    def _get_data_dir(self, outputs) -> Path:
        data_dir = getattr(self.dataparser, "data", None)
        if data_dir is not None:
            return Path(data_dir)

        # Fallback: parent of first image.
        return Path(outputs.image_filenames[0]).parent

    def _process_single_image(
        self,
        img_path: Path,
        image_tensor: torch.Tensor,
        save_freq_path: Optional[Path],
        debug_out_dir: Optional[Path],
    ) -> None:
        debug_levels = parse_debug_levels(self.debug_levels, self.n_levels)

        model, freq_map, level_map, debug_bundle = train_progressive_and_estimate_frequency_map(
            image_tensor=image_tensor,
            steps=self.steps_per_image,
            batch_size=self.train_batch_size,
            lr=self.lr,
            ssim_threshold=self.ssim_threshold,
            patch_size=self.patch_size,
            eval_patch_batch_size=self.eval_patch_batch_size,
            n_levels=self.n_levels,
            n_features=self.n_features,
            min_res=self.min_res,
            max_res=self.max_res,
            log2_hashmap_size=self.log2_hashmap_size,
            ssim_window_size=self.ssim_window_size,
            debug_levels=debug_levels if debug_out_dir is not None else None,
            debug_patch_count=self.debug_patch_count,
            debug_seed=self.debug_seed,
        )

        if save_freq_path is not None:
            torch.save(freq_map.detach().cpu(), save_freq_path)

        if debug_out_dir is not None:
            save_debug_artifacts(
                image_tensor=image_tensor,
                image_name=img_path.name,
                model=model,
                freq_map=freq_map,
                level_map=level_map,
                debug_bundle=debug_bundle,
                out_dir=debug_out_dir,
                min_res=self.min_res,
                max_res=self.max_res,
                render_full=self.debug_render_full or self.run_mode == "debug-overfit",
            )

        del model, freq_map, level_map, debug_bundle
        torch.cuda.empty_cache()

    def main(self):
        if self.device != "cuda":
            raise RuntimeError("LookCloser preprocessing requires CUDA.")

        CONSOLE.print("[bold green]Starting LookCloser preprocessing...[/bold green]")

        dataparser = self.dataparser.setup()
        outputs = dataparser.get_dataparser_outputs(split="train")

        data_dir = self._get_data_dir(outputs)
        output_dir = data_dir / self.output_name
        debug_root = data_dir / self.debug_dir_name

        device = torch.device(self.device)

        CONSOLE.print(f"Loaded {len(outputs.image_filenames)} train images.")
        CONSOLE.print(f"Data dir: {data_dir}")

        if self.run_mode == "debug-overfit":
            img_path = Path(outputs.image_filenames[0])
            CONSOLE.print(f"[bold yellow]Debug-overfit image:[/bold yellow] {img_path}")

            image = load_image_as_tensor(img_path, device=device)
            image = random_crop_image(image, self.debug_crop_size, seed=self.debug_seed)

            debug_out_dir = debug_root / f"{img_path.stem}_debug_overfit"
            self._process_single_image(
                img_path=img_path,
                image_tensor=image,
                save_freq_path=None,
                debug_out_dir=debug_out_dir,
            )

            CONSOLE.print("[bold green]Debug-overfit complete.[/bold green]")
            return

        # Normal preprocessing.
        output_dir.mkdir(parents=True, exist_ok=True)
        CONSOLE.print(f"Saving frequency maps to: {output_dir}")

        debug_count = 0

        for img_path_raw in track(outputs.image_filenames, description="Processing LookCloser frequency maps"):
            img_path = Path(img_path_raw)
            save_path = output_dir / f"{img_path.stem}.pt"

            if save_path.exists() and not self.force_recompute:
                continue

            image = load_image_as_tensor(img_path, device=device)

            debug_out_dir = None
            if self.debug_save and debug_count < self.debug_max_images:
                debug_out_dir = debug_root / img_path.stem
                debug_count += 1

            self._process_single_image(
                img_path=img_path,
                image_tensor=image,
                save_freq_path=save_path,
                debug_out_dir=debug_out_dir,
            )

            del image
            torch.cuda.empty_cache()

        CONSOLE.print("[bold green]LookCloser preprocessing complete.[/bold green]")
        CONSOLE.print(f"Frequency maps: {output_dir}")

        if self.debug_save:
            CONSOLE.print(f"Debug artifacts: {debug_root}")


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(LookCloserPreprocessConfig).main()


if __name__ == "__main__":
    entrypoint()