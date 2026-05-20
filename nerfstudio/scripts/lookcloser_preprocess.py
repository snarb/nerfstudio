# nerfstudio/scripts/lookcloser_preprocess.py

import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    _TCNN_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    # Keep module import and --help usable on machines without a visible CUDA GPU.
    # Actual preprocessing still fails early when the 2D HashGrid is constructed.
    tcnn = None
    _TCNN_IMPORT_ERROR = e

try:
    from pytorch_msssim import ssim as _pt_ssim  # type: ignore
except Exception:
    _pt_ssim = None

try:
    from torchvision.transforms import functional as TF
except Exception as e:
    raise ImportError("torchvision is required for image loading.") from e

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
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

        if tcnn is None:
            raise RuntimeError(
                "tinycudann could not be imported. LookCloser preprocessing requires CUDA and tinycudann."
            ) from _TCNN_IMPORT_ERROR

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
    if pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info):
        rgba = pil.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        pil = pil.convert("RGB")

    img = TF.to_tensor(pil).permute(1, 2, 0).contiguous()

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


def crop_image_with_coords(
    image: torch.Tensor,
    crop_size: int,
    crop_x: Optional[int] = None,
    crop_y: Optional[int] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Returns crop and (x0, y0, width, height) in the source image."""
    h, w, _ = image.shape
    if crop_size <= 0:
        return image, (0, 0, w, h)

    ch = min(int(crop_size), h)
    cw = min(int(crop_size), w)

    if crop_x is None or crop_y is None:
        rng = np.random.RandomState(seed)
        y0 = int(rng.randint(0, max(1, h - ch + 1)))
        x0 = int(rng.randint(0, max(1, w - cw + 1)))
    else:
        x0 = int(np.clip(crop_x, 0, max(0, w - cw)))
        y0 = int(np.clip(crop_y, 0, max(0, h - ch)))

    return image[y0 : y0 + ch, x0 : x0 + cw, :].contiguous(), (x0, y0, cw, ch)


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

    return torch.cat(outs, dim=0).view(h, w, 3).float()


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def get_scene_size_from_scene_box(scene_box) -> Optional[float]:
    """Returns the longest AABB side for LookCloser's max_res=2048*scene_size baseline."""
    aabb = getattr(scene_box, "aabb", None)
    if aabb is None:
        return None
    if torch.is_tensor(aabb):
        size = torch.max(aabb[1] - aabb[0]).item()
    else:
        arr = np.asarray(aabb)
        size = float(np.max(arr[1] - arr[0]))
    return float(size)


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


def colorize_level_map(level_map: torch.Tensor, n_levels: int) -> Image.Image:
    """Blue-to-red heatmap normalized by discrete assigned frequency level."""
    levels = level_map.detach().cpu().float().numpy()
    norm = levels / max(float(n_levels - 1), 1.0)
    norm = np.clip(norm, 0.0, 1.0)
    return colorize_normalized_levels(norm)


def colorize_normalized_levels(norm: np.ndarray) -> Image.Image:
    """Colorize normalized level values in [0, 1]."""
    norm = np.clip(norm, 0.0, 1.0)

    # Blue -> cyan -> yellow -> red. This makes mid/high levels visibly distinct,
    # unlike scalar resolution normalization where exponential schedules hide L10-L12.
    stops = np.array(
        [
            [35, 42, 200],
            [36, 160, 230],
            [250, 220, 60],
            [220, 40, 30],
        ],
        dtype=np.float32,
    ) / 255.0
    x = norm * (len(stops) - 1)
    idx0 = np.floor(x).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, len(stops) - 1)
    t = (x - idx0)[..., None]
    rgb = stops[idx0] * (1.0 - t) + stops[idx1] * t
    return Image.fromarray(to_uint8_np(rgb))


def colorize_level_map_quantile(level_map: torch.Tensor, n_levels: int) -> Image.Image:
    """
    Diagnostic heatmap with robust per-image contrast stretching.

    This does not change assigned levels or frequency-map semantics; it only makes
    local variation visible when the absolute level range is narrow.
    """
    levels = level_map.detach().cpu().float().numpy()
    lo, hi = np.percentile(levels, [5, 95])
    if hi <= lo:
        lo = float(np.min(levels))
        hi = float(np.max(levels))
    if hi <= lo:
        norm = np.zeros_like(levels, dtype=np.float32)
    else:
        norm = (levels - float(lo)) / float(hi - lo)
    return colorize_normalized_levels(norm)


def make_level_legend(n_levels: int, out_path: Path) -> None:
    cell_w = 32
    label_h = 22
    bar_h = 24
    width = cell_w * n_levels
    height = bar_h + label_h
    levels = torch.arange(n_levels, dtype=torch.float32).view(1, n_levels)
    legend = colorize_level_map(levels, n_levels).resize((width, bar_h), resample=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(legend, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for lvl in range(n_levels):
        x = lvl * cell_w
        draw.text((x + 4, bar_h + 4), str(lvl), fill=(0, 0, 0))
    canvas.save(out_path)


def upsample_heatmap_to_image(
    heatmap: Image.Image,
    image_h: int,
    image_w: int,
) -> Image.Image:
    return heatmap.resize((image_w, image_h), resample=Image.Resampling.NEAREST)


def save_freq_overlay(
    image: torch.Tensor,
    freq_map: torch.Tensor,
    level_map: torch.Tensor,
    out_dir: Path,
    min_res: float,
    max_res: float,
    n_levels: int,
    high_frequency_level: int = 13,
) -> None:
    h, w, _ = image.shape

    img_pil = Image.fromarray(to_uint8_np(image.detach().cpu().numpy()))
    scalar_heat_small = colorize_freq_map(freq_map, min_res=min_res, max_res=max_res)
    scalar_heat_big = upsample_heatmap_to_image(scalar_heat_small, h, w)

    level_heat_small = colorize_level_map(level_map, n_levels=n_levels)
    level_heat_big = upsample_heatmap_to_image(level_heat_small, h, w)
    level_quantile_small = colorize_level_map_quantile(level_map, n_levels=n_levels)
    level_quantile_big = upsample_heatmap_to_image(level_quantile_small, h, w)

    level_heat_small.save(out_dir / "level_heatmap_patch_grid.png")
    level_heat_big.save(out_dir / "level_heatmap.png")
    make_level_legend(n_levels, out_dir / "level_heatmap_legend.png")

    overlay = Image.blend(img_pil.convert("RGB"), level_heat_big.convert("RGB"), alpha=0.45)
    overlay.save(out_dir / "level_overlay.png")
    quantile_overlay = Image.blend(img_pil.convert("RGB"), level_quantile_big.convert("RGB"), alpha=0.45)
    level_quantile_small.save(out_dir / "level_heatmap_quantile_patch_grid.png")
    level_quantile_big.save(out_dir / "level_heatmap_quantile.png")
    quantile_overlay.save(out_dir / "level_overlay_quantile.png")

    threshold = int(np.clip(high_frequency_level, 0, n_levels - 1))
    high_mask_small = (level_map.detach().cpu().long() >= threshold).float()
    high_mask_big = upsample_heatmap_to_image(
        Image.fromarray(to_uint8_np(high_mask_small.numpy())),
        h,
        w,
    )
    high_mask_big.save(out_dir / f"high_frequency_mask_L{threshold}_plus.png")
    high_mask_np = np.asarray(high_mask_big, dtype=np.float32) / 255.0
    img_np = np.asarray(img_pil.convert("RGB"), dtype=np.float32) / 255.0
    red = np.zeros_like(img_np)
    red[..., 0] = 1.0
    high_overlay_np = img_np * (1.0 - 0.55 * high_mask_np[..., None]) + red * (0.55 * high_mask_np[..., None])
    Image.fromarray(to_uint8_np(high_overlay_np)).save(out_dir / f"high_frequency_overlay_L{threshold}_plus.png")

    # Compatibility names now point to the level-based diagnostic.
    level_heat_small.save(out_dir / "freq_heatmap_patch_grid.png")
    level_heat_big.save(out_dir / "freq_heatmap_fullres.png")
    level_heat_big.save(out_dir / "freq_heatmap.png")
    overlay.save(out_dir / "freq_overlay.png")

    scalar_heat_small.save(out_dir / "scalar_resolution_heatmap_patch_grid.png")
    scalar_heat_big.save(out_dir / "scalar_resolution_heatmap.png")


def save_freq_histogram(level_map: torch.Tensor, n_levels: int, out_path: Path) -> None:
    counts = torch.bincount(level_map.detach().cpu().long().flatten(), minlength=n_levels).numpy()
    width = max(420, n_levels * 32)
    height = 260
    margin_l, margin_r, margin_t, margin_b = 44, 16, 20, 44
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_count = max(int(counts.max()), 1)

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(0, 0, 0))
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(0, 0, 0))

    bar_gap = 2
    bar_w = max(1, int(plot_w / n_levels) - bar_gap)
    for lvl, count in enumerate(counts):
        x0 = margin_l + int(lvl * plot_w / n_levels) + bar_gap // 2
        x1 = x0 + bar_w
        bar_h = int((float(count) / max_count) * plot_h)
        y0 = margin_t + plot_h - bar_h
        draw.rectangle((x0, y0, x1, margin_t + plot_h), fill=(64, 114, 196))
        draw.text((x0, margin_t + plot_h + 6), str(lvl), fill=(0, 0, 0))

    draw.text((8, 6), "patch count by assigned level", fill=(0, 0, 0))
    draw.text((8, height - 18), f"max bin={max_count}", fill=(0, 0, 0))
    canvas.save(out_path)


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


def render_patches_nchw(
    model: InstantNGP2D,
    image_tensor: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    patch_size: int,
    level: int,
) -> torch.Tensor:
    h, w, _ = image_tensor.shape
    uv = make_patch_uv(xs, ys, h, w, patch_size)
    model.eval()
    with torch.no_grad():
        pred_flat = model.render_masked(uv, max_active_level=level)
    b = xs.shape[0]
    return pred_flat.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous().float()


def make_patch_audit_mosaic(
    title: str,
    coords: List[Tuple[int, int]],
    gt: torch.Tensor,
    assigned_pred: torch.Tensor,
    max_pred: torch.Tensor,
    assigned_levels: List[int],
    assigned_freqs: List[float],
    assigned_ssim: torch.Tensor,
    max_ssim: torch.Tensor,
    out_path: Path,
) -> None:
    gt_np = gt.detach().cpu().permute(0, 2, 3, 1).numpy()
    assigned_np = assigned_pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    max_np = max_pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    assigned_scores = assigned_ssim.detach().cpu().numpy()
    max_scores = max_ssim.detach().cpu().numpy()

    n, p = gt_np.shape[0], gt_np.shape[1]
    label_w = 210
    col_w = p
    row_h = max(p, 72)
    header_h = 28
    width = label_w + 3 * col_w
    height = header_h + n * row_h

    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(255, 255, 255))
    draw.text((label_w + 8, 7), "GT", fill=(255, 255, 255))
    draw.text((label_w + col_w + 8, 7), "assigned", fill=(255, 255, 255))
    draw.text((label_w + 2 * col_w + 8, 7), "max", fill=(255, 255, 255))

    for i in range(n):
        y = header_h + i * row_h
        x_coord, y_coord = coords[i]
        label = (
            f"x={x_coord} y={y_coord}\n"
            f"L{assigned_levels[i]} res={assigned_freqs[i]:.0f}\n"
            f"SSIM {assigned_scores[i]:.3f}/{max_scores[i]:.3f}"
        )
        draw.text((8, y + 7), label, fill=(235, 235, 235))
        canvas.paste(Image.fromarray(to_uint8_np(gt_np[i])), (label_w, y))
        canvas.paste(Image.fromarray(to_uint8_np(assigned_np[i])), (label_w + col_w, y))
        canvas.paste(Image.fromarray(to_uint8_np(max_np[i])), (label_w + 2 * col_w, y))

    canvas.save(out_path)


def save_uv_audit_mosaic(
    coords: List[Tuple[int, int]],
    gt: torch.Tensor,
    max_pred: torch.Tensor,
    max_ssim: torch.Tensor,
    out_path: Path,
) -> None:
    gt_np = gt.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred_np = max_pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    diff_np = np.abs(gt_np - pred_np)
    scores = max_ssim.detach().cpu().numpy()

    n, p = gt_np.shape[0], gt_np.shape[1]
    label_w = 160
    row_h = max(p, 56)
    header_h = 28
    width = label_w + 3 * p
    height = header_h + n * row_h

    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "fixed max-level UV audit", fill=(255, 255, 255))
    draw.text((label_w + 8, 7), "GT", fill=(255, 255, 255))
    draw.text((label_w + p + 8, 7), "max pred", fill=(255, 255, 255))
    draw.text((label_w + 2 * p + 8, 7), "diff", fill=(255, 255, 255))

    for i in range(n):
        y = header_h + i * row_h
        x_coord, y_coord = coords[i]
        draw.text((8, y + 7), f"x={x_coord} y={y_coord}\nSSIM {scores[i]:.3f}", fill=(235, 235, 235))
        canvas.paste(Image.fromarray(to_uint8_np(gt_np[i])), (label_w, y))
        canvas.paste(Image.fromarray(to_uint8_np(pred_np[i])), (label_w + p, y))
        diff_vis = diff_np[i] / max(float(diff_np[i].max()), 1e-8)
        canvas.paste(Image.fromarray(to_uint8_np(diff_vis)), (label_w + 2 * p, y))

    canvas.save(out_path)


def save_stats_json(
    path: Path,
    image_name: str,
    image_shape: Tuple[int, int],
    freq_map: torch.Tensor,
    level_map: torch.Tensor,
    min_res: float,
    max_res: float,
    n_levels: int,
    patch_size: int,
    stride: int,
    steps: int,
    batch_size: int,
    crop_coords: Optional[Tuple[int, int, int, int]] = None,
    debug_ssim_by_level: Optional[Dict[int, torch.Tensor]] = None,
    full_recon_metrics: Optional[Dict[str, float]] = None,
) -> None:
    freq_cpu = freq_map.detach().cpu().float()
    level_cpu = level_map.detach().cpu().long()

    hist = {}
    for lvl in torch.unique(level_cpu).tolist():
        hist[str(int(lvl))] = int((level_cpu == int(lvl)).sum().item())

    flat = freq_cpu.flatten().numpy()
    level_flat = level_cpu.flatten()
    total = max(int(level_flat.numel()), 1)
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    b = float(np.exp((np.log(max_res) - np.log(min_res)) / max(n_levels - 1, 1)))
    level_schedule = [float(min_res * (b ** lvl)) for lvl in range(n_levels)]

    stats = {
        "image": image_name,
        "shape": [int(freq_cpu.shape[0]), int(freq_cpu.shape[1])],
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "crop_coords_xywh": list(crop_coords) if crop_coords is not None else None,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "patch_size": int(patch_size),
        "stride": int(stride),
        "min": float(freq_cpu.min().item()),
        "max": float(freq_cpu.max().item()),
        "mean": float(freq_cpu.mean().item()),
        "median": float(freq_cpu.median().item()),
        "percentiles": {str(p): float(np.percentile(flat, p)) for p in percentiles},
        "fraction_min_level": float((level_flat == 0).sum().item() / total),
        "fraction_max_level": float((level_flat == int(n_levels - 1)).sum().item() / total),
        "number_of_non_empty_levels": int(torch.unique(level_cpu).numel()),
        "level_histogram": hist,
        "min_res": float(min_res),
        "max_res": float(max_res),
        "n_levels": int(n_levels),
        "level_resolution_schedule": level_schedule,
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
    steps: Optional[int],
    train_steps_per_level: Optional[int],
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
            pred = pred_flat.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous().float()
            scores = compute_ssim(gt.float(), pred, window_size=ssim_window_size, size_average=False)

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
                pred = pred_flat.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous().float()

                scores = compute_ssim(gt.float(), pred, window_size=ssim_window_size, size_average=False)
                ok = scores >= float(ssim_threshold)

                if ok.any():
                    ok_idxs = idxs[ok]
                    level_map[ok_idxs[:, 0], ok_idxs[:, 1]] = int(level)

        model.train()

    # Distribute train steps across levels. Prefer explicit per-level training so each
    # frequency band gets enough optimization before SSIM assignment.
    if train_steps_per_level is not None:
        per_level_steps = [int(train_steps_per_level)] * n_levels
        steps = int(train_steps_per_level) * n_levels
    else:
        steps = int(steps or 0)
        per_level = max(1, steps // n_levels)
        per_level_steps = [per_level] * n_levels
        per_level_steps[-1] += steps - per_level * n_levels

    global_step = 0

    # Train and evaluate each checkpoint with the same active prefix of HashGrid
    # levels. Full-level training followed by inference-only masking would not
    # measure the minimum level that can actually learn the patch.
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
            target = target.to(dtype=pred.dtype)
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

        pending_debug = bool(debug_levels and any(lvl > level for lvl in debug_levels))
        if (level_map >= 0).all() and not pending_debug:
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

def save_frequency_metadata(
    path: Path,
    image_name: str,
    image_shape: Tuple[int, int],
    crop_coords: Optional[Tuple[int, int, int, int]],
    patch_size: int,
    stride: int,
    min_res: int,
    max_res: int,
    n_levels: int,
    n_features: int,
    log2_hashmap_size: Optional[int],
) -> None:
    b = float(np.exp((np.log(max_res) - np.log(min_res)) / max(n_levels - 1, 1)))
    data = {
        "image": image_name,
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "crop_coords_xywh": list(crop_coords) if crop_coords is not None else None,
        "value_type": "scalar_resolution",
        "patch_size": int(patch_size),
        "stride": int(stride),
        "min_res": int(min_res),
        "max_res": int(max_res),
        "n_levels": int(n_levels),
        "n_features": int(n_features),
        "log2_hashmap_size": None if log2_hashmap_size is None else int(log2_hashmap_size),
        "per_level_scale": b,
        "level_resolution_schedule": [float(min_res * (b ** lvl)) for lvl in range(n_levels)],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _positions_from_flat_indices(
    flat_indices: Iterable[int],
    width_steps: int,
    patch_size: int,
) -> List[Tuple[int, int, int, int]]:
    out = []
    for flat_idx in flat_indices:
        iy = int(flat_idx) // width_steps
        ix = int(flat_idx) % width_steps
        out.append((iy, ix, ix * patch_size, iy * patch_size))
    return out


def _pick_level_positions(
    level_map: torch.Tensor,
    mode: Literal["min", "max", "random"],
    count: int,
    seed: int,
    patch_size: int,
) -> List[Tuple[int, int, int, int]]:
    level_cpu = level_map.detach().cpu().long()
    h_steps, w_steps = level_cpu.shape
    rng = np.random.RandomState(seed)

    if mode == "random":
        candidates = np.arange(h_steps * w_steps)
    else:
        target = int(level_cpu.min().item() if mode == "min" else level_cpu.max().item())
        candidates = (level_cpu.flatten() == target).nonzero(as_tuple=False).flatten().numpy()

    if candidates.size == 0:
        candidates = np.arange(h_steps * w_steps)

    if candidates.size > count:
        picked = rng.choice(candidates, size=count, replace=False)
    else:
        picked = candidates

    return _positions_from_flat_indices(picked.tolist(), w_steps, patch_size)


def save_patch_audit_artifacts(
    image_tensor: torch.Tensor,
    model: InstantNGP2D,
    freq_map: torch.Tensor,
    level_map: torch.Tensor,
    out_dir: Path,
    patch_size: int,
    count: int,
    seed: int,
    ssim_window_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = image_tensor.device

    for mode, filename, title in [
        ("min", "low_freq_patches.png", "low frequency patches"),
        ("max", "high_freq_patches.png", "high frequency patches"),
        ("random", "random_freq_patches.png", "random frequency patches"),
    ]:
        positions = _pick_level_positions(level_map, mode, count, seed + len(mode), patch_size)
        if not positions:
            continue

        iys = [p[0] for p in positions]
        ixs = [p[1] for p in positions]
        xs = torch.tensor([p[2] for p in positions], device=device, dtype=torch.long)
        ys = torch.tensor([p[3] for p in positions], device=device, dtype=torch.long)
        coords = [(int(x.item()), int(y.item())) for x, y in zip(xs, ys)]

        gt = extract_gt_patches(image_tensor, xs, ys, patch_size)
        assigned_levels = [int(level_map[iy, ix].item()) for iy, ix in zip(iys, ixs)]
        assigned_freqs = [float(freq_map[iy, ix].item()) for iy, ix in zip(iys, ixs)]

        assigned_preds = []
        assigned_scores = []
        for level in sorted(set(assigned_levels)):
            mask = torch.tensor([lvl == level for lvl in assigned_levels], device=device, dtype=torch.bool)
            pred = render_patches_nchw(model, image_tensor, xs[mask], ys[mask], patch_size, level)
            assigned_preds.append((mask.detach().cpu(), pred.detach().cpu()))

        assigned_pred = torch.empty_like(gt.detach().cpu())
        for mask_cpu, pred_cpu in assigned_preds:
            assigned_pred[mask_cpu] = pred_cpu

        max_pred = render_patches_nchw(
            model, image_tensor, xs, ys, patch_size, model.n_levels - 1
        ).detach().cpu()

        assigned_ssim = compute_ssim(
            gt.detach().cpu(),
            assigned_pred,
            window_size=ssim_window_size,
            size_average=False,
        )
        max_ssim = compute_ssim(
            gt.detach().cpu(),
            max_pred,
            window_size=ssim_window_size,
            size_average=False,
        )

        make_patch_audit_mosaic(
            title=title,
            coords=coords,
            gt=gt.detach().cpu(),
            assigned_pred=assigned_pred,
            max_pred=max_pred,
            assigned_levels=assigned_levels,
            assigned_freqs=assigned_freqs,
            assigned_ssim=assigned_ssim,
            max_ssim=max_ssim,
            out_path=out_dir / filename,
        )


def save_uv_audit_artifacts(
    image_tensor: torch.Tensor,
    model: InstantNGP2D,
    out_dir: Path,
    patch_size: int,
    count: int,
    seed: int,
    ssim_window_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w, _ = image_tensor.shape
    y_starts = compute_patch_starts(h, patch_size, patch_size)
    x_starts = compute_patch_starts(w, patch_size, patch_size)
    all_positions = [(iy, ix) for iy in range(len(y_starts)) for ix in range(len(x_starts))]
    rng = np.random.RandomState(seed)
    if len(all_positions) > count:
        picked = rng.choice(len(all_positions), size=count, replace=False)
        positions = [all_positions[int(i)] for i in picked]
    else:
        positions = all_positions

    device = image_tensor.device
    xs = torch.tensor([x_starts[ix] for _, ix in positions], device=device, dtype=torch.long)
    ys = torch.tensor([y_starts[iy] for iy, _ in positions], device=device, dtype=torch.long)
    coords = [(int(x.item()), int(y.item())) for x, y in zip(xs, ys)]

    gt = extract_gt_patches(image_tensor, xs, ys, patch_size)
    max_pred = render_patches_nchw(model, image_tensor, xs, ys, patch_size, model.n_levels - 1)
    max_ssim = compute_ssim(gt, max_pred, window_size=ssim_window_size, size_average=False)
    save_uv_audit_mosaic(coords, gt.detach().cpu(), max_pred.detach().cpu(), max_ssim.detach().cpu(), out_dir / "fixed_patches.png")

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
    log2_hashmap_size: int,
    patch_size: int,
    ssim_window_size: int,
    steps: int,
    batch_size: int,
    render_full: bool,
    crop_coords: Optional[Tuple[int, int, int, int]] = None,
    audit_patch_count: int = 16,
    uv_audit_patch_count: int = 10,
    audit_seed: int = 0,
    patch_audit_out_dir: Optional[Path] = None,
    high_frequency_level: int = 13,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w, _ = image_tensor.shape

    save_tensor_image_hwc(image_tensor, out_dir / "gt.png")
    save_freq_overlay(
        image_tensor,
        freq_map,
        level_map,
        out_dir,
        min_res=float(min_res),
        max_res=float(max_res),
        n_levels=model.n_levels,
        high_frequency_level=high_frequency_level,
    )
    save_freq_histogram(level_map, model.n_levels, out_dir / "freq_histogram.png")

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

    save_frequency_metadata(
        out_dir / "metadata.json",
        image_name=image_name,
        image_shape=(h, w),
        crop_coords=crop_coords,
        patch_size=patch_size,
        stride=patch_size,
        min_res=min_res,
        max_res=max_res,
        n_levels=model.n_levels,
        n_features=model.n_features,
        log2_hashmap_size=log2_hashmap_size,
    )

    save_patch_audit_artifacts(
        image_tensor=image_tensor,
        model=model,
        freq_map=freq_map,
        level_map=level_map,
        out_dir=patch_audit_out_dir if patch_audit_out_dir is not None else out_dir / "patch_audit",
        patch_size=patch_size,
        count=audit_patch_count,
        seed=audit_seed,
        ssim_window_size=ssim_window_size,
    )

    save_uv_audit_artifacts(
        image_tensor=image_tensor,
        model=model,
        out_dir=out_dir / "uv_audit",
        patch_size=patch_size,
        count=uv_audit_patch_count,
        seed=audit_seed,
        ssim_window_size=ssim_window_size,
    )

    save_stats_json(
        out_dir / "stats.json",
        image_name=image_name,
        image_shape=(h, w),
        freq_map=freq_map,
        level_map=level_map,
        min_res=float(min_res),
        max_res=float(max_res),
        n_levels=model.n_levels,
        patch_size=patch_size,
        stride=patch_size,
        steps=steps,
        batch_size=batch_size,
        crop_coords=crop_coords,
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

    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
    """Data parser config to load the dataset."""

    run_mode: Literal["preprocess", "debug-overfit", "sweep"] = "preprocess"
    """
    preprocess:
        process train images and save frequency maps.
    debug-overfit:
        run only one image crop, save visual debug artifacts, do not save dataset .pt maps.
    """

    output_name: str = "lookcloser_frequencies"
    """Directory name for frequency .pt maps, inside dataset data dir."""

    image_path: Optional[Path] = None
    """Optional direct image path for standalone HD/6K preprocessing debug runs."""

    output_root: Path = Path("lookcloser_debug_outputs")
    """Output root for direct image debug/sweep artifacts."""

    steps_per_image: Optional[int] = None
    """Optional legacy total 2D NGP training steps per image. Prefer train_steps_per_level."""

    train_steps_per_level: int = 1000
    """2D NGP optimization steps per active frequency level."""

    train_batch_size: int = 1 << 14
    """Random pixels sampled per training step."""

    lr: float = 1e-2
    """Adam learning rate."""

    ssim_threshold: float = 0.97
    """Patch is assigned to the first level where SSIM >= threshold."""

    patch_size: int = 8
    """Patch size and stride."""

    eval_patch_batch_size: int = 64
    """How many patches to evaluate in one SSIM batch."""

    n_levels: int = 16
    n_features: int = 2
    min_res: int = 16
    max_res: Optional[int] = None
    """Override maximum HashGrid resolution. If unset, uses max_res_base * scene_size."""

    max_res_base: int = 2048
    """Baseline HashGrid max resolution multiplier."""

    scene_size: Optional[float] = None
    """Scene size for direct image runs. Dataset runs infer this from the dataparser scene box."""

    log2_hashmap_size: int = 23

    ssim_window_size: int = 7

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

    crop_x: Optional[int] = None
    """Optional crop top-left x for direct image debug runs."""

    crop_y: Optional[int] = None
    """Optional crop top-left y for direct image debug runs."""

    audit_patch_count: int = 16
    """Patch count for low/high/random patch audit mosaics."""

    uv_audit_patch_count: int = 10
    """Fixed patch count for GT/max/diff UV audit."""

    high_frequency_level: int = 13
    """Assigned level threshold used for high-frequency mask and overlay debug artifacts."""

    sweep_steps_per_level_options: Tuple[int, ...] = (250, 500, 1000)
    """Sweep values for train_steps_per_level in direct image sweep mode."""

    sweep_ssim_threshold_options: Tuple[float, ...] = (0.90, 0.93, 0.95, 0.97)
    """Sweep values for SSIM threshold."""

    sweep_patch_size_options: Tuple[int, ...] = (16, 32, 64)
    """Sweep values for patch size."""

    sweep_max_res_options: Tuple[int, ...] = (2048, 4096)
    """Sweep values for max hash-grid resolution."""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if self.eval_patch_batch_size <= 0:
            raise ValueError("eval_patch_batch_size must be > 0.")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be > 0.")
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2.")
        if self.n_features <= 0:
            raise ValueError("n_features must be > 0.")
        if self.min_res <= 0:
            raise ValueError("min_res must be > 0.")
        if self.max_res is not None and self.max_res <= self.min_res:
            raise ValueError("Expected max_res > min_res.")
        if self.max_res_base <= 0:
            raise ValueError("max_res_base must be > 0.")
        if self.scene_size is not None and self.scene_size <= 0:
            raise ValueError("scene_size must be > 0 when provided.")
        if not (0.0 < self.ssim_threshold <= 1.0):
            raise ValueError("ssim_threshold must be in (0, 1].")
        if self.steps_per_image is not None and self.steps_per_image <= 0:
            raise ValueError("steps_per_image must be > 0 when provided.")
        if self.steps_per_image is None and self.train_steps_per_level <= 0:
            raise ValueError("train_steps_per_level must be > 0 when steps_per_image is not provided.")
        if self.ssim_window_size <= 0 or self.ssim_window_size % 2 == 0:
            raise ValueError("ssim_window_size must be a positive odd integer.")
        if self.audit_patch_count < 0 or self.uv_audit_patch_count < 0 or self.debug_patch_count < 0:
            raise ValueError("debug/audit patch counts must be >= 0.")
        if self.high_frequency_level < 0:
            raise ValueError("high_frequency_level must be >= 0.")
        for name, values in {
            "sweep_steps_per_level_options": self.sweep_steps_per_level_options,
            "sweep_patch_size_options": self.sweep_patch_size_options,
            "sweep_max_res_options": self.sweep_max_res_options,
        }.items():
            if not values or any(v <= 0 for v in values):
                raise ValueError(f"{name} must contain positive values.")
        if not self.sweep_ssim_threshold_options or any(
            not (0.0 < v <= 1.0) for v in self.sweep_ssim_threshold_options
        ):
            raise ValueError("sweep_ssim_threshold_options must contain values in (0, 1].")

    def _legacy_steps_per_level(self) -> Optional[int]:
        if self.steps_per_image is None:
            return self.train_steps_per_level
        return None

    def _total_training_steps(self) -> int:
        if self.steps_per_image is not None:
            return int(self.steps_per_image)
        return int(self.train_steps_per_level) * int(self.n_levels)

    def _effective_max_res(self, scene_size: Optional[float]) -> int:
        if self.max_res is not None:
            return int(self.max_res)
        effective_scene_size = float(scene_size if scene_size is not None else self.scene_size if self.scene_size is not None else 1.0)
        return int(round(float(self.max_res_base) * effective_scene_size))

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
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
        patch_audit_out_dir: Optional[Path] = None,
        scene_size: Optional[float] = None,
    ) -> None:
        debug_levels = parse_debug_levels(self.debug_levels, self.n_levels)
        max_res = self._effective_max_res(scene_size)

        model, freq_map, level_map, debug_bundle = train_progressive_and_estimate_frequency_map(
            image_tensor=image_tensor,
            steps=self.steps_per_image,
            train_steps_per_level=self._legacy_steps_per_level(),
            batch_size=self.train_batch_size,
            lr=self.lr,
            ssim_threshold=self.ssim_threshold,
            patch_size=self.patch_size,
            eval_patch_batch_size=self.eval_patch_batch_size,
            n_levels=self.n_levels,
            n_features=self.n_features,
            min_res=self.min_res,
            max_res=max_res,
            log2_hashmap_size=self.log2_hashmap_size,
            ssim_window_size=self.ssim_window_size,
            debug_levels=debug_levels if debug_out_dir is not None else None,
            debug_patch_count=self.debug_patch_count,
            debug_seed=self.debug_seed,
        )

        if save_freq_path is not None:
            torch.save(freq_map.detach().cpu(), save_freq_path)
            save_frequency_metadata(
                save_freq_path.with_suffix(".json"),
                image_name=img_path.name,
                image_shape=(int(image_tensor.shape[0]), int(image_tensor.shape[1])),
                crop_coords=crop_coords,
                patch_size=self.patch_size,
                stride=self.patch_size,
                min_res=self.min_res,
                max_res=max_res,
                n_levels=self.n_levels,
                n_features=self.n_features,
                log2_hashmap_size=self.log2_hashmap_size,
            )

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
                max_res=max_res,
                log2_hashmap_size=self.log2_hashmap_size,
                patch_size=self.patch_size,
                ssim_window_size=self.ssim_window_size,
                steps=self._total_training_steps(),
                batch_size=self.train_batch_size,
                render_full=self.debug_render_full or self.run_mode == "debug-overfit",
                crop_coords=crop_coords,
                audit_patch_count=self.audit_patch_count,
                uv_audit_patch_count=self.uv_audit_patch_count,
                audit_seed=self.debug_seed,
                patch_audit_out_dir=patch_audit_out_dir,
                high_frequency_level=self.high_frequency_level,
            )

        del model, freq_map, level_map, debug_bundle
        torch.cuda.empty_cache()

    def _load_direct_image_crop(self, device: torch.device) -> Tuple[Path, torch.Tensor, Tuple[int, int, int, int]]:
        if self.image_path is None:
            raise ValueError("image_path is required for direct debug/sweep runs.")
        img_path = Path(self.image_path)
        image = load_image_as_tensor(img_path, device=device)
        crop, crop_coords = crop_image_with_coords(
            image,
            crop_size=self.debug_crop_size,
            crop_x=self.crop_x,
            crop_y=self.crop_y,
            seed=self.debug_seed,
        )
        return img_path, crop, crop_coords

    def _run_direct_image(self, device: torch.device) -> None:
        img_path, image, crop_coords = self._load_direct_image_crop(device)
        self.output_root.mkdir(parents=True, exist_ok=True)

        if self.run_mode == "debug-overfit":
            out_dir = self.output_root / "overfit_hd"
            save_freq_path = None
        else:
            out_dir = self.output_root / "freq_hd"
            save_freq_path = out_dir / f"{img_path.stem}.pt"
            out_dir.mkdir(parents=True, exist_ok=True)

        self._process_single_image(
            img_path=img_path,
            image_tensor=image,
            save_freq_path=save_freq_path,
            debug_out_dir=out_dir,
            crop_coords=crop_coords,
            patch_audit_out_dir=self.output_root / "patch_audit" if self.run_mode == "preprocess" else None,
            scene_size=self.scene_size,
        )

    def _run_sweep(self, device: torch.device) -> None:
        img_path, image, crop_coords = self._load_direct_image_crop(device)
        sweep_root = self.output_root / "sweep_hd"
        sweep_root.mkdir(parents=True, exist_ok=True)
        summary_path = sweep_root / "sweep_summary.csv"

        original = {
            "steps_per_image": self.steps_per_image,
            "train_steps_per_level": self.train_steps_per_level,
            "ssim_threshold": self.ssim_threshold,
            "patch_size": self.patch_size,
            "max_res": self.max_res,
            "debug_render_full": self.debug_render_full,
        }

        rows = []
        phase_a = [
            (steps, thr, 32, max(self.sweep_max_res_options))
            for steps in self.sweep_steps_per_level_options
            for thr in self.sweep_ssim_threshold_options
        ]
        best_row = None

        try:
            self.debug_render_full = False
            for steps, threshold, patch_size, max_res in phase_a:
                row = self._run_one_sweep_case(
                    img_path, image, crop_coords, sweep_root, "A", steps, threshold, patch_size, max_res
                )
                rows.append(row)
                if best_row is None or self._sweep_score(row) > self._sweep_score(best_row):
                    best_row = row

            best_steps = int(best_row["train_steps_per_level"]) if best_row is not None else self.train_steps_per_level
            best_threshold = float(best_row["ssim_threshold"]) if best_row is not None else 0.95
            for patch_size in self.sweep_patch_size_options:
                for max_res in self.sweep_max_res_options:
                    row = self._run_one_sweep_case(
                        img_path, image, crop_coords, sweep_root, "B", best_steps, best_threshold, patch_size, max_res
                    )
                    rows.append(row)
        finally:
            self.steps_per_image = original["steps_per_image"]
            self.train_steps_per_level = original["train_steps_per_level"]
            self.ssim_threshold = original["ssim_threshold"]
            self.patch_size = original["patch_size"]
            self.max_res = original["max_res"]
            self.debug_render_full = original["debug_render_full"]

        fieldnames = [
            "phase",
            "run_dir",
            "train_steps_per_level",
            "ssim_threshold",
            "patch_size",
            "max_res",
            "mean",
            "median",
            "fraction_min_level",
            "fraction_max_level",
            "number_of_non_empty_levels",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        CONSOLE.print(f"[lookcloser-sweep] Saved summary: {summary_path}")

    def _run_one_sweep_case(
        self,
        img_path: Path,
        image: torch.Tensor,
        crop_coords: Tuple[int, int, int, int],
        sweep_root: Path,
        phase: str,
        steps: int,
        threshold: float,
        patch_size: int,
        max_res: int,
    ) -> Dict[str, object]:
        self.steps_per_image = None
        self.train_steps_per_level = int(steps)
        self.ssim_threshold = float(threshold)
        self.patch_size = int(patch_size)
        self.max_res = int(max_res)

        run_name = f"phase_{phase}_steps_{steps}_ssim_{threshold:.2f}_patch_{patch_size}_maxres_{max_res}"
        out_dir = sweep_root / run_name
        self._process_single_image(
            img_path=img_path,
            image_tensor=image,
            save_freq_path=None,
            debug_out_dir=out_dir,
            crop_coords=crop_coords,
            scene_size=self.scene_size,
        )

        stats_path = out_dir / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
        return {
            "phase": phase,
            "run_dir": str(out_dir),
            "train_steps_per_level": int(steps),
            "ssim_threshold": float(threshold),
            "patch_size": int(patch_size),
            "max_res": int(max_res),
            "mean": stats.get("mean"),
            "median": stats.get("median"),
            "fraction_min_level": stats.get("fraction_min_level"),
            "fraction_max_level": stats.get("fraction_max_level"),
            "number_of_non_empty_levels": stats.get("number_of_non_empty_levels"),
        }

    @staticmethod
    def _sweep_score(row: Dict[str, object]) -> float:
        non_empty = float(row.get("number_of_non_empty_levels") or 0.0)
        frac_min = float(row.get("fraction_min_level") or 0.0)
        frac_max = float(row.get("fraction_max_level") or 0.0)
        # Prefer non-collapsed maps. Visual audit remains the final choice.
        return non_empty - 4.0 * max(frac_min, frac_max)

    def main(self):
        if self.device != "cuda":
            raise RuntimeError("LookCloser preprocessing requires CUDA.")

        CONSOLE.print("[bold green]Starting LookCloser preprocessing...[/bold green]")

        device = torch.device(self.device)

        if self.image_path is not None:
            if self.run_mode == "sweep":
                self._run_sweep(device)
            else:
                self._run_direct_image(device)
            CONSOLE.print("[bold green]LookCloser direct image run complete.[/bold green]")
            return

        dataparser = self.dataparser.setup()
        outputs = dataparser.get_dataparser_outputs(split="train")
        scene_size = self.scene_size
        if scene_size is None:
            scene_size = get_scene_size_from_scene_box(getattr(outputs, "scene_box", None))

        data_dir = self._get_data_dir(outputs)
        output_dir = data_dir / self.output_name
        debug_root = data_dir / self.debug_dir_name

        CONSOLE.print(f"Loaded {len(outputs.image_filenames)} train images.")
        CONSOLE.print(f"Data dir: {data_dir}")

        if self.run_mode == "debug-overfit":
            img_path = Path(outputs.image_filenames[0])
            CONSOLE.print(f"[bold yellow]Debug-overfit image:[/bold yellow] {img_path}")

            image = load_image_as_tensor(img_path, device=device)
            image, crop_coords = crop_image_with_coords(
                image,
                self.debug_crop_size,
                crop_x=self.crop_x,
                crop_y=self.crop_y,
                seed=self.debug_seed,
            )

            debug_out_dir = debug_root / f"{img_path.stem}_debug_overfit"
            self._process_single_image(
                img_path=img_path,
                image_tensor=image,
                save_freq_path=None,
                debug_out_dir=debug_out_dir,
                crop_coords=crop_coords,
                scene_size=scene_size,
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
                scene_size=scene_size,
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
