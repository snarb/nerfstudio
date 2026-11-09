"""Actor-only image metrics with explicit receptive-field support.

RGB outside the frozen semantic mask cannot affect these scores. PSNR covers
every admitted pixel, including boundaries. SSIM and AlexNet LPIPS cover only
windows/features whose entire image footprint is admitted; report their support
alongside scores. These are foreground-domain metrics, not whole-image scores.
No background replacement or prediction-dependent mask is used.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.lpips import _normalize_tensor


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _feature_support(mask: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Propagate known input support through actual AlexNet spatial operators."""
    if isinstance(module, nn.Sequential):
        for child in module:
            mask = _feature_support(mask, child)
        return mask
    if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
        if isinstance(module, nn.Conv2d) and module.padding_mode != "zeros":
            raise ValueError("Only fixed zero-padded convolutions are supported")
        if isinstance(module, nn.MaxPool2d) and module.ceil_mode:
            raise ValueError("ceil-mode pooling needs a separate support audit")
        kernel = _pair(module.kernel_size)
        stride = _pair(module.stride or module.kernel_size)
        dilation = _pair(module.dilation)
        py, px = _pair(module.padding)
        # Beyond the raster, zero conv padding / -inf max-pool padding is fixed,
        # independent of any excluded image RGB, so it is known support.
        padded = F.pad(mask.float(), (px, px, py, py), value=1)
        counts = F.conv2d(padded, mask.new_ones((1, 1, *kernel), dtype=torch.float32),
                          stride=stride, dilation=dilation)
        return counts == math.prod(kernel)
    if isinstance(module, (nn.ReLU, nn.Identity, nn.Dropout)):
        return mask
    raise ValueError(f"Unreviewed spatial operator: {type(module).__name__}")


@torch.inference_mode()
def actor_image_metrics(prediction: torch.Tensor, target: torch.Tensor,
                        actor_mask: torch.Tensor, lpips_model: nn.Module) -> dict:
    """Score one NCHW RGB image in [0,1], using a frozen N1HW boolean mask.

    Pass a cached, eval-mode TorchMetrics AlexNet LPIPS model with resize=None.
    If any LPIPS level has no complete foreground footprint, LPIPS is unavailable
    (None), rather than silently discarding that level or scoring background.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction and target must share NCHW shape")
    if prediction.shape[:2] != (1, 3):
        raise ValueError("Score one RGB camera image at a time")
    if actor_mask.shape != (1, 1, *prediction.shape[-2:]) or actor_mask.dtype != torch.bool:
        raise ValueError("Frozen actor mask must be boolean N1HW")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("SSIM needs at least 11 pixels per image dimension")
    if not actor_mask.any():
        raise ValueError("Empty actor domain")
    for value in (prediction, target):
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("Expected finite display RGB in [0,1]")
    if (lpips_model.training or getattr(lpips_model, "pnet_type", None) != "alex"
            or getattr(lpips_model, "resize", None) is not None):
        raise ValueError("Expected eval-mode AlexNet LPIPS without resizing")

    pred, gt = prediction.float(), target.float()
    mse = (pred - gt).square()[actor_mask.expand_as(pred)].mean()
    psnr = float("inf") if mse == 0 else float(-10 * torch.log10(mse))
    # TorchMetrics uses an 11x11 Gaussian (sigma1.5), with reflected boundaries.
    support = F.conv2d(F.pad(actor_mask.float(), (5, 5, 5, 5), mode="reflect"),
                       pred.new_ones((1, 1, 11, 11))) == 121
    _, ssim_map = structural_similarity_index_measure(
        pred, gt, data_range=1.0, sigma=1.5, kernel_size=11, return_full_image=True)
    ssim = float(ssim_map[support.expand_as(ssim_map)].mean()) if support.any() else None

    p = lpips_model.scaling_layer(2 * pred - 1)
    q = lpips_model.scaling_layer(2 * gt - 1)
    mask = actor_mask
    layer_scores, layer_support = [], []
    for index in range(5):
        block = getattr(lpips_model.net, f"slice{index + 1}")
        p, q = block(p), block(q)
        mask = _feature_support(mask, block)
        distance = lpips_model.lins[index]((_normalize_tensor(p) - _normalize_tensor(q)).square())
        if distance.shape != mask.shape:
            raise ValueError("LPIPS feature/support raster mismatch")
        count = int(mask.sum())
        layer_support.append({"level": index, "valid_features": count,
                              "total_features": mask.numel()})
        layer_scores.append(float(distance[mask].mean()) if count else None)
    lpips = sum(layer_scores) if all(v is not None for v in layer_scores) else None
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips,
            "domain": "frozen actor mask; complete receptive fields for SSIM/LPIPS",
            "actor_pixels": int(actor_mask.sum()), "image_pixels": actor_mask.numel(),
            "ssim_valid_centers": int(support.sum()),
            "lpips_feature_support": layer_support, "lpips_per_layer": layer_scores,
            "boundary_scope": "PSNR includes every admitted boundary pixel; perceptual scores describe interiors"}
