"""Scene-linear HDR utilities shared by datasets, models, metrics, and preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor

from nerfstudio.data.utils.data_utils import load_exr_image

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = (2523.0 / 4096.0) * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = (2413.0 / 4096.0) * 32.0
PQ_C3 = (2392.0 / 4096.0) * 32.0
PQ_MAX_NITS = 10_000.0
BT709_LUMA = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class HDRCalibration:
    """Dataset-wide radiometric constants derived from the train split only."""

    linear_scale: float
    initial_radiance: float
    nits_per_scene_unit: float
    black_nits: float
    peak_nits: float
    log_mean_luminance: float
    luminance_q999: float
    luminance_q9999: float
    negative_channel_fraction: float
    nonfinite_channel_fraction: float
    sampled_pixel_count: int
    source: str = "robust_train_split"

    def as_metadata(self) -> dict:
        return asdict(self)


def pq_encode_nits(nits: Tensor, *, clamp_max: bool = True) -> Tensor:
    """Apply the SMPTE ST 2084 inverse EOTF to display-linear nits.

    The calculation intentionally runs in float32 outside autocast because the
    derivative near black is poorly behaved in float16.
    """

    with torch.autocast(device_type=nits.device.type, enabled=False):
        value = nits.float().clamp_min(0.0)
        if clamp_max:
            value = value.clamp_max(PQ_MAX_NITS)
        normalized = value / PQ_MAX_NITS
        powered = normalized.pow(PQ_M1)
        return ((PQ_C1 + PQ_C2 * powered) / (1.0 + PQ_C3 * powered)).pow(PQ_M2)


def pq_decode_nits(code: Tensor) -> Tensor:
    """Apply the SMPTE ST 2084 EOTF to a normalized PQ code."""

    with torch.autocast(device_type=code.device.type, enabled=False):
        value = code.float().clamp(0.0, 1.0)
        powered = value.pow(1.0 / PQ_M2)
        denominator = (PQ_C2 - PQ_C3 * powered).clamp_min(1e-12)
        normalized = ((powered - PQ_C1).clamp_min(0.0) / denominator).pow(1.0 / PQ_M1)
        return PQ_MAX_NITS * normalized


def scene_linear_to_pq(
    rgb: Tensor,
    *,
    nits_per_scene_unit: float,
    black_nits: float = 0.005,
    clamp_max: bool = True,
) -> Tensor:
    """Encode scene-linear RGB using one fixed dataset-wide scene-to-nits scale."""

    nits = float(black_nits) + float(nits_per_scene_unit) * rgb.float().clamp_min(0.0)
    return pq_encode_nits(nits, clamp_max=clamp_max)


def pq_to_scene_linear(
    code: Tensor,
    *,
    nits_per_scene_unit: float,
    black_nits: float = 0.005,
) -> Tensor:
    """Decode normalized PQ code into scene-linear RGB units."""

    return (pq_decode_nits(code) - float(black_nits)).clamp_min(0.0) / float(nits_per_scene_unit)


def hdr_display_preview(rgb: Tensor, exposure_ev: float = 0.0) -> Tensor:
    """Create a neutral SDR preview without changing the scene-linear master.

    This is a plain exposure multiplication followed by the IEC sRGB transfer
    function and clipping.  It is intentionally not a creative tone mapper.
    """

    linear = (rgb.float().clamp_min(0.0) * (2.0 ** float(exposure_ev))).clamp_max(1.0)
    return torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)


def activate_hdr_rgb(
    raw_rgb: Tensor,
    *,
    parameterization: str,
    linear_scale: float,
    initial_radiance: float,
    nits_per_scene_unit: float,
    black_nits: float = 0.005,
    peak_nits: float = PQ_MAX_NITS,
    softplus_beta: float = 1.0,
    pq_code_temperature: float = 1.0,
) -> Tensor:
    """Convert raw color-head logits to scene-linear radiance samples."""

    if parameterization == "sigmoid":
        return torch.sigmoid(raw_rgb)
    if linear_scale <= 0 or initial_radiance <= 0:
        raise ValueError("linear_scale and initial_radiance must be positive")
    if parameterization == "linear_softplus":
        normalized_initial = max(float(initial_radiance) / float(linear_scale), 1e-8)
        bias = np.log(np.expm1(float(softplus_beta) * normalized_initial)) / float(softplus_beta)
        return float(linear_scale) * torch.nn.functional.softplus(
            raw_rgb + float(bias), beta=float(softplus_beta), threshold=20.0
        )
    if parameterization == "pq_code":
        peak_code = pq_encode_nits(torch.as_tensor(float(peak_nits), device=raw_rgb.device)).item()
        initial_code = scene_linear_to_pq(
            torch.as_tensor(float(initial_radiance), device=raw_rgb.device),
            nits_per_scene_unit=nits_per_scene_unit,
            black_nits=black_nits,
        ).item()
        fraction = float(np.clip(initial_code / max(peak_code, 1e-8), 1e-6, 1.0 - 1e-6))
        bias = np.log(fraction / (1.0 - fraction))
        code = float(peak_code) * torch.sigmoid(raw_rgb / float(pq_code_temperature) + float(bias))
        return pq_to_scene_linear(
            code,
            nits_per_scene_unit=nits_per_scene_unit,
            black_nits=black_nits,
        ).to(dtype=raw_rgb.dtype)
    raise ValueError(f"Unknown RGB output parameterization: {parameterization!r}")


def calibrate_exr_paths(
    paths: Iterable[Path],
    *,
    sample_stride: int = 8,
    black_nits: float = 0.005,
    white_luminance: Optional[float] = None,
) -> HDRCalibration:
    """Derive deterministic scene-wide HDR scales from EXR training images."""

    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    luminance_parts = []
    channel_parts = []
    for path in paths:
        image = load_exr_image(Path(path))
        rgb = image[::sample_stride, ::sample_stride, :3]
        channel_parts.append(rgb.reshape(-1))
        luminance_parts.append(
            np.tensordot(rgb, np.asarray(BT709_LUMA, dtype=np.float32), axes=([-1], [0])).reshape(-1)
        )
    if not luminance_parts:
        raise ValueError("Cannot calibrate HDR data without training EXR images")

    luminance = np.concatenate(luminance_parts).astype(np.float64, copy=False)
    channels = np.concatenate(channel_parts).astype(np.float64, copy=False)
    finite_luminance = luminance[np.isfinite(luminance)]
    positive_luminance = finite_luminance[finite_luminance > 0]
    if positive_luminance.size == 0:
        raise ValueError("HDR calibration requires at least one positive finite luminance sample")

    log_mean = float(np.exp(np.mean(np.log(np.maximum(positive_luminance, 1e-8)))))
    q50 = float(np.quantile(positive_luminance, 0.5))
    q999 = float(np.quantile(positive_luminance, 0.999))
    q9999 = float(np.quantile(positive_luminance, 0.9999))
    linear_scale = max(q999, 1e-8)
    if white_luminance is not None:
        nits_per_scene_unit = float(white_luminance)
        source = "exr_white_luminance"
    else:
        nits_per_scene_unit = min(20.0 / max(log_mean, 1e-8), 4000.0 / max(q9999, 1e-8))
        source = "robust_train_split"

    finite_channels = np.isfinite(channels)
    return HDRCalibration(
        linear_scale=linear_scale,
        initial_radiance=max(q50, 1e-8),
        nits_per_scene_unit=float(nits_per_scene_unit),
        black_nits=float(black_nits),
        peak_nits=PQ_MAX_NITS,
        log_mean_luminance=log_mean,
        luminance_q999=q999,
        luminance_q9999=q9999,
        negative_channel_fraction=float(np.mean(channels[finite_channels] < 0)) if finite_channels.any() else 0.0,
        nonfinite_channel_fraction=float(1.0 - np.mean(finite_channels)),
        sampled_pixel_count=int(luminance.size),
        source=source,
    )
