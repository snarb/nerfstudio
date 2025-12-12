# Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LookCloser / FA-NeRF experimental integration.

This module wires a LookCloser-style model on top of the standard Nerfacto
implementation. The integration is intentionally incremental: each component of
the paper can be toggled independently to ease ablations and staged rollouts.

The implementation currently mirrors Nerfacto's rendering pathway while
exposing configuration flags and placeholder hooks for:

* Progressive image regression (2D frequency estimation)
* 3D frequency grid construction and updates
* Frequency-aware feature weighting in the field
* Adaptive ray marching based on frequency estimates
* Frequency-aware pixel sampling

Future changes can fill in the hooks with the algorithms described in the
LookCloser design document without needing to modify method registration or CLI
contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.nerfacto import (
    NerfactoModel,
    NerfactoModelConfig,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.utils import writer

try:
    import tinycudann as tcnn
except Exception:  # pragma: no cover - tcnn is optional for CPU-only runs
    tcnn = None


@writer.check_no_gradient("lookcloser_flags")
@dataclass
class LookCloserModelConfig(NerfactoModelConfig):
    """Model config for the LookCloser (FA-NeRF) integration.

    Each feature flag maps to a component of the paper and can be disabled to
    run controlled ablations or to fall back to vanilla Nerfacto behaviour.
    """

    _target: Type = field(default_factory=lambda: LookCloserModel)

    enable_progressive_image_regression: bool = True
    """Run the 2D progressive regression stage to estimate per-pixel frequencies."""

    enable_frequency_grid: bool = True
    """Construct and maintain the 3D frequency grid used by downstream modules."""

    enable_frequency_weighting: bool = True
    """Apply frequency-aware feature weighting inside the field before decoding."""

    enable_adaptive_ray_marching: bool = True
    """Use frequency-driven step sizes during ray marching."""

    enable_frequency_aware_sampling: bool = True
    """Bias pixel/ray sampling towards higher-frequency regions when available."""

    frequency_grid_resolution: int = 128
    """Resolution of the 3D frequency grid."""

    frequency_levels: int = 16
    """Number of discrete frequency levels (matches the hash-grid depth)."""


class LookCloserModel(NerfactoModel):
    """Drop-in LookCloser model built on Nerfacto.

    The class stores the component toggles and exposes small helpers to query
    them during training or rendering. The underlying rendering path is
    inherited from :class:`~nerfstudio.models.nerfacto.NerfactoModel` so users
    can gradually layer in the FA-NeRF features.
    """

    config: LookCloserModelConfig

    def __init__(self, config: LookCloserModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.lookcloser_flags = self._build_flag_report()

    def _build_flag_report(self) -> Dict[str, bool]:
        """Collect the enabled/disabled state for each component.

        Keeping this as a small helper makes it trivial to log or export the
        active configuration during experiments and ensures the flags are
        available without repeatedly reading the dataclass.
        """

        return {
            "progressive_image_regression": self.config.enable_progressive_image_regression,
            "frequency_grid": self.config.enable_frequency_grid,
            "frequency_weighting": self.config.enable_frequency_weighting,
            "adaptive_ray_marching": self.config.enable_adaptive_ray_marching,
            "frequency_aware_sampling": self.config.enable_frequency_aware_sampling,
        }

    def populate_modules(self) -> None:
        """Populate the Nerfacto modules and prepare optional LookCloser hooks."""

        super().populate_modules()
        if self.config.enable_frequency_grid:
            self.frequency_grid = FrequencyGridManager(
                aabb=self.scene_box.aabb,
                resolution=self.config.frequency_grid_resolution,
                num_levels=self.config.frequency_levels,
            )
        else:
            self.frequency_grid = None
        self._log_component_status()

    # ------------------------------------------------------------------
    # Core rendering overrides
    # ------------------------------------------------------------------
    def _apply_frequency_aware_deltas(self, ray_samples: RaySamples) -> RaySamples:
        """Clamp sampling deltas using Nyquist spacing derived from the frequency grid.

        The adaptive rule follows \delta=1/(2N_l) where N_l is the geometric
        resolution associated with the voxel frequency level. When the grid is
        absent or disabled, the original deltas are returned unchanged.
        """

        if not self.config.enable_adaptive_ray_marching or self.frequency_grid is None:
            return ray_samples

        positions = ray_samples.frustums.get_positions()
        l_values = self.frequency_grid.query(positions)
        n_levels = self.frequency_grid.level_to_resolution(l_values)
        target_deltas = torch.clamp(1.0 / (2.0 * n_levels), min=1e-4)
        ray_samples.deltas = torch.minimum(ray_samples.deltas, target_deltas)
        return ray_samples

    def _apply_frequency_weighting(self, ray_samples: RaySamples, field_outputs: Dict[str, torch.Tensor]) -> None:
        """Down-weight high-frequency features using the erf-based rule from Eq. 6.

        The weighting operates directly on the density and geometry features
        produced by the Nerfacto field. If the grid or weighting flag is
        disabled, the outputs are left untouched.
        """

        if (
            not self.config.enable_frequency_weighting
            or self.frequency_grid is None
            or FieldHeadNames.GEO_FEATURES not in field_outputs
        ):
            return

        positions = ray_samples.frustums.get_positions()
        l_grid = self.frequency_grid.query(positions).float()
        weights = FrequencyAwareWeighting.compute_weights(l_grid, num_levels=self.config.frequency_levels)
        geo_features = field_outputs[FieldHeadNames.GEO_FEATURES]
        field_outputs[FieldHeadNames.GEO_FEATURES] = geo_features * weights.unsqueeze(-1)

    def get_outputs(self, ray_bundle: RayBundle):
        """Inject frequency-aware deltas and feature weighting before rendering."""

        # Apply camera optimizer and run the standard proposal sampler
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        # Inject adaptive deltas before querying the field
        ray_samples = self._apply_frequency_aware_deltas(ray_samples)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        # Apply frequency-aware weighting on the geometry features
        self._apply_frequency_weighting(ray_samples, field_outputs)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        outputs["t_vals"] = ray_samples.frustums.starts
        return outputs

    # ------------------------------------------------------------------
    # Losses and training callbacks
    # ------------------------------------------------------------------
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Compute losses using the Charbonnier reconstruction objective and extras."""

        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"], gt_image=image
        )

        epsilon = 1e-4
        loss_dict = {}
        loss_dict["charbonnier_rgb"] = torch.sqrt((pred_rgb - gt_rgb) ** 2 + epsilon).mean()

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            if metrics_dict is not None and "distortion" in metrics_dict:
                loss_dict["distortion_loss"] = 0.01 * metrics_dict["distortion"]

            if self.config.predict_normals:
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )

            # Depth supervision if provided
            if "sparse_depth" in batch and batch["sparse_depth"] is not None:
                gt_depth_sparse = batch["sparse_depth"].to(self.device)
                valid = gt_depth_sparse > 0
                if valid.any() and self.step < 5000:
                    diff = outputs["depth"][valid] - gt_depth_sparse[valid]
                    loss_dict["depth_loss"] = torch.sqrt(diff**2 + epsilon).mean() * 0.001

            # Camera optimizer losses
            self.camera_optimizer.get_loss_dict(loss_dict)

            # Runtime grid update using available batch metadata
            self._maybe_update_frequency_grid(outputs, batch)

        return loss_dict

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        metrics_dict.update(self._boolean_metrics_from_flags())
        return metrics_dict

    def get_training_callbacks(self, training_callback_attributes):
        callbacks = super().get_training_callbacks(training_callback_attributes)
        if self.config.enable_frequency_grid:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=lambda step: setattr(self, "step", step),
                )
            )
        return callbacks

    # ------------------------------------------------------------------
    # Runtime frequency grid updates
    # ------------------------------------------------------------------
    def _maybe_update_frequency_grid(self, outputs, batch) -> None:
        if (
            not self.config.enable_frequency_grid
            or not self.config.enable_progressive_image_regression
            or self.frequency_grid is None
            or "f2d" not in batch
            or self.step % 1024 != 0
        ):
            return

        f2d_vals = batch["f2d"].to(self.device)
        depth = outputs.get("depth")
        if depth is None:
            return
        focals = batch.get("focal", None)
        if focals is None:
            return
        rays = batch.get("ray_bundle", None)
        if rays is None:
            return

        f3d = f2d_vals * (focals / (depth + 1e-6))
        levels = self.frequency_grid.freq_to_level(f3d)
        positions = rays.origins + rays.directions * depth
        self.frequency_grid.update_max(positions, levels)

    def _log_component_status(self) -> None:
        """Log a concise summary of which LookCloser components are active."""

        lines = [
            "LookCloser components:",
            f" - Progressive image regression: {'on' if self.config.enable_progressive_image_regression else 'off'}",
            f" - 3D frequency grid: {'on' if self.config.enable_frequency_grid else 'off'}",
            f" - Frequency-aware weighting: {'on' if self.config.enable_frequency_weighting else 'off'}",
            f" - Adaptive ray marching: {'on' if self.config.enable_adaptive_ray_marching else 'off'}",
            f" - Frequency-aware sampling: {'on' if self.config.enable_frequency_aware_sampling else 'off'}",
            f" - Grid resolution: {self.config.frequency_grid_resolution}",
            f" - Frequency levels: {self.config.frequency_levels}",
        ]

        # Use the writer to keep output consistent with Nerfstudio logging.
        writer.put_text("lookcloser/components", "\n".join(lines))

    def get_metrics_dict(self) -> Dict[str, float]:
        """Extend Nerfacto metrics with component toggles for experiment tracking."""

        metrics = super().get_metrics_dict()
        metrics.update(self._boolean_metrics_from_flags())
        return metrics

    def _boolean_metrics_from_flags(self) -> Dict[str, float]:
        """Represent the on/off state of each component as float metrics."""

        return {f"lookcloser/{name}": float(enabled) for name, enabled in self.lookcloser_flags.items()}

    # Placeholder hooks ---------------------------------------------------
    # These methods are intentionally lightweight. Future iterations can
    # override Nerfacto internals or plug in new samplers while continuing to
    # respect the component switches defined in the config.

    def maybe_use_frequency_grid(self) -> Optional[bool]:
        """Convenience helper consumed by future ray marching overrides."""

        return self.config.enable_frequency_grid

    def maybe_use_frequency_weighting(self) -> Optional[bool]:
        """Convenience helper consumed by future field queries."""

        return self.config.enable_frequency_weighting

    def maybe_use_adaptive_ray_marching(self) -> Optional[bool]:
        """Convenience helper consumed by future sampler overrides."""

        return self.config.enable_adaptive_ray_marching

    def maybe_use_frequency_aware_sampling(self) -> Optional[bool]:
        """Convenience helper consumed by future datamanager hooks."""

        return self.config.enable_frequency_aware_sampling

    def maybe_use_progressive_image_regression(self) -> Optional[bool]:
        """Convenience helper consumed by frequency estimation pipelines."""

        return self.config.enable_progressive_image_regression


# -----------------------------------------------------------------------------
# Frequency-aware feature weighting
# -----------------------------------------------------------------------------


class FrequencyAwareWeighting:
    """Implements the erf-based down-weighting curve from FA-NeRF Eq. 6."""

    @staticmethod
    def compute_weights(l_grid: torch.Tensor, num_levels: int = 16) -> torch.Tensor:
        l_min, l_max = 0.0, float(num_levels - 1)
        range_sq = (l_max - l_min) ** 2
        denom = torch.clamp((l_max - l_grid + 1.0) ** 2, min=1.0, max=range_sq)
        erf_arg = torch.sqrt(range_sq / denom)
        return torch.sqrt(1.0 - torch.exp(-(4.0 / math.pi) * (erf_arg**2)))


# -----------------------------------------------------------------------------
# Frequency grid utilities
# -----------------------------------------------------------------------------


class FrequencyGridManager:
    """Maintains a voxel grid of frequency levels and provides queries/updates."""

    def __init__(self, aabb: torch.Tensor, resolution: int = 128, num_levels: int = 16) -> None:
        self.aabb = aabb
        self.resolution = resolution
        self.num_levels = num_levels
        self.grid = torch.zeros((resolution, resolution, resolution), dtype=torch.float32)

    def to(self, device: torch.device) -> "FrequencyGridManager":
        self.grid = self.grid.to(device)
        self.aabb = self.aabb.to(device)
        return self

    def _get_base(self, min_res: int = 16, max_res: int = 2048) -> float:
        return math.exp((math.log(max_res) - math.log(min_res)) / (self.num_levels - 1))

    def level_to_resolution(self, level: torch.Tensor, min_res: int = 16, max_res: int = 2048) -> torch.Tensor:
        base = self._get_base(min_res=min_res, max_res=max_res)
        return min_res * (base ** level.float())

    def freq_to_level(self, scalar: torch.Tensor, min_res: int = 16, max_res: int = 2048) -> torch.Tensor:
        base = self._get_base(min_res=min_res, max_res=max_res)
        level = torch.log(torch.clamp(scalar, min=min_res) / min_res) / math.log(base)
        return torch.clamp(torch.round(level), 0, self.num_levels - 1)

    def freq_to_level_torch(self, scalar: torch.Tensor, min_res: int = 16, max_res: int = 2048) -> torch.Tensor:
        base = self._get_base(min_res=min_res, max_res=max_res)
        level = torch.log(torch.clamp(scalar, min=min_res) / min_res) / math.log(base)
        return torch.clamp(torch.round(level), 0, self.num_levels - 1).long()

    def world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        aabb_min, aabb_max = self.aabb[0], self.aabb[1]
        normed = (positions - aabb_min) / (aabb_max - aabb_min + 1e-6)
        idx = torch.clamp(normed * (self.resolution - 1), 0, self.resolution - 1)
        return idx.long()

    def query(self, positions: torch.Tensor) -> torch.Tensor:
        if self.grid.device != positions.device:
            self.to(positions.device)
        idx = self.world_to_grid(positions)
        ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
        return self.grid[ix, iy, iz]

    def update_max(self, positions: torch.Tensor, levels: torch.Tensor) -> None:
        idx = self.world_to_grid(positions)
        ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
        self.grid[ix, iy, iz] = torch.maximum(self.grid[ix, iy, iz], levels.to(self.grid.device).float())

    def initialize_from_sparse(
        self,
        sparse_points: Iterable[Tuple[int, torch.Tensor]],
        image_frequencies: Dict[Tuple[int, int], float],
        cameras: Dict[int, object],
    ) -> None:
        """Populate the grid using sparse SfM points and per-image frequencies."""

        point_freqs: Dict[int, List[float]] = {pid: [] for pid, _ in sparse_points}
        for (img_id, pt_id), f2d_scalar in image_frequencies.items():
            if img_id not in cameras:
                continue
            cam = cameras[img_id]
            if pt_id not in point_freqs:
                point_freqs[pt_id] = []
            cam_center = getattr(cam, "camera_center", None)
            if cam_center is None:
                continue
            pt_entry = dict(sparse_points).get(pt_id, None)
            if pt_entry is None:
                continue
            dist = torch.linalg.norm(pt_entry - cam_center)
            f3d = float(f2d_scalar) * float(getattr(cam, "focal_length", 1.0) / (dist + 1e-6))
            point_freqs[pt_id].append(f3d)

        for pt_id, freqs in point_freqs.items():
            if not freqs:
                continue
            median_f = float(np.median(freqs))
            level = self.freq_to_level(torch.tensor(median_f))
            pt_entry = dict(sparse_points).get(pt_id, None)
            if pt_entry is None:
                continue
            idx = self.world_to_grid(pt_entry)
            ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
            self.grid[ix, iy, iz] = torch.maximum(self.grid[ix, iy, iz], level.to(self.grid.device).float())


# -----------------------------------------------------------------------------
# 2D frequency estimation utilities (progressive image regression)
# -----------------------------------------------------------------------------


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    l_val = 1.0
    c1 = (0.01 * l_val) ** 2
    c2 = (0.03 * l_val) ** 2
    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


class InstantNGP2D(nn.Module):
    def __init__(
        self,
        n_levels: int = 16,
        n_features: int = 2,
        min_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 20,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.min_res = min_res
        self.b = math.exp((math.log(max_res) - math.log(min_res)) / (n_levels - 1))
        if tcnn is None:
            raise ImportError("tinycudann is required for InstantNGP2D")
        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": self.b,
            },
        )
        self.decoder = tcnn.Network(
            n_input_dims=n_levels * n_features,
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
        return float(self.min_res * (self.b**level_idx))

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoding(uv))

    def render_masked(self, uv_coords: torch.Tensor, max_active_level: int) -> torch.Tensor:
        features = self.encoding(uv_coords)
        cutoff = (max_active_level + 1) * self.n_features
        mask = torch.zeros_like(features)
        mask[:, :cutoff] = 1.0
        return self.decoder(features * mask)


def generate_uv_grid(y: int, x: int, h: int, w: int, size: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(y + 0.5, y + size - 0.5, steps=size, device=device)
    xs = torch.linspace(x + 0.5, x + size - 0.5, steps=size, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([grid_x / w, grid_y / h], dim=-1)
    return uv.reshape(-1, 2)


def train_2d_ngp(image: torch.Tensor, steps: int = 100) -> InstantNGP2D:
    model = InstantNGP2D().to(image.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    uv_full = generate_uv_grid(0, 0, h, w, h, image.device)[: pixels.shape[0]]
    for _ in range(steps):
        optimizer.zero_grad()
        preds = model(uv_full)
        loss = F.mse_loss(preds, pixels)
        loss.backward()
        optimizer.step()
    return model


def estimate_2d_frequencies(images: List[torch.Tensor]) -> List[torch.Tensor]:
    ssim_threshold = 0.95
    patch_size = 32
    stride = patch_size
    maps: List[torch.Tensor] = []
    for image in images:
        h, w, _ = image.shape
        model_2d = train_2d_ngp(image)
        freq_map = torch.zeros((h // stride, w // stride), dtype=torch.float32, device=image.device)
        for i, y in enumerate(range(0, h - patch_size + 1, stride)):
            for j, x in enumerate(range(0, w - patch_size + 1, stride)):
                patch_gt = image[y : y + patch_size, x : x + patch_size]
                uv_grid = generate_uv_grid(y, x, h, w, patch_size, device=image.device)
                patch_gt_fmt = patch_gt.permute(2, 0, 1).unsqueeze(0)
                found = False
                for level in range(16):
                    patch_pred = model_2d.render_masked(uv_grid, max_active_level=level)
                    patch_pred_fmt = patch_pred.view(patch_size, patch_size, 3).permute(2, 0, 1).unsqueeze(0)
                    score = compute_ssim(patch_gt_fmt, patch_pred_fmt)
                    if score > ssim_threshold:
                        freq_map[i, j] = model_2d.get_resolution_at_level(level)
                        found = True
                        break
                if not found:
                    freq_map[i, j] = model_2d.get_resolution_at_level(15)
        maps.append(freq_map)
    return maps


# -----------------------------------------------------------------------------
# Patch registry helpers
# -----------------------------------------------------------------------------


@dataclass
class PatchSample:
    img_idx: int
    uv_center: Tuple[int, int]
    f2d: float


def generate_patch_registry(dense_freq_maps: List[torch.Tensor], patch_size: int = 32) -> List[PatchSample]:
    registry: List[PatchSample] = []
    for img_idx, freq_map in enumerate(dense_freq_maps):
        h, w = freq_map.shape
        for i in range(h):
            for j in range(w):
                registry.append(PatchSample(img_idx=img_idx, uv_center=(i * patch_size, j * patch_size), f2d=float(freq_map[i, j])))
    return registry


def prepare_global_f2d_tensor(image_freq_map: Dict[Tuple[int, int], float], total_pixels: int) -> torch.Tensor:
    tensor = torch.zeros(total_pixels, dtype=torch.float32)
    for (img_id, pix_id), freq in image_freq_map.items():
        idx = img_id * total_pixels + pix_id if total_pixels > 0 else 0
        if idx < tensor.shape[0]:
            tensor[idx] = freq
    return tensor


# -----------------------------------------------------------------------------
# Frequency-aware pixel sampler
# -----------------------------------------------------------------------------


class FASPixelSampler(Sampler[List[int]]):
    def __init__(self, dense_freq_maps: List[torch.Tensor], batch_size: int = 4096, num_levels: int = 16) -> None:
        self.batch_size = batch_size
        self.num_levels = num_levels
        self.buckets: Dict[int, List[int]] = {l: [] for l in range(num_levels)}
        global_pixel_idx = 0
        base = math.exp((math.log(2048) - math.log(16)) / (num_levels - 1))
        for freq_map in dense_freq_maps:
            f_flat = freq_map.view(-1)
            levels = torch.log(torch.clamp(f_flat, min=16.0) / 16.0) / math.log(base)
            levels = torch.clamp(torch.round(levels), 0, num_levels - 1).long()
            indices = torch.arange(global_pixel_idx, global_pixel_idx + f_flat.shape[0])
            for l_idx in range(num_levels):
                mask = levels == l_idx
                if mask.any():
                    self.buckets[l_idx].extend(indices[mask].tolist())
            global_pixel_idx += f_flat.shape[0]
        ramp = np.linspace(1.0, 3.0, num_levels)
        probs = ramp / ramp.sum()
        self.samples_per_level = (probs * batch_size).astype(int)
        diff = batch_size - self.samples_per_level.sum()
        self.samples_per_level[-1] += diff

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        for l_idx in range(self.num_levels):
            count = int(self.samples_per_level[l_idx])
            bucket = self.buckets[l_idx]
            if bucket:
                chosen = np.random.choice(bucket, size=count, replace=True)
                batch.extend(chosen.tolist())
            else:
                batch.extend([0] * count)
        np.random.shuffle(batch)
        yield batch

    def __len__(self) -> int:
        return 1_000_000


# -----------------------------------------------------------------------------
# Adaptive ray marching (standalone utility)
# -----------------------------------------------------------------------------


def adaptive_step_march(
    ray_bundle: RayBundle,
    freq_grid: FrequencyGridManager,
    n_min: int = 16,
    n_max: int = 2048,
    num_levels: int = 16,
):
    rays_o = ray_bundle.origins
    rays_d = ray_bundle.directions
    b_val = math.exp((math.log(n_max) - math.log(n_min)) / (num_levels - 1))
    t_vals = torch.zeros((rays_o.shape[0], 1), device=rays_o.device)
    active = torch.ones(rays_o.shape[0], dtype=torch.bool, device=rays_o.device)
    while active.any():
        pos = rays_o[active] + rays_d[active] * t_vals[active]
        l = freq_grid.query(pos).long()
        n_l = n_min * (b_val ** l.float())
        dt = torch.clamp(1.0 / (2.0 * n_l), min=1e-4, max=0.1)
        t_vals[active] += dt
        finished = t_vals.squeeze(-1) > ray_bundle.fars.squeeze(-1)
        active = active & (~finished)
    return t_vals
