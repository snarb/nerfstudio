"""
LookCloser (FA-NeRF) Model Implementation.
Integrates Frequency-Aware Neural Radiance Fields with Adaptive Ray Marching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.lookcloser_field import LookCloserField
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
from nerfstudio.model_components.losses import (
    MSELoss,
    nerfstudio_distortion_loss,
)
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color


@dataclass
class LookCloserModelConfig(ModelConfig):
    """Configuration for LookCloser Model."""

    _target: Type = field(default_factory=lambda: LookCloserModel)

    # Grid parameters
    grid_resolution: int = 128
    """Resolution of the frequency voxel grid."""

    num_frequency_levels: int = 16
    """Number of discrete frequency levels."""

    min_res: float = 16.0
    """Minimum resolution (N_min)."""

    max_res: float = 2048.0
    """Maximum resolution (N_max)."""

    # Loss weights
    distortion_loss_mult: float = 0.01
    """Multiplier for Mip-NeRF 360 distortion loss."""

    depth_loss_mult: float = 0.001
    """Multiplier for sparse depth supervision."""

    depth_loss_steps: int = 5000
    """Number of steps to apply depth loss."""

    # Marching settings
    enable_adaptive_ray_marching: bool = True
    """Whether to use frequency-guided adaptive step sizes."""

    max_steps_per_ray: int = 1024
    """Maximum number of steps per ray for adaptive marching."""

    background_color: Literal["random", "last_sample", "black", "white"] = "random"
    """Background color strategy."""


class LookCloserModel(Model):
    """
    LookCloser: Frequency-Aware NeRF with Adaptive Ray Marching.

    This model maintains a 3D frequency grid that guides both the feature
    encoding capacity (via the Field) and the rendering step size (via Adaptive Ray Marching).
    """

    config: LookCloserModelConfig
    field: LookCloserField
    freq_grid: FrequencyGridManager

    def populate_modules(self):
        """Set up fields and modules."""
        super().populate_modules()

        # 1. Frequency Grid Manager (Persistent State)
        self.freq_grid = FrequencyGridManager(
            scene_box=self.scene_box,
            resolution=self.config.grid_resolution,
            num_levels=self.config.num_frequency_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
        )

        # 2. LookCloser Field (Frequency-Aware)
        self.field = LookCloserField(
            aabb=self.scene_box.aabb,
            freq_grid=self.freq_grid,
            num_levels=self.config.num_frequency_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
        )

        # 3. Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        # Metrics
        self.psnr = MSELoss()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        # Frequency grid is a buffer, not a parameter
        return param_groups

    def adaptive_ray_marching(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """
        Performs volumetric rendering with adaptive step sizes.
        Uses pre-allocated rectangular buffers to support efficient creation of
        padded RaySamples for the standard distortion loss.
        """
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        n_rays = rays_o.shape[0]
        device = rays_o.device

        # --- 1. Initialization ---
        t_vals = ray_bundle.nears.clone()

        # Accumulators for final image
        acc_rgb = torch.zeros((n_rays, 3), device=device)
        acc_depth = torch.zeros((n_rays, 1), device=device)
        acc_weights = torch.zeros((n_rays, 1), device=device)
        transmittance = torch.ones((n_rays, 1), device=device)

        # --- Pre-allocation for Padded History ---
        # We need (N_rays, Max_Steps) to satisfy RaySamples structure for distortion loss.
        max_steps = self.config.max_steps_per_ray

        # Buffers initialized to 0.
        # Note: Padding 0s here is handled safely in the Finalization step by clamping.
        history_weights = torch.zeros((n_rays, max_steps, 1), device=device)
        history_starts = torch.zeros((n_rays, max_steps, 1), device=device)
        history_ends = torch.zeros((n_rays, max_steps, 1), device=device)

        # Track insertion index per ray
        step_indices = torch.zeros(n_rays, dtype=torch.long, device=device)

        # Active mask
        active_mask = torch.ones(n_rays, dtype=torch.bool, device=device)

        # Constants
        N_min = self.config.min_res
        b_val = self.freq_grid.b
        min_step_size = 1e-4
        max_step_size = 0.1

        step_iter = 0

        # --- 2. Ray Marching Loop ---
        while active_mask.any() and step_iter < max_steps:
            # A. Current Positions
            # We work only on active rays to save compute on the Field query
            curr_t = t_vals[active_mask]
            curr_pos = rays_o[active_mask] + rays_d[active_mask] * curr_t

            # B. Adaptive Step Size Calculation
            l_indices = self.freq_grid.query(curr_pos).float()
            N_l = N_min * (b_val ** l_indices)

            # Nyquist step: delta = 1 / (2 * N_l)
            dt = 1.0 / (2.0 * N_l)
            dt = torch.clamp(dt, min=min_step_size, max=max_step_size)

            # C. Model Query (Density & Color)
            # Pass l_indices for Eq. 6 feature re-weighting
            view_dirs = rays_d[active_mask]
            density, rgb = self.field.query_points(curr_pos, view_dirs, l_grid=l_indices)

            # D. Volumetric Integration
            sigma = F.relu(density)
            alpha = 1.0 - torch.exp(-sigma * dt)

            curr_transmittance = transmittance[active_mask]
            weight = curr_transmittance * alpha

            # Accumulate Render
            acc_rgb[active_mask] += weight * rgb
            acc_depth[active_mask] += weight * curr_t
            acc_weights[active_mask] += weight

            # Update Transmittance
            transmittance[active_mask] *= (1.0 - alpha + 1e-10)

            # E. Store History (Scatter into Padded Buffers)
            # We use the 'step_indices' to place samples in the correct column per ray
            curr_step_idx = step_indices[active_mask]

            # Advanced indexing: [rows, cols]
            active_ray_idx = torch.nonzero(active_mask).squeeze(-1)

            history_weights[active_ray_idx, curr_step_idx] = weight
            history_starts[active_ray_idx, curr_step_idx] = curr_t
            history_ends[active_ray_idx, curr_step_idx] = curr_t + dt

            # Increment step indices for active rays
            step_indices[active_mask] += 1

            # F. Advance Rays
            t_vals[active_mask] += dt

            # G. Pruning
            opaque = transmittance < 1e-4
            out_of_bounds = t_vals > ray_bundle.fars
            newly_finished = (opaque | out_of_bounds).flatten() & active_mask
            active_mask = active_mask & (~newly_finished)

            step_iter += 1

        # --- 3. Finalization ---
        depth_final = acc_depth / (acc_weights + 1e-6)

        # Background composition
        if self.renderer_rgb.background_color == "white":
            acc_rgb = acc_rgb + transmittance
        elif self.renderer_rgb.background_color == "random":
             bg = torch.rand_like(acc_rgb)
             acc_rgb = acc_rgb + transmittance * bg

        # --- 4. Construct RaySamples for Loss ---
        # We need to normalize t -> s [0, 1] for MipNeRF 360 distortion loss
        # s = (t - near) / (far - near)

        # Expand near/far for broadcasting: (N, 1, 1)
        nears = ray_bundle.nears.unsqueeze(-1)
        fars = ray_bundle.fars.unsqueeze(-1)
        span = (fars - nears).clamp(min=1e-6)

        norm_starts = (history_starts - nears) / span
        norm_ends = (history_ends - nears) / span

        # Correctness Fix: Clamp to [0, 1] to handle padding logic cleanly.
        # Padded zeros (t=0) might become s < 0 if near > 0.
        # Clamping forces them to valid range [0, 1].
        # Since their weight is 0, they won't contribute to loss, but coordinates remain valid.
        norm_starts = norm_starts.clamp(min=0.0, max=1.0)
        norm_ends = norm_ends.clamp(min=0.0, max=1.0)

        # Create Frustums and RaySamples
        # We use dummy directions/origins for the samples as distortion loss doesn't use them.
        dummy_dirs = torch.zeros_like(history_starts).expand(-1, -1, 3)
        dummy_origins = torch.zeros_like(history_starts).expand(-1, -1, 3)

        frustums = Frustums(
            origins=dummy_origins,
            directions=dummy_dirs,
            starts=norm_starts,
            ends=norm_ends,
            pixel_area=torch.zeros_like(norm_starts) # Dummy
        )

        # camera_indices must match the dimensions (N_rays, Max_Steps, 1) or (N_rays, 1)
        # RaySamples expects one index per sample in the flattened structure,
        # or we can broadcast. Since RaySamples is a TensorDataclass,
        # we construct it with full shape (N, M, 1).
        loss_ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=torch.zeros_like(history_starts, dtype=torch.long),
            deltas=norm_ends - norm_starts,
            spacing_starts=norm_starts, # Explicitly used by nerfstudio_distortion_loss
            spacing_ends=norm_ends
        )

        return {
            "rgb": acc_rgb,
            "depth": depth_final,
            "accumulation": acc_weights,
            # Pass data needed for loss
            "loss_ray_samples": loss_ray_samples,
            "loss_weights": history_weights
        }

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        if self.config.enable_adaptive_ray_marching:
            return self.adaptive_ray_marching(ray_bundle)
        else:
            raise NotImplementedError("Only adaptive ray marching is supported.")

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        # 1. Charbonnier Reconstruction Loss
        epsilon = 1e-4
        diff_sq = (outputs["rgb"] - image) ** 2
        loss_dict["rgb_loss"] = torch.sqrt(diff_sq + epsilon).mean()

        # 2. Distortion Loss (Mip-NeRF 360)
        # Uses the standard Nerfstudio implementation which expects (RaySamples, weights)
        if self.config.distortion_loss_mult > 0:
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * nerfstudio_distortion_loss(
                ray_samples=outputs["loss_ray_samples"],
                weights=outputs["loss_weights"]
            )

        # 3. Depth Loss (Sparse Supervision)
        if (
            self.config.depth_loss_mult > 0
            and "depth_image" in batch
        ):
            gt_depth = batch["depth_image"].to(self.device)
            mask = gt_depth > 0
            if mask.any():
                pred_depth = outputs["depth"]
                depth_loss = F.mse_loss(pred_depth[mask], gt_depth[mask])
                loss_dict["depth_loss"] = self.config.depth_loss_mult * depth_loss

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [C, H, W] for metrics
        image = torch.moveaxis(image, -1, 0)
        rgb = torch.moveaxis(rgb, -1, 0)

        psnr = self.psnr(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
        }
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }
        return metrics_dict, images_dict