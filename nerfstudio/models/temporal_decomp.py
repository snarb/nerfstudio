# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Mask-gated static/dynamic DECOMPOSITION temporal Instant-NGP.

Extends :class:`TemporalInstantNGPModel` with the H2D decomposition field (separate static
3D branch and dynamic 4D branch) and a mask-gated dynamic-sparsity loss that penalizes the
dynamic branch's opacity on NON-person rays. This pushes the moving people into the 4D
dynamic branch and the background into the 3D static branch -> faster + fewer artifacts.

Runs on the WINNER streaming setup (VanillaPipeline + ParallelDataManager load_from_disk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import nerfacc
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.models.temporal_instant_ngp import TemporalInstantNGPModel, TemporalInstantNGPModelConfig


@dataclass
class TemporalDecompModelConfig(TemporalInstantNGPModelConfig):
    """Mask-gated static/dynamic decomposition temporal Instant-NGP config."""

    _target: Type = field(default_factory=lambda: TemporalDecompModel)
    decompose: bool = True
    """Use the H2D static/dynamic decomposition field + dynamic-sparsity loss.
    When True the field hypothesis is forced to ``H2D``."""
    dynamic_sparsity_mult: float = 0.05
    """Weight (lambda) of the mask-gated dynamic-sparsity loss. The loss penalizes the
    fraction of per-ray opacity coming from the dynamic branch on NON-person rays."""


class TemporalDecompModel(TemporalInstantNGPModel):
    """Decomposition temporal Instant-NGP model.

    The field exposes a per-sample DYNAMIC_DENSITY (sigma_d). ``get_outputs`` accumulates the
    per-sample dynamic opacity fraction ``sigma_d / (sigma_s + sigma_d + eps)`` along each ray
    to produce ``dynamic_accumulation`` in [0, 1]. ``get_loss_dict`` adds a mask-gated
    sparsity term that suppresses this dynamic accumulation on non-person rays.
    """

    config: TemporalDecompModelConfig

    def populate_modules(self):
        # Force the decomposition field hypothesis when decompose is on.
        if self.config.decompose:
            self.config.hypothesis = "H2D"
        super().populate_modules()

    def get_outputs(self, ray_bundle: RayBundle):  # type: ignore
        if not self.config.decompose:
            return super().get_outputs(ray_bundle)

        assert self.field is not None
        num_rays = len(ray_bundle)

        # Sampling: reuse ARM when enabled, else the fixed-step volumetric sampler.
        if self.config.enable_adaptive_ray_marching:
            from typing import cast

            coarse_step = float(self.config.adaptive_coarse_step_size or cast(float, self.config.render_step_size))
            with torch.no_grad():
                ray_samples, ray_indices, _stats = self.arm_sampler(
                    ray_bundle=ray_bundle,
                    render_step_size=coarse_step,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    alpha_thre=self.config.alpha_thre,
                    early_stop_eps=self.config.transmittance_threshold,
                    cone_angle=self.config.cone_angle,
                    adaptive_min_step_size=self.config.adaptive_min_step_size,
                    adaptive_max_step_size=self.config.adaptive_max_step_size,
                    adaptive_min_frequency_level=self.config.adaptive_min_frequency_level,
                    adaptive_max_frequency_level=self.config.adaptive_max_frequency_level,
                    adaptive_interval_level_mode=self.config.adaptive_interval_level_mode,
                    max_steps_per_ray=self.config.max_steps_per_ray,
                )
        else:
            with torch.no_grad():
                ray_samples, ray_indices = self.sampler(
                    ray_bundle=ray_bundle,
                    near_plane=self.config.near_plane,
                    far_plane=self.config.far_plane,
                    render_step_size=self.config.render_step_size,
                    alpha_thre=self.config.alpha_thre,
                    cone_angle=self.config.cone_angle,
                )

        field_outputs = self.field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        sigma_total = field_outputs[FieldHeadNames.DENSITY][..., 0]
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=sigma_total,
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB], weights=weights, ray_indices=ray_indices, num_rays=num_rays
        )
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays)
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        # Dynamic accumulation: per-ray sum of weight * (sigma_d / (sigma_s + sigma_d + eps)).
        sigma_d = field_outputs[FieldHeadNames.DYNAMIC_DENSITY][..., 0]
        dyn_frac = (sigma_d / (sigma_total + 1e-6)).clamp(0.0, 1.0)  # per-sample
        dynamic_accumulation = nerfacc.accumulate_along_rays(
            weights[..., 0], values=dyn_frac[..., None], ray_indices=ray_indices, n_rays=num_rays
        )

        return {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "dynamic_accumulation": dynamic_accumulation,
        }

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict=metrics_dict)
        if self.config.decompose and "dynamic_accumulation" in outputs:
            dyn_acc = outputs["dynamic_accumulation"]  # [num_rays, 1]
            is_person = batch.get("is_person", None)
            if is_person is None:
                # No labels: treat all as person (unknown) -> no suppression.
                is_person = torch.ones_like(dyn_acc)
            else:
                is_person = is_person.to(dyn_acc).reshape(dyn_acc.shape)
            # Penalize dynamic opacity ONLY on non-person rays.
            dyn_sparsity = ((1.0 - is_person) * dyn_acc).mean()
            loss_dict["dynamic_sparsity_loss"] = self.config.dynamic_sparsity_mult * dyn_sparsity
        return loss_dict
