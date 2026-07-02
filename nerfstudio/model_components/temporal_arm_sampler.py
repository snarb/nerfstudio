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

"""
Budget-aware Adaptive Ray Marching (ARM) sampler for the temporal model.

Ported from LookCloser's ``FrequencyAwareVolumetricSampler`` (the lever that drove structural
artifacts to ~0 on the static scene). Two changes vs. the original:
  * per-ray ``times`` are threaded into the density/sigma evaluation (4D field), and
  * the frequency signal is provided by a small pluggable callable (default: a constant
    ``fallback_frequency_level``) so ARM works WITHOUT the frequency-map preprocessing pipeline.

Pipeline: nerfacc occupancy coarse traversal -> per-interval frequency level -> Nyquist fine step
``dt = 1/(2*f)`` (normalized by ray speed) -> **budget-aware per-ray dt scaling** so the total sample
count per ray stays <= ``max_steps_per_ray`` (avoids front-loading ghost gaps) -> packed RaySamples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import nerfacc
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples


@dataclass
class ARMSamplerStats:
    num_samples: Tensor
    mean_samples_per_ray: Tensor
    max_samples_per_ray: Tensor
    saturation_rate: Tensor


class TemporalARMSampler(nn.Module):
    """Time-aware budget-aware adaptive ray marching over a 3D nerfacc occupancy grid.

    Args:
        occupancy_grid: nerfacc OccGridEstimator (3D, unchanged from the static model).
        density_fn: ``density_fn(positions, times=None) -> density`` of the temporal field.
        aabb: scene aabb ``[[x0,y0,z0],[x1,y1,z1]]`` used to normalize ray speed.
        min_res, max_res, num_levels: frequency-level -> resolution mapping ``N_l = min_res * b^l``.
        freq_level_fn: optional ``(positions) -> level`` callable; if None, a constant
            ``fallback_frequency_level`` is used everywhere (ARM without a frequency grid).
        fallback_frequency_level: constant level when ``freq_level_fn`` is None.
    """

    def __init__(
        self,
        occupancy_grid: nerfacc.OccGridEstimator,
        density_fn: Callable,
        aabb: Tensor,
        min_res: float = 16.0,
        max_res: float = 2048.0,
        num_levels: int = 16,
        freq_level_fn: Optional[Callable[[Tensor], Tensor]] = None,
        fallback_frequency_level: float = 8.0,
    ) -> None:
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.density_fn = density_fn
        aabb_size = (aabb[1] - aabb[0]).reshape(3).float()
        self.register_buffer("aabb_size_buf", aabb_size)
        self.min_res = float(min_res)
        self.num_levels = int(num_levels)
        self.growth_factor = (
            math.exp((math.log(max_res) - math.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1.0
        )
        self.freq_level_fn = freq_level_fn
        self.fallback_frequency_level = float(fallback_frequency_level)

    def level_to_freq(self, levels: Tensor) -> Tensor:
        """N_l = min_res * growth_factor**level (continuous in level)."""
        return self.min_res * torch.pow(torch.as_tensor(self.growth_factor, device=levels.device), levels)

    def _query_levels(self, positions: Tensor) -> Tensor:
        if self.freq_level_fn is not None:
            return self.freq_level_fn(positions).reshape(-1).float()
        return torch.full((positions.shape[0],), self.fallback_frequency_level, device=positions.device)

    def get_sigma_fn(self, origins: Tensor, directions: Tensor, times: Optional[Tensor]) -> Optional[Callable]:
        """Time-aware density callback for nerfacc visibility pruning during training."""
        if self.density_fn is None or not self.training:
            return None

        def sigma_fn(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
            positions = origins[ray_indices] + directions[ray_indices] * ((t_starts + t_ends)[:, None] * 0.5)
            if times is not None:
                return self.density_fn(positions, times=times[ray_indices]).squeeze(-1)
            return self.density_fn(positions).squeeze(-1)

        return sigma_fn

    @staticmethod
    def _empty(ray_bundle: RayBundle) -> Tuple[RaySamples, Tensor, ARMSamplerStats]:
        device = ray_bundle.origins.device
        ri = torch.zeros((0,), dtype=torch.long, device=device)
        empty = torch.zeros((0, 1), dtype=ray_bundle.origins.dtype, device=device)
        rs = RaySamples(
            frustums=Frustums(
                origins=torch.zeros((0, 3), dtype=ray_bundle.origins.dtype, device=device),
                directions=torch.zeros((0, 3), dtype=ray_bundle.directions.dtype, device=device),
                starts=empty,
                ends=empty,
                pixel_area=empty,
            ),
            camera_indices=torch.zeros((0, 1), dtype=torch.long, device=device),
            deltas=empty,
            spacing_starts=empty,
            spacing_ends=empty,
        )
        z = torch.tensor(0.0, device=device)
        return rs, ri, ARMSamplerStats(torch.tensor(0, device=device), z, z, z)

    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float,
        far_plane: float,
        alpha_thre: float,
        early_stop_eps: float,
        cone_angle: float,
        adaptive_min_step_size: float,
        adaptive_max_step_size: float,
        adaptive_min_frequency_level: float,
        adaptive_max_frequency_level: Optional[float],
        adaptive_interval_level_mode: str,
        max_steps_per_ray: int,
    ) -> Tuple[RaySamples, Tensor, ARMSamplerStats]:
        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        num_rays = rays_o.shape[0]
        device = rays_o.device
        times = ray_bundle.times
        t_min = ray_bundle.nears.contiguous().reshape(-1) if ray_bundle.nears is not None else None
        t_max = ray_bundle.fars.contiguous().reshape(-1) if ray_bundle.fars is not None else None

        with torch.no_grad():
            ray_indices, starts, ends = self.occupancy_grid.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                t_min=t_min,
                t_max=t_max,
                sigma_fn=self.get_sigma_fn(rays_o, rays_d, times),
                render_step_size=render_step_size,
                near_plane=near_plane,
                far_plane=far_plane,
                stratified=self.training,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
                early_stop_eps=early_stop_eps,
            )
            if starts.numel() == 0:
                return self._empty(ray_bundle)

            mid = 0.5 * (starts + ends)
            if adaptive_interval_level_mode == "midpoint":
                positions = rays_o[ray_indices] + rays_d[ray_indices] * mid[:, None]
                levels = self._query_levels(positions)
            elif adaptive_interval_level_mode == "max3":
                query_t = torch.stack((starts, mid, ends), dim=1)
                positions = rays_o[ray_indices, None, :] + rays_d[ray_indices, None, :] * query_t[..., None]
                levels = self._query_levels(positions.reshape(-1, 3)).reshape(-1, 3).amax(dim=1)
            else:
                raise ValueError(f"Unknown adaptive_interval_level_mode={adaptive_interval_level_mode!r}.")
            if adaptive_min_frequency_level > 0:
                levels = levels.clamp_min(adaptive_min_frequency_level)
            if adaptive_max_frequency_level is not None:
                levels = levels.clamp_max(adaptive_max_frequency_level)
            n_l = self.level_to_freq(levels).reshape(-1)

            dt_norm = 1.0 / (2.0 * n_l.clamp_min(1e-6))
            aabb_size = self.aabb_size_buf.to(device=device, dtype=rays_d.dtype).clamp_min(1e-6)
            norm_speed = torch.linalg.norm(rays_d[ray_indices] / aabb_size, dim=-1).clamp_min(1e-8)
            dt = (dt_norm / norm_speed).clamp(min=adaptive_min_step_size, max=adaptive_max_step_size)

            interval_lengths = (ends - starts).clamp_min(1e-8)
            # Budget-aware per-ray dt scaling (the front-loading fix): scale dt so per-ray sample
            # count stays <= max_steps_per_ray, distributing the budget along the full ray.
            counts_raw = torch.ceil(interval_lengths / dt).to(torch.long).clamp_min(1)
            if max_steps_per_ray > 0:
                per_ray_total = torch.zeros(num_rays, dtype=torch.long, device=device)
                per_ray_total.scatter_add_(0, ray_indices, counts_raw)
                over_budget = (per_ray_total.float() / max_steps_per_ray).clamp_min(1.0)
                dt = dt * over_budget[ray_indices]
                counts = torch.ceil(interval_lengths / dt).to(torch.long).clamp_min(1)
            else:
                counts = counts_raw
            total = int(counts.sum().item())
            if total == 0:
                return self._empty(ray_bundle)

            repeated_ray_indices = torch.repeat_interleave(ray_indices, counts)
            repeated_starts = torch.repeat_interleave(starts, counts)
            repeated_ends = torch.repeat_interleave(ends, counts)
            repeated_dt = torch.repeat_interleave(dt, counts)
            local_offsets = torch.arange(total, device=device) - torch.repeat_interleave(
                torch.cumsum(counts, dim=0) - counts, counts
            )
            local_offsets = local_offsets.to(starts.dtype)

            refined_starts = repeated_starts + repeated_dt * local_offsets
            refined_ends = torch.minimum(refined_starts + repeated_dt, repeated_ends)
            valid = refined_ends > refined_starts
            if max_steps_per_ray > 0:
                packed_all = nerfacc.pack_info(repeated_ray_indices, num_rays)
                ranks = torch.arange(total, device=device) - packed_all[repeated_ray_indices, 0]
                valid = valid & (ranks < max_steps_per_ray)

            repeated_ray_indices = repeated_ray_indices[valid]
            refined_starts = refined_starts[valid]
            refined_ends = refined_ends[valid]
            if refined_starts.numel() == 0:
                return self._empty(ray_bundle)

            packed = nerfacc.pack_info(repeated_ray_indices, num_rays)
            sample_counts = packed[:, 1]
            saturation_rate = (
                (sample_counts >= max_steps_per_ray).float().mean()
                if max_steps_per_ray > 0
                else torch.zeros((), device=device)
            )

            if t_min is None:
                ray_nears = torch.full_like(refined_starts, float(near_plane))
            else:
                ray_nears = t_min[repeated_ray_indices].to(dtype=refined_starts.dtype)
            if t_max is None:
                ray_fars = torch.full_like(refined_ends, float(far_plane))
            else:
                ray_fars = t_max[repeated_ray_indices].to(dtype=refined_ends.dtype)
            ray_spans = (ray_fars - ray_nears).clamp_min(1e-6)
            spacing_starts = ((refined_starts - ray_nears) / ray_spans).clamp(0.0, 1.0)
            spacing_ends = ((refined_ends - ray_nears) / ray_spans).clamp(0.0, 1.0)

        origins = rays_o[repeated_ray_indices]
        directions = rays_d[repeated_ray_indices]
        camera_indices = ray_bundle.camera_indices
        if camera_indices is not None:
            camera_indices = camera_indices.contiguous()[repeated_ray_indices]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=refined_starts[..., None],
                ends=refined_ends[..., None],
                pixel_area=ray_bundle[repeated_ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
            deltas=(refined_ends - refined_starts)[..., None],
            spacing_starts=spacing_starts[..., None],
            spacing_ends=spacing_ends[..., None],
        )
        if times is not None:
            ray_samples.times = times[repeated_ray_indices]

        stats = ARMSamplerStats(
            num_samples=torch.tensor(refined_starts.numel(), device=device),
            mean_samples_per_ray=sample_counts.float().mean(),
            max_samples_per_ray=sample_counts.max().float(),
            saturation_rate=saturation_rate,
        )
        return ray_samples, repeated_ray_indices, stats
