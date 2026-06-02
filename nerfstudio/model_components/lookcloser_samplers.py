"""Packed frequency-aware samplers for LookCloser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import nerfacc
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager


@dataclass
class FrequencyAwareSamplerStats:
    """Small profiling bundle for adaptive interval adjustment."""

    num_samples: Tensor
    mean_samples_per_ray: Tensor
    max_samples_per_ray: Tensor
    saturation_rate: Tensor


class FrequencyAwareVolumetricSampler(nn.Module):
    """Nerfacc occupancy traversal followed by vectorized frequency-aware subdivision."""

    def __init__(
        self,
        occupancy_grid: nerfacc.OccGridEstimator,
        freq_grid: FrequencyGridManager,
        density_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.occupancy_grid = occupancy_grid
        self.freq_grid = freq_grid
        self.density_fn = density_fn

    def get_sigma_fn(self, origins: Tensor, directions: Tensor) -> Optional[Callable]:
        """Returns a density callback for nerfacc visibility pruning during training."""
        if self.density_fn is None or not self.training:
            return None

        def sigma_fn(t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor) -> Tensor:
            positions = origins[ray_indices] + directions[ray_indices] * ((t_starts + t_ends)[:, None] * 0.5)
            return self.density_fn(positions).squeeze(-1)

        return sigma_fn

    @staticmethod
    def _empty_samples(ray_bundle: RayBundle) -> Tuple[RaySamples, Tensor, FrequencyAwareSamplerStats]:
        device = ray_bundle.origins.device
        ray_indices = torch.zeros((0,), dtype=torch.long, device=device)
        empty = torch.zeros((0, 1), dtype=ray_bundle.origins.dtype, device=device)
        ray_samples = RaySamples(
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
        zero = torch.tensor(0.0, device=device)
        stats = FrequencyAwareSamplerStats(
            num_samples=torch.tensor(0, device=device),
            mean_samples_per_ray=zero,
            max_samples_per_ray=zero,
            saturation_rate=zero,
        )
        return ray_samples, ray_indices, stats

    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float,
        far_plane: float,
        alpha_thre: float,
        cone_angle: float,
        adaptive_min_step_size: float,
        adaptive_max_step_size: float,
        adaptive_min_frequency_level: float,
        adaptive_max_frequency_level: Optional[float],
        max_steps_per_ray: int,
    ) -> Tuple[RaySamples, Tensor, FrequencyAwareSamplerStats]:
        """Generate packed adaptive samples for a ray bundle."""
        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        num_rays = rays_o.shape[0]
        device = rays_o.device

        t_min = ray_bundle.nears.contiguous().reshape(-1) if ray_bundle.nears is not None else None
        t_max = ray_bundle.fars.contiguous().reshape(-1) if ray_bundle.fars is not None else None

        with torch.no_grad():
            ray_indices, starts, ends = self.occupancy_grid.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                t_min=t_min,
                t_max=t_max,
                sigma_fn=self.get_sigma_fn(rays_o, rays_d),
                render_step_size=render_step_size,
                near_plane=near_plane,
                far_plane=far_plane,
                stratified=self.training,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )

            if starts.numel() == 0:
                return self._empty_samples(ray_bundle)

            mid = 0.5 * (starts + ends)
            positions = rays_o[ray_indices] + rays_d[ray_indices] * mid[:, None]
            levels = self.freq_grid.query(positions).reshape(-1).float()
            if adaptive_min_frequency_level > 0:
                levels = levels.clamp_min(adaptive_min_frequency_level)
            if adaptive_max_frequency_level is not None:
                levels = levels.clamp_max(adaptive_max_frequency_level)
            n_l = self.freq_grid.level_to_freq(levels).reshape(-1)

            dt_norm = 1.0 / (2.0 * n_l.clamp_min(1e-6))
            aabb_size = self.freq_grid.aabb_size_buf.to(device=device, dtype=rays_d.dtype).clamp_min(1e-6)
            norm_speed = torch.linalg.norm(rays_d[ray_indices] / aabb_size, dim=-1).clamp_min(1e-8)
            dt = (dt_norm / norm_speed).clamp(min=adaptive_min_step_size, max=adaptive_max_step_size)

            interval_lengths = (ends - starts).clamp_min(1e-8)
            counts = torch.ceil(interval_lengths / dt).to(torch.long).clamp_min(1)
            total = int(counts.sum().item())
            if total == 0:
                return self._empty_samples(ray_bundle)

            repeated_ray_indices = torch.repeat_interleave(ray_indices, counts)
            repeated_starts = torch.repeat_interleave(starts, counts)
            repeated_ends = torch.repeat_interleave(ends, counts)
            repeated_dt = torch.repeat_interleave(dt, counts)
            repeated_counts = torch.repeat_interleave(counts, counts)
            local_offsets = torch.arange(total, device=device) - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
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
            del repeated_counts

            if refined_starts.numel() == 0:
                return self._empty_samples(ray_bundle)

            packed = nerfacc.pack_info(repeated_ray_indices, num_rays)
            sample_counts = packed[:, 1]
            saturation_rate = (sample_counts >= max_steps_per_ray).float().mean() if max_steps_per_ray > 0 else torch.zeros((), device=device)

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
            spacing_starts=refined_starts[..., None],
            spacing_ends=refined_ends[..., None],
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[repeated_ray_indices]

        stats = FrequencyAwareSamplerStats(
            num_samples=torch.tensor(refined_starts.numel(), device=device),
            mean_samples_per_ray=sample_counts.float().mean(),
            max_samples_per_ray=sample_counts.max().float(),
            saturation_rate=saturation_rate,
        )
        return ray_samples, repeated_ray_indices, stats
