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
    packed_info: Tensor


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
    def _normalized_ray_speed(directions: Tensor, aabb_size: Tensor) -> Tensor:
        """Return world-ray speed after normalization into an anisotropic AABB."""
        return torch.linalg.norm(directions / aabb_size.clamp_min(1e-6), dim=-1).clamp_min(1e-8)

    @staticmethod
    def _pack_info_from_sorted_indices(ray_indices: Tensor, num_rays: int) -> Tensor:
        """Device-agnostic equivalent of ``nerfacc.pack_info`` for sorted indices."""
        counts = torch.bincount(ray_indices, minlength=num_rays)
        starts = torch.cumsum(counts, dim=0) - counts
        return torch.stack((starts, counts), dim=-1)

    @staticmethod
    def _merge_intervals_to_cap(
        starts: Tensor,
        ends: Tensor,
        target_dt: Tensor,
        ray_indices: Tensor,
        num_rays: int,
        cap: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Merge closest adjacent intervals instead of dropping a ray's far tail."""
        if cap <= 0 or starts.numel() == 0:
            return starts, ends, target_dt, ray_indices

        packed = FrequencyAwareVolumetricSampler._pack_info_from_sorted_indices(ray_indices, num_rays)
        if bool((packed[:, 1] <= cap).all()):
            return starts, ends, target_dt, ray_indices

        # Only rays above the cap need materialization.  The previous implementation
        # copied every non-empty ray through a Python loop once any ray overflowed;
        # that caused two device-to-host scalar synchronizations per training ray.
        overflow = packed[:, 1] > cap
        overflow_rows = torch.cat(
            (torch.nonzero(overflow, as_tuple=False), packed[overflow]), dim=1
        ).cpu().tolist()
        merged_starts = []
        merged_ends = []
        merged_dt = []
        merged_rays = []
        cursor = 0
        for ray_id, first, count in overflow_rows:
            if cursor < first:
                merged_starts.append(starts[cursor:first])
                merged_ends.append(ends[cursor:first])
                merged_dt.append(target_dt[cursor:first])
                merged_rays.append(ray_indices[cursor:first])
            ray_starts = starts[first : first + count].clone()
            ray_ends = ends[first : first + count].clone()
            ray_dt = target_dt[first : first + count].clone()

            # Repeatedly removing the first smallest gap is equivalent to removing
            # ``count - cap`` gaps in one stable ascending sort.  The remaining
            # boundaries define contiguous groups whose target dt is their minimum.
            gaps = (ray_starts[1:] - ray_ends[:-1]).clamp_min(0.0)
            remove_count = count - cap
            remove_order = torch.argsort(gaps, stable=True)
            removed_boundaries = torch.zeros_like(gaps, dtype=torch.bool)
            removed_boundaries[remove_order[:remove_count]] = True
            kept_boundaries = ~removed_boundaries
            group_ids = torch.cat(
                (
                    torch.zeros((1,), dtype=torch.long, device=starts.device),
                    torch.cumsum(kept_boundaries.to(torch.long), dim=0),
                )
            )
            group_starts = torch.cat(
                (torch.ones((1,), dtype=torch.bool, device=starts.device), kept_boundaries)
            )
            group_ends = torch.cat(
                (kept_boundaries, torch.ones((1,), dtype=torch.bool, device=starts.device))
            )
            merged_ray_dt = torch.full(
                (cap,), torch.inf, dtype=ray_dt.dtype, device=ray_dt.device
            )
            merged_ray_dt.scatter_reduce_(
                0, group_ids, ray_dt, reduce="amin", include_self=True
            )
            ray_starts = ray_starts[group_starts]
            ray_ends = ray_ends[group_ends]
            ray_dt = merged_ray_dt

            merged_starts.append(ray_starts)
            merged_ends.append(ray_ends)
            merged_dt.append(ray_dt)
            merged_rays.append(torch.full_like(ray_starts, ray_id, dtype=torch.long))
            cursor = first + count

        if cursor < starts.numel():
            merged_starts.append(starts[cursor:])
            merged_ends.append(ends[cursor:])
            merged_dt.append(target_dt[cursor:])
            merged_rays.append(ray_indices[cursor:])

        return (
            torch.cat(merged_starts),
            torch.cat(merged_ends),
            torch.cat(merged_dt),
            torch.cat(merged_rays),
        )

    @staticmethod
    def _allocate_interval_counts(
        counts_raw: Tensor,
        ray_indices: Tensor,
        num_rays: int,
        cap: int,
    ) -> Tensor:
        """Allocate a per-ray cap with minimum-one plus largest remainder."""
        if cap <= 0 or counts_raw.numel() == 0:
            return counts_raw

        interval_counts = torch.zeros(num_rays, dtype=torch.long, device=counts_raw.device)
        interval_counts.scatter_add_(0, ray_indices, torch.ones_like(counts_raw))
        if bool((interval_counts > cap).any()):
            raise ValueError("Each ray must have at most cap intervals before budget allocation.")

        extras = (counts_raw - 1).clamp_min(0)
        requested_total = torch.zeros(num_rays, dtype=torch.long, device=counts_raw.device)
        requested_total.scatter_add_(0, ray_indices, counts_raw)
        extras_total = torch.zeros_like(requested_total)
        extras_total.scatter_add_(0, ray_indices, extras)
        remaining = (cap - interval_counts).clamp_min(0)

        scale = torch.ones(num_rays, dtype=torch.float64, device=counts_raw.device)
        over = requested_total > cap
        scale[over] = remaining[over].to(torch.float64) / extras_total[over].clamp_min(1).to(torch.float64)
        quotas = extras.to(torch.float64) * scale[ray_indices]
        base_extras = torch.floor(quotas).to(torch.long)
        counts = 1 + base_extras

        allocated = torch.zeros_like(requested_total)
        allocated.scatter_add_(0, ray_indices, counts)
        target_total = torch.minimum(requested_total, torch.full_like(requested_total, cap))
        leftovers = (target_total - allocated).clamp_min(0)
        if bool((leftovers > 0).any()):
            remainders = quotas - base_extras.to(torch.float64)
            by_remainder = torch.argsort(remainders, descending=True, stable=True)
            grouped_order = by_remainder[torch.argsort(ray_indices[by_remainder], stable=True)]
            grouped_rays = ray_indices[grouped_order]
            grouped_packed = FrequencyAwareVolumetricSampler._pack_info_from_sorted_indices(
                grouped_rays, num_rays
            )
            ranks = torch.arange(grouped_order.numel(), device=counts.device) - grouped_packed[grouped_rays, 0]
            take = ranks < leftovers[grouped_rays]
            counts[grouped_order[take]] += 1

        return counts

    @staticmethod
    def _subdivide_intervals(
        starts: Tensor,
        ends: Tensor,
        ray_indices: Tensor,
        counts: Tensor,
        total: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Uniformly subdivide every interval so its complete extent is covered."""
        if total is None:
            total = int(counts.sum().item())
        if total == 0:
            return ray_indices[:0], starts[:0], ends[:0]
        repeated_ray_indices = torch.repeat_interleave(ray_indices, counts, output_size=total)
        repeated_starts = torch.repeat_interleave(starts, counts, output_size=total)
        repeated_ends = torch.repeat_interleave(ends, counts, output_size=total)
        interval_lengths = (ends - starts).clamp_min(1e-8)
        subdivision_dt = interval_lengths / counts.to(interval_lengths.dtype)
        repeated_dt = torch.repeat_interleave(subdivision_dt, counts, output_size=total)
        group_starts = torch.cumsum(counts, dim=0) - counts
        local_offsets = torch.arange(total, device=starts.device) - torch.repeat_interleave(
            group_starts, counts, output_size=total
        )
        refined_starts = repeated_starts + repeated_dt * local_offsets.to(starts.dtype)
        refined_ends = torch.minimum(refined_starts + repeated_dt, repeated_ends)
        valid = refined_ends > refined_starts
        return repeated_ray_indices[valid], refined_starts[valid], refined_ends[valid]

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
            packed_info=torch.zeros((len(ray_bundle), 2), dtype=torch.long, device=device),
        )
        return ray_samples, ray_indices, stats

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
        corrected_allocator: bool = False,
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
                early_stop_eps=early_stop_eps,
            )

            if starts.numel() == 0:
                return self._empty_samples(ray_bundle)

            mid = 0.5 * (starts + ends)
            if adaptive_interval_level_mode == "midpoint":
                positions = rays_o[ray_indices] + rays_d[ray_indices] * mid[:, None]
                levels = self.freq_grid.query(positions).reshape(-1).float()
            elif adaptive_interval_level_mode == "max3":
                query_t = torch.stack((starts, mid, ends), dim=1)
                positions = rays_o[ray_indices, None, :] + rays_d[ray_indices, None, :] * query_t[..., None]
                levels = self.freq_grid.query(positions.reshape(-1, 3)).reshape(-1, 3).float().amax(dim=1)
            else:
                raise ValueError(f"Unknown adaptive_interval_level_mode={adaptive_interval_level_mode!r}.")
            if adaptive_min_frequency_level > 0:
                levels = levels.clamp_min(adaptive_min_frequency_level)
            if adaptive_max_frequency_level is not None:
                levels = levels.clamp_max(adaptive_max_frequency_level)
            n_l = self.freq_grid.level_to_freq(levels).reshape(-1)

            dt_norm = 1.0 / (2.0 * n_l.clamp_min(1e-6))
            aabb_size = self.freq_grid.aabb_size_buf.to(device=device, dtype=rays_d.dtype).clamp_min(1e-6)
            norm_speed = self._normalized_ray_speed(rays_d[ray_indices], aabb_size)
            dt = (dt_norm / norm_speed).clamp(min=adaptive_min_step_size, max=adaptive_max_step_size)

            interval_lengths = (ends - starts).clamp_min(1e-8)
            # Budget-aware per-ray dt scaling: scale up dt for rays whose naive sample
            # count would exceed max_steps_per_ray. This distributes the budget
            # proportionally along the full ray (preserving relative frequency-based
            # density ratios) instead of front-loading early high-frequency intervals
            # and leaving a gap at the far end.
            counts_raw = torch.ceil(interval_lengths / dt).to(torch.long).clamp_min(1)
            if corrected_allocator and max_steps_per_ray > 0:
                starts, ends, dt, ray_indices = self._merge_intervals_to_cap(
                    starts=starts,
                    ends=ends,
                    target_dt=dt,
                    ray_indices=ray_indices,
                    num_rays=num_rays,
                    cap=max_steps_per_ray,
                )
                interval_lengths = (ends - starts).clamp_min(1e-8)
                counts_raw = torch.ceil(interval_lengths / dt).to(torch.long).clamp_min(1)
                counts = self._allocate_interval_counts(
                    counts_raw=counts_raw,
                    ray_indices=ray_indices,
                    num_rays=num_rays,
                    cap=max_steps_per_ray,
                )
            elif max_steps_per_ray > 0:
                per_ray_total = torch.zeros(num_rays, dtype=torch.long, device=device)
                per_ray_total.scatter_add_(0, ray_indices, counts_raw)
                over_budget = (per_ray_total.float() / max_steps_per_ray).clamp_min(1.0)
                dt = dt * over_budget[ray_indices]
                counts = torch.ceil(interval_lengths / dt).to(torch.long).clamp_min(1)
            else:
                counts = counts_raw
            total = int(counts.sum().item())
            if total == 0:
                return self._empty_samples(ray_bundle)

            if corrected_allocator:
                repeated_ray_indices, refined_starts, refined_ends = self._subdivide_intervals(
                    starts, ends, ray_indices, counts, total=total
                )
            else:
                repeated_ray_indices = torch.repeat_interleave(ray_indices, counts, output_size=total)
                repeated_starts = torch.repeat_interleave(starts, counts, output_size=total)
                repeated_ends = torch.repeat_interleave(ends, counts, output_size=total)
                repeated_dt = torch.repeat_interleave(dt, counts, output_size=total)
                local_offsets = torch.arange(total, device=device) - torch.repeat_interleave(
                    torch.cumsum(counts, dim=0) - counts, counts, output_size=total
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
                return self._empty_samples(ray_bundle)

            packed = nerfacc.pack_info(repeated_ray_indices, num_rays)
            sample_counts = packed[:, 1]
            saturation_rate = (sample_counts >= max_steps_per_ray).float().mean() if max_steps_per_ray > 0 else torch.zeros((), device=device)

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
                pixel_area=ray_bundle.pixel_area[repeated_ray_indices],
            ),
            camera_indices=camera_indices,
            deltas=(refined_ends - refined_starts)[..., None],
            spacing_starts=spacing_starts[..., None],
            spacing_ends=spacing_ends[..., None],
        )
        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[repeated_ray_indices]

        stats = FrequencyAwareSamplerStats(
            num_samples=torch.tensor(refined_starts.numel(), device=device),
            mean_samples_per_ray=sample_counts.float().mean(),
            max_samples_per_ray=sample_counts.max().float(),
            saturation_rate=saturation_rate,
            packed_info=packed,
        )
        return ray_samples, repeated_ray_indices, stats
