"""Correctness properties for LookCloser's optional corrected ARM allocator."""

import math

import pytest
import torch

from nerfstudio.model_components.lookcloser_samplers import FrequencyAwareVolumetricSampler


def _slow_allocate(raw: list[int], rays: list[int], num_rays: int, cap: int) -> list[int]:
    result = list(raw)
    for ray in range(num_rays):
        ids = [idx for idx, value in enumerate(rays) if value == ray]
        if not ids or sum(raw[idx] for idx in ids) <= cap:
            continue
        if len(ids) > cap:
            raise ValueError("slow allocator expects intervals to be merged first")
        extras = [raw[idx] - 1 for idx in ids]
        remaining = cap - len(ids)
        total_extras = sum(extras)
        quotas = [extra * remaining / total_extras for extra in extras]
        allocated = [1 + math.floor(quota) for quota in quotas]
        leftover = cap - sum(allocated)
        order = sorted(range(len(ids)), key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i))
        for local_idx in order[:leftover]:
            allocated[local_idx] += 1
        for idx, value in zip(ids, allocated):
            result[idx] = value
    return result


def _slow_merge(starts, ends, dt, rays, num_rays, cap):
    merged_starts, merged_ends, merged_dt, merged_rays = [], [], [], []
    for ray in range(num_rays):
        ids = torch.nonzero(rays == ray, as_tuple=False).flatten()
        if not len(ids):
            continue
        ray_starts = starts[ids].clone()
        ray_ends = ends[ids].clone()
        ray_dt = dt[ids].clone()
        while ray_starts.numel() > cap:
            gaps = (ray_starts[1:] - ray_ends[:-1]).clamp_min(0.0)
            merge_at = int(torch.argmin(gaps).item())
            ray_ends[merge_at] = ray_ends[merge_at + 1]
            ray_dt[merge_at] = torch.minimum(ray_dt[merge_at], ray_dt[merge_at + 1])
            keep = torch.ones(ray_starts.shape[0], dtype=torch.bool, device=ray_starts.device)
            keep[merge_at + 1] = False
            ray_starts, ray_ends, ray_dt = ray_starts[keep], ray_ends[keep], ray_dt[keep]
        merged_starts.append(ray_starts)
        merged_ends.append(ray_ends)
        merged_dt.append(ray_dt)
        merged_rays.append(torch.full_like(ray_starts, ray, dtype=torch.long))
    return tuple(map(torch.cat, (merged_starts, merged_ends, merged_dt, merged_rays)))


def test_minimum_one_largest_remainder_regression() -> None:
    raw = torch.tensor([2, 2], dtype=torch.long)
    rays = torch.tensor([0, 0], dtype=torch.long)
    counts = FrequencyAwareVolumetricSampler._allocate_interval_counts(raw, rays, 1, 3)
    assert counts.tolist() == [2, 1]


@pytest.mark.parametrize("seed", range(12))
def test_allocator_matches_slow_reference(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    per_ray_intervals = torch.randint(0, 8, (9,), generator=generator)
    rays = torch.repeat_interleave(torch.arange(9), per_ray_intervals)
    raw = torch.randint(1, 20, (len(rays),), generator=generator)
    cap = 9
    keep = (
        torch.cat([torch.nonzero(rays == ray, as_tuple=False).flatten()[:cap] for ray in range(9)])
        if len(rays)
        else torch.empty(0, dtype=torch.long)
    )
    rays = rays[keep]
    raw = raw[keep]
    actual = FrequencyAwareVolumetricSampler._allocate_interval_counts(raw, rays, 9, cap)
    assert actual.tolist() == _slow_allocate(raw.tolist(), rays.tolist(), 9, cap)
    totals = torch.zeros(9, dtype=torch.long)
    totals.scatter_add_(0, rays, actual)
    assert bool((totals <= cap).all())
    assert bool((actual >= 1).all())


def test_empty_intervals() -> None:
    counts = FrequencyAwareVolumetricSampler._allocate_interval_counts(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), 4, 3
    )
    assert counts.numel() == 0


def test_intervals_over_cap_are_merged_without_losing_far_tail() -> None:
    starts = torch.tensor([0.0, 2.0, 4.0, 6.0])
    ends = starts + 1.0
    dt = torch.tensor([0.5, 0.25, 0.75, 1.0])
    rays = torch.zeros(4, dtype=torch.long)
    merged_starts, merged_ends, merged_dt, merged_rays = (
        FrequencyAwareVolumetricSampler._merge_intervals_to_cap(starts, ends, dt, rays, 1, 2)
    )
    assert merged_starts.tolist() == [0.0, 6.0]
    assert merged_ends.tolist() == [5.0, 7.0]
    assert merged_dt.tolist() == [0.25, 1.0]
    assert merged_rays.tolist() == [0, 0]
    assert merged_starts[0] == starts[0]
    assert merged_ends[-1] == ends[-1]


def test_merge_fast_path_preserves_unaffected_rays_and_tie_order() -> None:
    starts = torch.tensor([0.0, 2.0, 0.0, 2.0, 4.0, 6.0, 0.0, 3.0])
    ends = starts + 1.0
    dt = torch.tensor([0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.8, 0.9])
    rays = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
    merged = FrequencyAwareVolumetricSampler._merge_intervals_to_cap(
        starts, ends, dt, rays, num_rays=4, cap=2
    )
    merged_starts, merged_ends, merged_dt, merged_rays = merged
    assert merged_rays.tolist() == [0, 0, 1, 1, 2, 2]
    assert merged_starts.tolist() == [0.0, 2.0, 0.0, 6.0, 0.0, 3.0]
    assert merged_ends.tolist() == [1.0, 3.0, 5.0, 7.0, 1.0, 4.0]
    assert torch.equal(merged_dt, torch.tensor([0.5, 0.6, 0.3, 0.2, 0.8, 0.9]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("seed", range(8))
def test_vectorized_gap_merge_matches_iterative_reference(seed: int, device: str) -> None:
    generator = torch.Generator().manual_seed(seed)
    interval_counts = torch.randint(1, 14, (7,), generator=generator)
    rays = torch.repeat_interleave(torch.arange(7), interval_counts)
    lengths = torch.randint(1, 4, (rays.numel(),), generator=generator).float()
    gaps = torch.randint(0, 4, (rays.numel(),), generator=generator).float()
    starts = torch.empty_like(lengths)
    cursor = 0
    for count in interval_counts.tolist():
        local_lengths = lengths[cursor : cursor + count]
        local_gaps = gaps[cursor : cursor + count]
        local_starts = torch.cumsum(local_lengths + local_gaps, dim=0) - local_lengths - local_gaps
        starts[cursor : cursor + count] = local_starts
        cursor += count
    ends = starts + lengths
    dt = torch.rand(rays.numel(), generator=generator).clamp_min(0.01)
    starts, ends, dt, rays = (tensor.to(device) for tensor in (starts, ends, dt, rays))
    cap = 5
    actual = FrequencyAwareVolumetricSampler._merge_intervals_to_cap(
        starts, ends, dt, rays, num_rays=7, cap=cap
    )
    expected = _slow_merge(starts, ends, dt, rays, num_rays=7, cap=cap)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_subdivision_has_no_internal_or_tail_gaps() -> None:
    starts = torch.tensor([0.0, 3.0])
    ends = torch.tensor([2.0, 7.0])
    rays = torch.tensor([0, 0], dtype=torch.long)
    counts = torch.tensor([3, 2], dtype=torch.long)
    _, refined_starts, refined_ends = FrequencyAwareVolumetricSampler._subdivide_intervals(
        starts, ends, rays, counts
    )
    assert torch.allclose(refined_starts[[0, 3]], starts)
    assert torch.allclose(refined_ends[[2, 4]], ends)
    assert torch.allclose(refined_ends[[0, 1, 3]], refined_starts[[1, 2, 4]])


def test_anisotropic_aabb_normalized_speed() -> None:
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    aabb_size = torch.tensor([2.0, 1.0, 4.0])
    speed = FrequencyAwareVolumetricSampler._normalized_ray_speed(directions, aabb_size)
    assert torch.allclose(speed, torch.tensor([0.5, 1.0]))
