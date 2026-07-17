"""Exact trajectory tests for the opt-in consolidated FAS H2D path."""

import numpy as np
import pytest
import torch

from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSampler, LookCloserPixelSamplerConfig


def _sampler(*, consolidate_h2d: bool, group_size: int = 1) -> LookCloserPixelSampler:
    config = LookCloserPixelSamplerConfig(
        num_rays_per_batch=257,
        num_levels=16,
        patch_size=8,
        stride=5,
        fas_strength=0.75,
        fas_patch_group_size=group_size,
        fas_consolidate_h2d=consolidate_h2d,
    )
    sampler = LookCloserPixelSampler(config)
    sampler.is_initialized = True
    sampler.current_fas_strength = 0.75
    # At least twelve FAS samples per level: empty levels therefore exercise the
    # fallback path instead of disappearing through zero probability.
    sampler.probs = np.full(16, 1.0 / 16.0, dtype=np.float64)
    sampler.buckets = {}
    for level in range(16):
        if level in {0, 7, 15}:
            sampler.buckets[level] = torch.empty((0, 3), dtype=torch.int32)
        else:
            sampler.buckets[level] = torch.tensor(
                [
                    [level % 4, level % 5, (level + 1) % 6],
                    [(level + 1) % 4, (level + 2) % 5, (level + 3) % 6],
                    [(level + 2) % 4, (level + 4) % 5, (level + 5) % 6],
                ],
                dtype=torch.int32,
            )
    sampler.image_shapes = {0: (31, 37), 1: (33, 39), 2: (35, 41), 3: (37, 43)}
    return sampler


def _run_with_states(sampler: LookCloserPixelSampler, device: str):
    output = sampler.sample_method(257, 4, 37, 43, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return output, cpu_state, cuda_states


def test_consolidated_h2d_is_opt_in_and_cpu_keeps_exact_legacy_path() -> None:
    assert LookCloserPixelSamplerConfig().fas_consolidate_h2d is False

    legacy = _sampler(consolidate_h2d=False, group_size=3)
    consolidated = _sampler(consolidate_h2d=True, group_size=3)
    torch.manual_seed(8147)
    initial_cpu_state = torch.get_rng_state().clone()
    expected, expected_cpu_state, _ = _run_with_states(legacy, "cpu")

    torch.set_rng_state(initial_cpu_state)
    actual, actual_cpu_state, _ = _run_with_states(consolidated, "cpu")

    assert torch.equal(actual, expected)
    assert torch.equal(actual_cpu_state, expected_cpu_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("group_size", [1, 3])
def test_consolidated_h2d_cuda_matches_legacy_indices_and_rng_states(group_size: int) -> None:
    # Warm CUDA initialization before snapshotting state so context setup is not
    # part of either compared trajectory.
    torch.empty(1, device="cuda")
    torch.manual_seed(4919)
    torch.cuda.manual_seed_all(7621)
    torch.cuda.synchronize()
    initial_cpu_state = torch.get_rng_state().clone()
    initial_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

    legacy = _sampler(consolidate_h2d=False, group_size=group_size)
    expected, expected_cpu_state, expected_cuda_states = _run_with_states(legacy, "cuda")

    torch.set_rng_state(initial_cpu_state)
    torch.cuda.set_rng_state_all(initial_cuda_states)
    consolidated = _sampler(consolidate_h2d=True, group_size=group_size)
    actual, actual_cpu_state, actual_cuda_states = _run_with_states(consolidated, "cuda")

    assert torch.equal(actual, expected)
    assert torch.equal(actual_cpu_state, expected_cpu_state)
    assert len(actual_cuda_states) == len(expected_cuda_states)
    assert all(
        torch.equal(actual_state, expected_state)
        for actual_state, expected_state in zip(actual_cuda_states, expected_cuda_states)
    )
