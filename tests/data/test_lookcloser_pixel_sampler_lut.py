"""Shape-LUT parity for mixed/legacy frequency metadata."""

import types

import numpy as np
import pytest
import torch

from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSampler, LookCloserPixelSamplerConfig


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_image_shape_lut_matches_dictionary_lookup_for_mixed_and_invalid_ids(device: str) -> None:
    sampler = LookCloserPixelSampler(LookCloserPixelSamplerConfig(num_rays_per_batch=8))
    sampler.image_shapes = {0: (120, 200), 2: (80, 160), 5: (60, 90)}
    indices = torch.tensor([0, 1, 2, 5, -1, 8], dtype=torch.long, device=device)
    heights, widths = sampler._image_shapes_for_indices(
        indices, num_images=6, image_height=100, image_width=180
    )
    assert heights.cpu().tolist() == [120, 100, 80, 60, 100, 100]
    assert widths.cpu().tolist() == [200, 180, 160, 90, 180, 180]


def test_image_shape_lut_cache_is_stable_for_repeated_gathers() -> None:
    sampler = LookCloserPixelSampler(LookCloserPixelSamplerConfig(num_rays_per_batch=4))
    sampler.image_shapes = {1: (72, 96)}
    indices = torch.tensor([1, 0, 1, 0])
    first = sampler._image_shapes_for_indices(indices, 2, 64, 64)
    second = sampler._image_shapes_for_indices(indices, 2, 64, 64)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert len(sampler._image_shape_lut_cache) == 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_full_fas_sample_indices_and_rng_match_dictionary_lookup(device: str) -> None:
    config = LookCloserPixelSamplerConfig(num_rays_per_batch=32, num_levels=2, patch_size=4, stride=4)
    sampler = LookCloserPixelSampler(config)
    sampler.is_initialized = True
    sampler.current_fas_strength = 1.0
    sampler.probs = np.array([0.4, 0.6])
    sampler.buckets = {
        0: torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
        1: torch.tensor([[0, 2, 2], [1, 3, 3]], dtype=torch.int32),
    }
    sampler.image_shapes = {0: (14, 15), 1: (16, 17)}

    def dictionary_lookup(self, img_idx, num_images, image_height, image_width):
        heights = torch.tensor(
            [self.image_shapes.get(int(index.item()), (image_height, image_width))[0] for index in img_idx],
            device=img_idx.device,
            dtype=torch.long,
        )
        widths = torch.tensor(
            [self.image_shapes.get(int(index.item()), (image_height, image_width))[1] for index in img_idx],
            device=img_idx.device,
            dtype=torch.long,
        )
        return heights, widths

    fast_lookup = sampler._image_shapes_for_indices
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    sampler._image_shapes_for_indices = types.MethodType(dictionary_lookup, sampler)
    reference = sampler.sample_method(32, 2, 16, 17, device=device)
    reference_cpu_state = torch.get_rng_state()
    reference_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    sampler._image_shapes_for_indices = fast_lookup
    actual = sampler.sample_method(32, 2, 16, 17, device=device)
    assert torch.equal(actual, reference)
    assert torch.equal(torch.get_rng_state(), reference_cpu_state)
    if reference_cuda_state is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), reference_cuda_state)
        )


def test_fas_training_patches_are_spatially_contiguous_and_row_major() -> None:
    config = LookCloserPixelSamplerConfig(
        num_rays_per_batch=18,
        num_levels=2,
        patch_size=4,
        stride=4,
        training_patch_size=3,
    )
    sampler = LookCloserPixelSampler(config)
    sampler.is_initialized = True
    sampler.current_fas_strength = 1.0
    sampler.probs = np.array([0.5, 0.5])
    sampler.buckets = {
        0: torch.tensor([[0, 1, 1]], dtype=torch.int32),
        1: torch.tensor([[1, 2, 2]], dtype=torch.int32),
    }
    sampler.image_shapes = {0: (16, 17), 1: (15, 18)}

    indices = sampler.sample_method(18, 2, 16, 18)
    patches = indices.reshape(2, 3, 3, 3)
    expected_y = torch.arange(3).view(3, 1).expand(3, 3)
    expected_x = torch.arange(3).view(1, 3).expand(3, 3)
    for patch in patches:
        assert bool((patch[..., 0] == patch[0, 0, 0]).all())
        assert torch.equal(patch[..., 1] - patch[0, 0, 1], expected_y)
        assert torch.equal(patch[..., 2] - patch[0, 0, 2], expected_x)
        image_index = int(patch[0, 0, 0])
        height, width = sampler.image_shapes[image_index]
        assert int(patch[..., 1].max()) < height
        assert int(patch[..., 2].max()) < width


def test_nondivisible_eval_batch_preserves_unstructured_sampling() -> None:
    config = LookCloserPixelSamplerConfig(
        num_rays_per_batch=10,
        num_levels=1,
        training_patch_size=3,
    )
    sampler = LookCloserPixelSampler(config)
    sampler.is_initialized = True
    sampler.current_fas_strength = 1.0
    sampler.probs = np.array([1.0])
    sampler.buckets = {0: torch.tensor([[0, 0, 0]], dtype=torch.int32)}
    indices = sampler.sample_method(10, 1, 16, 16)
    assert indices.shape == (10, 3)
