"""Tests for the opt-in static-camera RayGenerator cache."""

import pytest
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components.ray_generators import RayGenerator


def _make_perspective_cameras() -> Cameras:
    num_cameras = 3
    camera_to_worlds = torch.eye(4).expand(num_cameras, -1, -1)[:, :3].clone()
    camera_to_worlds[1, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    camera_to_worlds[2, :3, :3] = torch.tensor(
        [
            [0.9523336, 0.0, 0.3050586],
            [0.0, 1.0, 0.0],
            [-0.3050586, 0.0, 0.9523336],
        ]
    )
    distortion_params = torch.tensor(
        [
            [0.01, -0.002, 0.0003, -0.0004, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.004, 0.001, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    return Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=torch.tensor([8.0, 9.0, 10.0]),
        fy=torch.tensor([7.0, 8.0, 9.0]),
        cx=torch.tensor([3.0, 3.2, 2.8]),
        cy=torch.tensor([2.0, 2.1, 1.9]),
        width=7,
        height=5,
        distortion_params=distortion_params,
        camera_type=CameraType.PERSPECTIVE,
        times=torch.tensor([0.1, 0.2, 0.3]),
        metadata={"appearance": torch.arange(6, dtype=torch.float32).reshape(3, 2)},
    )


def _assert_ray_bundles_equal(expected, actual) -> None:
    for field in ("origins", "directions", "pixel_area", "camera_indices", "times"):
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        assert (expected_value is None) == (actual_value is None)
        if expected_value is not None:
            assert torch.equal(expected_value, actual_value), field

    assert expected.metadata.keys() == actual.metadata.keys()
    for key in expected.metadata:
        assert torch.equal(expected.metadata[key], actual.metadata[key]), key


def test_static_ray_cache_is_byte_exact_rng_free_and_nonpersistent() -> None:
    cameras = _make_perspective_cameras()
    rng_state = torch.random.get_rng_state().clone()

    cached_generator = RayGenerator(cameras, cache_rays=True, cache_chunk_size=11)

    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert cached_generator.cache_build_seconds >= 0.0
    # Pixel fields: (directions 3 + area 1 + norm 1) * fp32, plus
    # per-camera origins, times, and two-value appearance metadata.
    assert cached_generator.cache_num_bytes == 3 * 5 * 7 * 5 * 4 + 3 * (3 + 1 + 2) * 4
    assert cached_generator.state_dict() == {}

    indices = torch.tensor(
        [
            [2, 4, 6],
            [0, 0, 0],
            [1, 2, 3],
            [2, 1, 5],
            [1, 2, 3],  # duplicate lookup
            [0, 4, 1],
            [-1, -1, -1],  # preserve normal PyTorch negative-index semantics
        ],
        dtype=torch.long,
    )
    expected = RayGenerator(cameras)(indices)
    actual = cached_generator(indices)
    _assert_ray_bundles_equal(expected, actual)


def test_ray_cache_is_disabled_by_default() -> None:
    cameras = _make_perspective_cameras()
    generator = RayGenerator(cameras)

    assert generator.cache_rays is False
    assert generator.cache_num_bytes == 0
    assert generator.cache_build_seconds == 0.0
    assert generator._cached_directions is None

    indices = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
    expected = cameras.generate_rays(
        camera_indices=indices[:, :1],
        coords=generator.image_coords[indices[:, 1], indices[:, 2]],
    )
    _assert_ray_bundles_equal(expected, generator(indices))


def test_ray_cache_fails_closed_for_unsupported_cameras_and_metadata() -> None:
    cameras = _make_perspective_cameras()
    cameras.camera_type[1] = CameraType.FISHEYE.value
    with pytest.raises(ValueError, match="perspective"):
        RayGenerator(cameras, cache_rays=True)

    cameras = _make_perspective_cameras()
    cameras.width = cameras.width.clone()
    cameras.width[1] -= 1
    with pytest.raises(ValueError, match="jagged"):
        RayGenerator(cameras, cache_rays=True)

    cameras = _make_perspective_cameras()
    cameras.metadata = {"nested": {"appearance": cameras.metadata["appearance"]}}
    with pytest.raises(TypeError, match="top-level"):
        RayGenerator(cameras, cache_rays=True)


def test_ray_cache_rejects_multidimensional_camera_batches() -> None:
    camera_to_worlds = torch.eye(4).expand(2, 2, -1, -1)[..., :3, :].clone()
    cameras = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=8.0,
        fy=8.0,
        cx=3.0,
        cy=2.0,
        width=7,
        height=5,
        camera_type=CameraType.PERSPECTIVE,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        RayGenerator(cameras, cache_rays=True)
