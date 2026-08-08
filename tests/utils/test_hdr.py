"""Tests for scene-linear HDR and ST 2084 utilities."""

import math

import cv2
import numpy as np
import torch

from nerfstudio.utils.hdr import calibrate_exr_paths, pq_decode_nits, pq_encode_nits, scene_linear_to_pq


def test_pq_reference_points_and_round_trip():
    nits = torch.tensor([0.0, 0.005, 100.0, 1000.0, 10000.0])
    code = pq_encode_nits(nits)
    assert float(code[0]) < 1e-5
    assert math.isclose(float(code[2]), 0.5080784, rel_tol=0, abs_tol=2e-6)
    torch.testing.assert_close(pq_decode_nits(code), nits, atol=5e-2, rtol=5e-5)


def test_scene_to_pq_has_finite_black_gradient():
    rgb = torch.tensor([[-1.0, 0.0, 1e-6, 1.0]], requires_grad=True)
    encoded = scene_linear_to_pq(rgb, nits_per_scene_unit=100.0)
    encoded.sum().backward()
    assert torch.isfinite(encoded).all()
    assert torch.isfinite(rgb.grad).all()
    assert float(rgb.grad[0, 2]) > 0


def test_calibration_is_train_wide_and_preserves_hdr_statistics(tmp_path):
    paths = []
    for index, value in enumerate((0.01, 0.02)):
        rgb = np.full((16, 16, 3), value, dtype=np.float32)
        rgb[0, 0] = (-0.1, 2.0, 4.0)
        path = tmp_path / f"{index}.exr"
        assert cv2.imwrite(str(path), rgb[..., ::-1])
        paths.append(path)
    calibration = calibrate_exr_paths(paths, sample_stride=1)
    assert calibration.linear_scale > 0.01
    assert calibration.nits_per_scene_unit > 0
    assert calibration.negative_channel_fraction > 0
    assert calibration.sampled_pixel_count == 512
