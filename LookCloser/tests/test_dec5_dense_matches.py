from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_hair import project
from audit_dec5_calibration import fundamental, distances
from study_dec5_dense_matches import triangulate, crop_to_camera
from render_dec5_global_hull import bilinear


def cameras():
    frames = []
    for center in ([0, 0, 0], [.3, .05, 0]):
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = center
        frames.append(dict(transform_matrix=pose.tolist(), fl_x=920., fl_y=910., cx=480., cy=270., w=960, h=540))
    return frames


def test_known_calibrated_points_triangulate_and_respect_epipolar_lines():
    a, b = cameras()
    points = np.array([[0., 0., 3.], [.2, -.3, 4.], [-.5, .3, 6.]])
    uv_a, _ = project(points, a)
    uv_b, _ = project(points, b)
    recovered, angles = triangulate(uv_a, uv_b, a, b)
    np.testing.assert_allclose(recovered, points, atol=1e-11)
    assert (angles > 1).all()
    np.testing.assert_allclose(distances(uv_a, uv_b, fundamental(a, b)), 0, atol=1e-10)
    shifted = uv_b+np.array([0., 5.])
    assert (distances(uv_a, shifted, fundamental(a, b)) > 4).all()


def test_crop_coordinates_invert_actual_pixel_rotation():
    image = np.arange(35).reshape(5, 7)
    crop = [1, 2, 5, 6]
    rotated = np.rot90(image)[crop[1]:crop[3], crop[0]:crop[2]]
    yy, xx = np.indices(rotated.shape)
    uv = crop_to_camera(np.stack((xx+.5, yy+.5), -1), dict(w=7), crop)
    np.testing.assert_array_equal(image[np.floor(uv[..., 1]).astype(int), np.floor(uv[..., 0]).astype(int)], rotated)


def test_cpu_transfer_interpolation_matches_frozen_torch_sampler():
    rng = np.random.default_rng(1)
    image = rng.random((17, 21, 3), dtype=np.float32)
    uv = rng.uniform([-.5, -.5], [21.5, 17.5], (500, 2)).astype(np.float32)
    grid = torch.from_numpy(uv/np.array([21., 17.], np.float32)*2-1)[None, None]
    expected = F.grid_sample(torch.from_numpy(image).permute(2, 0, 1)[None], grid, align_corners=False)[0, :, 0].T.numpy()
    np.testing.assert_allclose(bilinear(image, uv), expected, atol=2e-6)
