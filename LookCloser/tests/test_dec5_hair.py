import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_hair import (VALIDATION, TEST, camera_split, cv_pose, unproject, project,
                      orientations, orientation_cost, composite_layers, confidence_override,
                      volume_depths, ray_to_z, resize_pinhole, mvs_camera_model, diverse_witnesses)


def camera():
    return dict(transform_matrix=np.diag([1., -1., -1., 1.]).tolist(),
                cx=40., cy=30., fl_x=100., fl_y=100., w=80, h=60)


def test_union_split_has_no_historical_leakage():
    rows = [{'physical_camera': x} for x in VALIDATION+TEST+('train',)]
    split = camera_split(rows)
    assert split['train'] == 'train'
    assert all(split[x] != 'train' for x in VALIDATION+TEST)
    with pytest.raises(ValueError):
        camera_split(rows+rows[:1])


def test_camera_z_roundtrip_and_ray_distance_are_distinct():
    f = camera()
    uv = np.array([[40., 30.], [140., 30.]])
    xyz = unproject(uv, np.array([2., 2.]), f)
    np.testing.assert_allclose(xyz, [[0, 0, 2], [2, 0, 2]])
    projected, z = project(xyz, f)
    np.testing.assert_allclose(projected, uv)
    np.testing.assert_allclose(z, [2, 2])
    np.testing.assert_allclose(ray_to_z([2, 2*np.sqrt(2)], uv, f), [2, 2])
    with pytest.raises(ValueError):
        unproject(uv, z, dict(f, k1=.1))


def test_colmap_sparse_camera_must_follow_actual_mvs_raster_size():
    f = camera()
    small = resize_pinhole(f, 40, 30)
    uv = np.array([[20.5, 14.5], [30.5, 15.5]])
    np.testing.assert_allclose(unproject(uv, [2., 3.], small),
                               unproject(uv*2, [2., 3.], f))
    with pytest.raises(ValueError):
        resize_pinhole(f, 40, 29)


def test_zero_distortion_must_not_trigger_colmap_opencv_resampling():
    assert mvs_camera_model([dict(camera(), camera_model='OPENCV')]) == 'PINHOLE'
    assert mvs_camera_model([dict(camera(), k1=.01)]) == 'OPENCV'
    with pytest.raises(ValueError):
        mvs_camera_model([dict(camera(), camera_model='OPENCV_FISHEYE')])


def test_orientation_witnesses_are_not_a_cluster_of_nearly_identical_views():
    angles = np.deg2rad(np.linspace(-30,30,61))
    centers = np.stack((np.sin(angles),np.zeros_like(angles),np.cos(angles)),-1)
    indices = diverse_witnesses(centers,30,np.zeros(3))
    assert len(indices) == 8 and 30 not in indices
    assert np.rad2deg(angles[indices].max()-angles[indices].min()) >= 40


def test_strand_tangent_not_gradient_and_rotation_modulo_pi():
    pytest.importorskip('cv2')
    yy, xx = np.mgrid[:64, :64]
    rgb = ((np.sin(xx*.5)+1)*.5).astype(np.float32)
    tangent, q = orientations(rgb)
    assert np.median(tangent[8:-8, 8:-8, 0]) < -.95  # vertical strands
    assert np.median(q[8:-8, 8:-8]) > .9
    flat, flat_q = orientations(np.zeros((32, 32), np.float32))
    assert np.max(flat_q) == 0


def test_orientation_requires_three_views_and_normalizes_weight():
    normals = np.array([[[1., 0, 0]], [[0., 1, 0]], [[1., 1, 0]]])/np.sqrt(2)
    a, count = orientation_cost(normals, np.ones((3, 1)))
    b, _ = orientation_cost(normals, np.ones((3, 1))*10)
    np.testing.assert_allclose(a, 0, atol=1e-12)
    np.testing.assert_allclose(a, b)
    assert count[0] == 3
    c, _ = orientation_cost(normals[:2], np.ones((2, 1)))
    assert np.isinf(c[0])
    parallel = np.broadcast_to([1., 0, 0], (3, 1, 3))
    weak, _ = orientation_cost(parallel, np.ones((3, 1)))
    assert np.isinf(weak[0])  # Multiple identical constraints do not resolve direction.


def test_two_layer_premultiplication_and_roi_preservation():
    base = np.ones((2, 2, 3), np.float32)*.7
    alpha = np.array([[.5, 1], [0, .25]], np.float32)
    foreground = np.zeros_like(base)
    foreground[..., 0] = alpha
    background = np.zeros_like(base)
    background[..., 2] = 1
    roi = np.array([[True, False], [True, True]])
    front = np.ones((2, 2)); back = np.ones((2, 2))*2
    back[1, 1] = .5
    out, valid = composite_layers(base, foreground, alpha, background, front, back, True, roi)
    np.testing.assert_allclose(out[0, 0], [.5, 0, .5])
    np.testing.assert_array_equal(out[~valid], base[~valid])
    np.testing.assert_allclose(out[1, 0], [0, 0, 1])


def test_depth_override_unknown_is_not_background():
    base = np.array([1., 2., 3.])
    out, take = confidence_override(base, [0., 4., np.nan], [True]*3, [True]*3)
    np.testing.assert_array_equal(out, [1., 4., 3.])


def test_volume_mean_can_lie_between_two_surfaces():
    result = volume_depths(np.array([[1., 2., 3., 4., 5.]]),
                           np.array([[0., .4, 0., .4, 0.]]))
    np.testing.assert_allclose(result['expected'], 3)
    np.testing.assert_allclose(result['front'], 2)
    np.testing.assert_allclose(result['peak1'], 2)
    np.testing.assert_allclose(result['peak2'], 4)
    np.testing.assert_allclose(result['unconditioned'], 2.4)
    assert volume_depths(np.arange(5)[None], np.zeros((1, 5)))['front'][0] == 0


def test_subpixel_volume_peak_recovers_off_center_gaussian_and_keeps_boundary():
    t = np.arange(5, dtype=float)[None]
    weights = np.exp(-.5*((t-2.3)/.6)**2)
    weights /= weights.sum()
    result = volume_depths(t, weights)
    np.testing.assert_allclose(result['peak1'], 2.)
    np.testing.assert_allclose(result['peak1_subpixel'], 2.3, atol=1e-10)
    edge = volume_depths(t, np.array([[1., 0., 0., 0., 0.]]))
    np.testing.assert_array_equal(edge['peak1_subpixel'], edge['peak1'])
