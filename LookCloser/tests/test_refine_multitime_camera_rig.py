from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
pytest.importorskip('pycolmap')
from refine_multitime_camera_rig import align_similarity, make_reconstruction, reprojection_profile, optimize


def test_similarity_restores_camera_gauge():
    rng = np.random.default_rng(21)
    source = rng.normal(size=(12, 3))
    r = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    target = 2 * source @ r.T + [1, 2, 3]
    s, rotation, t = align_similarity(source, target)
    np.testing.assert_allclose(s * source @ rotation.T + t, target, atol=1e-12)


def scene():
    cameras = []
    for i, x in enumerate([-.3, 0, .3, .6]):
        w2c = np.eye(4)
        w2c[0, 3] = -x
        pose = np.linalg.inv(w2c) @ np.diag([1., -1., -1., 1.])
        cameras.append(dict(physical_camera=str(i), transform_matrix=pose.tolist(), w=640, h=480, fl_x=800., fl_y=800., cx=320., cy=240.))
    rng = np.random.default_rng(22)
    tracks = []
    for time in ['one', 'two']:
        for xyz in rng.uniform([-.3, -.3, 3], [.3, .3, 6], (30, 3)):
            observations = []
            for i, x in enumerate([-.3, 0, .3, .6]):
                q = xyz - [x, 0, 0]
                xy = q[:2] / q[2] * 800 + [319.5, 239.5]
                observations.append(dict(camera=i, xy=xy.tolist()))
            tracks.append(dict(frame_id=time, xyz=xyz.tolist(), observations=observations))
    return cameras, tracks


def test_temporal_points_share_exactly_one_camera_parameter_block():
    cameras, tracks = scene()
    reconstruction = make_reconstruction(cameras, tracks)
    assert reconstruction.num_cameras() == reconstruction.num_reg_images() == 4
    assert reconstruction.num_points3D() == 60
    assert len(reprojection_profile(reconstruction, tracks)) == 8
    assert max(r['p90'] for r in reprojection_profile(reconstruction, tracks)) < 1e-10


def test_exact_scene_bundle_adjustment_is_identity():
    cameras, tracks = scene()
    reconstruction = make_reconstruction(cameras, tracks)
    optimize(reconstruction, 'poses', iterations=10, threads=1)
    assert max(r['p90'] for r in reprojection_profile(reconstruction, tracks)) < 1e-8


def test_regularized_prior_attaches_to_real_pose_and_keeps_exact_scene():
    pytest.importorskip('pyceres')
    cameras, tracks = scene()
    reconstruction = make_reconstruction(cameras, tracks)
    optimize(reconstruction, 'pinhole', iterations=10, threads=1, regularized=True)
    assert max(r['p90'] for r in reprojection_profile(reconstruction, tracks)) < 1e-7
    for i in range(1, 5):
        np.testing.assert_allclose(reconstruction.cameras[i].params, [800, 800, 320, 240], atol=1e-6)


def test_shared_regularized_fit_repairs_small_camera_error_within_bounds():
    pytest.importorskip('pyceres')
    cameras, tracks = scene()
    cameras[-1]['fl_x'] *= 1.03
    cameras[-1]['transform_matrix'][0][3] += .025
    reconstruction = make_reconstruction(cameras, tracks)
    before = max(r['p90'] for r in reprojection_profile(reconstruction, tracks))
    optimize(reconstruction, 'focal', threads=1, regularized=True)
    after = max(r['p90'] for r in reprojection_profile(reconstruction, tracks))
    assert after < before / 10
    for index, frame in enumerate(cameras, 1):
        focal = reconstruction.cameras[index].params[:2]
        initial = np.array([frame['fl_x'], frame['fl_y']])
        assert np.all(focal >= initial * .95 - 1e-7)
        assert np.all(focal <= initial * 1.05 + 1e-7)
