import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from temporal_rigid_patch import transfer_points, project_native, warp_rigid, fit_rigid


def test_flow_composition_keeps_out_of_frame_paths_invalid():
    from temporal_rigid_patch import compose_flow_fields
    a = np.zeros((8, 10, 2), np.float32); a[..., 0] = 2
    b = -a
    flow, valid = compose_flow_fields([a, b])
    np.testing.assert_allclose(flow[:, :8], 0)
    assert valid[:, :8].all() and not valid[:, 8:].any()


def test_point_gauge_transfer_inverts_scale_and_translation():
    a = np.eye(4); a[:3, 3] = [.3, -.2, .1]
    b = np.eye(4); b[:3, :3] = Rotation.from_euler('z', .3).as_matrix(); b[:3, 3] = [-.1, .3, .2]
    source = dict(dataparser_scale=.2, dataparser_transform=a[:3].tolist())
    target = dict(dataparser_scale=.7, dataparser_transform=b[:3].tolist())
    raw = np.array([[1., 2., 3.], [2., 1., 1.]])
    actual = transfer_points(.2 * (raw + a[:3, 3]), source, target)
    np.testing.assert_allclose(actual, .7 * (raw @ b[:3, :3].T + b[:3, 3]), atol=1e-12)


def test_rigid_fit_transfers_to_camera_not_used_in_fit():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(50, 3)) * .1 + [0, 0, -2.]
    cameras = []
    for x in [-.3, .3, .6]:
        pose = np.eye(4); pose[0, 3] = x
        cameras.append(dict(transform_matrix=pose.tolist(), fl_x=700., fl_y=700., cx=400., cy=300.))
    truth = np.array([.05, -.08, .02, .1, -.07, .03])
    moved = warp_rigid(points, truth, points.mean(0))
    pi = np.tile(np.arange(len(points)), 3); ci = np.repeat(np.arange(3), len(points))
    observations = np.concatenate([project_native(moved, c)[0] for c in cameras])
    fitted, center, errors = fit_rigid(points, cameras, pi, ci, observations, np.zeros(6), [0, 1])
    np.testing.assert_allclose(warp_rigid(points, fitted, center), moved, atol=1e-8)
    assert errors[ci == 2].max() < 1e-6
