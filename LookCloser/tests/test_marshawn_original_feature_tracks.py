import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from marshawn_original_feature_tracks import recover_track, triangulate_detections


def source_fixture():
    rows, features = {}, {}
    for i, x in enumerate([-1., -.3, .4, 1.]):
        name = chr(65+i)
        pose = np.eye(4); pose[:3, 3] = [x, 0., 3.]
        rows[name] = dict(transform_matrix=pose.tolist(), fl_x=100., fl_y=100., cx=50., cy=50.)
        uv = np.array([50.-100.*x/3., 50.+[.1, -.15, .12, -.07][i]], np.float32)
        descriptors = np.zeros((3, 8), np.float32)
        descriptors[0, i] = 100.  # Same UV, different SIFT orientation; not original correspondence.
        descriptors[1] = 1.      # Correct shared descriptor.
        descriptors[2] = 50.
        features[name] = dict(uv=np.array([uv, uv, [300., 300.]], np.float32), des=descriptors)
    cameras = list(rows)
    pairs = {(a, b) for i, a in enumerate(cameras) for b in cameras[i+1:]}
    xyz = triangulate_detections([(c, 1) for c in cameras], rows, features)
    return xyz, cameras, rows, features, pairs


def test_duplicate_orientation_recovered_by_descriptor_graph_with_real_residuals():
    xyz, cameras, rows, features, pairs = source_fixture()
    result = recover_track(xyz, cameras, rows, features, pairs)
    assert [o['keypoint_index'] for o in result['observations']] == [1]*4
    assert result['original_xyz_max_abs'] < 1e-12
    assert result['fit_error_max_hd_px'] > .01
    assert len(result['leave_one_source_view_out']) == 4
    assert max(o['heldout_raw_feature_error_hd_px'] for o in result['leave_one_source_view_out']) > .01


def test_projected_uv_alone_cannot_certify_a_track():
    xyz, cameras, rows, features, _ = source_fixture()
    with pytest.raises(ValueError, match='Descriptor graph'):
        recover_track(xyz, cameras, rows, features, set())


def test_wrong_original_xyz_or_camera_membership_rejected():
    xyz, cameras, rows, features, pairs = source_fixture()
    with pytest.raises(ValueError, match='Descriptor graph'):
        recover_track(xyz+np.array([.001, 0., 0.]), cameras, rows, features, pairs)
    with pytest.raises(ValueError, match='distinct'):
        recover_track(xyz, cameras[:2], rows, features, pairs)
