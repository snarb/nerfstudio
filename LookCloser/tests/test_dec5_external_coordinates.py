"""Check coordinate bridges against external library behavior."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_hair import integer_center_pinhole, unproject
from review_dec5_hair_path import remap, pair_metrics


def test_open3d_rgbd_points_match_colmap_half_pixel_rays():
    o3d = pytest.importorskip('open3d')
    camera = dict(w=4, h=3, fl_x=8., fl_y=9., cx=2., cy=1.5,
                  transform_matrix=np.diag([1., -1., -1., 1.]).tolist())
    depth = np.ones((3, 4), np.float32)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(4, 3, *integer_center_pinhole(camera))
    points = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), intrinsic,
                                                           depth_scale=1., depth_trunc=10.)
    yy, xx = np.indices(depth.shape)
    expected = unproject(np.stack((xx+.5, yy+.5), -1), depth, camera)
    np.testing.assert_allclose(np.asarray(points.points), expected.reshape(-1, 3), atol=1e-7)


def test_large_hair_roi_remap_preserves_all_samples_and_missing_values():
    source = np.arange(48, dtype=np.float32).reshape(6, 8)
    indices = np.arange(100003) % 48
    uv = np.stack((indices % 8 + .5, indices // 8 + .5), -1)
    uv[-1] = -100
    actual = remap(source, uv, True)
    np.testing.assert_array_equal(actual[:-1], source.ravel()[indices[:-1]])
    assert actual[-1] == -1


def test_constant_candidate_depth_offset_is_not_a_temporal_dropout():
    previous = dict(front_depth=np.full((2, 2), 2.), source_id=np.zeros((2, 2), np.int16),
                    hair_probability=np.ones((2, 2)))
    following = {k:v.copy() for k,v in previous.items()}
    yy, xx = np.indices((2, 2)); yy, xx = yy.ravel(), xx.ravel()
    uv = np.stack((xx+.5, yy+.5), -1)
    result, _ = pair_metrics(previous, following, yy, xx, uv, np.ones(4), np.ones(4, bool), np.full(4, 2.))
    assert result['depth_exposure_dropouts'] == 0
    following['source_id'][0, 0] = -1
    following['front_depth'][0, 0] = 0
    result, _ = pair_metrics(previous, following, yy, xx, uv, np.ones(4), np.ones(4, bool), np.full(4, 2.))
    assert result['anchors'] == 4
    assert result['visibility_toggles'] == result['depth_exposure_dropouts'] == 1
