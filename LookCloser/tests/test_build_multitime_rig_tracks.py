from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_multitime_rig_tracks import merge_tracks, triangulate_track, unique_feature_indices


def test_duplicate_sift_orientations_use_one_location():
    xy = np.array([[1, 2], [1, 2], [3, 4], [1, 2]])
    assert unique_feature_indices(xy).tolist() == [0, 2]


def test_track_union_forbids_two_features_in_one_camera():
    tracks, conflicts = merge_tracks([((0, 1), (1, 2)), ((1, 2), (2, 3)), ((0, 8), (2, 3))])
    assert tracks == [[(0, 1), (1, 2), (2, 3)]] and conflicts == 1


def test_track_cycles_do_not_duplicate_observations():
    tracks, conflicts = merge_tracks([((0, 1), (1, 2)), ((1, 2), (2, 3)), ((2, 3), (0, 1))])
    assert len(tracks) == 1 and len(tracks[0]) == 3 and conflicts == 0


def synthetic():
    k = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]])
    projections = [k @ np.c_[np.eye(3), [t, 0., 0.]] for t in [-.3, 0., .3]]
    xyz = np.array([.1, -.1, 4.])
    observations = []
    for camera, p in enumerate(projections):
        q = p @ np.r_[xyz, 1.]
        observations.append((camera, q[:2] / q[2]))
    return xyz, projections, observations


def test_dlt_seed_recovers_known_point():
    xyz, projections, observations = synthetic()
    seed = triangulate_track(observations, projections)
    np.testing.assert_allclose(seed[0], xyz, atol=1e-12)
    assert seed[1] < 1e-10 and seed[2] > .5


def test_bad_seed_correspondence_is_rejected():
    _, projections, observations = synthetic()
    observations[0] = (0, observations[0][1] + [0, 50.])
    assert triangulate_track(observations, projections) is None
