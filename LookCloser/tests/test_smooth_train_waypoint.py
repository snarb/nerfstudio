import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from smooth_train_waypoint import periodic_bump, displace_poses, motion_summary


def test_periodicity_and_compact_support():
    x = np.linspace(-100, 200, 500)
    np.testing.assert_allclose(periodic_bump(x, 147, 150, 35),
                               periodic_bump(x + 150, 147, 150, 35), atol=1e-14)
    assert periodic_bump(147, 147, 150, 35) == 1
    assert periodic_bump(50, 147, 150, 35) == 0
    with pytest.raises(ValueError): periodic_bump(x, 0, 150, 80)


def test_exact_waypoint_and_no_pose_freeze():
    poses = np.repeat(np.eye(4)[None], 150, axis=0)
    phase = np.arange(150) * 2 * np.pi / 150
    poses[:, 0, 3] = np.cos(phase)
    poses[:, 1, 3] = np.sin(phase)
    target = poses[147].copy(); target[0, 3] -= .2
    changed, weights = displace_poses(poses, target, 147, 35)
    np.testing.assert_allclose(changed[147], target)
    np.testing.assert_array_equal(changed[weights == 0], poses[weights == 0])
    assert motion_summary(changed)['step_min'] > .01
    np.testing.assert_allclose(changed[:, :3, :3], poses[:, :3, :3])
