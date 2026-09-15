"""Opt-in periodic, local pose displacement through an exact train waypoint.

This changes the camera shot, not geometry. Work in one calibration coordinate
system; normalize to each actor frame only after evaluating the path.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def periodic_bump(samples, center, period, half_width):
    if not 0 < half_width < period / 2:
        raise ValueError('Require 0 < half_width < period/2')
    distance = (np.asarray(samples) - center + period / 2) % period - period / 2
    return np.where(abs(distance) < half_width,
                    np.cos(np.pi * distance / (2 * half_width)) ** 4, 0.)


def displace_poses(poses, target, center, half_width):
    poses = np.asarray(poses, dtype=float)
    target = np.asarray(target, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or target.shape != (4, 4):
        raise ValueError('Expected homogeneous camera poses')
    if not np.isfinite(poses).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite pose')
    if not 0 <= center < len(poses):
        raise ValueError('Waypoint index outside path')
    weight = periodic_bump(np.arange(len(poses)), center, len(poses), half_width)
    delta = target[:3, 3] - poses[center, :3, 3]
    rotation = Rotation.from_matrix(target[:3, :3] @ poses[center, :3, :3].T).as_rotvec()
    result = poses.copy()
    result[:, :3, 3] += weight[:, None] * delta
    result[:, :3, :3] = Rotation.from_rotvec(weight[:, None] * rotation).as_matrix() @ poses[:, :3, :3]
    # Preserve unchanged records exactly, including their binary rotation values.
    result[weight == 0] = poses[weight == 0]
    # Stored calibrated rotations carry float32 normalization round-off;
    # scipy projects them to SO(3). Preserve the actual saved waypoint.
    np.testing.assert_allclose(result[center], target, atol=2e-7)
    result[center] = target
    return result, weight


def motion_summary(poses):
    poses = np.asarray(poses)
    position = poses[:, :3, 3]
    velocity = np.roll(position, -1, axis=0) - position
    speed = np.linalg.norm(velocity, axis=1)
    axes = poses[:, :3, 2]
    angle = np.degrees(np.arccos(np.clip(axes @ axes.T, -1, 1)))
    return dict(position_extent=np.ptp(position, axis=0).tolist(),
                view_angle_extent_degrees=float(angle.max()),
                step_min=float(speed.min()), step_max=float(speed.max()),
                step_median=float(np.median(speed)),
                loop_seam_step=float(speed[-1]),
                max_second_difference=float(np.linalg.norm(
                    np.roll(velocity, -1, axis=0) - velocity, axis=1).max()))
