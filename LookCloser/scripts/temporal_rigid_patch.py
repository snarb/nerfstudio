"""Rigid multiview temporal registration in a common normalized camera gauge."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def compose_flow_fields(fields):
    """Compose pixel displacements without turning an out-of-frame path valid."""
    import cv2
    h, w = fields[0].shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    origin = np.stack([xx, yy], -1); position = origin.copy(); valid = np.ones((h, w), bool)
    for field in fields:
        if field.shape != origin.shape or not np.isfinite(field).all(): raise ValueError('Invalid flow field')
        valid &= (position[..., 0] >= 0) & (position[..., 0] <= w - 1) & (position[..., 1] >= 0) & (position[..., 1] <= h - 1)
        position += cv2.remap(field.astype(np.float32), position[..., 0], position[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid &= (position[..., 0] >= 0) & (position[..., 0] <= w - 1) & (position[..., 1] >= 0) & (position[..., 1] <= h - 1)
    return position - origin, valid


def transfer_points(points, source_metadata, target_metadata):
    def gauge(metadata):
        transform = np.eye(4); transform[:3] = metadata['dataparser_transform']
        scale = np.diag([metadata['dataparser_scale']] * 3 + [1.])
        return scale @ transform
    matrix = gauge(target_metadata) @ np.linalg.inv(gauge(source_metadata))
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def project_native(points, camera):
    pose = np.asarray(camera['transform_matrix']) @ np.diag([1., -1., -1., 1.])
    local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    uv = local[:, :2] / np.maximum(z[:, None], 1e-9)
    return uv * [camera['fl_x'], camera['fl_y']] + [camera['cx'], camera['cy']], z


def warp_rigid(points, parameters, center):
    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
    return (np.asarray(points) - center) @ rotation.T + center + parameters[3:]


def fit_rigid(points, cameras, point_indices, camera_indices, observations, initial, fit_camera_indices):
    points = np.asarray(points, float); center = points.mean(0)
    pi = np.asarray(point_indices); ci = np.asarray(camera_indices); observations = np.asarray(observations)
    selected = np.isin(ci, fit_camera_indices)
    if selected.sum() < 30 or len(np.unique(ci[selected])) < 2:
        raise ValueError('Insufficient multiview correspondences')
    def residual(parameters):
        moved = warp_rigid(points, parameters, center); predicted = np.empty_like(observations)
        for index, camera in enumerate(cameras):
            mask = ci == index
            predicted[mask] = project_native(moved[pi[mask]], camera)[0]
        return predicted - observations
    result = least_squares(lambda p: residual(p)[selected].ravel(), initial,
                           loss='soft_l1', f_scale=2., max_nfev=300)
    if not result.success or not np.isfinite(result.x).all(): raise ValueError('Rigid fit failed')
    errors = np.linalg.norm(residual(result.x), axis=1)
    return result.x, center, errors
