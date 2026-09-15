"""Bounded multiview silhouette proposal, not measured surface depth.

Out-of-image/annotation rays are unknown, never background votes. The returned
implicit field describes a possible occupancy envelope, not its unique shape.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def signed_pixels(mask):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError('Need a nontrivial two-dimensional silhouette')
    return (distance_transform_edt(mask) - distance_transform_edt(~mask)).astype(np.float32)


def combine_silhouettes(projected, depths, fields, domains, minimum_views=3,
                        margin_pixels=0.):
    """Minimum signed silhouette distance, with explicit availability count.

    Coordinates address pixel centers, as in joint_temporal_texture.project().
    Positive margin dilates a silhouette; it is uncertainty, not evidence.
    """
    projected = np.asarray(projected)
    depths = np.asarray(depths)
    if projected.ndim != 3 or projected.shape[2] != 2 or depths.shape != projected.shape[:2]:
        raise ValueError('Expected camera x point x xy and camera x point depth')
    if len(fields) != len(projected) or len(domains) != len(fields) or minimum_views < 1:
        raise ValueError('Invalid camera inventory or minimum view count')
    n = projected.shape[1]
    minimum = np.full(n, np.inf, np.float32)
    available = np.zeros(n, np.uint16)
    positive = np.zeros(n, np.uint16)
    for uv, z, field, domain in zip(projected, depths, fields, domains):
        h, w = field.shape
        if domain.shape != field.shape:
            raise ValueError('Domain and silhouette dimensions differ')
        finite = np.isfinite(uv).all(1) & np.isfinite(z)
        valid = finite & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        xy = uv[ids].T[::-1]
        # Require the whole interpolation footprint to be annotated.
        known = map_coordinates(domain.astype(np.float32), xy, order=1, mode='constant', cval=0) >= .99999
        ids = ids[known]
        distance = map_coordinates(field, uv[ids].T[::-1], order=1, mode='constant', cval=-1e6) + margin_pixels
        minimum[ids] = np.minimum(minimum[ids], distance)
        available[ids] += 1
        positive[ids] += distance >= 0
    minimum[available < minimum_views] = -1.
    return minimum, available, positive


def remove_box_caps(vertices, triangles, lower, upper, spacing):
    """Do not mistake artificial AABB closures for observed silhouettes."""
    v = np.asarray(vertices)
    near = ((v <= np.asarray(lower) + 1.1*spacing) |
            (v >= np.asarray(upper) - 1.1*spacing)).any(1)
    return ~near[np.asarray(triangles)].any(1)
