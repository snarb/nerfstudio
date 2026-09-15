"""Deterministic, angularly distributed observed seeds for a bounded body prior."""
import numpy as np


def select_seeds(query, normal, points, normals, *, radius=.006, sectors=8, per_sector=3):
    """Use nearby compatible seeds around a hole, not only its nearest one edge.

    Selection does not certify the query; the unchanged cross-validated hull
    certificate must subsequently accept it. Returned indices refer to points.
    """
    q = np.asarray(query, float)
    n = np.asarray(normal, float)
    p = np.asarray(points, float)
    sn = np.asarray(normals, float)
    if p.shape != sn.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Expected matching Nx3 seed points/normals')
    if radius <= 0 or sectors < 1 or per_sector < 1:
        raise ValueError('Invalid neighborhood limits')
    if not np.isfinite(q).all() or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-9:
        raise ValueError('Invalid query/normal')
    n = n / np.linalg.norm(n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, axis); u /= np.linalg.norm(u)
    w = np.cross(n, u)
    delta = p - q
    distance = np.linalg.norm(delta, axis=1)
    good = np.isfinite(p).all(1) & np.isfinite(sn).all(1) & (distance <= radius) & (sn @ n >= .5)
    ids = np.flatnonzero(good)
    angle = np.arctan2(delta[ids] @ w, delta[ids] @ u)
    bins = np.floor((angle + np.pi) * sectors / (2 * np.pi)).astype(int) % sectors
    selected = []
    for sector in range(sectors):
        candidate = ids[bins == sector]
        order = np.lexsort((candidate, distance[candidate]))
        selected.extend(candidate[order[:per_sector]].tolist())
    return np.asarray(selected, dtype=np.int64)
