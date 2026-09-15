"""Train-only calibrated plane-patch comparison, not a depth estimator.

Pixel coordinates are native RGB indices (principal point minus one half).
Unphotographed, grazing, occluded or low-texture comparisons abstain.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def project_rgb(points, row):
    p = np.asarray(row['transform_matrix'], float)
    q = (np.asarray(points)-p[:3, 3]) @ p[:3, :3]
    z = -q[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.stack([row['fl_x']*q[..., 0]/z+row['cx']-.5,
                       -row['fl_y']*q[..., 1]/z+row['cy']-.5], -1)
    return uv, z


def patch_points(row, centers, normals, radius):
    """N planes through world points; return world grid and fixed query UV."""
    centers, normals = np.asarray(centers, float), np.asarray(normals, float)
    if centers.ndim != 2 or centers.shape[1] != 3 or normals.shape != centers.shape:
        raise ValueError('Expected N world centers and normals')
    if radius < 1:
        raise ValueError('Positive patch radius required')
    pose = np.asarray(row['transform_matrix'], float)
    uv, z = project_rgb(centers, row)
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    uv = uv[:, None, :] + np.column_stack([x.ravel(), y.ravel()])[None]
    rays = np.stack([(uv[..., 0]-row['cx']+.5)/row['fl_x'],
                     -(uv[..., 1]-row['cy']+.5)/row['fl_y'], -np.ones(uv.shape[:-1])], -1)
    rays = rays @ pose[:3, :3].T
    normal_size = np.linalg.norm(normals, axis=1)
    normal = normals / np.maximum(normal_size[:, None], 1e-12)
    denominator = np.einsum('nki,ni->nk', rays, normal)
    numerator = np.einsum('ni,ni->n', centers-pose[:3, 3], normal)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = numerator[:, None] / denominator
    valid = (z > 0) & (normal_size > 1e-8) & (np.abs(denominator) > .05).all(1) & (t > 0).all(1)
    points = pose[:3, 3] + t[..., None] * rays
    valid &= np.isfinite(points).all(axis=(1, 2))
    return points, uv, valid


def photographed_rgb(image, uv):
    image = np.asarray(image, float)
    uv = np.asarray(uv)
    h, w = image.shape[:2]
    good = np.isfinite(uv).all(-1) & (uv[..., 0] >= 1) & (uv[..., 0] <= w-2) & (uv[..., 1] >= 1) & (uv[..., 1] <= h-2)
    xy = np.where(np.isfinite(uv), uv, -10)
    values = np.stack([map_coordinates(image[..., c], xy.reshape(-1, 2).T[::-1],
                                      order=1, mode='constant', cval=np.nan).reshape(uv.shape[:-1])
                       for c in range(3)], -1)
    values[~good] = np.nan
    return values, good.all(-1)


def patch_ncc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('Expected equally shaped N by patch pixels by RGB arrays')
    a = a-a.mean(1, keepdims=True)
    b = b-b.mean(1, keepdims=True)
    product = np.sqrt(np.sum(a*a, axis=(1, 2))*np.sum(b*b, axis=(1, 2)))
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.sum(a*b, axis=(1, 2))/product
    scores[product < 1e-8] = np.nan
    return scores


def compare_planes(query, witnesses, images, depths, near, far, normals, radius):
    """Warp identical query footprints through parallel competing depth planes.

    The caller must provide collinear near/far centers. A single center depth
    occlusion check abstains; it cannot certify all patch pixels as one surface.
    """
    nearp, uv, nearvalid = patch_points(query, near, normals, radius)
    farp, faruv, farvalid = patch_points(query, far, normals, radius)
    np.testing.assert_allclose(uv, faruv, atol=1e-5)
    query_rgb, query_valid = photographed_rgb(images[query['physical_camera']], uv)
    variation = np.std(query_rgb, axis=1).mean(1)
    near_scores, far_scores, availability = [], [], []
    patches = []
    center = nearp.shape[1]//2
    for row in witnesses:
        sampled, usable = [], []
        for points, hypothesis_valid in [(nearp, nearvalid), (farp, farvalid)]:
            target_uv, z = project_rgb(points, row)
            rgb, inside = photographed_rgb(images[row['physical_camera']], target_uv)
            integer = np.rint(target_uv[:, center]+.5).astype(int)
            h, w = depths[row['physical_camera']].shape
            xy = np.clip(integer, [0, 0], [w-1, h-1])
            observed = depths[row['physical_camera']][xy[:, 1], xy[:, 0]]
            occluded = (observed > 0) & np.isfinite(observed) & (observed < z[:, center]-.003)
            known = hypothesis_valid & inside & (z > 0).all(1) & ~occluded
            sampled.append(rgb)
            usable.append(known)
        good = query_valid & usable[0] & usable[1]
        a, b = patch_ncc(query_rgb, sampled[0]), patch_ncc(query_rgb, sampled[1])
        a[~good], b[~good] = np.nan, np.nan
        near_scores.append(a); far_scores.append(b); availability.append(good)
        patches.append(sampled)
    return dict(near=np.array(near_scores).T, far=np.array(far_scores).T,
                available=np.array(availability).T, query_std=variation,
                query_rgb=query_rgb, patches=np.array(patches))


def evidence_decision(near, far, available, query_std):
    """Both plane choices and both scales must agree in >=3 same witnesses."""
    near, far, available = np.asarray(near), np.asarray(far), np.asarray(available, bool)
    if near.shape != far.shape or near.shape != available.shape or near.ndim != 3 or near.shape[0] != 4:
        raise ValueError('Expected four configurations by events by witnesses')
    std = np.asarray(query_std)
    if std.shape != near.shape[:2]:
        raise ValueError('Expected per-configuration query variation')
    good = available & np.isfinite(near) & np.isfinite(far) & (std[..., None] >= 2)
    nearer = good & (near >= .65) & (near-far >= .2)
    farther = good & (far >= .65) & (far-near >= .2)
    near_votes = nearer.all(0).sum(1)
    far_votes = farther.any(0).sum(1)
    return dict(near_votes=near_votes, far_votes=far_votes,
                reject_far=(near_votes >= 3) & (far_votes < 2))
