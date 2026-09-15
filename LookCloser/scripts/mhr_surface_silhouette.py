"""Train-camera silhouette residuals on fixed barycentric surface samples.

Opt-in auxiliary term: edge midpoints and centroid of every face touching an
active vertex, equally weighted. No target image, ray, or residual ROI is used.
This is finite sampling, not a continuous silhouette containment certificate.
"""
import numpy as np
from scipy import sparse
from conform_mhr_measured_surface import barycentric_matrix
from fit_mhr_silhouette_conformance import silhouette_samples

BARY = np.array([[.5, .5, 0.], [.5, 0., .5], [0., .5, .5], [1/3, 1/3, 1/3]])


def association(triangles, active):
    triangles = np.asarray(triangles, int)
    active = np.asarray(active, bool)
    faces = triangles[active[triangles].any(1)]
    return barycentric_matrix(np.repeat(faces, len(BARY), axis=0),
                              np.tile(BARY, (len(faces), 1)), len(active))


def linearize(vertices, triangles, active, rows, sdfs, weight=16.):
    """Return weighted Jacobian and -residual for active xyz increments.

    Robust weights are frozen within one Gauss–Newton/IRLS step. Fixed vertices
    contribute to projected sample positions but have no variable columns.
    """
    vertices = np.asarray(vertices, float)
    active = np.asarray(active, bool)
    if not np.isfinite(vertices).all() or not np.isfinite(weight) or weight <= 0:
        raise ValueError('Finite geometry and positive finite weight required')
    if len(rows) != len(sdfs) or not rows or not active.any():
        raise ValueError('Expected paired cameras/SDFs and active vertices')
    samples = association(triangles, active)
    if not samples.shape[0]:
        raise ValueError('No active surface samples')
    points = samples @ vertices
    active_samples = samples[:, active]
    blocks, residuals = [], []
    for ids, values, gradients in silhouette_samples(points, rows, sdfs):
        take = values > 0
        excess = values[take]
        robust = 1 / np.sqrt(1 + (excess / 8.) ** 2)
        weights = np.sqrt(weight * robust / (len(rows) * len(points))) / 2.
        mapping = active_samples[ids[take]].tocoo()
        gradient = gradients[take] * weights[:, None]
        block = sparse.coo_matrix(
            ((mapping.data[:, None] * gradient[mapping.row]).ravel(),
             (np.repeat(mapping.row, 3), (3 * mapping.col[:, None] + np.arange(3)).ravel())),
            shape=(len(excess), 3 * int(active.sum()))).tocsr()
        blocks.append(block)
        residuals.append(-excess * weights)
    return sparse.vstack(blocks, format='csr'), np.concatenate(residuals)


def augment_optimizer(source):
    replacements = {
        "system = sparse.vstack([data, smooth, magnitude, sil], format='csr')":
        "surface, surface_rhs = surface_silhouette(current, triangles, active, rows, sdfs)\n"
        "        system = sparse.vstack([data, smooth, magnitude, sil, surface], format='csr')",
        'rhs = np.r_[data_rhs, -smooth@displacement, -magnitude@displacement, rr]':
        'rhs = np.r_[data_rhs, -smooth@displacement, -magnitude@displacement, rr, surface_rhs]',
        'silhouette_rows=cursor,': 'surface_silhouette_rows=surface.shape[0], silhouette_rows=cursor,'}
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source
