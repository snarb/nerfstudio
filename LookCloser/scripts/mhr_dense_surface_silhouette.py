"""Nested, geometry-only barycentric quadrature for the frozen surface residual."""
from types import FunctionType
import numpy as np
from conform_mhr_measured_surface import barycentric_matrix
import mhr_surface_silhouette as base


def barycentric_lattice(order):
    if order not in (2, 4, 8):
        raise ValueError('Supported frozen lattice orders: 2, 4, 8')
    if order == 2:
        return base.BARY.copy()
    points = barycentric_lattice(order // 2).tolist()
    seen = {tuple(p) for p in points}
    for i in range(order + 1):
        for j in range(order - i + 1):
            k = order - i - j
            if max(i, j, k) == order:
                continue  # vertices already have their independent weight16 term
            p = (i/order, j/order, k/order)
            if p not in seen:
                seen.add(p); points.append(p)
    return np.asarray(points, float)


def association(triangles, active, order):
    triangles = np.asarray(triangles, int); active = np.asarray(active, bool)
    faces = triangles[active[triangles].any(1)]
    bary = barycentric_lattice(order)
    return barycentric_matrix(np.repeat(faces, len(bary), axis=0),
                              np.tile(bary, (len(faces), 1)), len(active))


def linearizer(order):
    """Clone only the association binding; residual/Jacobian code is unchanged."""
    barycentric_lattice(order)  # validate before any fitting/output side effect
    def samples(triangles, active):
        return association(triangles, active, order)
    original = base.linearize
    return FunctionType(original.__code__, dict(original.__globals__, association=samples),
                        name=f'dense_surface_order{order}', argdefs=original.__defaults__)
