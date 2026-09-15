"""Conforming, geometry-preserving midpoint subdivision; no smoothing or repair.

Selected triangles split all their edges. Neighbours split exactly the shared
marked edges, without propagating selection across the entire mesh. Original
parent IDs remain available for confidence attribution and coverage checks.
"""
import numpy as np


def subdivide(vertices, triangles, selected, parent_ids=None):
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    selected = np.asarray(selected)
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all()
            or t.ndim != 2 or t.shape[1] != 3 or not len(t)
            or t.min() < 0 or t.max() >= len(v)
            or selected.shape != (len(t),) or selected.dtype != bool):
        raise ValueError('Invalid finite triangle mesh or boolean selection')
    parents = np.arange(len(t)) if parent_ids is None else np.asarray(parent_ids)
    if parents.shape != (len(t),) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError('Invalid parent IDs')
    edges = np.sort(t[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    unique, inverse = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    edge_ids = inverse.reshape(-1, 3)
    marked = np.zeros(len(unique), bool)
    marked[edge_ids[selected].ravel()] = True
    midpoint_ids = np.full(len(unique), -1, np.int64)
    midpoint_ids[marked] = len(v) + np.arange(marked.sum())
    new_v = np.concatenate([v, v[unique[marked]].mean(1)])
    mids = midpoint_ids[edge_ids]
    children, ancestry = [], []
    for face, mid, parent in zip(t, mids, parents):
        mask = mid >= 0
        count = int(mask.sum())
        if count == 0:
            pieces = [face.tolist()]
        elif count == 3:
            a, b, c = face; ab, bc, ca = mid
            pieces = [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        else:
            # Rotate without reversing orientation. Two marked edges become AB, BC.
            start = int(np.flatnonzero(mask)[0]) if count == 1 else (int(np.flatnonzero(~mask)[0])+1) % 3
            a, b, c = np.roll(face, -start)
            ab, bc, _ = np.roll(mid, -start)
            pieces = [[a, ab, c], [ab, b, c]] if count == 1 else [[b, bc, ab], [a, ab, c], [ab, bc, c]]
        children.extend(pieces)
        ancestry.extend([int(parent)] * len(pieces))
    return new_v, np.asarray(children, np.int64), np.asarray(ancestry, np.int64)


def verify_coverage(original_v, original_t, vertices, triangles, parents):
    """Check original vertices, orientation, plane, barycentric support and area."""
    ov, ot, v, t, p = map(np.asarray, (original_v, original_t, vertices, triangles, parents))
    np.testing.assert_array_equal(v[:len(ov)], ov)
    if p.shape != (len(t),) or p.min() < 0 or p.max() >= len(ot):
        raise ValueError('Invalid ancestry')
    base = ov[ot]
    normal = np.cross(base[:, 1]-base[:, 0], base[:, 2]-base[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    if (lengths <= 0).any():
        raise ValueError('Degenerate original face')
    child = v[t]
    child_normal = np.cross(child[:, 1]-child[:, 0], child[:, 2]-child[:, 0])
    if (np.einsum('ij,ij->i', child_normal, normal[p]) <= 0).any():
        raise ValueError('Reversed or degenerate child')
    a = base[p, 1]-base[p, 0]; b = base[p, 2]-base[p, 0]
    offset = child-base[p, 0, None, :]
    plane = np.einsum('ijk,ik->ij', offset, normal[p]/lengths[p, None])
    np.testing.assert_allclose(plane, 0, atol=1e-12, rtol=0)
    aa = np.einsum('ij,ij->i', a, a); ab = np.einsum('ij,ij->i', a, b); bb = np.einsum('ij,ij->i', b, b)
    da = np.einsum('ijk,ik->ij', offset, a); db = np.einsum('ijk,ik->ij', offset, b)
    denom = (aa*bb-ab*ab)[:, None]
    u = (bb[:, None]*da-ab[:, None]*db)/denom
    w = (aa[:, None]*db-ab[:, None]*da)/denom
    if (u < -1e-7).any() or (w < -1e-7).any() or (u+w > 1+1e-7).any():
        raise ValueError('Child outside original face')
    totals = np.bincount(p, weights=np.linalg.norm(child_normal, axis=1), minlength=len(ot))
    np.testing.assert_allclose(totals, lengths, atol=1e-15, rtol=1e-9)
    return dict(original_vertices_exact=True, child_orientation_preserved=True,
                child_planes_preserved=True, child_barycentrics_inside=True,
                per_parent_area_preserved=True, max_plane_error=float(abs(plane).max()))
