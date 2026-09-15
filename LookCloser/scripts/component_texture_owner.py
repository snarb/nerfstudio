"""Coverage-first hard source ownership on connected added-mesh regions.

No color averaging, geometry mutation, registration or permission to use an
invisible source. Unsupported faces keep the original graph solution.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from hard_surface_texture import face_adjacency


def components(triangles, original_count):
    triangles = np.asarray(triangles)
    if not 0 <= original_count <= len(triangles):
        raise ValueError('Invalid original face count')
    n = len(triangles)-original_count
    if not n:
        return np.empty(0, int)
    edges = face_adjacency(triangles)
    edges = edges[(edges >= original_count).all(1)]-original_count
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(adjacency, directed=False, return_labels=True)[1]


def choose_owners(labels, quality, region_ids, areas, original_count, *, minimum_faces=100, minimum_coverage=.8, coverage_slack=.01):
    labels = np.asarray(labels);quality = np.asarray(quality);areas = np.asarray(areas)
    if quality.ndim != 2 or quality.shape[1] != len(labels) or areas.shape != labels.shape or region_ids.shape != (len(labels)-original_count,):
        raise ValueError('Incompatible face/source arrays')
    if not np.isfinite(quality).all() or not np.isfinite(areas).all() or (quality < 0).any() or (areas < 0).any():
        raise ValueError('Nonfinite/negative source evidence')
    output = labels.copy();records = []
    for region in np.unique(region_ids):
        ids = original_count+np.flatnonzero(region_ids == region)
        if len(ids) < minimum_faces or areas[ids].sum() <= 0:
            continue
        weights = areas[ids]/areas[ids].sum();q = quality[:, ids]
        coverage = (q > 0) @ weights
        normalized = q/np.maximum(q.max(0), 1e-12)
        mean_quality = normalized @ weights
        near_best = coverage >= coverage.max()-coverage_slack
        order = np.lexsort((np.arange(len(coverage)), -mean_quality, ~near_best))
        owner = int(order[0]);accepted = bool(coverage[owner] >= minimum_coverage)
        changed = 0
        if accepted:
            supported = ids[q[owner] > 0]
            changed = int((output[supported] != owner).sum());output[supported] = owner
        records.append(dict(region=int(region), faces=len(ids), owner=owner, owner_coverage=float(coverage[owner]),
            accepted=accepted, changed_faces=changed, coverage=coverage.tolist(), mean_quality=mean_quality.tolist()))
    np.testing.assert_array_equal(output[:original_count], labels[:original_count])
    changed = np.flatnonzero(output != labels)
    if len(changed):
        assert (quality[output[changed], changed] > 0).all()
    return output, records
