"""Exploratory large-patch qualifier; both depth alternatives get best plane fit.

Use only over a previously proposed surface; never generate depth from this cost.
Report: experiments/dec5_independent_plane_patch_guard.md.
"""
import numpy as np
from plane_patch_evidence import compare_planes


def best_plane_decision(near, far, available, query_std):
    near, far, available = np.asarray(near), np.asarray(far), np.asarray(available, bool)
    if near.shape != far.shape or near.shape != available.shape or near.ndim != 3 or near.shape[0] != 2:
        raise ValueError('Expected two plane models by events by witnesses')
    if np.asarray(query_std).shape != near.shape[:2]:
        raise ValueError('Expected query variation for each plane model')
    known = available.all(0) & np.isfinite(near).all(0) & np.isfinite(far).all(0)
    a, b = near.max(0), far.max(0)
    near_votes = (known & (a >= .65) & (a-b >= .2)).sum(1)
    far_votes = (known & (b >= .65) & (b-a >= .2)).sum(1)
    reject = (near_votes >= 3) & (far_votes < 2) & (np.asarray(query_std).min(0) >= 4)
    return dict(reject_far=reject, near_votes=near_votes, far_votes=far_votes)


def score_events(query, witnesses, images, depths, near, far, normals):
    """Bound memory in native31x31 comparisons; retain compact scores."""
    chunks = []
    for lo in range(0, len(near), 64):
        hi = min(lo+64, len(near));count = hi-lo
        ns = [np.tile(np.asarray(query['transform_matrix'])[:3, 2], (count, 1)), normals[lo:hi]]
        results = []
        for normal in ns:
            q = compare_planes(query, witnesses, images, depths, near[lo:hi], far[lo:hi], normal, 15)
            results.append({k: q[k] for k in ['near', 'far', 'available', 'query_std']})
        arrays = {k: np.array([r[k] for r in results]) for k in results[0]}
        chunks.append(best_plane_decision(**arrays))
    return {k: np.concatenate([r[k] for r in chunks]) for k in chunks[0]} if chunks else dict(
        reject_far=np.empty(0, bool), near_votes=np.empty(0, int), far_votes=np.empty(0, int))
