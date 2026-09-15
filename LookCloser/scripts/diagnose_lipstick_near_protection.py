"""Read-only test of near-depth evidence protecting an observed false fin.

The old train-view diagnostic selects observations, not geometry to delete.
All candidates/meshes remain unchanged; corroboration is not ground truth.
"""
from pathlib import Path
import numpy as np
from joint_temporal_texture import read, sha, atomic_json
from review_full_block_transfer import ROOT as DEPTH_ROOT
from study_confidence_depth_prior import load_real, project_integer, support, unproject
from prune_measured_free_surface import near_tap_evidence

ROOT = Path('/mnt/data/dec5_lipstick_fin_near_protection/000995')
DIAGNOSTIC = Path('/mnt/data/dec5_lipstick_fin_depth/000995')
PRUNING = Path('/mnt/data/dec5_measured_free_center_pruning/000995')


def protected_faces(near, other_view_counts, minimum_other_views):
    near, other_view_counts = np.asarray(near), np.asarray(other_view_counts)
    if near.ndim != 3 or near.shape[2] != 4 or near.shape != other_view_counts.shape:
        raise ValueError('Expected cameras x triangles x four samples')
    if minimum_other_views < 0:
        raise ValueError('Negative corroboration threshold')
    return (near & (other_view_counts >= minimum_other_views)).any(axis=(0, 2))


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    dq, dr = read(DIAGNOSTIC/'request.json'), read(DIAGNOSTIC/'result.json')
    pq, pr = read(PRUNING/'request.json'), read(PRUNING/'result.json')
    assert dq['source_mesh_sha256'] == pq['mesh_sha256'] == sha(pq['mesh'])
    assert pr['request_sha256'] == sha(PRUNING/'request.json')
    assert dr['request_sha256'] == sha(DIAGNOSTIC/'request.json')
    for name, digest in pr['hashes'].items():
        assert sha(PRUNING/name) == digest
    assert sha(DIAGNOSTIC/'evidence.npz') == dr['hashes']['evidence.npz']
    rows, depths, receipt = load_real(DEPTH_ROOT, '000995')
    assert receipt == pq['depth_receipt'] == dq['depth_receipt']
    diagnostic = np.load(DIAGNOSTIC/'evidence.npz')
    prune = np.load(PRUNING/'evidence.npz')
    ids = diagnostic['triangle_ids']
    points = diagnostic['points']
    samples = prune['sample_indices'][ids]
    np.testing.assert_array_equal(prune['points'][samples], points)
    near = np.zeros((len(rows), len(ids), 4), bool)
    counts = np.zeros(near.shape, np.int16)
    observations = []
    for ci, (row, depth) in enumerate(zip(rows, depths)):
        flat = points.reshape(-1, 3)
        uv, z = project_integer(row, flat)
        hit = near_tap_evidence(depth, uv, z, tolerance=.0015, radius=0)
        near[ci] = hit.reshape(-1, 4)
        take = np.flatnonzero(hit)
        xy = np.rint(uv[take]).astype(int)
        observed = depth[xy[:, 1], xy[:, 0]]
        if len(take):
            witnesses = unproject(row, xy[:, 0], xy[:, 1], observed)
            corroboration, _ = support(witnesses, row, rows, depths)
            counts[ci].reshape(-1)[take] = corroboration
            for j, index in enumerate(take):
                observations.append(dict(camera=row['physical_camera'], triangle=int(ids[index//4]),
                    sample=int(index%4), pixel=xy[j].tolist(), measured_depth=float(observed[j]),
                    query_depth=float(z[index]), delta=float(observed[j]-z[index]),
                    other_agreeing_views=int(corroboration[j])))
        atomic_json(ROOT/'progress.json', dict(camera=ci+1, total=len(rows), geometry_changed=False))
    np.testing.assert_array_equal(near.sum(0), prune['near_counts'][samples])
    stable = prune['stable_far_counts'][samples]
    was_removed = np.isin(ids, prune['removed_triangle_ids'])
    remaining = ~was_removed
    rules = []
    for minimum in [0, 1, 2, 3]:
        protected = protected_faces(near, counts, minimum)
        rules.append(dict(minimum_other_near_views=minimum,
            protected_diagnostic_faces=int(protected.sum()),
            remaining_protected=int((remaining & protected).sum()),
            remaining_without_protection=int((remaining & ~protected).sum()),
            remaining_without_protection_and_stable_far_gate=int((remaining & ~protected & (stable >= 6).all(1)).sum()),
            representative_protected=bool(protected[ids == dr['representative_triangle']][0])))
    np.savez_compressed(ROOT/'evidence.npz', triangle_ids=ids, points=points, near=near,
        other_view_counts=counts, stable_far_counts=stable, previously_removed=was_removed)
    bindings = [DIAGNOSTIC/'request.json', DIAGNOSTIC/'result.json', DIAGNOSTIC/'evidence.npz',
                PRUNING/'request.json', PRUNING/'result.json', PRUNING/'evidence.npz']
    atomic_json(ROOT/'result.json', dict(script_sha256=sha(__file__),
        helpers={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in
            ['prune_measured_free_surface.py', 'study_confidence_depth_prior.py', 'joint_temporal_texture.py']},
        input_hashes={str(p):sha(p) for p in bindings}, depth_receipt=receipt,
        evidence_sha256=sha(ROOT/'evidence.npz'), diagnostic_triangles=len(ids),
        previously_remaining=int(remaining.sum()), near_policy_replayed_exactly=True,
        threshold_controls=rules, representative_triangle=dr['representative_triangle'],
        representative_near_observations=[r for r in observations if r['triangle']==dr['representative_triangle']],
        observations=observations, geometry_changed=False, no_removal_proposed_as_safe=True,
        heldout_used=False, diagnostic_not_quality_metrics=True,
        note='Stable farther footprints alone are not the full deletion gate; no pruning is run here.'))
    print(rules, flush=True)
    print('representative', [r for r in observations if r['triangle']==dr['representative_triangle']], flush=True)


if __name__ == '__main__':
    main()
