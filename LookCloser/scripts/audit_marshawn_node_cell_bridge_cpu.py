"""CPU synthetic contract proof on actual native bounds and fixed witnesses.

No fitted head is available or inferred. Synthetic node arrays are never saved
as production semantic predictions, and sparse witness retention is not a
certificate of complete hair/feet coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from marshawn_semantic_node_cell_bridge import LogitNodeCellBridge, fresh_dilation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(grid_path, query_paths, output):
    with np.load(grid_path, allow_pickle=False) as data:
        bounds = data['cascade_aabbs'].astype(np.float64).reshape(-1, 2, 3)
        resolution = data['resolution'].astype(np.int64)
        if resolution.ndim == 0:
            resolution = np.repeat(resolution, 3)
    points = np.array([p['witness_xyz'] for path in query_paths
                       for p in json.loads(Path(path).read_text())['pixels']], dtype=np.float64)
    if points.shape != (109, 3):
        raise ValueError('Exact sealed clothing103 plus shoe6 required')
    # Match the declared buffer dtype of FrozenActorGrid, not native cascade
    # bounds rounded to a convenient decimal or an assumed identical phase.
    head_bounds = np.array([[-.84, -.21, -.07], [.15, .42, .63]], dtype=np.float32).astype(np.float64)
    node_resolution = np.array([128]*3)
    q = (points-head_bounds[0])/(head_bounds[1]-head_bounds[0])*(node_resolution-1)
    if (q < 0).any() or (q > node_resolution-1).any():
        raise ValueError('Witness outside declared semantic head bounds')
    cases = dict(all_positive=np.full((128,)*3, 10., dtype=np.float32),
                 all_negative=np.full((128,)*3, -10., dtype=np.float32),
                 isolated_knot=np.full((128,)*3, -10., dtype=np.float32),
                 synthetic_witness_basis=np.full((128,)*3, -10., dtype=np.float32))
    cases['isolated_knot'][64, 64, 64] = 10
    # Deliberately constructed positive bases test geometry/axis conversion;
    # these are NOT learned semantic labels or a proposed fit.
    for lo in np.floor(q).astype(int):
        hi = np.minimum(lo+2, node_resolution)
        cases['synthetic_witness_basis'][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 10
    rows = {}
    for name, nodes in cases.items():
        gate = LogitNodeCellBridge(nodes, head_bounds, world_to_box=np.eye(4), selector='hard_theta05')
        core = gate.map_cascades(bounds, resolution)
        halo = np.stack([fresh_dilation(x, 1) for x in core])
        per_cascade = []
        for index, box in enumerate(bounds):
            ids = np.floor((points-box[0])/(box[1]-box[0])*resolution).astype(int)
            in_domain = ((ids >= 0) & (ids < resolution)).all(1)
            occupied = np.zeros(len(points), dtype=bool)
            occupied_halo = occupied.copy()
            occupied[in_domain] = core[index][tuple(ids[in_domain].T)]
            occupied_halo[in_domain] = halo[index][tuple(ids[in_domain].T)]
            per_cascade.append(dict(native_bounds=box.tolist(),
                cells_radius0=int(core[index].sum()), cells_radius1=int(halo[index].sum()),
                witness_observations_inside=int(in_domain.sum()),
                witnesses_radius0=int(occupied.sum()), witnesses_radius1=int(occupied_halo.sum())))
            if name in ('all_positive', 'synthetic_witness_basis') and not occupied[in_domain].all():
                raise AssertionError('Constructed positive source point lost')
        assert (halo | core == halo).all()
        rows[name] = per_cascade
    receipt = dict(passed=True, CPU_only=True, GPU_used=False, actual_head_evaluated=False,
        source_hashes={str(p): sha(p) for p in [grid_path, *query_paths, Path(__file__),
            Path(__file__).with_name('marshawn_semantic_node_cell_bridge.py')]},
        semantic_bounds=head_bounds.tolist(), semantic_node_resolution=node_resolution.tolist(),
        native_resolution=resolution.tolist(), semantic_node_pitch=((head_bounds[1]-head_bounds[0])/127).tolist(),
        native_cell_pitches=((bounds[:, 1]-bounds[:, 0])/resolution).tolist(),
        source_observations=len(points), unique_positions=len(np.unique(points, axis=0)),
        cases=rows, coordinate_guard=2e-7, numerical_logit_guard=1e-5,
        certification='Conservative mathematical selector-support envelope under declared arithmetic slack; not a CUDA deployment certificate.',
        limitations=['No actual trained direct-LC head values were used.',
                     'Synthetic witness positives are a coordinate test, never semantic evidence.',
                     'Outside head bounds is zero by explicit chosen-head policy, not proved empty space.',
                     'GPU interpolation/threshold arithmetic still needs actual deployment parity.',
                     'Skipping zero-density cells can change native sample phase/tail; support proof alone does not establish rendered identity.',
                     'No model, grid, masks, labels, training or source checkpoint was changed.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--queries', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.grid, args.queries, args.output)
