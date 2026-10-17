"""Evaluation-only association of curvature phase and local LPIPS change."""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import read, sha, write
from edge_position_quality_luster_hybrid import quality
from diagnose_transfer_regions_luster_hybrid import summarize


def run(root, name):
    candidate = root / 'candidates' / name
    config = read(candidate / 'config.json')
    rows = []
    for target in [7, 12, 95, 150]:
        prefix = f'cam{target:03d}'
        lp, wp = candidate / f'{prefix}_sources.npy', candidate / f'{prefix}_confidence.npy'
        mp = root / 'evaluation' / f'{name}_spatial' / f'{prefix}_lpips_maps.npz'
        labels, weight = np.load(lp, mmap_mode='r'), np.load(wp, mmap_mode='r')
        cost = np.full(labels.shape, .5, np.float32)
        known = np.zeros(labels.shape, bool)
        inputs = {str(p): sha(p) for p in [lp, wp, mp, candidate / f'{prefix}_receipt.json']}
        for source in np.unique(labels[labels >= 0]):
            d = quality(root, target, int(source), config)
            receipt = read(d / 'receipt.json')
            x0, y0, x1, y1 = receipt['box']
            sl = np.s_[y0:y1, x0:x1]
            selected = labels[sl] == source
            cost[sl][selected] = np.load(d / 'cost.npy', mmap_mode='r')[selected]
            known[sl][selected] = np.load(d / 'known.npy', mmap_mode='r')[selected]
            inputs[str(d / 'receipt.json')] = sha(d / 'receipt.json')
        with np.load(mp) as maps:
            x0, y0, x1, y1 = map(int, maps['crop'])
            sl = np.s_[y0:y1, x0:x1]
            active = maps['mask'] & (weight[sl] > 0)
            reliable = active & known[sl]
            baseline, hybrid = maps['baseline'], maps['hybrid']
            delta = hybrid - baseline
            edges = [0., .1, .2, .3, .4, .5, .6, .8, 1.0001]
            groups = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                selection = reliable & (cost[sl] >= lo) & (cost[sl] < hi)
                groups.append(dict(min_cost=lo, max_cost=hi, **summarize(selection, delta, baseline, hybrid)))
            assert sum(g['pixels'] for g in groups) == int(reliable.sum())
            unknown = summarize(active & ~known[sl], delta, baseline, hybrid)
        rows.append(dict(camera=target, cost_groups=groups, unknown=unknown, inputs=inputs))
        print('edge_position_diagnostic_complete', target, flush=True)
        write(root / 'evaluation' / f'{name}_edge_position_association_partial.json', dict(complete=False, rows=rows))
    write(root / 'evaluation' / f'{name}_edge_position_association.json', dict(candidate=name,
        usage='evaluation_only_never_construction', code_sha256=sha(Path(__file__)), rows=rows,
        limitation='LPIPS receptive fields span regions; correlations are not causal. No gate, labels or prediction are emitted.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', default='single_source_contradiction400_once_seam')
    a = p.parse_args()
    run(a.root, a.name)
