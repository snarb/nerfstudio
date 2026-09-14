"""Matched boundary-feather control; fixed train skin ROI, never face metrics."""
from pathlib import Path
import argparse
import review_confidence_boundary_completion as scorer
from joint_temporal_texture import read, sha, atomic_json


def run(output, frames, shape):
    scorer.ROOTS['all_edges'] = Path('/mnt/data/dec5_forearm_contrastive_guard_supported' if shape == 'plane'
                                    else '/mnt/data/dec5_forearm_admission_quadric_bounded')
    scorer.ROOTS['observed_ring'] = Path('/mnt/data/dec5_forearm_' + shape + '_observed_ring')
    scorer.run(output, frames, variants=['all_edges', 'observed_ring'])
    result = read(output / 'metrics.json')
    result.update(wrapper_sha256=sha(__file__), admission_shape=shape)
    atomic_json(output / 'metrics.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', choices=['plane', 'quadric'], required=True)
    p.add_argument('--frames', nargs='+', default=['001037'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.output, a.frames, a.shape)
