"""Read-only measurement of conservative traversal cover versus source support.

Neither mask is physical occupancy. This audit never rewrites a checkpoint or
qualifies removal of a cell: it identifies the additional spatial allowance
introduced by applying a whole-cell traversal cover as a point density gate.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.ndimage import maximum_filter

from marshawn_semantic_cell_admission import WholeCellAdmission


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(source_path, archive_path):
    with np.load(source_path, allow_pickle=False) as z:
        source = z['semantic_gate_pad1'].copy()
        base = z['semantic_gate0'].copy()
        bounds = z['aabb'].copy()
    with np.load(archive_path, allow_pickle=False) as z:
        admission = z['admission'][0].copy()
        active = z['active'][0].copy()
        target_bounds = z['cascade_aabbs'][0].reshape(2, 3)
    assert source.dtype == admission.dtype == active.dtype == np.bool_
    assert source.shape == admission.shape == active.shape == (128, 128, 128)
    np.testing.assert_allclose(bounds, target_bounds, rtol=0, atol=2e-7)
    gate = WholeCellAdmission(source, bounds)
    np.testing.assert_array_equal(gate.map_target_grid(admission.shape, target_bounds), admission)
    assert not np.any(active & ~admission)
    twice = maximum_filter(source, size=3, mode='constant', cval=0)
    # Centers are far from source-cell boundaries. Query the original source
    # with its conservative epsilon; do not assume equal integer cell indices.
    centers = target_bounds[0] + (np.argwhere(active) + .5) / np.array(active.shape) * (target_bounds[1] - target_bounds[0])
    source_admits_active_centers = gate.point_admitted(centers)
    pitch = (target_bounds[1] - target_bounds[0]) / np.array(active.shape)
    return {
        'kind': 'READ_ONLY_TRAVERSAL_COVER_SPATIAL_ALLOWANCE_AUDIT',
        'source_sha256': digest(source_path),
        'actual_grid_sha256': digest(archive_path),
        'base_cells': int(base.sum()),
        'source_pad1_cells': int(source.sum()),
        'whole_cell_admission_cells': int(admission.sum()),
        'cover_equals_another_26_neighbor_dilation': bool(np.array_equal(twice, admission)),
        'cover_vs_second_dilation_xor': int(np.count_nonzero(twice ^ admission)),
        'active_cells': int(active.sum()),
        'active_centers_outside_original_source_support': int(np.count_nonzero(~source_admits_active_centers)),
        'active_center_outside_fraction': float(np.mean(~source_admits_active_centers)),
        'floor_frame_cell_pitch': pitch.tolist(),
        'source_support_volume': float(source.sum() * np.prod((bounds[1] - bounds[0]) / np.array(source.shape))),
        'covered_support_volume': float(admission.sum() * np.prod(pitch)),
        'physical_empty_cells_certified': 0,
        'production_changes': False,
        'interpretation': 'The mapped support conservatively covers intersecting source cells for traversal. Applying that entire cover to point density admits extra boundary space. This is intentional conservatism, not a proven implementation error or a certificate that the extra space is empty.',
        'candidate_requires': ['Keep conservative traversal cover unchanged.', 'Independently qualify source-coordinate point support with floating boundary tolerance and UNKNOWN policy.', 'Retain all positive witnesses, feet anchors, and whole TRAIN silhouette ray coverage.', 'Compare real renders and dense ray integration before any promotion.'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.archive)
    result['script_sha256'] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
