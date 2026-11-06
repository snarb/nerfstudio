"""Occluding non-actor surfaces must not carve hidden actor geometry."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from marshawn_semantic_occluders import merge_occluder_unknown
from mask_visual_hull_aabb import MaskCamera, classify_cells


def test_foreground_mat_and_bag_cannot_vote_clear_behind_surface():
    size = 32
    support = np.zeros((size, size), bool)
    unknown = np.zeros_like(support)
    valid = np.ones_like(support)
    pose = np.eye(4)
    cell_lo = np.array([[-.02, -.02, -2.02]])
    cell_hi = np.array([[.02, .02, -1.98]])
    before = MaskCamera('TRAIN', pose, 20, 20, 16, 16,
                        support, unknown, valid).prepare(0)
    assert classify_cells([before], cell_lo, cell_hi)[2][0] == 1
    mat = np.zeros_like(support)
    mat[14:19, 14:19] = True
    fg, veto, eligible, receipt = merge_occluder_unknown(
        support, unknown, valid, {'opaque_mat_or_bag': mat})
    after = MaskCamera('TRAIN', pose, 20, 20, 16, 16,
                       fg, veto, eligible).prepare(0)
    assert classify_cells([after], cell_lo, cell_hi)[2][0] == 0
    assert not unknown.any() and not support.any()
    assert receipt['removed_clear_pixels'] == 25


def test_uncertain_contact_preserves_actor_support_and_no_new_clear_pixels():
    support = np.eye(8, dtype=bool)
    unknown = np.zeros_like(support)
    valid = np.ones_like(support)
    occluder = np.zeros_like(support)
    occluder[2:5, 2:5] = True
    fg, veto, _, receipt = merge_occluder_unknown(
        support, unknown, valid, {'contact': occluder})
    assert np.array_equal(fg, support)
    assert np.array_equal(veto, occluder)
    assert receipt['sources']['contact']['overlaps_support'] == 3
    assert receipt['removed_clear_pixels'] == 6


if __name__ == '__main__':
    test_foreground_mat_and_bag_cannot_vote_clear_behind_surface()
    test_uncertain_contact_preserves_actor_support_and_no_new_clear_pixels()
    print('2 source occlusion regression checks passed')
