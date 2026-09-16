from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_local_mhr_frontier import cohort


def test_skin_cohort_excludes_observed_and_non_skin_rays():
    d = np.ones((30, 30)); d[14, 14] = 0; d[2, 2] = 0; d[10, 10] = np.inf
    skin = np.zeros((30, 30), np.uint8); skin[5:25, 5:25] = 230
    mask, interior = cohort(d, [0, 0, 30, 30], skin)
    assert mask.sum() == interior.sum() == 2
    assert mask[14, 14] and mask[10, 10] and not mask[2, 2]


def test_crop_and_erosion_are_separate():
    d = np.ones((30, 30)); d[5, 5] = d[15, 15] = 0
    skin = np.zeros_like(d, dtype=np.uint8); skin[5:25, 5:25] = 255
    mask, interior = cohort(d, [0, 0, 30, 30], skin)
    assert mask.sum() == 2 and interior.sum() == 1
    mask, interior = cohort(d, [10, 10, 20, 20], skin)
    assert mask.sum() == interior.sum() == 1


def test_virtual_view_does_not_claim_skin_ground_truth():
    d = np.ones((20, 20)); d[10, 10] = 0; d[0:3, 5] = 0
    mask, interior = cohort(d, [0, 0, 20, 20])
    assert mask.sum() == 1 and mask[10, 10] and not interior.any()
