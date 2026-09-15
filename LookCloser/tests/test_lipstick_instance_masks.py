"""Coordinate/API controls; these tests do not certify semantic mask quality."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from study_lipstick_instance_masks import binary_masks, portrait_xy
from audit_lipstick_instance_witnesses import signed_mask_distance


def test_portrait_pixels_match_numpy_rotation():
    a = np.arange(15).reshape(3, 5)
    for y in range(3):
        for x in range(5):
            px, py = portrait_xy([x, y], a.shape[1])
            assert np.rot90(a)[py, px] == a[y, x]


def test_float_thresholded_sam_api_masks_become_boolean():
    masks = np.array([[[0., 1.], [1., 0.]]], dtype=np.float32)
    result = binary_masks(masks)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, masks == 1)


@pytest.mark.parametrize('values', [[.1, .9], [np.nan], [np.inf], [-1]])
def test_unthresholded_or_nonfinite_masks_fail_closed(values):
    with pytest.raises(ValueError):
        binary_masks(values)


def test_signed_distance_separates_interior_and_exterior():
    mask = np.zeros((15, 15), np.uint8)
    mask[4:11, 4:11] = 1
    d = signed_mask_distance(mask)
    assert d[7, 7] >= 4 and d[0, 0] < -2
    assert d[4, 7] == 1 and d[3, 7] == -1
