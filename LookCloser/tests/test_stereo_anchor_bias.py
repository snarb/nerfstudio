import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from stereo_anchor_bias import spatial_folds, fit_and_validate


def anchors():
    y, x = np.mgrid[0:768:4, 0:768:4]
    uv = np.column_stack((x.ravel(), y.ravel()))
    expected = 100 + uv[:, 1] * .02
    return uv, expected


def test_constant_offset_transfers_to_unseen_blocks():
    uv, expected = anchors()
    result = fit_and_validate(uv, expected - 2, expected, 800, -1100)
    assert result['status'] == 'passes_anchor_bias_gate'
    assert result['correction'] == 2
    assert result['corrected_depth_error']['p90'] < 1e-12


def test_spatially_inconsistent_error_is_not_global_bias():
    uv, expected = anchors(); fold, _ = spatial_folds(uv)
    result = fit_and_validate(uv, expected + np.where(fold % 2, 5, -5), expected, 800, -1100)
    assert result['status'] == 'reject_bias_only_model'


def test_large_or_insufficient_fit_fails_closed():
    uv, expected = anchors()
    assert fit_and_validate(uv, expected - 10, expected, 800, -1100)['status'] == 'reject_bias_only_model'
    assert fit_and_validate(uv[:10], expected[:10], expected[:10], 800, -1100)['status'] == 'insufficient_spatial_anchors'
