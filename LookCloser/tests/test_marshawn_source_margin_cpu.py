import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_marshawn_source_margin_cpu import admitted, integrate
from marshawn_source_point_admission import SourcePointAdmission


def test_zero_margin_matches_proven_selector_and_nested_buffers():
    rng = np.random.default_rng(19)
    mask = rng.random((11, 9, 8)) > .8
    bounds = np.array([[-.84, -.21, -.07], [.15, .42, .63]])
    points = rng.uniform(bounds[0]-.01, bounds[1]+.01, (5000, 3)).astype(np.float32)
    zero = admitted(points, mask, bounds, margin=0)
    expected = SourcePointAdmission(torch.from_numpy(mask), bounds)(torch.from_numpy(points)).numpy()
    np.testing.assert_array_equal(zero, expected)
    for margin in [.0005, .001, .002]:
        new = admitted(points, mask, bounds, margin=margin)
        assert new[zero].all()
        zero = new


def test_geometric_distance_not_whole_cell_dilation_and_unknown_unchanged():
    mask = np.zeros((4, 4, 4), bool); mask[2, 2, 2] = True
    bounds = [[0, 0, 0], [.04, .04, .04]]
    points = [[.019, .019, .019], [.0189, .025, .025], [.015, .025, .025], [-.001, 0, 0]]
    assert admitted(points, mask, bounds, margin=.001).tolist() == [True, False, False, True]
    assert admitted([[.025]*3], mask, bounds, margin=0).all()


def test_half_pitch_cap_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        admitted([[0, 0, 0]], np.ones((2, 2, 2), bool), [[0]*3, [.01]*3], margin=.0025)
    with pytest.raises(ValueError):
        admitted([[np.nan, 0, 0]], np.ones((2, 2, 2), bool), [[0]*3, [1]*3], margin=0)


def test_integral_identity_removal_and_shape_guard():
    start = np.array([0., 1.]); end = start+1; sigma = np.array([1., 2.]); rgb = np.array([[1., 0, 0], [0, 1., 0]])
    weights, color = integrate(start, end, sigma, rgb, np.ones(2, bool))
    np.testing.assert_allclose(weights, [1-np.exp(-1), np.exp(-1)*(1-np.exp(-2))])
    removed, _ = integrate(start, end, sigma, rgb, np.array([False, True]))
    np.testing.assert_allclose(removed, [0, 1-np.exp(-2)])
    with pytest.raises(ValueError):
        integrate(start, end, sigma, rgb, np.ones(1, bool))


def test_empty_native_ray_is_zero_not_an_occupied_or_unknown_claim():
    weights, rgb = integrate(np.empty(0), np.empty(0), np.empty(0), np.empty((0, 3)), np.empty(0, bool))
    assert weights.shape == (0,)
    np.testing.assert_array_equal(rgb, np.zeros(3))
