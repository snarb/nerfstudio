from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_multitime_rig_holdout import symmetric_epipolar_error


def test_symmetric_distance_is_direction_and_scale_invariant():
    matrix = np.array([[0., 0, 0], [0, 0, -1.], [0, 1., 0]])
    a = np.array([[10., 20.], [30., 40.]])
    b = np.array([[12., 21.], [60., 43.]])
    expected = np.array([1., 3.])
    np.testing.assert_allclose(symmetric_epipolar_error(matrix, a, b), expected)
    np.testing.assert_allclose(symmetric_epipolar_error(-matrix.T * 12, b, a), expected)
