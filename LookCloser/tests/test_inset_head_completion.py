import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_inset_head_completion import inset_vertices


def test_inset_is_bounded_and_does_not_mutate():
    v = np.array([[1., 0, 0], [0., 2., 0], [0., 0., 0]])
    original = v.copy()
    value = inset_vertices(v, np.zeros(3), .1)
    np.testing.assert_allclose(value, [[.9, 0, 0], [0, 1.9, 0], [0, 0, 0]])
    np.testing.assert_array_equal(v, original)
    np.testing.assert_array_equal(inset_vertices(v, np.zeros(3), 0), v)
    assert np.linalg.norm(inset_vertices(v, np.zeros(3), 10)[0]) == .75
