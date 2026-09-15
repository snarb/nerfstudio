from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_lipstick_near_protection import protected_faces


def test_raw_near_observation_and_corroborated_observation_are_distinct():
    near = np.zeros((3, 2, 4), bool)
    count = np.zeros_like(near, dtype=int)
    near[0, 0, 3] = True
    near[1, 1, 0] = True
    count[1, 1, 0] = 2
    np.testing.assert_array_equal(protected_faces(near, count, 0), [True, True])
    np.testing.assert_array_equal(protected_faces(near, count, 1), [False, True])
    np.testing.assert_array_equal(protected_faces(near, count, 3), [False, False])


def test_corroboration_without_a_near_measurement_never_protects():
    near = np.zeros((3, 1, 4), bool)
    count = np.full(near.shape, 20)
    assert not protected_faces(near, count, 1).any()


def test_invalid_layout_or_threshold_rejected():
    with pytest.raises(ValueError):
        protected_faces(np.zeros((2, 4)), np.zeros((2, 4)), 0)
    with pytest.raises(ValueError):
        protected_faces(np.zeros((3, 2, 4)), np.zeros((3, 2, 4)), -1)
