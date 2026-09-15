import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_mhr_admission_stages import stage_ids


def test_maps_semantic_indices_to_original_proposals():
    stages = stage_ids(10, np.array([2, 5, 9]), np.array([True, False, False]),
        np.array([True, False, True]), np.array([2]), np.array([9]))
    np.testing.assert_array_equal(stages['interpolated_initial'], [2, 9])
    np.testing.assert_array_equal(stages['interpolated_final'], [9])


def test_rejects_final_outside_initial():
    with pytest.raises(AssertionError):
        stage_ids(10, np.array([2, 5]), np.array([True, False]),
            np.array([True, True]), np.array([5]), np.array([2]))
