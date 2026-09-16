import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_query_support_quorum import removable


def test_requires_all_four_queries_and_all_far_gates():
    near=np.array([[0,2,0,0],[0,3,0,0],[0,2,0,0],[0,2,0,0]])
    stable=np.full((4,4),6);far=stable.copy();stable[2,2]=5;far[3,3]=5
    np.testing.assert_array_equal(removable(near,stable,far),[True,False,False,False])


def test_corroborating_measurement_does_not_increase_query_votes():
    near=np.array([[0,2,0,0]]);far=np.full((1,4),28)
    assert removable(near,far,far)[0]
    with pytest.raises(ValueError):removable(near,far[:,:3],far)
