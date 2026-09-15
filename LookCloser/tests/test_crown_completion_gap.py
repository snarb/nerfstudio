from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_crown_completion_gap import stages


def test_nested_stage_labels():
    np.testing.assert_array_equal(stages(5,[1,2,3],[2,3],[3]),[0,1,2,3,0])
    with pytest.raises(ValueError):stages(3,[1],[2],[])
    with pytest.raises(ValueError):stages(3,[1],[1],[2])
    with pytest.raises(ValueError):stages(3,[-1],[],[])
