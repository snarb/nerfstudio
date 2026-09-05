from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_camera_color_fits_common_samples import held_pair_residuals


def test_comparison_uses_identical_declared_held_points_only():
    colors=np.zeros((2,6,3));colors[1]=.2
    valid=np.ones((2,6),bool);held=np.array([True,False,True,False,True,False])
    valid[1,4]=False
    expected=held_pair_residuals(colors,valid,held,[(0,1)],['a','b'])
    assert expected['count']==2 and np.isclose(expected['display_l1_median'],.2)
    colors[:,~held]=1000;colors[:,4]=np.nan
    assert held_pair_residuals(colors,valid,held,[(0,1)],['a','b'])==expected


def test_comparison_fails_on_missing_or_nonfinite_held_evidence():
    colors=np.zeros((2,4,3));valid=np.ones((2,4),bool);held=np.ones(4,bool)
    colors[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        held_pair_residuals(colors,valid,held,[(0,1)],['a','b'])
    with pytest.raises(ValueError,match='common held'):
        held_pair_residuals(colors,valid,held,[],['a','b'])
