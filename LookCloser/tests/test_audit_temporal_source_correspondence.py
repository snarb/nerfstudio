from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_temporal_source_correspondence import fit_residual_trend


def test_known_time_offset_and_zero_offset_are_recovered():
    x=np.arange(-2,3)
    for offset in [-.75,0.,1.25]:
        result=fit_residual_trend(x,.6*(x-offset))
        assert result['qualified'] and abs(result['zero_crossing_available_frames']-offset)<1e-10


def test_static_or_inconsistent_residual_does_not_identify_a_time_offset():
    x=np.arange(-2,3)
    assert not fit_residual_trend(x,np.full(5,.7))['qualified']
    assert not fit_residual_trend(x,np.array([1,-1,1,-1,1]))['qualified']


def test_duplicate_or_nonfinite_temporal_observations_fail():
    with pytest.raises(ValueError):fit_residual_trend([0,0,1],[0,1,2])
    with pytest.raises(ValueError):fit_residual_trend([0,1,2],[0,float('nan'),1])
