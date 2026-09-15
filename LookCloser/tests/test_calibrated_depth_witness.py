from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrated_depth_witness import response_gains


def test_renderer_gain_gauge_and_global_channel_offset_invariance():
    a=np.array([[.2,.1,-.3],[-.1,.3,.4],[.25,-.2,.1]],dtype=np.float64)
    expected=np.exp(a-a.mean(0))
    np.testing.assert_array_equal(response_gains(a),expected)
    np.testing.assert_allclose(response_gains(a+[2,-3,1]),expected,rtol=1e-14,atol=0)
    np.testing.assert_allclose(np.prod(response_gains(a),axis=0),np.ones(3),rtol=1e-14)
    np.testing.assert_array_equal(response_gains(a,False),np.exp(a))


def test_rejects_invalid_profiles():
    for a in [np.zeros(3),np.zeros((4,2)),np.full((3,3),np.nan)]:
        with pytest.raises(ValueError):response_gains(a)
