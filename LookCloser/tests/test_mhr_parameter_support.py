import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_mhr_head_parameter_support import differences


def test_centered_linear_and_quadratic_components():
    base=np.arange(12,dtype=float).reshape(4,3)
    linear=np.ones((2,4,3));linear[1]*=2
    quadratic=np.ones_like(linear)*.25
    d,r=differences(base,base+linear+quadratic,base-linear+quadratic)
    np.testing.assert_allclose(d,linear)
    np.testing.assert_allclose(r,quadratic)


def test_reject_nonfinite_or_bad_shape():
    with pytest.raises(ValueError):differences(np.zeros((4,3)),np.zeros((2,4,3)),np.zeros((1,4,3)))
    with pytest.raises(ValueError):differences(np.zeros((4,3)),np.full((2,4,3),np.nan),np.zeros((2,4,3)))
