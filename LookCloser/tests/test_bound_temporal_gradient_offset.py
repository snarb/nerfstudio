import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bound_temporal_gradient_offset import bounded_rgb


def test_bounded_correction_preserves_support_and_nonzero_channels():
    before=np.array([[[1,3,255],[32,60,120]],[[0,0,0],[1,200,254]]],np.uint8)
    offset=np.array([[[-2,-2,2],[.01,-.02,.03]],[[1,1,1],[-1,-1,-1]]])
    support=np.array([[True,True],[False,False]])
    result=bounded_rgb(before,offset,support)
    assert np.array_equal(result[~support],before[~support])
    assert (result[before>0]>0).all()
    assert np.array_equal(result[0,1],np.rint(before[0,1]+offset[0,1]*255).astype(np.uint8))


def test_zero_offset_identity_and_invalid_inputs():
    rgb=np.arange(72,dtype=np.uint8).reshape(4,6,3);mask=np.ones((4,6),bool)
    assert np.array_equal(bounded_rgb(rgb,np.zeros_like(rgb,float),mask),rgb)
    with pytest.raises(ValueError):bounded_rgb(rgb,np.full_like(rgb,np.nan,float),mask)
    with pytest.raises(ValueError):bounded_rgb(rgb,np.zeros_like(rgb,float),mask,maximum_ratio=.5)
