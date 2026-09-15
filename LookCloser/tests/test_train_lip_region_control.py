import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from infer_train_lip_regions import cycles, area
from study_coherent_lip_texture import depth_samples


def test_closed_landmark_cycles_and_largest_area():
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4)]
    assert cycles(edges)==[[0,1,2,3],[4,5,6,7]]
    points=np.array([[0,0],[4,0],[4,3],[0,3]])
    assert area(points)==area(points[::-1])==12
    with pytest.raises(AssertionError):cycles([(0,1),(1,2)])


def test_integer_pixel_does_not_mix_zero_weight_neighbors():
    d=np.array([[3.,0.],[0.,0.]],np.float32)
    uv,center,taps,weight=depth_samples(d,np.array([[0.,0.]],np.float32))
    np.testing.assert_array_equal(center,[3.])
    assert (((taps>0)&(abs(taps-3)<.009))|(weight==0)).all()


def test_fractional_native_sampling_and_center_snap():
    d=np.array([[1.,2.],[3.,4.]],np.float32)
    uv,center,_,_=depth_samples(d,np.array([[.5,.5],[.0001,0]],np.float32))
    np.testing.assert_allclose(center,[2.5,1.])
    np.testing.assert_array_equal(uv[1],[0,0])
