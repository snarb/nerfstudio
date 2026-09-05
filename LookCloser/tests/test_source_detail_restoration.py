from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from source_detail_restoration import relative_restoration_amount,restore_source_detail


def fixture():
    rgb=np.full((32,36,3),.4,np.float32)
    return rgb,np.ones(rgb.shape[:2],bool),np.ones(rgb.shape[:2]),np.full(rgb.shape[:2],.125)


def test_constant_color_and_zero_amount_are_identity():
    rgb,valid,depth,amount=fixture()
    out,_,_=restore_source_detail(rgb,valid,depth,amount)
    np.testing.assert_array_equal(out,rgb)
    rng=np.random.default_rng(1);rgb=rng.uniform(.1,.9,rgb.shape).astype(np.float32)
    out,_,_=restore_source_detail(rgb,valid,depth,amount*0)
    np.testing.assert_array_equal(out,rgb)


def test_guarded_pixels_do_not_read_hidden_rgb_or_another_depth_layer():
    rgb,valid,depth,amount=fixture();valid[:,18]=False;depth[16:]=2
    a,applied,_=restore_source_detail(rgb,valid,depth,amount)
    poisoned=rgb.copy();poisoned[:,18]=.9
    b,_,_=restore_source_detail(poisoned,valid,depth,amount)
    np.testing.assert_array_equal(a[valid],b[valid])
    assert not applied[:,17:20].any() and not applied[15:17].any()
    assert not applied[[0,-1]].any() and not applied[:,[0,-1]].any()


def test_relative_amount_uses_only_available_sources_and_cancels_gauge():
    variance=np.array([[0,0,0],[1,1,1],[-100,-100,-100]],float)
    valid=np.array([[1,1,0],[1,0,1],[0,0,0]],bool)
    a=relative_restoration_amount(variance,valid,.1)
    np.testing.assert_array_equal(a,[[0,0,0],[.125,0,0],[0,0,0]])
    np.testing.assert_array_equal(a,relative_restoration_amount(variance+17,valid,.1))
    np.testing.assert_array_equal(relative_restoration_amount(variance,np.zeros_like(valid),.1),0)


def test_small_gaussian_blur_is_partly_recovered_without_new_source_rgb():
    from scipy.ndimage import gaussian_filter
    y,x=np.mgrid[:64,:64];sharp=.3+.08*np.cos(x*.7)+.06*np.sin(y*.6)
    blurry=gaussian_filter(sharp,.6)
    from patchmatch_color_calibration import encode_exposed_linear,decode_exposed_linear
    rgb=np.repeat(encode_exposed_linear(blurry)[...,None],3,-1)
    out,_,_=restore_source_detail(rgb,np.ones((64,64),bool),np.ones((64,64)),np.full((64,64),.125))
    restored=decode_exposed_linear(out)[...,0]
    assert np.mean((restored[4:-4,4:-4]-sharp[4:-4,4:-4])**2)<np.mean((blurry[4:-4,4:-4]-sharp[4:-4,4:-4])**2)


@pytest.mark.parametrize('field',['rgb','depth','amount'])
def test_nonfinite_inputs_fail_closed(field):
    rgb,valid,depth,amount=fixture();args=dict(rgb=rgb,valid=valid,depth=depth,amount=amount)
    args[field].flat[0]=float('nan')
    with pytest.raises(ValueError):restore_source_detail(**args)
