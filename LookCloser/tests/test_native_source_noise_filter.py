from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_source_noise_filter import high_frequency_floor,filter_native_rgb


def test_haar_floor_scales_with_independent_channel_noise():
    rng=np.random.default_rng(12)
    rgb=np.rint(120+rng.normal(0,3,(256,256,3))).clip(0,255).astype(np.uint8)
    floor=high_frequency_floor(rgb)
    expected=3*np.linalg.norm([.2126,.7152,.0722])
    assert abs(floor['sigma_rgb8']-expected)<.3
    assert not floor['physical_sensor_noise_identified']


def test_constant_image_and_disabled_filter_are_exact():
    image=np.full((128,128,3),123,np.uint8)
    out,stats=filter_native_rgb(image,1)
    np.testing.assert_array_equal(out,image)
    noisy=np.random.default_rng(1).integers(0,256,(32,32,3),dtype=np.uint8)
    out,stats=filter_native_rgb(noisy,0)
    np.testing.assert_array_equal(out,noisy)
    assert stats['exact_identity']


def test_noise_is_reduced_without_moving_step_boundary():
    rng=np.random.default_rng(4)
    reference=np.full((128,128,3),90.);reference[:,64:]=160
    noisy=np.rint(reference+rng.normal(0,2,reference.shape)).clip(0,255).astype(np.uint8)
    out,stats=filter_native_rgb(noisy,2,floor={'sigma_rgb8':2.})
    assert np.mean((out-reference)**2)<.4*np.mean((noisy-reference)**2)
    assert abs(float(out[:,:64].mean())-90)<.2 and abs(float(out[:,64:].mean())-160)<.2
    assert np.max(out[:,63])<110 and np.min(out[:,64])>140
    assert not stats['source_camera_averaging'] and not stats['random_detail_added']


@pytest.mark.parametrize('strength',[-1,3,np.nan,np.inf])
def test_invalid_strength_fails_closed(strength):
    with pytest.raises(ValueError):filter_native_rgb(np.zeros((32,32,3),np.uint8),strength)
