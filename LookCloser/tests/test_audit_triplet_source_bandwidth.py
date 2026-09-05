from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_triplet_source_bandwidth import instrumental_moments,solve_instrumental_variance


@pytest.mark.parametrize('sigma',[0.,.6])
def test_independent_camera_noise_does_not_become_blur_in_instrumental_moments(sigma):
    m=np.zeros((2,2));v=np.zeros(2)
    for seed in range(100):
        rng=np.random.default_rng(seed)
        scene=cv2.GaussianBlur(rng.normal(0,1,(64,64)).astype(np.float32),(0,0),1.2)*.1+.4
        a=scene+rng.normal(0,.015,scene.shape)
        b=(cv2.GaussianBlur(scene,(0,0),sigma) if sigma else scene)+rng.normal(0,.003,scene.shape)
        c=scene+rng.normal(0,.008,scene.shape)
        matrix,rhs=instrumental_moments(a,b,c);m+=matrix;v+=rhs
    result=solve_instrumental_variance(m,v)
    assert result['valid'] and abs(result['variance']-sigma*sigma)<.05


def test_offset_and_exposure_are_not_blur_when_three_sources_have_shared_texture():
    rng=np.random.default_rng(5);a=cv2.GaussianBlur(rng.random((48,48)),(0,0),.8);b=a*1.3+.2;c=a*.7-.1
    result=solve_instrumental_variance(*instrumental_moments(a,b,c))
    assert result['valid'] and abs(result['gain']-1.3)<1e-10 and abs(result['variance'])<1e-10


def test_flat_and_nonfinite_instruments_fail_closed():
    a=np.ones((48,48))
    assert not solve_instrumental_variance(*instrumental_moments(a,a,a))['valid']
    a[0,0]=float('nan')
    with pytest.raises(ValueError):instrumental_moments(a,a,a)


def test_ill_conditioned_moments_are_not_reported_as_a_bandwidth_measurement():
    result=solve_instrumental_variance(np.diag([1.,1e-5]),np.array([1.,0]))
    assert not result['valid'] and result['reason']=='ill_conditioned'
