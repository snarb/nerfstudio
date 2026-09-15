import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_surface_certificate import certify


def seeds():
    x,y=np.meshgrid(np.linspace(-.002,.002,5),np.linspace(-.002,.002,5))
    return np.column_stack([x.ravel(),y.ravel(),100*(x*x+y*y).ravel()])


def test_interpolation_and_extrapolation():
    ok,stats=certify([0,0,0],[0,0,1],seeds())
    assert ok and stats['loo_p90']<1e-12
    assert not certify([.004,0,0],[0,0,1],seeds())[0]


def test_rejects_missing_and_inconsistent_support():
    assert not certify([0,0,0],[0,0,1],seeds()[:5])[0]
    noisy=seeds();noisy[:,2]+=np.where(np.arange(len(noisy))%2,.002,-.002)
    assert not certify([0,0,0],[0,0,1],noisy)[0]
