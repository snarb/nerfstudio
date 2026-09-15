import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from admit_mhr_local_patch_depth import interpolation_admission
from guard_jaw_measured_depth import initial_admission
from local_surface_certificate import certify

def test_direct_guard_requires_real_sample_support():
    votes=np.ones((2,10),int);votes[0,:]=2;free=np.zeros((3,2,10),bool)
    np.testing.assert_array_equal(initial_admission(votes,free,np.array([2,2]),np.array([0,0])),[True,False])

def test_interpolation_requires_every_vertex_and_keeps_veto():
    pp=np.array([[0,1,2],[1,2,3]]);cert=np.array([True,True,True,False]);free=np.zeros((3,2,10),bool)
    keep,_=interpolation_admission(np.array([False,False]),cert,pp,free,np.array([2,2]),np.array([0,0]));np.testing.assert_array_equal(keep,[True,False])
    free[0,0,4]=True
    keep,_=interpolation_admission(np.array([False,False]),cert,pp,free,np.array([2,2]),np.array([0,0]));assert not keep.any()

def test_interpolation_never_overrides_available_semantic_disagreement():
    keep,_=interpolation_admission(np.array([False]),np.ones(3,bool),np.array([[0,1,2]]),np.zeros((2,1,10),bool),np.array([20]),np.array([1]));assert not keep[0]

def test_seed_certificate_rejects_extrapolation_and_offset():
    x,y=np.meshgrid(np.linspace(-.002,.002,5),np.linspace(-.002,.002,5));seeds=np.c_[x.ravel(),y.ravel(),np.zeros(25)]
    assert certify([0,0,0],[0,0,1],seeds)[0]
    assert not certify([.004,0,0],[0,0,1],seeds)[0]
    assert not certify([0,0,.001],[0,0,1],seeds)[0]
