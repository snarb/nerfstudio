import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prune_measured_free_surface import near_tap_evidence, removable


def test_any_near_tap_protects_even_when_rest_missing():
    d=np.zeros((7,7),np.float32);d[1,1]=2
    assert near_tap_evidence(d,np.array([[3.,3.]]),np.array([2.]))[0]
    assert not near_tap_evidence(d,np.array([[4.,4.]]),np.array([2.]))[0]
    assert not near_tap_evidence(d,np.array([[3.,3.]]),np.array([1.]))[0]
    assert not near_tap_evidence(d,np.array([[3.,3.]]),np.array([2.]),radius=0)[0]
    assert near_tap_evidence(d,np.array([[1.,1.]]),np.array([2.]),radius=0)[0]


def test_all_four_samples_and_both_far_certificates_required():
    near=np.zeros((4,4),int);stable=np.full((4,4),6);trusted=stable.copy()
    near[1,2]=1;stable[2,3]=5;trusted[3,0]=5
    assert removable(near,stable,trusted).tolist()==[True,False,False,False]
    with pytest.raises(ValueError):removable(near[:,:3],stable,trusted)


def test_missing_invalid_and_outside_are_not_near_measurements():
    d=np.zeros((7,7));d[3,3]=np.nan
    assert not near_tap_evidence(d,np.array([[3.,3.],[np.nan,3.],[-100.,0.]]),np.ones(3)).any()
