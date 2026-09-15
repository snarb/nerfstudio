import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_interior_texture_sources import prefer_interior


def test_preference_never_creates_visibility_or_loses_last_source():
    q=np.array([[10.,5.,0.,0.],[2.,0.,0.,2.],[0.,1.,0.,0.]])
    m=np.array([[3.,1.,50.,1.],[20.,30.,60.,1.],[100.,2.,50.,100.]])
    out=prefer_interior(q,m)
    np.testing.assert_array_equal(out[:,0],[0.,2.,0.])
    np.testing.assert_array_equal(out[:,1:],q[:,1:])
    assert np.array_equal((q>0).any(0),(out>0).any(0))
    assert not ((out>0)&(q==0)).any()


def test_uniformly_interior_unchanged_and_threshold_inclusive():
    q=np.array([[1.,2.],[3.,4.]])
    np.testing.assert_array_equal(prefer_interior(q,np.full_like(q,16.)),q)


def test_invalid_evidence_rejected():
    with pytest.raises(ValueError):prefer_interior(np.ones((2,1)),np.ones((1,2)))
    with pytest.raises(ValueError):prefer_interior(np.array([[np.nan]]),np.ones((1,1)))
    with pytest.raises(ValueError):prefer_interior(np.array([[-1.]]),np.ones((1,1)))
