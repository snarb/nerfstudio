from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_stereo_input_exposure import effective_gains,paired_ncc


def test_ncc_affine_brightness_invariance():
    a=np.arange(121,dtype=float)[None]/121
    ncc,usable=paired_ncc(a,a*.5+.2)
    assert usable.all() and np.allclose(ncc,1)


def test_flat_patch_is_penalized_not_removed():
    a=np.stack([np.ones(121),np.arange(121)/121])
    ncc,usable=paired_ncc(a,a)
    assert len(ncc)==2 and ncc[0]==-1 and not usable[0] and usable[1]


def test_ncc_rejects_changed_shape_and_nonfinite():
    with pytest.raises(ValueError):paired_ncc(np.zeros((2,4)),np.zeros((1,4)))
    with pytest.raises(ValueError):paired_ncc(np.array([[1,np.nan]]),np.ones((1,2)))


def test_ingest_cancel_has_one_train_only_exposure():
    values=effective_gains([2,8],[.5,2])
    assert np.allclose(values['original'],[2,8])
    assert np.allclose(values['cancel_ingest'],[4,4])
    assert np.allclose(values['scalar_fit'],[1,16])


@pytest.mark.parametrize('values',[[],[0,1],[np.inf,1]])
def test_invalid_gains_fail_closed(values):
    with pytest.raises(ValueError):effective_gains(values,np.ones(len(values)))
