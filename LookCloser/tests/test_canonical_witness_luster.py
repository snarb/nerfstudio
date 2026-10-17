import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from canonical_witness_luster_hybrid import canonical_reference


def test_zero_preserves_reference_exactly_without_mutating_inputs():
    rng=np.random.default_rng(31)
    witness=rng.random((64,64,3),dtype=np.float32);base=witness*.7
    before=witness.copy();mask=np.ones((64,64),bool)
    got=canonical_reference(witness,base,mask,strength=0)
    assert np.array_equal(got,witness) and np.array_equal(witness,before)


def test_constant_view_color_is_removed():
    base=np.full((64,64,3),.4,np.float32)
    photo=base+np.array([.1,-.05,.2],np.float32)
    got=canonical_reference(photo,base,np.ones((64,64),bool))
    np.testing.assert_allclose(got,base,atol=2e-7)


def test_unsupported_color_cannot_affect_supported_reference():
    base=np.full((64,64,3),.4,np.float32);photo=base+.1
    mask=np.ones((64,64),bool);mask[20:24,20:24]=False
    changed=photo.copy();changed[~mask]=1000
    first=canonical_reference(photo,base,mask)
    second=canonical_reference(changed,base,mask)
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(second[~mask],base[~mask])


def test_detail_impulse_keeps_its_position():
    base=np.full((65,65,3),.4,np.float32);photo=base+.1
    photo[32,27]+=.1
    got=canonical_reference(photo,base,np.ones((65,65),bool))
    assert np.unravel_index(np.argmax(got[...,0]),got.shape[:2])==(32,27)
    assert got[32,27,0]>.49
