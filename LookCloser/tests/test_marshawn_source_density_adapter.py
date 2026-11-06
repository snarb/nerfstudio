"""Source point gating preserves native normalization and independent traversal state."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from marshawn_source_density_adapter import install_source_density_admission

class Field:
    def _normalize_positions(self,p):
        selector=((p>=0)&(p<=1)).all(-1)
        return p*2,selector,p.shape,p

def test_source_gate_changes_only_selector_and_keeps_native_outside_rejection():
    f=Field();mask=torch.zeros((2,2,2),dtype=torch.bool);mask[0,0,0]=True
    p=torch.tensor([[.25,.25,.25],[.75,.75,.75],[1.1,.2,.2]],requires_grad=True)
    before=f._normalize_positions(p);lookup=install_source_density_admission(f,mask,[[0,0,0],[1,1,1]])
    after=f._normalize_positions(p)
    for i in [0,3]:torch.testing.assert_close(before[i],after[i])
    assert before[2]==after[2]
    assert after[1].tolist()==[True,False,False]
    assert lookup(p).tolist()==[True,False,True] # Outside source UNKNOWN, native bounds still reject.
    after[0][after[1]].sum().backward()
    torch.testing.assert_close(p.grad,torch.tensor([[2.,2.,2.],[0.,0.,0.],[0.,0.,0.]]))
    with pytest.raises(ValueError,match='already'):install_source_density_admission(f,mask,[[0,0,0],[1,1,1]])

def test_allones_identity_rng_and_float64_source_bounds():
    f=Field();mask=torch.ones((2,2,2),dtype=torch.bool);p=torch.tensor([[.2,.3,.4],[-.1,0.,0.],[1.,1.,1.]])
    old=f._normalize_positions(p);rng=torch.get_rng_state().clone();lookup=install_source_density_admission(f,mask,[[0,0,0],[1,1,1]])
    got=f._normalize_positions(p)
    assert lookup.bounds.dtype==torch.float64
    assert torch.equal(rng,torch.get_rng_state())
    for i in [0,1,3]:torch.testing.assert_close(old[i],got[i])
