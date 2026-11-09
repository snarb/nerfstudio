import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.marshawn_packed_distortion_candidate import (
    PackedDistortionCapture, load_frozen_formula, normalized_spacing,
    packet_terms, ray_aabb_intervals,
)

SOURCE = Path('/fsx/tmp/marshawn_001660_hd/code_sfm/nerfstudio/models/lookcloser.py')


@pytest.fixture
def formula():
    return load_frozen_formula(SOURCE, hashlib.sha256(SOURCE.read_bytes()).hexdigest())


def quadratic(starts, ends, weights, indices, num_rays):
    rows=[]
    for ray in range(num_rays):
        keep=indices==ray
        s,e,w=starts[keep].reshape(-1),ends[keep].reshape(-1),weights[keep].reshape(-1)
        mid=(s+e)/2
        rows.append((w[:,None]*w[None,:]*(mid[:,None]-mid[None,:]).abs()).sum()
                    +(w.square()*(e-s)/3).sum())
    return torch.stack(rows)[:,None]


def test_frozen_value_and_weight_gradient_with_empty_rays(formula):
    indices=torch.tensor([0,0,2,4,4,4])
    starts=torch.tensor([.1,.5,.2,.01,.3,.8],dtype=torch.float64)[:,None]
    ends=starts+.02
    w=torch.tensor([.2,.7,.5,.1,.2,.6],dtype=torch.float64,requires_grad=True)[:,None]
    actual=formula(starts,ends,w,indices,6)
    expected=quadratic(starts,ends,w,indices,6)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),w,retain_graph=True)[0],
                               torch.autograd.grad(expected.sum(),w)[0],rtol=1e-12,atol=1e-14)
    assert actual[[1,3,5]].count_nonzero()==0


@pytest.mark.parametrize('num_rays',[0,4])
def test_all_empty_packet_has_finite_zero_gradient(formula,num_rays):
    w=torch.empty((0,1),dtype=torch.float64,requires_grad=True)
    packet=dict(starts=w.detach(),ends=w.detach(),weights=w,ray_indices=torch.empty(0,dtype=torch.long),
                num_rays=num_rays,near=torch.full((num_rays,),.01),far=torch.ones(num_rays))
    for term in packet_terms(packet,formula).values():
        assert term.item()==0
        assert torch.autograd.grad(term,w,retain_graph=True)[0].numel()==0


def test_frozen_formula_rejects_changed_source():
    with pytest.raises(ValueError,match='SHA'):
        load_frozen_formula(SOURCE,'0'*64)


@pytest.mark.parametrize('weight',[float('nan'),float('inf'),-.01])
def test_invalid_weight_fails_before_formula(formula,weight):
    w=torch.empty((0,1),requires_grad=True)
    packet=dict(starts=w,ends=w,weights=w,ray_indices=torch.empty(0,dtype=torch.long),
                num_rays=0,near=torch.empty(0),far=torch.empty(0))
    with pytest.raises(ValueError,match='Finite'):
        packet_terms(packet,formula,weight)


def test_thin_ray_after_opaque_packet_stays_positive(formula):
    n=4096
    ids=torch.cat([torch.zeros(n,dtype=torch.long),torch.ones(2,dtype=torch.long)])
    starts=torch.cat([torch.linspace(0,.99,n,dtype=torch.float64),torch.tensor([.2,.8],dtype=torch.float64)])[:,None]
    ends=starts+1e-4
    w=torch.cat([torch.full((n,),.9,dtype=torch.float64),torch.full((2,),1e-8,dtype=torch.float64)]).requires_grad_()[:,None]
    actual=formula(starts,ends,w,ids,2)[1].sum()
    expected=quadratic(starts[-2:],ends[-2:],w[-2:],torch.zeros(2,dtype=torch.long),1).sum()
    assert actual>0
    grad=torch.autograd.grad(actual,w,retain_graph=True)[0][-2:]
    ref=torch.autograd.grad(expected,w)[0][-2:]
    assert (grad>0).all()
    torch.testing.assert_close(grad,ref,rtol=1e-4,atol=1e-12)


def test_roi_and_inverse_depth_definition():
    o=torch.tensor([[0.,0.,0.],[2.,0.,0.]])
    d=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    n,f,valid=ray_aabb_intervals(o,d,torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]]),.01,1000)
    assert valid.tolist()==[True,False]
    assert n[0]==.01 and f[0]==1
    index=torch.tensor([0,0]);near=torch.tensor([.01]);far=torch.tensor([7.])
    t=torch.tensor([[2.],[4.]])
    linear,_=normalized_spacing(t,t+.003,index,near,far,'linear')
    inverse,_=normalized_spacing(t,t+.003,index,near,far,'inverse_depth')
    assert (inverse[1]-inverse[0]) < (linear[1]-linear[0])/50


class DummyNative(torch.nn.Module):
    def __init__(self):
        super().__init__();self.parameter=torch.nn.Parameter(torch.tensor(.5))
        self.renderer_depth=DummyDepth()
        self.occupancy_grid=SimpleNamespace(aabbs=torch.tensor([[-1.,-1.,-1.,1.,1.,1.]]))
        self.config=SimpleNamespace(near_plane=.01,far_plane=1000)
    def forward(self,ray_bundle):
        weights=self.parameter.sigmoid().reshape(1,1)
        samples=SimpleNamespace(frustums=SimpleNamespace(starts=torch.tensor([[.1]]),ends=torch.tensor([[.2]])))
        return self.renderer_depth(weights=weights,ray_samples=samples,ray_indices=torch.tensor([0]),num_rays=1)


class DummyDepth(torch.nn.Module):
    def forward(self,**kwargs):
        return kwargs['weights'].sum()


def test_hook_graph_rng_state_and_exception_cleanup(formula):
    model=DummyNative();model.train()
    ray=SimpleNamespace(origins=torch.zeros(1,3),directions=torch.tensor([[0.,0.,1.]]))
    rng=torch.get_rng_state().clone();state={k:v.clone()for k,v in model.state_dict().items()}
    hooks=(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks))
    with PackedDistortionCapture(model)as capture:
        model(ray)
        assert capture.calls==1 and capture.packet['weights'].grad_fn is not None
        term=packet_terms(capture.packet,formula)['linear']
        assert torch.autograd.grad(term,model.parameter)[0].isfinite()
    assert hooks==(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(state[k],v)for k,v in model.state_dict().items())
    with pytest.raises(RuntimeError):
        with PackedDistortionCapture(model):
            raise RuntimeError('probe failure')
    assert hooks==(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks))
    model.eval()
    with PackedDistortionCapture(model)as capture:
        model(ray)
        assert capture.calls==0
