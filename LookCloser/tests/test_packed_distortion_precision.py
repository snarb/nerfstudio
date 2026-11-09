"""Analytic thin-surface invariants against an independent quadratic oracle."""
import pytest
import torch
from nerfstudio.models.lookcloser import LookCloserModel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='nerfacc packed metadata requires CUDA')


def pairwise(starts,ends,weights):
    mid=(starts.double()+ends.double())*.5;w=weights.double()
    return (w[:,None]*w[None,:]*(mid[:,None]-mid[None,:]).abs()).sum()+(w.square()*(ends.double()-starts.double())/3).sum()


def test_thin_ray_objective_and_gradient_are_independent_of_4095_unrelated_opaque_rays():
    results=[]
    for prefix in [0,4095]:
        starts=torch.full((prefix+2,),.91298,device='cuda');ends=starts+.00002
        starts[-2:]=torch.tensor([.64701,.64703],device='cuda');ends[-2:]=starts[-2:]+.00001
        weights=torch.ones(prefix+2,device='cuda');weights[-2:]=torch.tensor([.37,.63],device='cuda');weights.requires_grad_()
        ids=torch.cat((torch.arange(prefix,device='cuda'),torch.full((2,),prefix,device='cuda')))
        actual=LookCloserModel._packed_distortion_loss(starts[:,None],ends[:,None],weights[:,None],ids,prefix+1)[-1,0]
        expected=pairwise(starts[-2:],ends[-2:],weights[-2:])
        gradient=torch.autograd.grad(actual,weights,retain_graph=True)[0]
        oracle_gradient=torch.autograd.grad(expected,weights)[0]
        assert actual.dtype==weights.dtype and actual>=0
        torch.testing.assert_close(actual.double(),expected,rtol=1e-6,atol=1e-11)
        torch.testing.assert_close(gradient,oracle_gradient,rtol=1e-5,atol=1e-10)
        results.append((actual.detach(),gradient[-2:]))
    for a,b in zip(results[0],results[1]):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-10)


def test_ragged_empty_and_dense_rays_match_pairwise_values_and_gradients():
    counts=[0,1,7,0,17];starts=[];ends=[];indices=[]
    for ray,count in enumerate(counts):
        if not count:continue
        edge=torch.linspace(.1,.9,count+1,device='cuda');starts.append(edge[:-1]);ends.append(edge[1:]);indices.extend([ray]*count)
    starts,ends=torch.cat(starts),torch.cat(ends);ids=torch.tensor(indices,device='cuda')
    weights=torch.linspace(.001,.05,len(ids),device='cuda').requires_grad_()
    actual=LookCloserModel._packed_distortion_loss(starts[:,None],ends[:,None],weights[:,None],ids,len(counts))[:,0]
    expected=torch.stack([pairwise(starts[ids==i],ends[ids==i],weights[ids==i]) for i in range(len(counts))])
    torch.testing.assert_close(actual.double(),expected,rtol=1e-6,atol=1e-10)
    a=torch.autograd.grad(actual.sum(),weights,retain_graph=True)[0];b=torch.autograd.grad(expected.sum(),weights)[0]
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-9)
