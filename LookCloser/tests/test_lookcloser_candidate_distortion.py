import torch
import pytest
from types import SimpleNamespace
from nerfstudio.models.lookcloser import LookCloserModel, LookCloserModelConfig

def test_distortion_derivative_is_twice_objective_under_uniform_weight_gain():
    w=torch.tensor([[[.2],[.3]]]);starts=torch.tensor([[[.1],[.6]]]);ends=starts+.1
    gain=torch.zeros(1,1,requires_grad=True)
    original=LookCloserModel._dense_distortion_loss(starts,ends,w)
    changed=LookCloserModel._dense_distortion_loss(starts,ends,w*gain.exp()[:,None]);changed.sum().backward()
    torch.testing.assert_close(gain.grad,2*original)


def test_neutral_distortion_keeps_value_and_shape_gradient_but_not_opacity_pressure():
    starts=torch.tensor([[[.1],[.6]]]);ends=starts+.1
    weights=torch.tensor([[[.2],[.3]]],requires_grad=True)
    raw=LookCloserModel._dense_distortion_loss(starts,ends,weights)
    neutral=LookCloserModel._opacity_neutral_distortion(raw,weights.sum(-2))
    torch.testing.assert_close(neutral,raw,atol=0,rtol=0)
    old=torch.autograd.grad(raw.sum(),weights,retain_graph=True)[0]
    new=torch.autograd.grad(neutral.sum(),weights)[0]
    torch.testing.assert_close((new*weights).sum(),torch.tensor(0.),atol=1e-7,rtol=0)
    tangent=torch.tensor([[[1.],[-1.]]])
    torch.testing.assert_close((old*tangent).sum(),(new*tangent).sum())


def test_neutral_distortion_empty_and_tiny_rays_remain_finite():
    weights=torch.tensor([[[0.],[0.]],[[1e-12],[2e-12]]],requires_grad=True)
    starts=torch.tensor([[[.1],[.6]]]).expand_as(weights);ends=starts+.1
    raw=LookCloserModel._dense_distortion_loss(starts,ends,weights)
    neutral=LookCloserModel._opacity_neutral_distortion(raw,weights.sum(-2));neutral.sum().backward()
    assert torch.isfinite(neutral).all() and torch.isfinite(weights.grad).all() and neutral[0]==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='nerfacc packed distortion requires CUDA')
def test_objective_dense_packed_parity_with_empty_ray_and_default_preserved():
    model=LookCloserModel.__new__(LookCloserModel);torch.nn.Module.__init__(model)
    model.device_indicator_param=torch.nn.Parameter(torch.empty(0,device='cuda'),requires_grad=False)
    model.config=LookCloserModelConfig(depth_loss_mult=0)
    starts=torch.tensor([[[.1],[.6]],[[.1],[.6]],[[.2],[.7]]],device='cuda');ends=starts+.1
    index=torch.tensor([0,0,2,2],device='cuda');selected=torch.tensor([0,2],device='cuda')
    reference=None
    for neutral in [False,True]:
        results=[]
        for packed in [False,True]:
            weight=torch.tensor([[[.2],[.3]],[[0.],[0.]],[[.1],[.15]]],device='cuda',requires_grad=True)
            rgb=torch.zeros(3,3,device='cuda');out=dict(rgb=rgb,loss_weights=weight,
                loss_ray_samples=SimpleNamespace(spacing_starts=starts,spacing_ends=ends))
            if packed:out.update(packed_weights=weight[selected].reshape(-1,1),packed_ray_indices=index,
                packed_spacing_starts=starts[selected].reshape(-1,1),packed_spacing_ends=ends[selected].reshape(-1,1))
            model.config.opacity_neutral_distortion=neutral
            value=model.get_loss_dict(out,dict(image=rgb))['distortion_loss']
            gradient=torch.autograd.grad(value,weight)[0];results.append((value.detach(),gradient))
        torch.testing.assert_close(results[0][0],results[1][0]);torch.testing.assert_close(results[0][1],results[1][1],atol=1e-8,rtol=1e-5)
        if not neutral:
            reference=results[0][0]
            expected=.01*LookCloserModel._dense_distortion_loss(starts,ends,weight.detach()).mean()
            torch.testing.assert_close(reference,expected,atol=0,rtol=0)
        else:torch.testing.assert_close(results[0][0],reference,atol=0,rtol=0)
