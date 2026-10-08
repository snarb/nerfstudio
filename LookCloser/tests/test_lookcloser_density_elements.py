from types import SimpleNamespace
import pytest
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.lookcloser_opacity import density_element_loss
from nerfstudio.models.lookcloser_opacity import LookCloserOpacityModel,LookCloserOpacityModelConfig


def test_partial_elements_are_suppressed_and_dense_elements_remain_opaque():
    sigma=torch.tensor([35.,100.,700.],requires_grad=True)
    density_element_loss(sigma,.04,.99).backward()
    assert (sigma.grad[:2]>0).all()
    assert sigma.grad[2]<=0 and abs(float(sigma.grad[2]))<1e-9


def test_sparse_endpoint_is_cheaper_than_unnecessary_opaque_support():
    assert density_element_loss(torch.zeros(1),.04)==0
    assert density_element_loss(torch.full((1,),1000.),.04)>.98


def test_empty_elements_have_zero_differentiable_penalty():
    sigma=torch.empty(0,requires_grad=True)
    value=density_element_loss(sigma,.04);value.backward()
    assert value==0 and sigma.grad.numel()==0


@pytest.mark.parametrize('length,bias', [(0.,.99),(.04,1.),(.04,-.1)])
def test_invalid_material_controls_are_rejected(length,bias):
    with pytest.raises(ValueError):density_element_loss(torch.ones(1),length,bias)


def test_auxiliary_element_sampling_keeps_rgb_and_rng_and_samples_active_cells():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.));self.points=None
        def density_fn(self,p):self.points=p.detach();return self.scale.expand(len(p),1)
    model=LookCloserOpacityModel.__new__(LookCloserOpacityModel);torch.nn.Module.__init__(model)
    model.device_indicator_param=torch.nn.Parameter(torch.empty(0))
    model.config=LookCloserOpacityModelConfig(density_element_regularization=.01,density_element_samples=64)
    model.current_train_step=38000;model.field=Field()
    grid=torch.zeros(1,4,4,4,dtype=torch.bool);grid[0,0,0,0]=True;grid[0,3,3,3]=True
    model.occupancy_grid=SimpleNamespace(binaries=grid,aabbs=torch.tensor([[0.,0.,0.,1.,1.,1.]]),resolution=torch.tensor([4,4,4]))
    original=dict(rgb=torch.tensor([[.1,.2,.3],[.4,.5,.6]]),packed_weights=torch.ones(2,1),
                  packed_ray_indices=torch.arange(2),packed_spacing_starts=torch.zeros(2,1),packed_spacing_ends=torch.ones(2,1))
    model._volume_outputs=lambda _:original.copy()
    rays=RayBundle(origins=torch.zeros(2,3),directions=torch.tensor([[0.,0.,1.]]).expand(2,3),pixel_area=torch.ones(2,1))
    torch.manual_seed(314);state=torch.get_rng_state().clone()
    result=model.get_outputs(rays)
    assert torch.equal(state,torch.get_rng_state())
    assert torch.equal(result['rgb'],original['rgb'])
    assert torch.equal(result['packed_weights'],original['packed_weights'])
    points=model.field.points
    assert (((points>=0)&(points<.25)).all(-1)|((points>=.75)&(points<1)).all(-1)).all()
    loss=density_element_loss(result['density_element_sigma'],result['density_element_length'])
    loss.backward();assert model.field.scale.grad>0
    model.config.density_element_regularization=0
    assert 'density_element_sigma' not in model.get_outputs(rays)
