import math
from types import SimpleNamespace
import torch

from nerfstudio.model_components.lookcloser_opacity import density_band_loss
from nerfstudio.models.lookcloser_opacity import LookCloserOpacityModel,LookCloserOpacityModelConfig


def test_band_opacity_prefers_dense_surface_and_rejects_front_fog():
    eps=torch.tensor([.001])
    dense=torch.full((1,9),-math.log(.01)/.002,requires_grad=True)
    free=torch.zeros(1,6,requires_grad=True)
    good=density_band_loss(dense,eps,free)
    assert float(good)<1e-12
    assert density_band_loss(dense*.1,eps,free)>good+.5
    assert density_band_loss(dense,eps,free+3000)>good+.5


def test_smaller_interval_requires_more_density():
    sigma=torch.full((1,9),2303.)
    free=torch.zeros(1,6)
    assert density_band_loss(sigma,torch.tensor([.001]),free)<1e-6
    assert density_band_loss(sigma,torch.tensor([.0005]),free)>.24


def test_gradients_increase_surface_density_and_decrease_free_density():
    sigma=torch.full((1,9),100.,requires_grad=True)
    free=torch.full((1,6),1000.,requires_grad=True)
    density_band_loss(sigma,torch.tensor([.001]),free).backward()
    assert (sigma.grad<0).all() and (free.grad>0).all()


def test_empty_qualified_set_has_no_penalty():
    sigma=torch.empty(0,9,requires_grad=True);free=torch.empty(0,6,requires_grad=True)
    value=density_band_loss(sigma,torch.empty(0),free)
    assert value==0
    value.backward();assert sigma.grad.numel()==free.grad.numel()==0


def test_model_band_queries_only_valid_rays_and_preserves_rgb_sampler_output():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(100.));self.points=[]
        def density_fn(self,p):
            self.points.append(p.detach());return self.scale.expand(len(p),1)
    model=LookCloserOpacityModel.__new__(LookCloserOpacityModel);torch.nn.Module.__init__(model)
    model.config=LookCloserOpacityModelConfig(depth_prior_strength=10,depth_prior_objective='density_band',volume_depth_guided_sampling=False)
    model.field=Field()
    original=dict(rgb=torch.tensor([[.1,.2,.3],[.4,.5,.6]]),depth=torch.ones(2,1),accumulation=torch.ones(2,1),
        packed_weights=torch.ones(2,1),packed_ray_indices=torch.arange(2),
        packed_spacing_starts=torch.zeros(2,1),packed_spacing_ends=torch.ones(2,1))
    model._volume_outputs=lambda _:original.copy()
    rays=SimpleNamespace(origins=torch.zeros(2,3),directions=torch.tensor([[0.,0.,1.],[0.,0.,1.]]),
        nears=torch.full((2,1),.05),fars=torch.full((2,1),2.),
        metadata=dict(depth_prior=torch.tensor([[1.],[float('nan')]]),depth_uncertainty=torch.tensor([[.001],[float('nan')]]),depth_valid=torch.tensor([[True],[False]])))
    result=model.get_outputs(rays)
    assert torch.equal(result['rgb'],original['rgb'])
    assert torch.equal(result['packed_weights'],original['packed_weights'])
    assert result['density_band_sigma'].shape==(1,9)
    assert result['density_free_sigma'].shape==(1,6)
    assert all(p.isfinite().all() for p in model.field.points)
    assert torch.allclose(model.field.points[0][:,2],torch.linspace(.998,1.002,9))
    assert (model.field.points[1][:,2]<=.996).all()


def test_band_outside_renderable_range_stays_unknown():
    class Field(torch.nn.Module):
        def density_fn(self,p):raise AssertionError('Out-of-range priors must not query density')
    model=LookCloserOpacityModel.__new__(LookCloserOpacityModel);torch.nn.Module.__init__(model)
    model.config=LookCloserOpacityModelConfig(depth_prior_strength=10,depth_prior_objective='density_band',volume_depth_guided_sampling=False)
    model.field=Field()
    model._volume_outputs=lambda _:dict(rgb=torch.zeros(1,3),packed_weights=torch.ones(1,1),packed_ray_indices=torch.zeros(1,dtype=torch.long),packed_spacing_starts=torch.zeros(1,1),packed_spacing_ends=torch.ones(1,1))
    rays=SimpleNamespace(origins=torch.zeros(1,3),directions=torch.tensor([[0.,0.,1.]]),nears=torch.tensor([[.05]]),fars=torch.tensor([[1.]]),metadata=dict(depth_prior=torch.tensor([[1.]]),depth_uncertainty=torch.tensor([[.001]]),depth_valid=torch.tensor([[True]])))
    result=model.get_outputs(rays)
    assert result['density_band_sigma'].numel()==0
