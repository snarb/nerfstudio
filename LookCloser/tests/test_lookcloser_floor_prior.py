import math
from types import SimpleNamespace

import pytest
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.lookcloser_opacity import floor_ray_geometry_loss
from nerfstudio.models.lookcloser_opacity import LookCloserOpacityModel,LookCloserOpacityModelConfig


def test_floor_geometry_prefers_empty_space_and_opaque_backing():
    free=torch.zeros(2,65,requires_grad=True)
    band=torch.full((2,9),-math.log(.01)/.02,requires_grad=True)
    value=floor_ray_geometry_loss(free,torch.ones(2),band,torch.full((2,),.01))
    assert value<1e-10
    assert floor_ray_geometry_loss(free+1,torch.ones(2),band,torch.full((2,),.01))>.99
    assert floor_ray_geometry_loss(free,torch.ones(2),band*.1,torch.full((2,),.01))>.8


def test_floor_integral_hits_layers_between_sparse_probes_with_correct_gradient():
    free=torch.zeros(1,65);free[0,22:25]=100;free.requires_grad_()
    band=torch.ones(1,9,requires_grad=True)
    value=floor_ray_geometry_loss(free,torch.ones(1),band,torch.tensor([.01]))
    value.backward()
    assert value>4 and (free.grad>0).all() and (band.grad<0).all()


def test_floor_free_space_penalty_does_not_saturate():
    band=torch.full((1,9),1000.)
    a=floor_ray_geometry_loss(torch.ones(1,65)*100,torch.ones(1),band,torch.tensor([.01]))
    b=floor_ray_geometry_loss(torch.ones(1,65)*200,torch.ones(1),band,torch.tensor([.01]))
    assert torch.allclose(b,2*a)


@pytest.mark.parametrize('sampling',['uniform','stratified'])
def test_floor_auxiliary_queries_preserve_native_rgb_and_rng(sampling):
    class Field(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.));self.points=[]
        def density_fn(self,p):self.points.append(p.detach());return self.scale.expand(len(p),1)
    m=LookCloserOpacityModel.__new__(LookCloserOpacityModel);torch.nn.Module.__init__(m)
    m.device_indicator_param=torch.nn.Parameter(torch.empty(0));m.current_train_step=38000;m.field=Field()
    m.config=LookCloserOpacityModelConfig(floor_ray_regularization=.001,floor_rays_per_batch=2,floor_free_sampling=sampling)
    m.floor_origins=torch.zeros(1,3);m.floor_directions=torch.tensor([[0.,0.,1.]])
    m.floor_depth=torch.tensor([1.]);m.floor_tolerance=torch.tensor([.1])
    m.floor_admission=torch.zeros(1,2,2,2,dtype=torch.bool)
    m.occupancy_grid=SimpleNamespace(binaries=m.floor_admission.clone())
    original=dict(rgb=torch.ones(2,3),packed_weights=torch.ones(2,1),packed_ray_indices=torch.arange(2),
        packed_spacing_starts=torch.zeros(2,1),packed_spacing_ends=torch.ones(2,1))
    m._volume_outputs=lambda _:original.copy()
    rays=RayBundle(origins=torch.zeros(2,3),directions=torch.ones(2,3),pixel_area=torch.ones(2,1))
    torch.manual_seed(77);state=torch.get_rng_state().clone();result=m.get_outputs(rays)
    assert torch.equal(state,torch.get_rng_state()) and torch.equal(result['rgb'],original['rgb'])
    assert result['floor_free_sigma'].shape==(2,65) and result['floor_band_sigma'].shape==(2,9)
    assert m.field.points[0][:,2].max()<=.90001 and m.field.points[1][:,2].min()>=.89999
    original_points=m.field.points[0].clone()
    m.get_outputs(rays)
    assert torch.equal(original_points,m.field.points[2])
    if sampling=='stratified':
        f=(original_points[:,2].reshape(2,65)-m.config.near_plane)/(.9-m.config.near_plane)
        assert ((f*65).floor()==torch.arange(65)[None]).all()
        m.current_train_step+=1;m.get_outputs(rays)
        assert not torch.equal(original_points,m.field.points[4])


def test_floor_stratified_estimator_uses_equal_interval_weights():
    free=torch.tensor([[2.,0.,0.]],requires_grad=True)
    band=torch.ones(1,9)*1000
    value=floor_ray_geometry_loss(free,torch.ones(1),band,torch.tensor([.01]),free_monte_carlo=True)
    assert torch.allclose(value,torch.tensor(2/3))
    value.backward()
    assert torch.allclose(free.grad,torch.full_like(free,1/3))


def test_floor_invalid_interval_shapes_are_rejected():
    with pytest.raises(ValueError):floor_ray_geometry_loss(torch.ones(1,1),torch.ones(1),torch.ones(1,9),torch.ones(1))


def test_checkpoint_selection_excludes_other_floor_rendering_protocols(tmp_path):
    from scripts.run_luster_opacity import floor_evaluation_protocol,qualify_floor_history
    manifest=tmp_path/'manifest.json';manifest.write_text('{"archive_sha256":"a"}')
    request={'model':{'floor_ray_regularization':.001,'floor_ray_prior_path':str(manifest)}}
    digest=floor_evaluation_protocol(request)
    old={'step':38000,'eval_all_psnr':100.}
    current={'step':40000,'eval_all_psnr':20.,'floor_evaluation_protocol':digest}
    assert qualify_floor_history([old,current],request)==[current]
    assert qualify_floor_history([old,current],{})==[old]
    # Objective coefficient and sampling do not affect admission at evaluation.
    request['model'].update(floor_ray_regularization=.0001,floor_free_sampling='stratified')
    assert qualify_floor_history([current],request)==[current]
    manifest.write_text('{"archive_sha256":"b"}')
    assert qualify_floor_history([current],request)==[]
