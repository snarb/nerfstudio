"""Behavioral contracts for the opt-in teacher/student workflow."""
import gzip
from pathlib import Path

import numpy as np
import pytest
import torch

from nerfstudio.model_components.mesh_distillation import (
    camera_z_to_distance, compose_actor_background, projected_frequency,
    ray_box, weighted_charbonnier,
)
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


def test_background_composition_uses_premultiplied_actor():
    a=torch.tensor([[.1,.2,.3],[.2,.3,.4],[0.,0.,0.]])
    b=torch.tensor([[.8,.6,.2]]).expand_as(a)
    out=compose_actor_background(a,torch.tensor([[1.],[.5],[0.]]),b)
    torch.testing.assert_close(out[0],a[0]);torch.testing.assert_close(out[2],b[2])
    torch.testing.assert_close(out[1],a[1]+b[1]*.5)


def test_unknown_pixels_have_zero_gradient_but_valid_black_is_supervised():
    pred=torch.full((3,3),.5,requires_grad=True)
    target=torch.tensor([[0.,0.,0.],[float('nan')]*3,[0.,0.,0.]])
    weight=torch.tensor([[1.],[0.],[.25]])
    loss=weighted_charbonnier(pred,target,weight);loss.backward()
    assert torch.isfinite(loss)
    assert torch.all(pred.grad[0]>0) and torch.all(pred.grad[1]==0)
    torch.testing.assert_close(pred.grad[2],pred.grad[0]*.25)


def test_zero_weight_batch_is_finite_and_inert():
    x=torch.ones(2,3,requires_grad=True)
    loss=weighted_charbonnier(x,torch.zeros_like(x),torch.zeros(2,1));loss.backward()
    assert loss==0 and torch.all(x.grad==0)


def test_selected_checkpoint_stays_immutable_when_latest_advances(tmp_path):
    import sys,os
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from mesh_distillation_background import publish_hardlink
    latest=tmp_path/'latest.pt';first=tmp_path/'candidate_1.pt';best=tmp_path/'best.pt'
    torch.save({'step':1,'parameter':torch.tensor([.3])},latest)
    publish_hardlink(latest,first);publish_hardlink(first,best)
    old_bytes=best.read_bytes();publish_hardlink(first,best)
    assert not (tmp_path/'best.pt.link').exists()
    temporary=tmp_path/'latest.tmp';torch.save({'step':2,'parameter':torch.tensor([.7])},temporary);temporary.replace(latest)
    assert best.read_bytes()==old_bytes and first.read_bytes()==old_bytes
    assert torch.load(latest,weights_only=True)['step']==2
    second=tmp_path/'candidate_2.pt';publish_hardlink(latest,second);publish_hardlink(second,best)
    assert os.path.samefile(latest,second) and os.path.samefile(second,best)
    assert torch.load(first,weights_only=True)['step']==1 and torch.load(best,weights_only=True)['step']==2


def test_frequency_projection_is_invariant_to_scene_scale_and_pixel_resolution():
    r=torch.tensor([128.]);fx=torch.tensor([9600.]);fy=torch.tensor([9600.]);w=torch.tensor([1920.]);h=torch.tensor([1080.]);z=torch.tensor([.8]);size=torch.tensor([.15,.1,.08])
    a=projected_frequency(r,fx,fy,w,h,z,size)
    b=projected_frequency(r,fx*2,fy*2,w*2,h*2,z*7,size*7)
    torch.testing.assert_close(a,b)
    assert a<8192


def test_camera_z_at_oblique_ray_is_converted_exactly_once():
    z=torch.tensor([[2.],[0.]])
    norm=torch.tensor([[5**.5],[3.]])
    torch.testing.assert_close(camera_z_to_distance(z,norm),torch.tensor([[2*5**.5],[0.]]))


def test_gzip_depth_is_scaled_and_resized_like_numpy(tmp_path):
    values=np.array([[0.,2.],[3.,4.]],dtype=np.float32)
    p=tmp_path/'depth.npy.gz'
    with gzip.open(p,'wb') as f:np.save(f,values)
    a=get_depth_image_from_path(p,4,4,.5)
    assert a.shape==(4,4,1) and a[0,0]==0 and a[-1,-1]==2


def test_parallel_ray_box_misses_and_inside_rays():
    box=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
    o=torch.tensor([[0.,0.,-2.],[2.,0.,-2.],[0.,0.,0.]])
    d=torch.tensor([[0.,0.,1.]]).expand_as(o)
    near,far,hit=ray_box(o,d,box)
    assert hit.tolist()==[True,False,True]
    assert near[0]==1 and far[0]==3 and near[2]==0


def test_plane_slab_intervals_cover_front_back_parallel_and_inside_rays():
    from nerfstudio.model_components.mesh_distillation import ray_plane_slab
    slab=torch.tensor([0.,0.,1.,0.,.1,.3])
    origins=torch.tensor([[0.,0.,1.],[0.,0.,0.],[0.,0.,.2],[0.,0.,.5],[0.,0.,.2]])
    directions=torch.tensor([[0.,0.,-1.],[0.,0.,1.],[1.,0.,0.],[1.,0.,0.],[0.,0.,-1.]])
    near,far,hit=ray_plane_slab(origins,directions,slab)
    assert hit.tolist()==[True,True,True,False,True]
    torch.testing.assert_close(near[[0,1,2,4]],torch.tensor([.7,.1,0.,0.]))
    torch.testing.assert_close(far[[0,1,4]],torch.tensor([.9,.3,.1]))
    assert torch.isposinf(far[2])
    # Rotate coordinates and the plane together; geometry must not change.
    rotation=torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    rotated=slab.clone();rotated[:3]=rotation@slab[:3]
    other=ray_plane_slab(origins@rotation.T,directions@rotation.T,rotated)
    for a,b in zip((near,far,hit),other):torch.testing.assert_close(a,b)


def test_frozen_plane_texture_resolution_survives_checkpoint_geometry():
    from nerfstudio.model_components.mesh_distillation import SeparateBackground
    geometry=dict(plane=[0,0,1,0],basis=[[1,0,0],[0,1,0]],uv_bounds=[[-1,-1],[1,1]],
                  bounds=[[-1,-1,-1],[1,1,1]],actor_bounds=[[-.1,-.1,.5],[.1,.1,.7]],
                  behind_limit=.4,texture_resolution=32)
    source=SeparateBackground('plane',geometry)
    assert source.texture.shape==(1,3,32,32)
    with torch.no_grad():source.texture.fill_(torch.logit(torch.tensor(.7)))
    restored=SeparateBackground('plane',geometry);restored.load_state_dict(source.state_dict())
    actual=restored(torch.tensor([[0.,0.,1.]]),torch.tensor([[0.,0.,-1.]]))
    torch.testing.assert_close(actual['rgb'],torch.full((1,3),.7))
    assert actual['valid'].item() and actual['depth'].item()==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Checks CUDA autocast geometry')
def test_plane_geometry_and_texture_are_identical_under_training_autocast():
    from nerfstudio.model_components.mesh_distillation import SeparateBackground,ray_plane_slab
    plane=[.2969928,-.9459547,-.1302493,.5339599]
    slab=torch.tensor([*plane,.003,.3],device='cuda')
    origins=torch.tensor([[.075,-.5,.2],[.1,-.4,.05]],device='cuda')
    directions=torch.tensor([[.1,1.,0.],[0.,1.,.1]],device='cuda');directions/=directions.norm(dim=-1,keepdim=True)
    geometry=dict(plane=plane,basis=[[1,0,0],[0,0,1]],uv_bounds=[[-1,-1],[1,1]],bounds=[[-1,-1,-1],[1,1,1]],actor_bounds=[[-.1,-.1,.5],[.1,.1,.7]],behind_limit=.4,texture_resolution=64)
    wall=SeparateBackground('plane',geometry).cuda()
    with torch.no_grad():wall.texture.normal_(std=2.)
    first=ray_plane_slab(origins,directions,slab);plain=wall(origins,directions)
    with torch.autocast('cuda',dtype=torch.float16):
        second=ray_plane_slab(origins,directions,slab);mixed=wall(origins,directions)
    for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for key in plain:torch.testing.assert_close(plain[key],mixed[key],atol=0,rtol=0)


def test_observed_grid_does_not_permanently_assign_unknown_voxels_maximum():
    from nerfstudio.pipelines.mesh_distillation_pipeline import ObservedFrequencyGrid
    from nerfstudio.data.scene_box import SceneBox
    grid=ObservedFrequencyGrid(SceneBox(torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])),resolution=8)
    point=torch.tensor([[0.,0.,0.]])
    assert grid.query(point).item()==15
    grid.update_max(point,torch.tensor([3.]))
    assert grid.query(point).item()==3
    state=grid.state_dict();restored=ObservedFrequencyGrid(SceneBox(torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])),resolution=8)
    restored.load_state_dict(state);restored.frozen=True
    assert restored.query(point).item()==3
    with pytest.raises(RuntimeError,match='frozen'):restored.update_max(point,torch.tensor([5.]))


def test_normalized_density_preserves_optical_thickness_and_is_finite():
    from nerfstudio.fields.lookcloser_field import LookCloserField
    holder=type('DensityHolder',(),{})()
    holder.aabb=torch.tensor([[-.1,-.1,-.1],[.1,.1,.1]])
    holder.normalized_exponential_density=True
    logits=torch.tensor([[-5.],[0.],[100.]],dtype=torch.float16)
    density=LookCloserField.activate_density(holder,logits)
    holder.aabb*=7
    scaled=LookCloserField.activate_density(holder,logits)
    torch.testing.assert_close(density*.02,scaled*.14)
    assert torch.isfinite(density).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='TCNN field requires CUDA')
def test_view_independent_field_color_is_invariant_to_ray_direction():
    from nerfstudio.fields.lookcloser_field import LookCloserField
    from nerfstudio.pipelines.mesh_distillation_pipeline import ObservedFrequencyGrid
    from nerfstudio.data.scene_box import SceneBox
    box=SceneBox(torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]],device='cuda'))
    grid=ObservedFrequencyGrid(box,resolution=8).cuda()
    field=LookCloserField(box.aabb,grid,num_levels=2,min_res=4,max_res=8,log2_hashmap_size=5,
                          enable_feature_reweighting=False).cuda().eval()
    field.correct_sh_directions=True
    point=torch.tensor([[.1,.2,.3],[-.2,.1,.4]],device='cuda')
    first=torch.tensor([[0.,0.,1.],[1.,0.,0.]],device='cuda');second=-first
    assert not torch.equal(field.encode_directions(first),field.encode_directions(second))
    field.view_independent_color=True
    with torch.no_grad():a=field.query_points(point,first);b=field.query_points(point,second)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    # World-space slab gating is identical in direct ray marching and density
    # proposals, including points inside the AABB but behind the wall.
    field.register_buffer('density_slab',torch.tensor([0.,0.,1.,0.,.35,.5],device='cuda'))
    with torch.no_grad():direct=field.query_points(point,first)[0];proposal=field.density_fn(point)
    torch.testing.assert_close(direct,proposal,rtol=0,atol=0)
    assert direct[0]==0 and direct[1]>0


def test_foreground_support_uses_xyz_axis_order_and_conservative_interpolation():
    from nerfstudio.fields.lookcloser_field import LookCloserField
    holder=type('SupportHolder',(),{})()
    holder.foreground_support=torch.zeros(4,5,6);holder.foreground_support[1,2,4]=1
    coordinates=torch.tensor([[1/3,2/4,4/5],[0.,0.,0.]])
    actual=LookCloserField.foreground_support_at(holder,coordinates)
    torch.testing.assert_close(actual,torch.tensor([[1.],[0.]]),atol=1e-6,rtol=0)


def test_fixed_sampling_jitters_training_and_preserves_eval_and_constant_integral():
    from types import SimpleNamespace
    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.models.lookcloser import LookCloserModel
    class ConstantField:
        def query_points(self,positions,directions,**kwargs):
            return torch.ones(len(positions),1),torch.full((len(positions),3),.4)
    holder=SimpleNamespace(config=SimpleNamespace(fixed_num_samples_per_ray=16,fixed_stratified_sampling=True),
        training=True,field=ConstantField(),renderer_rgb=SimpleNamespace(background_color='black'))
    rays=RayBundle(origins=torch.zeros(2,3),directions=torch.tensor([[0.,0.,1.]]).expand(2,-1),
        pixel_area=torch.ones(2,1),nears=torch.ones(2,1),fars=torch.full((2,1),3.))
    a=LookCloserModel.fixed_ray_marching(holder,rays);b=LookCloserModel.fixed_ray_marching(holder,rays)
    assert not torch.equal(a['sample_distances'],b['sample_distances'])
    edges=torch.linspace(1.,3.,17)
    assert (a['sample_distances'][...,0]>=edges[:-1]).all() and (a['sample_distances'][...,0]<=edges[1:]).all()
    torch.testing.assert_close(a['accumulation'],torch.full((2,1),1-np.exp(-2)))
    torch.testing.assert_close(a['rgb'],b['rgb'])
    holder.training=False
    c=LookCloserModel.fixed_ray_marching(holder,rays);d=LookCloserModel.fixed_ray_marching(holder,rays)
    torch.testing.assert_close(c['sample_distances'],d['sample_distances'])
    torch.testing.assert_close(c['sample_distances'][0,:,0],(edges[:-1]+edges[1:])*.5)


def test_importance_intervals_cover_ray_and_resolve_a_thin_surface():
    from types import SimpleNamespace
    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.models.lookcloser import LookCloserModel
    class SlabField:
        def __init__(self): self.amplitude=torch.tensor(100.,requires_grad=True)
        def density_fn(self,points):
            # Analytic slab slightly wider than one coarse interval.
            return self.amplitude*((points[:,2:3]>.43)&(points[:,2:3]<.47)).float()
        def query_points(self,positions,directions,**kwargs):
            return self.density_fn(positions),torch.full((len(positions),3),.6)
    field=SlabField()
    holder=SimpleNamespace(config=SimpleNamespace(fixed_num_samples_per_ray=32,
        fixed_stratified_sampling=True,fixed_importance_samples=256),training=False,
        field=field,renderer_rgb=SimpleNamespace(background_color='black'))
    rays=RayBundle(origins=torch.zeros(2,3),directions=torch.tensor([[0.,0.,1.]]).expand(2,-1),
        pixel_area=torch.ones(2,1),nears=torch.zeros(2,1),fars=torch.ones(2,1))
    out=LookCloserModel.fixed_ray_marching(holder,rays)
    samples=out['loss_ray_samples'];starts=samples.spacing_starts;ends=samples.spacing_ends
    assert (ends>=starts).all() and torch.all(starts[:,0]==0) and torch.all(ends[:,-1]==1)
    torch.testing.assert_close(ends[:,:-1],starts[:,1:])
    assert (out['sample_distances'].squeeze(-1).sub(.45).abs()<.04).float().mean()>.65
    # Compare integration to an independent high-resolution quadrature.
    z=(torch.arange(20000)+.5)/20000
    exact_tau=field.density_fn(torch.stack([z*0,z*0,z],-1)).mean()
    expected=-torch.expm1(-exact_tau)
    torch.testing.assert_close(out['accumulation'],expected.expand(2,1),atol=.003,rtol=0)
    torch.testing.assert_close(out['rgb'],.6*out['accumulation'].expand(-1,3))
    out['rgb'].sum().backward();assert torch.isfinite(field.amplitude.grad) and field.amplitude.grad>0
    repeated=LookCloserModel.fixed_ray_marching(holder,rays)
    torch.testing.assert_close(out['sample_distances'],repeated['sample_distances'])
    holder.training=True
    jittered=LookCloserModel.fixed_ray_marching(holder,rays)
    assert not torch.equal(out['sample_distances'],jittered['sample_distances'])
    # A transparent proposal still covers the full ray and remains finite.
    with torch.no_grad():field.amplitude.zero_()
    empty=LookCloserModel.fixed_ray_marching(holder,rays)
    assert torch.isfinite(empty['rgb']).all() and torch.all(empty['accumulation']==0)


def test_surface_depth_outputs_do_not_change_radiance_and_match_exponential_median():
    from types import SimpleNamespace
    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.models.lookcloser import LookCloserModel
    class ConstantField:
        def query_points(self,positions,directions,**kwargs):
            return torch.ones(len(positions),1),torch.full((len(positions),3),.4)
    holder=SimpleNamespace(config=SimpleNamespace(fixed_num_samples_per_ray=4096,
        fixed_stratified_sampling=False,fixed_depth_estimator='expected'),
        training=False,field=ConstantField(),renderer_rgb=SimpleNamespace(background_color='black'))
    rays=RayBundle(origins=torch.zeros(1,3),directions=torch.tensor([[0.,0.,1.]]),
        pixel_area=torch.ones(1,1),nears=torch.ones(1,1),fars=torch.full((1,1),3.))
    baseline=LookCloserModel.fixed_ray_marching(holder,rays)
    holder.config.fixed_depth_estimator='median';median=LookCloserModel.fixed_ray_marching(holder,rays)
    exact=1-np.log(1-.5*(1-np.exp(-2)))
    assert abs(float(median['depth'])-exact)<.001
    torch.testing.assert_close(median['rgb'],baseline['rgb'],rtol=0,atol=0)
    holder.config.fixed_num_samples_per_ray=16
    coarse_median=LookCloserModel.fixed_ray_marching(holder,rays)
    assert abs(float(coarse_median['depth'])-exact)<1e-5
    holder.config.fixed_num_samples_per_ray=4096
    holder.config.fixed_depth_estimator='mode';mode=LookCloserModel.fixed_ray_marching(holder,rays)
    assert abs(float(mode['depth'])-1)<.001
    torch.testing.assert_close(mode['rgb'],baseline['rgb'],rtol=0,atol=0)


def test_photographic_wall_is_view_independent_with_physical_depth(tmp_path):
    import json,hashlib,sys
    from PIL import Image
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_distillation_clean_wall import PhotographicBrickWall
    geometry=dict(plane=[0.,0.,1.,0.],basis=[[1.,0.,0.],[0.,1.,0.]],origin_uv=[0.,0.],brick_size=[2.,1.],stagger=.5)
    (tmp_path/'geometry.json').write_text(json.dumps(geometry))
    texture=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],np.uint8)
    Image.fromarray(texture).save(tmp_path/'brick.png')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'receipt.json').write_text(json.dumps(dict(actual_eval_rgb_used=False,geometry_sha256=digest(tmp_path/'geometry.json'),texture_sha256=digest(tmp_path/'brick.png'))))
    wall=PhotographicBrickWall(tmp_path)
    # The same physical point from separated cameras must have one texture value.
    target=torch.tensor([[.7,.3,0.],[.7,.3,0.]])
    origins=torch.tensor([[0.,0.,2.],[1.,-.5,3.]])
    directions=torch.nn.functional.normalize(target-origins,dim=-1)
    out=wall(origins,directions)
    assert out['valid'].all()
    torch.testing.assert_close(origins+directions*out['depth'],target,atol=5e-7,rtol=0)
    torch.testing.assert_close(out['rgb'][0],out['rgb'][1],atol=1e-6,rtol=0)
    # One course + half a brick has the same sample in the staggered pattern.
    shifted=target+torch.tensor([1.,1.,0.])
    out2=wall(shifted+torch.tensor([0.,0.,2.]),torch.tensor([[0.,0.,-1.],[0.,0.,-1.]]))
    torch.testing.assert_close(out2['rgb'],out['rgb'],atol=1e-6,rtol=0)
