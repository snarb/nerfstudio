"""Occluded foreground plane: only cameras outside the nearest eight see background."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_partial_neural_visibility_cannot_silently_mix_missing_source_cameras():
    from study_dec5_hair_geometry import Renderer
    renderer=object.__new__(Renderer)
    renderer.scene=SimpleNamespace(centers=np.c_[np.arange(10),np.zeros((10,2))],
        frames=[dict(physical_camera=str(i)) for i in range(10)],allowed_visibility_sources={str(i) for i in range(8)})
    camera=dict(transform_matrix=np.eye(4).tolist())
    assert renderer.source_order(camera).tolist()==list(range(8))
    renderer.scene.allowed_visibility_sources.remove('7')
    with pytest.raises(ValueError,match='does not cover'):renderer.source_order(camera)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_native_rgb_sampling_preserves_scaled_calibration_with_coarse_visibility():
    from study_dec5_hair_geometry import Renderer
    camera=dict(h=8,w=8,fl_x=20.,fl_y=20.,cx=4.,cy=4.,transform_matrix=np.eye(4).tolist())
    renderer=object.__new__(Renderer);renderer.scene=SimpleNamespace(cameras=[camera],centers=np.zeros((1,3)))
    renderer.depth=[torch.full((1,8,8),2.,device='cuda')];renderer.semantic=[torch.ones((3,8,8),device='cuda')/3]
    yy,xx=torch.meshgrid(torch.arange(16,device='cuda'),torch.arange(16,device='cuda'),indexing='ij')
    renderer.rgb=[torch.stack(((xx+.5)/16,(yy+.5)/16,torch.zeros_like(xx).float()))]
    result=renderer.warp(camera,torch.full((8,8),2.,device='cuda'))
    torch.testing.assert_close(result['rgb'][4,4],torch.tensor([9/16,9/16,0.],device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_background_uses_its_own_visible_cameras_and_stereo_surface():
    from study_dec5_hair_geometry import Renderer
    from study_dec5_layer_background import LayerRenderer
    scene=SimpleNamespace(cameras=[],centers=[])
    renderer=object.__new__(LayerRenderer);renderer.scene=scene
    renderer.visibility_depth_sampling='bilinear'
    renderer.depth=[];renderer.background_observed=[];renderer.semantic=[];renderer.rgb=[]
    for i in range(10):
        pose=np.eye(4);pose[0,3]=i*.002
        scene.cameras.append(dict(h=8,w=8,fl_x=20.,fl_y=20.,cx=4.,cy=4.,transform_matrix=pose.tolist()))
        scene.centers.append(pose[:3,3])
        # The head-only field places a front hit everywhere. Original stereo
        # sees the background plane in two more distant cameras.
        renderer.depth.append(torch.ones((1,8,8),device='cuda'))
        renderer.background_observed.append(torch.full((1,8,8),2. if i>=8 else 1.,device='cuda'))
        sem=torch.zeros((3,8,8),device='cuda');sem[2 if i>=8 else 0]=1;renderer.semantic.append(sem)
        color=torch.tensor([.2,.4,.6] if i==8 else [.9,.1,.2],device='cuda')[:,None,None].expand(3,8,8)
        renderer.rgb.append(color)
    scene.centers=np.asarray(scene.centers);depth=torch.full((8,8),2.,device='cuda');before=depth.clone()
    old=Renderer.warp_layer(renderer,scene.cameras[0],depth)
    new=renderer.warp_layer(scene.cameras[0],depth)
    assert int(old['source_id'][4,4])==-1
    assert int(new['source_id'][4,4])==8 and int(new['support'][4,4])==2
    torch.testing.assert_close(new['rgb'][4,4],torch.tensor([.2,.4,.6],device='cuda'))
    assert torch.equal(depth,before)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_depth_discontinuity_does_not_create_a_visible_intermediate_surface():
    from study_dec5_hair_geometry import Renderer
    target=dict(h=8,w=8,fl_x=2.,fl_y=2.,cx=4.,cy=4.,transform_matrix=np.eye(4).tolist())
    pose=np.eye(4);pose[0,3]=-.5
    source=dict(target,transform_matrix=pose.tolist())
    renderer=object.__new__(Renderer)
    renderer.scene=SimpleNamespace(cameras=[source],centers=np.array([pose[:3,3]]))
    renderer.depth=[torch.cat((torch.ones((1,8,4)),3*torch.ones((1,8,4))),-1).cuda()]
    renderer.semantic=[torch.ones((3,8,8),device='cuda')/3]
    renderer.rgb=[torch.full((3,8,8),.5,device='cuda')]
    depth=torch.full((8,8),2.,device='cuda')
    renderer.visibility_depth_sampling='bilinear'
    interpolated=renderer.warp(target,depth)
    renderer.visibility_depth_sampling='nearest'
    discrete=renderer.warp(target,depth)
    # The target ray projects exactly between depth-1 and depth-3 source pixels.
    # A surface at depth 2 exists only in the bilinear interpolation.
    assert int(interpolated['source_id'][4,3])==0
    assert int(discrete['source_id'][4,3])==-1
    assert torch.equal(depth,torch.full_like(depth,2.))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_foreground_is_premultiplied_before_interpolating_transparent_neighbors():
    from study_dec5_hair_geometry import Renderer
    target=dict(h=8,w=8,fl_x=2.,fl_y=2.,cx=4.,cy=4.,transform_matrix=np.eye(4).tolist())
    pose=np.eye(4);pose[0,3]=-.5
    source=dict(target,transform_matrix=pose.tolist())
    renderer=object.__new__(Renderer)
    renderer.scene=SimpleNamespace(cameras=[source],centers=np.array([pose[:3,3]]))
    renderer.depth=[torch.full((1,8,8),2.,device='cuda')]
    renderer.semantic=[torch.ones((3,8,8),device='cuda')/3]
    alpha=torch.ones((1,8,8),device='cuda');alpha[:,:,4:]=0
    color=torch.full((3,8,8),.8,device='cuda');color[:,:,4:]=100
    renderer.alpha=[alpha];renderer.foreground=[color];renderer.rgb=[color]
    renderer.known=[torch.ones_like(alpha)]
    result=renderer.warp_layer(target,torch.full((8,8),2.,device='cuda'),foreground=True,decontaminate=True,premultiply=True)
    torch.testing.assert_close(result['alpha'][4,3],torch.tensor(.5,device='cuda'))
    torch.testing.assert_close(result['rgb'][4,3],torch.full((3,),.4,device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_rejected_ray_visibility_callback_restores_plane_and_protects_domain():
    from study_dec5_hair_geometry import Renderer
    camera=dict(h=8,w=8,fl_x=20.,fl_y=20.,cx=4.,cy=4.,transform_matrix=np.eye(4).tolist())
    renderer=object.__new__(Renderer);renderer.scene=SimpleNamespace(cameras=[camera],centers=np.zeros((1,3)))
    renderer.depth=[torch.full((1,8,8),2.,device='cuda')];renderer.semantic=[torch.ones((3,8,8),device='cuda')/3];renderer.rgb=[torch.full((3,8,8),.4,device='cuda')]
    depth=torch.ones((8,8),device='cuda');baseline=renderer.warp(camera,depth)
    assert int(baseline['source_id'][4,4])==-1
    def exact_plane(index,uv,observed,semantics,eligible):
        value=observed.clone();value[eligible]=1.;return value
    renderer.visibility_depth_resampler=exact_plane
    renderer.visibility_resampling_domain=torch.ones((8,8),device='cuda',dtype=torch.bool);renderer.visibility_resampling_domain[4,4]=False
    candidate=renderer.warp(camera,depth)
    assert int(candidate['source_id'][3,3])==0 and int(candidate['source_id'][4,4])==-1
    torch.testing.assert_close(candidate['rgb'][3,3],torch.full((3,),.4,device='cuda'))
    renderer.visibility_depth_resampler=lambda index,uv,observed,semantics,eligible:observed
    unchanged=renderer.warp(camera,depth)
    for key in baseline:assert torch.equal(baseline[key],unchanged[key]),key


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_zero_alpha_requires_three_views_and_prefers_positive_sources():
    from study_dec5_hair_geometry import Renderer
    camera=dict(h=8,w=8,fl_x=20.,fl_y=20.,cx=4.,cy=4.,transform_matrix=np.eye(4).tolist())
    renderer=object.__new__(Renderer);renderer.scene=SimpleNamespace(cameras=[camera]*3,centers=np.zeros((3,3)))
    renderer.depth=[torch.full((1,8,8),2.,device='cuda') for _ in range(3)]
    renderer.semantic=[torch.ones((3,8,8),device='cuda')/3 for _ in range(3)]
    renderer.rgb=[torch.full((3,8,8),.8,device='cuda') for _ in range(3)];renderer.foreground=renderer.rgb
    renderer.alpha=[torch.zeros((1,8,8),device='cuda') for _ in range(3)];renderer.known=[torch.ones_like(v) for v in renderer.alpha]
    depth=torch.full((8,8),2.,device='cuda')
    unknown=renderer.warp_layer(camera,depth,foreground=True,premultiply=True)
    assert unknown['source_id'][4,4]==-1
    empty=renderer.warp_layer(camera,depth,foreground=True,premultiply=True,allow_zero_alpha=True)
    assert empty['source_id'][4,4]==0 and empty['alpha'][4,4]==0 and empty['support'][4,4]==3
    renderer.alpha[1].fill_(.5);renderer.alpha[2].fill_(.75)
    hair=renderer.warp_layer(camera,depth,foreground=True,premultiply=True,allow_zero_alpha=True)
    assert hair['source_id'][4,4]==1 and hair['support'][4,4]==2
    torch.testing.assert_close(hair['rgb'][4,4],torch.full((3,),.4,device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Renderer requires CUDA')
def test_native_matte_raster_is_independent_of_visibility_resolution(tmp_path):
    from study_dec5_hair_geometry import Renderer
    renderer=object.__new__(Renderer);renderer.scene=SimpleNamespace(frames=[dict(h=8,w=8,physical_camera='source')])
    renderer.rgb=[torch.zeros((3,16,16),device='cuda')]
    np.savez_compressed(tmp_path/'source.npz',alpha=np.ones((16,16),np.float16),foreground=np.ones((16,16,3),np.float32)*.4,original_shape=[16,16,3],portrait_crop=[0,0,16,16])
    renderer.load_mattes(tmp_path)
    assert renderer.alpha[0].shape==(1,16,16) and renderer.foreground[0].shape==(3,16,16)
    assert bool((renderer.known[0]==1).all())
