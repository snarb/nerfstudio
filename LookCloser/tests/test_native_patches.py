from types import SimpleNamespace

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.native_patches import NativeOpaquePatches, patch_objective


def fixture():
    yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
    image=torch.stack([xx,yy,torch.zeros_like(xx)],-1).to(torch.uint8)
    core=torch.ones(1,16,16,dtype=torch.bool);core[:,6:9,6:9]=False
    targets=SimpleNamespace(images=image[None],core=core)
    cameras=Cameras(camera_to_worlds=torch.eye(4)[:3][None],fx=16.,fy=16.,cx=8.,cy=8.,width=16,height=16)
    return targets,cameras


def test_native_patches_preserve_ray_centers_and_exclude_uncertain_pixels():
    targets,cameras=fixture();sampler=NativeOpaquePatches(targets,size=12)
    rays,rgb,metadata=sampler.sample(16,cameras)
    yx=metadata['native_yx'];indices=metadata['indices'];base=yx//2
    assert targets.core[indices,base[...,0],base[...,1]].all()
    torch.testing.assert_close(rgb*255,targets.images[indices,yx[...,0],yx[...,1]].float())
    camera_directions=rays.directions*rays.metadata['directions_norm']
    coords=metadata['base_coords'].reshape(-1,2)
    torch.testing.assert_close(camera_directions[:,0],(coords[:,1]-8)/16)
    torch.testing.assert_close(camera_directions[:,1],-(coords[:,0]-8)/16)
    torch.testing.assert_close(metadata['base_coords'][:,:,1,1]-metadata['base_coords'][:,:,0,1],torch.full((16,12),.5))


def test_patch_sampling_has_independent_reproducible_random_state():
    targets,cameras=fixture();a=NativeOpaquePatches(targets,12);b=NativeOpaquePatches(targets,12)
    before=torch.get_rng_state().clone()
    _,rgb_a,_=a.sample(8,cameras)
    torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
    torch.rand(100)
    _,rgb_b,_=b.sample(8,cameras)
    torch.testing.assert_close(rgb_a,rgb_b,rtol=0,atol=0)


def test_partial_confidence_cannot_be_promoted_to_fully_trusted_patch_rgb():
    targets,cameras=fixture();confidence=torch.ones_like(targets.core,dtype=torch.float32)
    confidence[:,:8]=.5
    sampler=NativeOpaquePatches(targets,12,confidence=confidence)
    _,_,metadata=sampler.sample(8,cameras)
    base=metadata['native_yx']//2
    assert (confidence[metadata['indices'],base[...,0],base[...,1]]==1).all()


def test_spatial_objective_is_finite_and_backpropagates_to_structure():
    target=torch.rand(2,16,16,3)
    exact=patch_objective(target,target,.2)
    prediction=target.roll(2,dims=2).clone().requires_grad_()
    value=patch_objective(prediction,target,.2)
    assert value>exact and torch.isfinite(value)
    value.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum()>0
