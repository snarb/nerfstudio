"""Subpixel target/ray alignment and preservation of uncertain boundaries."""
import torch

from nerfstudio.model_components.native_training import sample_native_rgb, NativeTrainingTargets
from nerfstudio.cameras.cameras import Cameras


def test_observed_unknown_exclusion_precedes_native_erosion(tmp_path):
    import hashlib,json
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    image=tmp_path/'train.png';Image.fromarray(np.full((7,7,3),128,np.uint8)).save(image)
    mask=tmp_path/'mask.png';Image.fromarray(np.full((7,7),255,np.uint8)).save(mask)
    native=tmp_path/'native.npy';np.save(native,np.full((14,14,3),128,np.uint8))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(train_only=True,records=[dict(stem='train',file='native.npy',hd_sha256=sha(image),native_sha256=sha(native))])))
    ds=SimpleNamespace(image_filenames=[image],metadata=dict(distillation_root=str(tmp_path),
        distillation_rows=[dict(mask_path='mask.png',alpha_file_path='mask.png')]))
    excluded=torch.zeros(1,7,7,dtype=torch.bool);excluded[0,3,3]=True
    original=NativeTrainingTargets(manifest,ds,'cpu')
    modified=NativeTrainingTargets(manifest,ds,'cpu',exclude=excluded)
    assert original.core[0,2:5,2:5].all()
    assert not modified.core[0,2:5,2:5].any()
    assert modified.core[0,1,1] and not modified.core[0,0].any()


def test_native_sampling_preserves_pixel_centers_and_linear_subpixel_ramp():
    image = torch.zeros((1, 4, 6, 3), dtype=torch.uint8)
    image[0, :, :, 0] = torch.arange(6)*20
    image[0, :, :, 1] = torch.arange(4)[:, None]*30
    coords = torch.tensor([[.5, .5], [1., 1.], [1.5, 2.5]])
    sampled = sample_native_rgb(image, torch.zeros(3, dtype=torch.long), coords, 2, 3)
    torch.testing.assert_close(sampled*255, torch.tensor([[10., 15., 0.], [30., 45., 0.], [90., 75., 0.]]))


def test_native_targets_do_not_change_uncertain_rays_or_targets():
    target = NativeTrainingTargets.__new__(NativeTrainingTargets)
    target.images = torch.full((1, 8, 8, 3), 204, dtype=torch.uint8)
    target.core = torch.zeros((1, 4, 4), dtype=torch.bool); target.core[0, 1, 1] = True
    target.height = target.width = 4
    cameras = Cameras(camera_to_worlds=torch.eye(4)[:3][None], fx=10., fy=10., cx=2., cy=2., width=4, height=4)
    coords = torch.tensor([[1.5, 1.5], [2.5, 2.5]])
    expected = cameras.generate_rays(camera_indices=torch.zeros((2, 1), dtype=torch.long), coords=coords)
    old = torch.full((2, 3), .2)
    batch = dict(indices=torch.tensor([[0, 1, 1], [0, 2, 2]]), mask=torch.ones(2, 1),
                 image=old, foreground_target=old, alpha_target=torch.tensor([[1.], [.5]]))
    rays, new = target.apply(cameras, batch)
    torch.testing.assert_close(new['image'][0], torch.full((3,), .8))
    torch.testing.assert_close(new['image'][1], old[1], rtol=0, atol=0)
    torch.testing.assert_close(new['foreground_target'][1], old[1], rtol=0, atol=0)
    torch.testing.assert_close(rays.directions[1], expected.directions[1], rtol=0, atol=0)
    assert not torch.equal(rays.directions[0], expected.directions[0])
    torch.testing.assert_close(batch['image'], old, rtol=0, atol=0)


def test_native_composite_uses_same_ray_photo_and_background_and_falls_back(monkeypatch):
    from nerfstudio.model_components.observed_background import ObservedBackgroundTargets
    target=NativeTrainingTargets.__new__(NativeTrainingTargets)
    y,x=torch.meshgrid(torch.arange(10),torch.arange(10),indexing='ij')
    target.images=torch.stack([x*10,y*20,x*0],-1).to(torch.uint8)[None]
    target.core=torch.zeros(1,5,5,dtype=torch.bool);target.height=target.width=5
    plates=ObservedBackgroundTargets.__new__(ObservedBackgroundTargets)
    yy,xx=torch.meshgrid(torch.arange(5),torch.arange(5),indexing='ij')
    plates.rgb=torch.stack([xx*20,yy*40,xx*0],-1).to(torch.uint8)[None]
    plates.valid=torch.ones(1,5,5,dtype=torch.bool);plates.valid[0,4,4]=False
    camera=Cameras(camera_to_worlds=torch.eye(4)[:3][None],fx=10.,fy=10.,cx=2.,cy=2.,width=5,height=5)
    batch=dict(indices=torch.tensor([[0,1,1],[0,3,3],[0,2,2]]),mask=torch.ones(3,1),
        image=torch.full((3,3),.2),foreground_target=torch.full((3,3),.3),alpha_target=torch.full((3,1),.4),
        observed_background_valid=torch.tensor([[1.],[1.],[0.]]),observed_background_rgb=torch.full((3,3),.6))
    monkeypatch.setattr(torch,'rand_like',lambda x:torch.full_like(x,.75))
    rays,got=target.apply(camera,batch,observed=plates)
    coords=torch.tensor([[1.75,1.75],[3.5,3.5],[2.5,2.5]])
    expected=camera.generate_rays(camera_indices=torch.zeros(3,1,dtype=torch.long),coords=coords)
    torch.testing.assert_close(rays.directions,expected.directions)
    torch.testing.assert_close(got['image'][0]*255,torch.tensor([30.,60.,0.]))
    torch.testing.assert_close(got['observed_background_rgb'][0]*255,torch.tensor([25.,50.,0.]))
    torch.testing.assert_close(got['image'][1:],batch['image'][1:],rtol=0,atol=0)
    torch.testing.assert_close(got['observed_background_rgb'][1:],batch['observed_background_rgb'][1:],rtol=0,atol=0)
    torch.testing.assert_close(got['foreground_target'],batch['foreground_target'],rtol=0,atol=0)
    assert got['native_observed_valid'].flatten().tolist()==[True,False,False]
    assert target.observed_sampling_summary['fallback_incomplete_footprint']==1
