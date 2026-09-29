import hashlib
import json
import numpy as np
import pytest
import torch

from nerfstudio.model_components.stereo_depth_evidence import conditional_depth_interval,opaque_surface_intervals,StereoDepthTargets


def test_conditional_depth_has_shape_gradient_but_no_opacity_scale_preference():
    weights=torch.tensor([[[.2],[.3]]],requires_grad=True)
    distances=torch.tensor([[[2.],[2.5]]]);target=torch.tensor([[2.]]);sigma=torch.tensor([[.1]]);valid=torch.ones(1,1)
    value=conditional_depth_interval(weights,distances,target,sigma,valid)
    torch.testing.assert_close(value,conditional_depth_interval(weights*.4,distances,target,sigma,valid))
    torch.testing.assert_close(value,conditional_depth_interval(weights,distances*7,target*7,sigma*7,valid))
    value.backward();assert weights.grad[0,0,0]<0 and weights.grad[0,1,0]>0
    torch.testing.assert_close((weights.grad*weights).sum(),torch.tensor(0.),atol=1e-6,rtol=0)


def test_unknown_targets_do_not_contaminate_and_correct_interval_is_flat():
    weights=torch.ones(2,2,1,requires_grad=True)
    distances=torch.tensor([[[1.],[1.1]],[[9.],[10.]]]);target=torch.tensor([[1.],[float('nan')]])
    sigma=torch.tensor([[.1],[float('nan')]]);valid=torch.tensor([[1.],[0.]])
    value=conditional_depth_interval(weights,distances,target,sigma,valid)
    assert value==0;value.backward();assert not weights.grad.any()


def test_subpixel_neighborhoods_exclude_opacity_and_surface_edges():
    depths=torch.ones(2,9,9);sigmas=torch.full_like(depths,.01);opaque=torch.ones(9,9,dtype=torch.bool)
    depths[:,:,5:]=2
    opaque[2,2]=False;depths[:,0,0]=float('nan')
    z,s,valid=opaque_surface_intervals(depths,sigmas,opaque)
    assert not valid[:,4:6].any() and not valid[1:4,1:4].any()
    assert valid[6,2] and valid[6,7]
    assert torch.isfinite(z).all() and torch.isfinite(s).all()
    torch.testing.assert_close(z[6,2],torch.tensor(1.))
    assert not valid[0].any() and not valid[-1].any()


def test_stereo_loader_binds_original_photos_calibration_and_depth(tmp_path):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    root=tmp_path/'data';root.mkdir();(root/'transforms.json').write_text('{}')
    images=[]
    for i in range(4):
        path=root/f'{i}.png';path.write_bytes(bytes([i]));images.append(path)
    stage=tmp_path/'stage';stage.mkdir()
    (stage/'request.json').write_text(json.dumps(dict(heldout_used=False,data_manifest_sha256=digest(root/'transforms.json'))))
    path=tmp_path/'camera_00_disjoint_00_02.npz'
    np.savez(path,depths=np.ones((2,5,5),np.float32),sigmas=np.full((2,5,5),.01,np.float32))
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(arguments=dict(stereo=str(stage)),uses_eval=False,staged_request_sha256=digest(stage/'request.json'),
        source_image_sha256={str(i):digest(p) for i,p in enumerate(images)},disjoint_pairs=[dict(index=0,pairs=[[0,1],[2,3]])],depth_artifacts={path.name:digest(path)})))
    class Dataset:
        metadata=dict(distillation_root=root,distillation_split='train',distillation_rows=[dict(h=5,w=5)]*4)
        image_filenames=images
        def __getitem__(self,index):
            return {k:torch.full((5,5,1),0. if k=='empty_mask' else 1.) for k in ['alpha_target','mask','confidence','alpha_valid','empty_mask']}
    ds=Dataset();loaded=StereoDepthTargets(receipt,ds)
    out=loaded.apply(dict(indices=torch.tensor([[0,2,2],[1,2,2],[0,0,0]])))
    torch.testing.assert_close(out['stereo_depth_valid'],torch.tensor([[1.],[0.],[0.]]))
    old=images[2].read_bytes();images[2].write_bytes(b'changed')
    with pytest.raises(ValueError,match='photograph'):StereoDepthTargets(receipt,ds)
    images[2].write_bytes(old);(root/'transforms.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='identity'):StereoDepthTargets(receipt,ds)
    (root/'transforms.json').write_text('{}');path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='arrays'):StereoDepthTargets(receipt,ds)
