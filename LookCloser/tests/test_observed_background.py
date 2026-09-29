import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from nerfstudio.model_components.observed_background import ObservedBackgroundTargets,observed_composite_objective


def test_subpixel_background_requires_only_nonzero_interpolation_support():
    targets=ObservedBackgroundTargets.__new__(ObservedBackgroundTargets)
    y,x=torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')
    targets.rgb=torch.stack([x*20,y*30,x*0],-1).to(torch.uint8)[None]
    targets.valid=torch.ones(1,4,4,dtype=torch.bool);targets.valid[0,2,2]=False
    coords=torch.tensor([[1.5,1.5],[1.75,1.75],[2.25,2.25],[.25,.5],[.5,.5],[1.25,1.25]])
    rgb,safe=targets.sample_subpixel(torch.zeros(6,dtype=torch.long),coords)
    assert safe.tolist()==[True,False,False,False,True,True]
    torch.testing.assert_close(rgb[0]*255,torch.tensor([20.,30.,0.]))
    torch.testing.assert_close(rgb[-1]*255,torch.tensor([15.,22.5,0.]))
    assert not rgb[~safe].any()


def test_composite_gradients_recover_dark_foreground_and_respect_intervals():
    alpha=torch.tensor([[.2]],requires_grad=True)
    value=observed_composite_objective(alpha*.1,alpha,torch.full((1,3),.8),torch.full((1,3),.2),torch.ones(1,1),torch.full((1,1),.02))
    value.backward();assert alpha.grad.item()<0
    alpha.grad.zero_()
    value=observed_composite_objective(alpha*.1,alpha,torch.full((1,3),.8),torch.full((1,3),.66),torch.ones(1,1),torch.full((1,1),.02))
    value.backward();assert value==0 and alpha.grad.item()==0


def test_unknown_nan_background_has_no_value_or_gradient():
    rgb=torch.zeros(2,3,requires_grad=True);a=torch.zeros(2,1,requires_grad=True)
    bg=torch.tensor([[.7,.7,.7],[float('nan')]*3]);valid=torch.tensor([[1.],[0.]])
    value=observed_composite_objective(rgb,a,bg,torch.full((2,3),.5),valid,torch.zeros(2,1))
    value.backward();assert torch.isfinite(rgb.grad).all() and not rgb.grad[1].any() and not a.grad[1].any()


def test_plate_identity_and_opaque_ray_exclusion(tmp_path):
    root=tmp_path/'data';root.mkdir();plate=tmp_path/'plates';(plate/'train').mkdir(parents=True)
    rgb=np.full((2,3,3),100,np.uint8);Image.fromarray(rgb).save(root/'train.png')
    Image.fromarray(rgb).save(plate/'train/background.png');Image.fromarray(np.full((2,3),255,np.uint8)).save(plate/'train/valid.png')
    row=dict(physical_camera='a',file_path='train.png',w=3,h=2,fl_x=10.,fl_y=10.,transform_matrix=np.eye(4).tolist())
    meta=dict(frames=[row],train_filenames=['train.png']);(root/'transforms.json').write_text(json.dumps(meta))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(physical_camera='a',background_sha256=digest(plate/'train/background.png'),valid_sha256=digest(plate/'train/valid.png'),background_residual_p90=.02)
    receipt=dict(arguments=dict(data=str(root)),source_manifest_sha256=digest(root/'transforms.json'),records=[record],uses_eval_cameras=False,uses_target_frame_for_background=False,target_conversion_byte_exact=True)
    (plate/'receipt.json').write_text(json.dumps(receipt))
    target=tmp_path/'train_target.png';Image.fromarray(rgb).save(target)
    target_root=tmp_path/'target';target_root.mkdir();(target_root/'transforms.json').write_text(json.dumps(meta))
    ds=SimpleNamespace(image_filenames=[target],metadata=dict(distillation_split='train',distillation_rows=[row],distillation_root=str(target_root)))
    targets=ObservedBackgroundTargets(plate,ds)
    batch=dict(indices=torch.tensor([[0,0,0],[0,1,1]]),alpha_target=torch.tensor([[1.],[.3]]))
    got=targets.apply(batch);assert got['observed_background_valid'].flatten().tolist()==[0.,1.]
    torch.testing.assert_close(got['observed_background_error'],torch.full((2,1),.02))
    record['supervision_error']=.03
    (plate/'receipt.json').write_text(json.dumps(receipt))
    extended=ObservedBackgroundTargets(plate,ds).apply(batch)
    torch.testing.assert_close(extended['observed_background_error'],torch.full((2,1),.03))
    matte=tmp_path/'mattes'/target.stem;matte.mkdir(parents=True)
    tri=np.array([[128,255,0],[128,255,0]],dtype=np.uint8)
    Image.fromarray(tri).save(matte/'trimap.png')
    Image.fromarray(np.full((2,3),255,np.uint8)).save(target_root/'alpha.png')
    row['alpha_file_path']='alpha.png'
    proof=dict(actual_eval_used=False,physical_camera='a',source_sha256=digest(target),
               alpha_sha256=digest(target_root/'alpha.png'),trimap_sha256=digest(matte/'trimap.png'))
    (matte/'receipt.json').write_text(json.dumps(proof))
    provenance=ObservedBackgroundTargets(plate,ds,matte.parent)
    opaque=dict(indices=torch.tensor([[0,0,0],[0,0,1],[0,0,2]]),alpha_target=torch.ones(3,1))
    assert provenance.apply(opaque)['observed_background_valid'].flatten().tolist()==[1.,0.,0.]
    assert provenance.opaque_override[0].tolist()==[[True,False,False],[True,False,False]]
    # Observed unknowns may override saturation; unobserved ones may not.
    provenance.valid[0,0,0]=False
    assert not provenance.apply(opaque)['observed_background_valid'].any()
    proof['actual_eval_used']=True;(matte/'receipt.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='train-only'):ObservedBackgroundTargets(plate,ds,matte.parent)
    proof['actual_eval_used']=False;proof['alpha_sha256']='wrong'
    (matte/'receipt.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='identity mismatch'):ObservedBackgroundTargets(plate,ds,matte.parent)
    global_row=dict(row);global_row.pop('fl_x');ds.metadata['distillation_rows']=[global_row]
    (target_root/'transforms.json').write_text(json.dumps(dict(meta,fl_x=11.)))
    with pytest.raises(ValueError,match='calibration changed'):ObservedBackgroundTargets(plate,ds)
    ds.metadata['distillation_rows']=[row]
    Image.fromarray(rgb+1).save(target)
    with pytest.raises(ValueError,match='photograph changed'):ObservedBackgroundTargets(plate,ds)
    Image.fromarray(rgb).save(target)
    Image.fromarray(rgb+1).save(plate/'train/background.png')
    with pytest.raises(ValueError,match='plate identity'):ObservedBackgroundTargets(plate,ds)


def test_model_replaces_conflicting_matte_and_empty_terms_only_on_observed_rays(monkeypatch):
    from nerfstudio.models.lookcloser import LookCloserModel
    from nerfstudio.pipelines.mesh_distillation_pipeline import DistillationModel,DistillationModelConfig
    monkeypatch.setattr(LookCloserModel,'get_loss_dict',lambda *args,**kwargs:{})
    model=object.__new__(DistillationModel);torch.nn.Module.__init__(model)
    model.device_indicator_param=torch.nn.Parameter(torch.zeros(1));model.config=DistillationModelConfig()
    model.config.matte_opacity_weight=.3;model.config.empty_opacity_weight=.1
    model.config.depth_loss_steps=0;model.current_train_step=0
    alpha=torch.full((2,1),.5,requires_grad=True);rgb=alpha*torch.full((2,3),.2)
    out=dict(rgb=rgb,actor_rgb=rgb,accumulation=alpha,optical_thickness=-torch.log1p(-alpha))
    batch=dict(image=torch.full((2,3),.5),confidence=torch.ones(2,1),mask=torch.ones(2,1),
               alpha_valid=torch.ones(2,1),foreground_target=torch.zeros(2,3),alpha_target=torch.zeros(2,1),empty_mask=torch.ones(2,1))
    baseline=model.get_loss_dict(out,batch)
    inactive=dict(batch,observed_background_rgb=torch.full((2,3),.8),observed_background_valid=torch.zeros(2,1),observed_background_error=torch.zeros(2,1))
    got=model.get_loss_dict(out,inactive)
    for key in baseline:torch.testing.assert_close(got[key],baseline[key],rtol=0,atol=0)
    active=dict(inactive,observed_background_valid=torch.ones(2,1));got=model.get_loss_dict(out,active)
    assert got['known_empty']==0 and got['matte_opacity']==0 and got['rgb_loss']==0
    assert got['observed_composite']==0
    sum(got.values()).backward();assert torch.isfinite(alpha.grad).all() and not alpha.grad.any()
    model.config.optimize_training_cameras=True
    with pytest.raises(ValueError,match='fixed training cameras'):model.get_loss_dict(out,active)
