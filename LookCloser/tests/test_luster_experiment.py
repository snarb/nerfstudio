"""Calibration, selection and retained density/SH contracts for Luster."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_luster_frame import resize_intrinsics, ray_box_hits, repair_background_strip
from run_luster_experiment import select_best
from nerfstudio.fields.lookcloser_field import LookCloserField
from nerfstudio.models.lookcloser import LookCloserModel
from run_luster_experiment import load_background_lookup
from PIL import Image
import pytest


def test_temporal_strip_repair_preserves_connected_moving_arm():
    mask=np.zeros((40,60),dtype=np.uint8)
    mask[10:35,25:45]=255
    mask[15:20,5:30]=255  # Hand extends into the formerly cleared strip.
    mask[1:8,1:3]=255    # Disconnected background stand.
    mask[1:8,17:23]=255  # A stand fragment crosses the old strip boundary.
    repaired=repair_background_strip(mask,width=20)
    np.testing.assert_array_equal(repaired[15:20,5:30],mask[15:20,5:30])
    assert not repaired[1:8,1:3].any()
    assert not repaired[1:8,17:23].any()
    np.testing.assert_array_equal(repaired[10:35,25:45],mask[10:35,25:45])


def test_background_penalty_only_pushes_trusted_background_transparent():
    opacity=torch.tensor([[.2],[.7],[30.]],requires_grad=True)
    value=LookCloserModel.background_opacity_penalty(opacity,torch.tensor([[True],[False],[True]]))
    value.backward()
    assert opacity.grad[0]>0 and opacity.grad[2]>0 and opacity.grad[1]==0
    assert torch.isfinite(value)
    # Thickness30 rounds alpha to1 in FP32, but must retain a clearing gradient.
    assert (1-torch.exp(-opacity[2].detach())).item()==1.


def test_background_thickness_matches_unsaturated_alpha_bce():
    thickness=torch.tensor([[.1],[1.],[3.]])
    alpha=1-torch.exp(-thickness)
    penalty=LookCloserModel.background_opacity_penalty(thickness,torch.ones_like(thickness,dtype=torch.bool))
    torch.testing.assert_close(penalty,-torch.log1p(-alpha).mean())


def test_background_penalty_empty_mask_has_zero_gradient():
    opacity=torch.tensor([[1.],[.4]],requires_grad=True)
    value=LookCloserModel.background_opacity_penalty(opacity,torch.zeros_like(opacity,dtype=torch.bool))
    value.backward()
    assert value==0 and torch.equal(opacity.grad,torch.zeros_like(opacity))
    with pytest.raises(ValueError):
        LookCloserModel.background_opacity_penalty(opacity,torch.ones_like(opacity))


def test_background_lookup_respects_camera_order_soft_margin_and_exclusions(tmp_path):
    (tmp_path/'masks').mkdir()
    filenames=[]
    for cid,shape in [(97,(5,8)),(12,(8,5)),(164,(5,8))]:
        name=f'cam_{cid:03d}_000470.png';mask=np.zeros(shape,dtype='uint8');mask[2,2]=1
        Image.fromarray(mask).save(tmp_path/'masks'/name);filenames.append(Path(name))
    lookup=load_background_lookup(SimpleNamespace(image_filenames=filenames),dict(data=str(tmp_path),background_mask_margin=1,background_mask_exclude_cameras=[164]))
    result=lookup(torch.tensor([[0,0,7],[1,7,4],[0,1,1],[0,2,2],[2,0,7]]))
    assert result.shape==(5,1) and result.dtype==torch.bool
    assert result[:,0].tolist()==[True,True,False,False,False]


def test_resize_projection_uses_actual_axis_scales():
    original=(3000,4096);target=(1406,1920)
    k=np.array([4693.,4686.,1505.,2000.])
    scaled=np.array(resize_intrinsics(k,original,target))
    xyz=np.array([[.1,.2,1.],[-.4,.6,2.]])
    source=xyz[:,:2]/xyz[:,2:] * k[:2]+k[2:]
    resized=xyz[:,:2]/xyz[:,2:] * scaled[:2]+scaled[2:]
    np.testing.assert_allclose(resized,source*np.array(target)/np.array(original),atol=1e-10)


def test_ray_bounds_do_not_accept_box_behind_camera():
    hit,positive=ray_box_hits(np.array([0.,0.,2.]),np.array([[0.,0.,1.],[0.,0.,-1.]]),np.array([[-1.,-1.,-1.],[1.,1.,1.]]))
    assert (hit&positive).tolist()==[False,True]


def test_checkpoint_psnr_window_before_lpips():
    rows=[dict(step=1,eval_all_psnr=30.,eval_all_lpips=.2),
          dict(step=2,eval_all_psnr=29.95,eval_all_lpips=.19),
          dict(step=3,eval_all_psnr=29.8,eval_all_lpips=.1)]
    assert select_best(rows)['step']==2


def test_exp_casts_logits_before_bias_and_activation():
    field=SimpleNamespace(density_activation='trunc_exp',density_normalization='none')
    logits=torch.tensor([12.,15.],dtype=torch.float16)
    result=LookCloserField.activate_density(field,logits)
    assert result.dtype==torch.float32 and torch.isfinite(result).all()
    torch.testing.assert_close(result,torch.exp(logits.float()+1.))


def test_corrected_sh_input_domain():
    field=SimpleNamespace(correct_sh_directions=True,direction_encoding=lambda x:x)
    direction=torch.tensor([[-1.,0.,1.]])
    torch.testing.assert_close(LookCloserField.encode_directions(field,direction),torch.tensor([[0.,.5,1.]]))
