"""Calibration, selection and retained density/SH contracts for Luster."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_luster_frame import resize_intrinsics, ray_box_hits
from run_luster_experiment import select_best
from nerfstudio.fields.lookcloser_field import LookCloserField


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
