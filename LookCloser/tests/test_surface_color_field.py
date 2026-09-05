from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from surface_color_field import solve_surface_field,correct_surface_colors
from patchmatch_color_calibration import apply_camera_gain


def test_harmonic_gain_extends_through_missing_observations():
    target=torch.full((1,3,30,40),.2);observed=torch.ones((1,1,30,40),dtype=torch.bool)
    observed[:,:,8:22,12:28]=False
    field,stats=solve_surface_field(target,observed,torch.ones((30,40)),smoothness=16)
    assert stats['converged']
    torch.testing.assert_close(field,torch.full_like(field,.2),atol=5e-4,rtol=0)


def test_gain_does_not_leak_between_disconnected_depth_layers():
    target=torch.full((1,3,20,40),.3);observed=torch.ones((1,1,20,40),dtype=torch.bool)
    observed[:,:,:,20:]=False
    depth=torch.ones((20,40));depth[:,20:]=2
    field,stats=solve_surface_field(target,observed,depth,smoothness=16)
    assert stats['converged']
    torch.testing.assert_close(field[:,:,:,20:],torch.zeros_like(field[:,:,:,20:]),atol=1e-7,rtol=0)


def test_hidden_primary_not_used_and_single_source_detail_preserved():
    primary=torch.full((3,40,60),.45);primary[:,::2]=.5
    secondary=apply_camera_gain(primary,[1.25,.9,1.1])
    visible=torch.ones((40,60),dtype=torch.bool);visible[10:30,20:40]=False
    out,stats=correct_surface_colors([primary,secondary],[visible,torch.ones_like(visible)],torch.ones((40,60)),smoothness=16,holdout=False)
    assert stats['converged'] and out[0] is primary and not stats['source_averaging']
    torch.testing.assert_close(out[1],primary,atol=1e-4,rtol=0)
    poison=primary.clone();poison[:,10:30,20:40]=.1
    changed,_=correct_surface_colors([poison,secondary],[visible,torch.ones_like(visible)],torch.ones((40,60)),smoothness=16,holdout=False)
    torch.testing.assert_close(changed[1],out[1],atol=1e-7,rtol=0)


def test_no_shared_observation_leaves_gain_one():
    first=torch.full((3,20,20),.4);second=torch.full_like(first,.6)
    a=torch.zeros((20,20),dtype=torch.bool);a[:,:8]=True
    b=torch.zeros_like(a);b[:,12:]=True
    out,stats=correct_surface_colors([first,second],[a,b],torch.ones((20,20)),smoothness=16)
    assert stats['converged']
    torch.testing.assert_close(out[1],second)
