from pathlib import Path
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from angular_surface_color import fit_angular_coefficients,angular_log_gain,calibrated_camera_rgb,validate_camera_response


def test_mesh_field_recovers_known_direction_gain_and_ignores_held_rgb():
    xx,yy=np.meshgrid(np.arange(6),np.arange(5));vertices=np.c_[xx.ravel(),yy.ravel(),xx.ravel()*0]*.0005
    triangles=[]
    for y in range(4):
        for x in range(5):
            i=y*6+x;triangles.extend([[i,i+1,i+6],[i+1,i+7,i+6]])
    directions=torch.tensor([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])[:,None].expand(-1,30,-1)
    true=torch.tensor([[.1,.15,.2],[-.2,.1,.05],[.3,.05,-.1]])
    logrgb=torch.einsum('cnk,kj->cnj',directions,true)+torch.arange(30)[None,:,None]*.01
    valid=torch.ones((6,30),dtype=torch.bool);held=torch.zeros(30,dtype=torch.bool);held[13:16]=True
    actual,stats=fit_angular_coefficients(logrgb,directions,valid,vertices,np.array(triangles),held,smoothness=1.,ridge=.0001)
    assert stats['converged']
    torch.testing.assert_close(actual,true[None].expand(30,-1,-1),atol=1e-4,rtol=0)
    poison=logrgb.clone();poison[:,held]=10
    other,_=fit_angular_coefficients(poison,directions,valid,vertices,np.array(triangles),held,smoothness=1.,ridge=.0001)
    torch.testing.assert_close(actual,other)


def test_angular_gain_is_identity_for_same_camera_and_clamped():
    world=torch.zeros((2,2,3));field=torch.ones((2,2,3,3))*100
    center=torch.tensor([0.,0.,1.])
    assert (angular_log_gain(field,world,center,center)==0).all()
    assert angular_log_gain(field,world,center,-center).max()<=np.log(2)+1e-7


def test_spatial_camera_response_matches_renderer_and_is_opt_in():
    from patchmatch_color_calibration import apply_camera_gain
    rgb=torch.full((3,12,18),.4)
    row={'rgb_gain':[1.,1.,1.],'exposure_gain':[1.1,1.1,1.1],
         'spatial_log_gain_grid':[[0.,.2],[-.1,.3]]}
    torch.testing.assert_close(calibrated_camera_rgb(rgb,row),rgb)
    expected=apply_camera_gain(rgb,row['exposure_gain'],row['spatial_log_gain_grid'])
    actual=calibrated_camera_rgb(rgb,row,'spatial')
    torch.testing.assert_close(actual,expected)
    assert (actual-rgb).abs().max()>.01
    with pytest.raises(ValueError):calibrated_camera_rgb(rgb,row,'unknown')


def test_angular_fit_response_must_match_renderer_mode():
    validate_camera_response({},'rgb')
    validate_camera_response({'camera_color_model':'spatial'},'spatial')
    with pytest.raises(ValueError):validate_camera_response({'camera_color_model':'rgb'},'spatial')
    with pytest.raises(ValueError):validate_camera_response({'camera_color_model':'spatial'},'rgb')
