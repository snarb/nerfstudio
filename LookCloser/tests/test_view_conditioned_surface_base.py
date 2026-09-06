from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from view_conditioned_surface_base import direction_features,fit_directional_base,interpolation_gate


def fixture():
    y,x=np.mgrid[:4,:5];v=np.c_[x.ravel(),y.ravel(),np.zeros(20)]*.0005
    t=[]
    for j in range(3):
        for i in range(4):
            k=j*5+i;t.extend([[k,k+1,k+5],[k+1,k+6,k+5]])
    d=torch.randn((18,1,3),generator=torch.Generator().manual_seed(45),dtype=torch.float64).repeat(1,20,1)
    visible=torch.ones((18,20),dtype=torch.bool)
    return v,np.asarray(t),d,visible


@pytest.mark.parametrize('degree,size',[(0,1),(1,4),(2,9)])
def test_features_normalize_directions(degree,size):
    _,_,d,_=fixture();a=direction_features(d,degree)
    assert a.shape==(18,20,size)
    torch.testing.assert_close(a,direction_features(d*3,degree),rtol=1e-10,atol=1e-10)


def test_invalid_direction_rejected():
    with pytest.raises(ValueError):direction_features(torch.zeros((3,3)),1)
    with pytest.raises(ValueError):direction_features(torch.ones((3,3)),3)


@pytest.mark.parametrize('degree',[0,1,2])
def test_constant_camera_rgb_is_constant_for_every_query(degree):
    v,t,d,visible=fixture();rgb=torch.full_like(d,.4)
    coeff,stats=fit_directional_base(rgb,d,visible,v,t,degree=degree)
    assert stats['converged']
    pred=torch.einsum('cnk,nkl->cnl',direction_features(d.float(),degree),coeff)
    torch.testing.assert_close(pred,rgb.float(),rtol=0,atol=1e-6)


def test_linear_radiance_predicts_a_camera_not_in_fit():
    v,t,d,visible=fixture();phi=direction_features(d,1)
    coeff=torch.tensor([[.4,.35,.3],[.05,.03,-.02],[-.02,.04,.03],[.03,-.01,.02]],dtype=torch.float64)
    rgb=phi@coeff;visible[-1]=False
    fitted,stats=fit_directional_base(rgb,d,visible,v,t,degree=1,ridge=1e-6)
    assert stats['converged']
    pred=torch.einsum('nk,nkl->nl',phi[-1].float(),fitted)
    torch.testing.assert_close(pred,rgb[-1].float(),atol=2e-5,rtol=0)


def test_withheld_camera_values_cannot_change_fit():
    v,t,d,visible=fixture();rgb=torch.rand(d.shape,generator=torch.Generator().manual_seed(3))*.5
    visible[-3:]=False;a,_=fit_directional_base(rgb,d,visible,v,t,degree=1)
    rgb[~visible]=float('nan');b,_=fit_directional_base(rgb,d,visible,v,t,degree=1)
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_bad_visible_values_fail():
    v,t,d,visible=fixture();rgb=torch.full_like(d,.4);rgb[0,0,0]=float('nan')
    with pytest.raises(ValueError):fit_directional_base(rgb,d,visible,v,t,degree=1)


def gate_rows(errors):
    return [dict(degree=i,fit={'converged':True},cameras=[dict(camera='train_'+str(j),vertices=20,mean_abs_rgb=e) for j,e in enumerate(es)]) for i,es in enumerate(errors)]


def test_gate_prefers_small_model_until_quadratic_adds_five_percent():
    rows=gate_rows([[.1,.1,.1],[.08,.08,.08],[.078,.078,.078]])
    assert interpolation_gate(rows)['selected_degree']==1
    rows[2]['cameras']=gate_rows([[.07,.07,.07]])[0]['cameras']
    assert interpolation_gate(rows)['selected_degree']==2


def test_gate_rejects_no_useful_improvement():
    assert not interpolation_gate(gate_rows([[.1]*3,[.099]*3,[.095]*3]))['eligible_for_render']


def test_gate_rejects_changed_inventory_and_nonfinite_errors():
    rows=gate_rows([[.1]*3,[.08]*3,[.07]*3]);rows[2]['cameras'][0]['vertices']=19
    with pytest.raises(ValueError):interpolation_gate(rows)
    rows=gate_rows([[.1]*3,[.08]*3,[.07]*3]);rows[2]['cameras'][0]['mean_abs_rgb']=float('nan')
    with pytest.raises(ValueError):interpolation_gate(rows)
