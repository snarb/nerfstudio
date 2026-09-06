from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from visible_source_blending import blend_visible_sources,visibility_weights,to_linear,to_display


def fixture():
    torch.set_num_threads(4)
    a=torch.full((3,32,40),.3);b=torch.full_like(a,.5)
    valid=[torch.ones((32,40),dtype=torch.bool)]*2
    return a,b,valid,torch.zeros((32,40),dtype=torch.long),torch.ones((32,40))


def test_constant_two_source_mix_is_explicit_linear_display_average():
    a,b,valid,labels,depth=fixture()
    out,weights,stats=blend_visible_sources([a,b],valid,labels,depth,[1,1])
    expected=to_display((to_linear(a)+to_linear(b))/2)
    torch.testing.assert_close(out['full_rgb'],expected,atol=1e-6,rtol=0)
    torch.testing.assert_close(out['low_band'],expected,atol=2e-6,rtol=0)
    assert stats['source_rgb_averaging'] and torch.allclose(weights,torch.full_like(weights,.5))


def test_invalid_source_does_not_contribute_and_empty_support_stays_black():
    a,b,valid,labels,depth=fixture();valid=[valid[0].clone(),valid[1].clone()]
    valid[1][:,:20]=False;b=b.clone();b[:,:,:20]=100
    valid[0][:,:3]=False;labels[:,:3]=-1
    out,weights,_=blend_visible_sources([a,b],valid,labels,depth,[1,1])
    assert not weights[1,:,:20].any() and not weights[:,:,:3].any()
    assert not out['full_rgb'][:,:,:3].any()
    torch.testing.assert_close(out['full_rgb'][:,:,3:20],a[:,:,3:20],atol=1e-6,rtol=0)


def test_feathered_visibility_handoff_has_no_hard_color_step():
    a,b,valid,labels,depth=fixture();valid=[valid[0].clone(),valid[1].clone()]
    valid[0][:,20:]=False;labels[:,20:]=1
    out,_,_=blend_visible_sources([a,b],valid,labels,depth,[1,1],feather_pixels=8)
    edge=float((out['full_rgb'][:,16,20]-out['full_rgb'][:,16,19]).abs().max())
    assert edge<.03


def test_low_band_retains_selected_high_frequency_detail():
    a,b,valid,labels,depth=fixture();a[:]=.4;b[:]=.4
    yy,xx=torch.meshgrid(torch.arange(32),torch.arange(40),indexing='ij')
    a=a+(((yy+xx)%2)*.04-.02)
    out,_,_=blend_visible_sources([a,b],valid,labels,depth,[1,1])
    assert float(out['low_band'][:,8:24,8:32].std())>1.8*float(out['full_rgb'][:,8:24,8:32].std())


def test_far_only_visible_source_still_has_normalized_weight():
    valid=[torch.zeros((8,8),dtype=torch.bool),torch.zeros((8,8),dtype=torch.bool),torch.ones((8,8),dtype=torch.bool)]
    weights,_=visibility_weights(valid,[0,0,1000])
    assert torch.equal(weights[2],torch.ones((8,8)))


@pytest.mark.parametrize('distances',[[1,np.nan],[-1,1],[1]])
def test_bad_camera_prior_fails_closed(distances):
    _,_,valid,_,_=fixture()
    with pytest.raises(ValueError):visibility_weights(valid,distances)
