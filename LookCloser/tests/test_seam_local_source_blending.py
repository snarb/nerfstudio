from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from seam_local_source_blending import seam_local_weights,blend_seam_sources


def fixture():
    torch.set_num_threads(4)
    valid=np.ones((2,32,80),bool);labels=np.zeros((32,80),np.int32);labels[:,40:]=1
    return valid,labels,np.ones((32,80),np.float32)


def test_blend_is_local_normalized_and_only_visible():
    valid,labels,depth=fixture()
    weights,band,stats=seam_local_weights(valid,labels,depth,radius=8)
    np.testing.assert_allclose(weights.sum(0),1,atol=1e-7)
    assert (weights[0,:,:30]==1).all() and not weights[1,:,:30].any()
    assert (weights[1,:,50:]==1).all() and not weights[0,:,50:].any()
    assert stats['mixed_pixels']>0 and not band[:,:30].any()


def test_true_depth_boundary_is_not_a_texture_seam():
    valid,labels,depth=fixture();depth[:,40:]=2
    weights,band,stats=seam_local_weights(valid,labels,depth,radius=8)
    assert not band.any() and not stats['mixed_pixels']
    np.testing.assert_array_equal(weights,np.arange(2)[:,None,None]==labels)


def test_source_visibility_boundary_fades_without_invisible_rgb():
    valid,labels,depth=fixture();valid[0,:,40:]=False
    weights,_,_=seam_local_weights(valid,labels,depth,radius=8)
    assert not weights[0,:,40:].any()
    assert weights[0,16,39]<.15


def test_no_seam_and_missing_surface_exact_identity():
    valid,labels,depth=fixture();labels[:]=0;labels[:,:4]=-1;depth[:,:4]=0;valid[:,:,:4]=False
    a=torch.full((3,32,80),.3);b=torch.full_like(a,.5)
    out,weights,band,stats=blend_seam_sources([a,b],torch.tensor(valid),torch.tensor(labels),torch.tensor(depth),radius=8)
    expected=a.clone();expected[:,:,:4]=0
    for value in out.values():assert torch.equal(value,expected)
    assert not band.any() and not stats['mixed_pixels']


def test_outside_mixture_rgb_unchanged_and_full_seam_step_reduced():
    valid,labels,depth=fixture();a=torch.full((3,32,80),.3);b=torch.full_like(a,.5)
    out,weights,band,stats=blend_seam_sources([a,b],torch.tensor(valid),torch.tensor(labels),torch.tensor(depth),radius=8)
    for value in out.values():
        assert torch.equal(value[:,:,:30],a[:,:,:30]) and torch.equal(value[:,:,50:],b[:,:,50:])
    assert float((out['full_rgb'][:,16,40]-out['full_rgb'][:,16,39]).abs().max())<.03


@pytest.mark.parametrize('bad',['negative_depth','invisible_selected','fractional_label','bad_radius'])
def test_invalid_geometry_fails_closed(bad):
    valid,labels,depth=fixture();kwargs={}
    if bad=='negative_depth':depth[0,0]=-1
    if bad=='invisible_selected':valid[0,0,0]=False
    if bad=='fractional_label':labels=labels.astype(float)+.1
    if bad=='bad_radius':kwargs['radius']=0
    with pytest.raises(ValueError):seam_local_weights(valid,labels,depth,**kwargs)
