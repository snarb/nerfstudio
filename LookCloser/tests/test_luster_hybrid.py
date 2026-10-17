"""Meaningful geometry and leakage guards for frozen-field native transfer."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from luster_hybrid_common import assert_source, project, unproject
from render_luster_hybrid import packed_quantiles
from transfer_luster_hybrid import footprint_metrics,merge_details


def camera():
    return dict(fl_x=130.,fl_y=117.,cx=60.3,cy=41.7,w=120,h=80,transform_matrix=np.eye(4).tolist())


def test_pixel_centers_and_z_roundtrip():
    row=camera();xy=np.array([[0,0],[119,79],[55.2,37.8]],np.float32)
    z=np.array([1.,2.,3.],np.float32)
    world=unproject(z,row,xy)
    uv,actual=project(world,row)
    np.testing.assert_allclose(uv,xy,atol=1e-5)
    np.testing.assert_allclose(actual,z)
    assert np.linalg.norm(world[0])>z[0]


def test_source_split_and_self_exclusion():
    rows={i:dict(train=i not in (12,95,150)) for i in (7,8,12,95,150)}
    assert_source(rows,7,8)
    for i in (7,12,95,150):
        with pytest.raises(ValueError):assert_source(rows,7,i)


class Rays(SimpleNamespace):
    def __len__(self):return len(self.nears)


def test_quantiles_continuous_exponential_and_empty():
    rays=Rays(nears=torch.tensor([[2.],[2.],[2.]]),fars=torch.tensor([[4.],[4.],[4.]]))
    mass=1-np.exp(-6.)
    out=dict(packed_ray_indices=torch.tensor([0,2]),packed_weights=torch.tensor([[mass],[.5]]),
             packed_spacing_starts=torch.tensor([[0.],[0.]]),packed_spacing_ends=torch.tensor([[1.],[1.]]))
    got=packed_quantiles(out,rays)
    expected=2-np.log(1-np.array([.1,.5,.9])*mass)/3
    np.testing.assert_allclose(got[0].numpy(),expected,atol=1e-6)
    assert torch.isnan(got[1:]).all()


def test_quantiles_detect_two_layers_without_midpoint_surface():
    rays=Rays(nears=torch.tensor([[0.]]),fars=torch.tensor([[10.]]))
    out=dict(packed_ray_indices=torch.tensor([0,0]),packed_weights=torch.tensor([[.6],[.39]]),
             packed_spacing_starts=torch.tensor([[.1],[.8]]),packed_spacing_ends=torch.tensor([[.11],[.81]]))
    got=packed_quantiles(out,rays)[0]
    assert 1<got[1]<1.1 and got[2]-got[0]>6.8


def test_bilinear_footprint_rejects_one_wrong_depth_and_outside():
    rgb=np.ones((3,3,3),np.float32)
    depth=np.ones((3,3),np.float32)
    source=dict(alpha=depth.copy(),depth_z_q10=depth.copy(),depth_z_q50=depth.copy(),depth_z_q90=depth.copy())
    source['depth_z_q50'][1,1]=2
    uv=np.array([[[.5,.5],[-.1,0],[0,0]]],np.float32);z=np.ones((1,3),np.float32)
    _,valid,_,err,*_=footprint_metrics(rgb,source,uv,z,None,None,None,None,None)
    assert err[0,0]==1 and not valid[0,1] and valid[0,2] and err[0,2]==0


@pytest.mark.parametrize('mode',['direct','detail','multiband'])
def test_exact_fallback_and_no_filter_background_leak(mode):
    base=np.full((80,80,3),.4,np.float32);texture=np.full_like(base,.5)
    texture[:20]=1.;texture[60:]=0.
    labels=np.full((80,80),-1,np.int16);labels[20:60,20:60]=8
    confidence=(labels>=0).astype(np.float32)
    result,weight=merge_details(base,texture,labels,confidence,mode=mode)
    assert np.array_equal(result[weight==0],base[weight==0])
    if mode!='direct':np.testing.assert_allclose(result,base,atol=1e-6)


def test_area_footprint_preserves_magnification_and_coordinate_mapping():
    from antialias_luster_hybrid import footprint_transform
    angle=.71
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    jac=np.stack([np.eye(2)*.7,rotation@np.diag([3.,.5])@rotation.T])
    source,target,singular=footprint_transform(jac)
    np.testing.assert_allclose(source[0],0,atol=1e-12)
    np.testing.assert_allclose(jac@target,source,atol=1e-12)
    # Only the minified axis contributes additional integration variance.
    np.testing.assert_allclose(np.linalg.eigvalsh(source[1]@source[1].T),[0,8],atol=1e-12)


def test_high_frequency_corroboration_rejects_wrong_phase_and_noise():
    from coherent_luster_hybrid import corroboration
    rng=np.random.default_rng(27)
    first=rng.normal(0,.05,(64,64,3)).astype(np.float32)
    support=np.ones((64,64),bool)
    match,energy=corroboration(first,first*2,support,3.)
    opposite,_=corroboration(first,-first,support,3.)
    noise,_=corroboration(first,rng.normal(0,.05,first.shape).astype(np.float32),support,3.)
    assert match[8:-8,8:-8].min()>.99 and energy.mean()>.03
    assert opposite[8:-8,8:-8].max()<-.99
    assert abs(noise[8:-8,8:-8].mean())<.1


def test_supported_detail_does_not_depend_on_distant_selected_pixels():
    from supported_luster_hybrid import merge_supported,CONFIG
    shape=(80,80);base=np.full((*shape,3),.4,np.float32)
    texture=base.copy();texture[:,40:]=.6
    a={k:np.ones(shape,np.float32) for k in ['valid','alpha','resolution_ratio']}
    a.update({k:np.zeros(shape,np.float32) for k in ['depth_error','roundtrip','uncertainty','score']});a['rgb']=texture
    labels=np.full(shape,-1,np.int16);labels[20:60,20:60]=8
    other=labels.copy();other[10:20,10:70]=8
    result,w=merge_supported(base,[(8,a)],labels,np.ones(shape,np.float32),CONFIG)
    result2,w2=merge_supported(base,[(8,a)],other,np.ones(shape,np.float32),CONFIG)
    both=(w==1)&(w2==1)
    np.testing.assert_array_equal(result[both],result2[both])
    np.testing.assert_array_equal(result[w==0],base[w==0])


def test_registration_recovers_small_shift_without_gt():
    from align_luster_hybrid import registration
    import cv2
    rng=np.random.default_rng(4)
    base=cv2.GaussianBlur(rng.random((128,128,3),dtype=np.float32),(0,0),1.)
    source=np.roll(base,1,axis=1)
    flow,mask,_=registration(base,source,np.ones((128,128),bool))
    assert mask[24:-24,24:-24].mean()>.8
    assert abs(np.median(flow[24:-24,24:-24,0])-1)<.2
    empty,valid,_=registration(base,source,np.zeros((128,128),bool))
    assert not valid.any() and not empty.any()


def test_view_bracketing_uses_symmetric_sources_without_extrapolation():
    from bracket_luster_hybrid import bracket_weight
    angle=np.deg2rad(20)
    a=np.array([np.sin(angle),0,np.cos(angle)])
    b=np.array([-np.sin(angle),0,np.cos(angle)])
    target=np.array([0,0,1.])
    weight,error=bracket_weight(a,b,target)
    assert abs(weight-.5)<1e-8 and abs(error-(1-np.cos(angle)))<1e-8
    weight,_=bracket_weight(a,b,np.array([1,0,0]))
    assert weight==1


@pytest.mark.parametrize('module',['foreground_luster_hybrid','foreground_noise_luster_hybrid'])
def test_source_foreground_guard_separates_background_hole(module):
    import importlib
    evidence=importlib.import_module(module).foreground_evidence
    rng=np.random.default_rng(8);plate=rng.uniform(.6,.8,(400,400,3)).astype(np.float32)
    photo=plate*.95+.02;empty=np.ones((400,400),bool);empty[120:280,120:280]=False
    photo[120:280,120:280]=[.1,.4,.7]
    photo[180:220,180:220]=plate[180:220,180:220]*.95+.02
    mask,_=evidence(photo,plate,empty)
    assert mask[130:170,130:170].all() and not mask[190:210,190:210].any()
    assert not mask[empty].any()


def test_detail_energy_gate_preserves_base_when_source_is_blurry():
    from bracket_adaptive_luster_hybrid import merge_adaptive,CONFIG
    y,x=np.mgrid[:80,:80];pattern=np.sin(x*1.2).astype(np.float32)
    base=np.repeat((.5+.1*pattern)[...,None],3,axis=-1)
    source=np.repeat((.5+.03*pattern)[...,None],3,axis=-1)
    labels=np.full((80,80),-1,np.int16);labels[10:70,10:70]=10
    result,weight=merge_adaptive(base,source,labels,np.ones((80,80),np.float32),CONFIG)
    np.testing.assert_array_equal(result,base)
    assert not weight.any()
    stronger=np.repeat((.5+.2*pattern)[...,None],3,axis=-1)
    result,weight=merge_adaptive(base,stronger,labels,np.ones((80,80),np.float32),CONFIG)
    assert weight[20:60,20:60].min()>.99
    assert result[20:60,20:60].std()>base[20:60,20:60].std()*1.5
    np.testing.assert_array_equal(result[weight==0],base[weight==0])


def test_pair_interpolation_moves_sources_toward_common_view():
    from pairalign_luster_hybrid import intermediate_lookups
    forward=np.zeros((64,64,2),np.float32);forward[...,0]=2
    ua,ub=intermediate_lookups(forward,-forward,np.full((64,64),.5,np.float32))
    x=np.broadcast_to(np.arange(16,48)[None,:],(32,32))
    np.testing.assert_allclose(ua[16:48,16:48,0],x-1)
    np.testing.assert_allclose(ub[16:48,16:48,0],x+1)


def test_pair_flow_rejects_excess_displacement_and_inconsistent_reverse():
    from pairalign_luster_hybrid import safe_flow
    forward=np.zeros((64,64,2),np.float32);forward[...,0]=2
    support=np.ones((64,64),bool)
    assert safe_flow(forward,-forward,support,4)[16:48,16:48].all()
    assert not safe_flow(forward,-forward,support,1).any()
    assert not safe_flow(forward,forward,support,4).any()
