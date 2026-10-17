"""Meaningful geometry and leakage guards for frozen-field native transfer."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_interval_depth_cost_respects_spread_active_taps_and_invalid_depth():
    from depth_interval_quality_luster_hybrid import interval_cost
    z=np.ones((4,1,4),np.float32);lo=z-.1;hi=z+.1;taps=np.ones_like(z,bool)
    expected=np.array([[1.,1.1,1.1,1.1]],np.float32);spread=np.array([[0,0,.5,0]],np.float32);floor=np.full((1,4),.01,np.float32)
    first=interval_cost(expected,z,lo,hi,taps,spread,floor)
    assert first[0,0]==0 and 0<first[0,2]<first[0,1]
    z[3,0,3]=np.nan;taps[3,0,3]=False
    got=interval_cost(expected,z,lo,hi,taps,spread,floor)
    np.testing.assert_allclose(got[0,3],first[0,3])
    taps[3,0,3]=True
    assert interval_cost(expected,z,lo,hi,taps,spread,floor)[0,3]==1e6
    taps[:]=False
    np.testing.assert_array_equal(interval_cost(expected,z,lo,hi,taps,spread,floor),1e6)


def test_census_cost_contrast_gain_reversal_unknown_and_boundary():
    from source_census_quality_luster_hybrid import census_cost,shifted
    yy,xx=np.mgrid[:32,:32];value=(xx+yy).astype(np.float32)*.01
    valid=np.ones(value.shape,bool)
    np.testing.assert_allclose(census_cost(2*value+.2,value,valid)[8:-8,8:-8],0)
    np.testing.assert_allclose(census_cost(-value,value,valid)[8:-8,8:-8],1)
    np.testing.assert_allclose(census_cost(value,np.zeros_like(value),valid),.5)
    valid[:]=False
    np.testing.assert_allclose(census_cost(-value,value,valid),.5)
    impulse=np.zeros((5,5),np.float32);impulse[0,0]=1
    assert shifted(impulse,1,0).sum()==0 and shifted(impulse,0,1).sum()==0
    assert shifted(impulse,-1,0)[1,0]==1


def test_source_surface_witnesses_exclude_all_targets_and_unavailable():
    from source_surface_confidence_luster_hybrid import witness_candidates
    rows={i:dict(train=i not in (12,95,150)) for i in [7,8,9,10,11,12,95,150]}
    assert witness_candidates(rows,8,[7,8,9,10,12,95,150])==[9,10]


def test_source_surface_requires_two_distinct_known_witnesses():
    from source_surface_confidence_luster_hybrid import two_witness_cost,sample_cost
    costs=np.array([[[.1,.1,.2]],[[.3,.3,.4]],[[.9,.9,.6]]],np.float32)
    valid=np.array([[[1,1,0]],[[1,0,0]],[[1,0,0]]],bool)
    got,known=two_witness_cost(costs,valid)
    np.testing.assert_allclose(got,[[.2,.5,.5]])
    np.testing.assert_array_equal(known,[[1,0,0]])
    data=np.array([[.1,.2],[.3,.4]],np.float32)
    uv=np.array([[[10,20],[11,21],[10.5,20.5],[50,50],[np.nan,20]]],np.float32)
    result=sample_cost(data,uv,[10,20,12,22])
    np.testing.assert_allclose(result,[[.1,.4,.25,.5,.5]],atol=1e-6)

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


def test_source_model_residual_keeps_base_when_no_new_detail_exists():
    from residual_luster_hybrid import merge_residual
    rng=np.random.default_rng(9);base=rng.uniform(.2,.8,(64,64,3)).astype(np.float32)
    labels=np.full((64,64),-1,np.int16);labels[10:54,10:54]=1
    weight=(labels>=0).astype(np.float32)
    np.testing.assert_array_equal(merge_residual(base,np.zeros_like(base),labels,weight,4),base)
    residual=np.full_like(base,.02)
    np.testing.assert_allclose(merge_residual(base,residual,labels,weight,4),base,atol=1e-7)
    y,x=np.mgrid[:64,:64];residual=np.repeat((.05*np.sin(x*1.3))[...,None],3,axis=-1).astype(np.float32)
    result=merge_residual(base,residual,labels,weight,4)
    np.testing.assert_array_equal(result[weight==0],base[weight==0])
    assert np.abs(result-base).mean()>.005


def test_band_boost_preserves_coarse_reference_at_zero_boost():
    from bands_luster_hybrid import merge_bands
    from transfer_luster_hybrid import merge_details
    rng=np.random.default_rng(12);base=rng.uniform(.2,.6,(64,64,3)).astype(np.float32)
    photo=rng.uniform(.2,.6,(64,64,3)).astype(np.float32)
    labels=np.full((64,64),-1,np.int16);labels[8:56,8:56]=1
    reference,weight=merge_details(base,photo,labels,np.full((64,64),.7,np.float32),'detail',8.,1.)
    result,_=merge_bands(base,photo,labels,weight,8.,1.,0.)
    np.testing.assert_allclose(result,reference,atol=2e-7)
    flat,_=merge_bands(base,base+.02,labels,weight,8.,1.,1.)
    np.testing.assert_allclose(flat,base,atol=2e-7)
    result,w=merge_bands(base,photo,labels,weight,8.,1.,1.)
    np.testing.assert_array_equal(result[w==0],base[w==0])


@pytest.mark.parametrize('module',['single_source_luster_hybrid','graph_single_luster_hybrid','scale_single_luster_hybrid','risk_single_luster_hybrid','phase_single_luster_hybrid','quality_risk_luster_hybrid','fine_phase_single_luster_hybrid','phase_witness_luster_hybrid'])
def test_automatic_single_source_selection_and_occlusion_fallback(tmp_path,monkeypatch,module):
    """A full predictor run must select one photo and reject a depth-occluded strip."""
    import importlib
    single=importlib.import_module(module)
    from luster_hybrid_common import write,read
    (tmp_path/'logs').mkdir()
    h=w=80
    first=np.full((h,w,3),[.46,.4,.4],np.float32)
    second=np.full((h,w,3),[.4,.4,.46],np.float32)
    base=first.copy();base[:,40:]=second[:,40:]
    base+=.002
    rows={7:dict(train=True),8:dict(train=True),9:dict(train=True)}
    monkeypatch.setattr(single,'native_rows',lambda:rows)
    monkeypatch.setattr(single,'geometry',lambda root,target:dict(rgb=base))
    for cid in [8,9]:(tmp_path/f'photo{cid}.jpg').write_bytes(bytes([cid]))
    monkeypatch.setattr(single,'source_path',lambda cid:tmp_path/f'photo{cid}.jpg')
    write(tmp_path/'preflight/sources.json',dict(sources={'7':[8,9]}))
    for cid,rgb in [(8,first),(9,second)]:
        d=tmp_path/f'warps/cam007/src{cid:03d}';d.mkdir(parents=True)
        a={k:np.ones((h,w),np.float32) for k in ['valid','alpha','resolution_ratio']}
        a.update({k:np.zeros((h,w),np.float32) for k in ['depth_error','roundtrip','uncertainty','score']})
        a['depth_error'][35:45]=1.;a['rgb']=rgb
        for key,value in a.items():np.save(d/f'{key}.npy',value)
        write(d/'receipt.json',dict(box=[0,0,w,h],arrays=list(a)))
    if module=='phase_witness_luster_hybrid':
        from luster_hybrid_common import sha
        q=tmp_path/'quality';q.mkdir();dummy=q/'source_receipt.json';write(dummy,{})
        for key in ['source_phase_cost','view_phase_cost','src008_cost','src009_cost']:
            np.save(q/f'{key}.npy',np.zeros((h,w),np.float32))
        write(q/'receipt.json',dict(source_model_receipt=str(dummy),source_model_receipt_sha256=sha(dummy)))
        monkeypatch.setattr(single,'quality',lambda *args:q)
        monkeypatch.setattr(single,'evidence',lambda *args:q)
    single.combine(tmp_path,7,'check',{**single.DEFAULT,'count':2,'mode':'direct'})
    d=tmp_path/'candidates/check';result=np.load(d/'cam007.npy');labels=np.load(d/'cam007_sources.npy');weight=np.load(d/'cam007_confidence.npy')
    assert labels[15,15]==8 and labels[15,65]==9
    assert (labels[35:45]==-1).all()
    np.testing.assert_array_equal(result[weight==0],base[weight==0])
    for cid,rgb in [(8,first),(9,second)]:
        chosen=(labels==cid)&(weight==1)
        assert chosen.sum()>500
        np.testing.assert_allclose(result[chosen],rgb[chosen],atol=1e-7)
    receipt=read(d/'cam007_receipt.json')
    assert receipt['texture_source_count_per_pixel']==1 and not receipt['lookup_shift']


def test_surface_differential_reports_foreshortening_and_backside():
    from surface_quality_luster_hybrid import differential_quality
    y,x=np.mgrid[:31,:31]
    world=np.stack([(x-15)*.01,(y-15)*.01,np.full_like(x,-1)],-1).astype(np.float32)
    uv=np.stack([2*x,.5*y],-1).astype(np.float32)
    result=differential_quality(world,uv,np.zeros(3),np.zeros(3))
    np.testing.assert_allclose(result['min_axis'],.5)
    np.testing.assert_allclose(result['max_axis'],2.)
    assert result['incidence'][15,15]==1
    behind=differential_quality(world,uv,np.zeros(3),np.array([0,0,-2]))
    assert behind['incidence'][15,15]==-1
    world[15,15]=np.nan
    invalid=differential_quality(world,uv,np.zeros(3),np.zeros(3))
    assert not invalid['valid'][15,15] and invalid['min_axis'][15,15]==0


def test_missing_bands_keep_existing_geometry_phase_and_admit_new_texture():
    from missing_bands_luster_hybrid import merge_missing
    from missing_single_luster_hybrid import DEFAULT
    y,x=np.mgrid[:80,:80];pattern=np.repeat((.08*np.sin(x*1.2))[...,None],3,axis=-1).astype(np.float32)
    flat=np.full_like(pattern,.5);labels=np.full((80,80),-1,np.int16);labels[10:70,10:70]=8
    unchanged,weight=merge_missing(flat,flat,labels,DEFAULT)
    np.testing.assert_array_equal(unchanged,flat);assert not weight.any()
    changed,weight=merge_missing(flat,flat+pattern,labels,DEFAULT)
    assert changed[25:55,25:55].std()>.03
    np.testing.assert_array_equal(changed[weight==0],flat[weight==0])
    wrong,weight=merge_missing(flat+pattern,flat-pattern*2,labels,DEFAULT)
    np.testing.assert_allclose(wrong[25:55,25:55],(flat+pattern)[25:55,25:55],atol=1e-6)


def test_photo_patch_agreement_rejects_phase_conflicts_and_missing_support():
    from patch_consistency_luster_hybrid import agreement
    y,x=np.mgrid[:40,:40]
    texture=np.repeat((.1*np.sin(x))[...,None],3,axis=-1).astype(np.float32)
    valid=np.ones((40,40),bool)
    assert agreement(texture,texture,valid).max()==0
    assert agreement(texture,-texture,valid)[10:30,10:30].mean()>1.9
    assert np.isinf(agreement(texture,texture,np.zeros_like(valid))).all()
    flat=np.zeros_like(texture)
    assert agreement(flat,flat,valid).max()==0


def test_surface_mode_uses_pdf_density_not_largest_bin_mass():
    from probe_depth_luster_hybrid import packed_mode
    rays=Rays(nears=torch.tensor([[2.],[2.]]),fars=torch.tensor([[12.],[12.]]))
    out=dict(packed_ray_indices=torch.tensor([0,0]),packed_weights=torch.tensor([[.2],[.79]]),
             packed_spacing_starts=torch.tensor([[.1],[.3]]),packed_spacing_ends=torch.tensor([[.101],[.8]]))
    got=packed_mode(out,rays)
    assert abs(got[0].item()-3.)<1e-6
    assert torch.isnan(got[1])


def test_adaptive_seam_requires_photo_agreement_and_keeps_single_source():
    from seam_single_luster_hybrid import merge
    from scale_single_luster_hybrid import DEFAULT
    shape=(80,80);y,x=np.mgrid[:80,:80];base=np.full((*shape,3),.4,np.float32)
    first=base+np.repeat((.04*np.sin(x*np.pi/2))[...,None],3,-1).astype(np.float32)
    def record(rgb):
        a={k:np.ones(shape,np.float32) for k in ['valid','alpha','resolution_ratio']}
        a.update({k:np.zeros(shape,np.float32) for k in ['depth_error','roundtrip','uncertainty','score']});a['rgb']=rgb;return a
    labels=np.full(shape,-1,np.int16);labels[8:72,8:40]=8;labels[8:72,40:72]=9
    config={**DEFAULT,'filter_support':'geometry','seam_fade':3.,'adaptive_seam':True,'seam_tolerance':.01}
    result,w=merge(base,[(8,record(first)),(9,record(first))],labels,config)
    assert np.all(w[20:60,39:41]==1)
    conflict,w2=merge(base,[(8,record(first)),(9,record(2*base-first))],labels,config)
    assert np.all(w2[20:60,39:41]<.5)
    np.testing.assert_array_equal(result[w==0],base[w==0])
    np.testing.assert_array_equal(conflict[w2==0],base[w2==0])
    # The diagnostic control reconstructs the established merge exactly.
    texture=first.copy();texture[:,40:]=2*base[:,40:]-first[:,40:]
    reference,wr=merge_details(base,texture,labels,(labels>=0).astype(np.float32),'detail',8.,1.)
    control,wc=merge(base,[(8,record(first)),(9,record(2*base-first))],labels,{**config,'filter_support':'labels','adaptive_seam':False})
    np.testing.assert_array_equal(control,reference)
    np.testing.assert_array_equal(wc,wr)


def test_cubic_native_sampling_keeps_centers_range_and_rejects_hidden_tap():
    from cubic_luster_hybrid import sample
    y,x=np.mgrid[:12,:12];photo=np.repeat((x/11.)[...,None],3,-1).astype(np.float32)
    depth=np.ones((12,12),np.float32);source={key:depth.copy() for key in ['alpha','depth_z_q10','depth_z_q50','depth_z_q90']}
    uv=np.array([[[4.,5.],[4.25,5.5],[.25,5.5]]],np.float32);expected=np.ones((1,3),np.float32)
    rgb,valid,alpha,err=sample(photo,source,uv,expected)
    np.testing.assert_allclose(rgb[0,:2,0],[4/11,4.25/11],atol=1e-7)
    assert valid[0,0] and valid[0,1] and not valid[0,2]
    source['alpha'][4,3]=0
    _,_,alpha,_=sample(photo,source,uv,expected)
    assert alpha[0,0]==1 and alpha[0,1]==0  # zero-weight tap does not invalidate exact center
    photo[:,5:]=1.;photo[:,:5]=0.
    rgb,_,_,_=sample(photo,source,uv,expected)
    assert rgb.min()>=0 and rgb.max()<=1


def test_guided_single_source_correction_preserves_fallback_and_source_isolation():
    from guided_single_luster_hybrid import merge
    y,x=np.mgrid[:80,:100];base=np.repeat((.3+.002*x)[...,None],3,axis=-1).astype(np.float32)
    labels=np.full((80,100),-1,np.int16);labels[8:72,8:45]=8;labels[8:72,55:92]=9
    photo=base+.04;config=dict(sigma=8.,guided_epsilon=.001,gain=1.)
    corrected,weight=merge(base,photo,labels,config)
    np.testing.assert_allclose(corrected,base,atol=2e-6)
    photo[labels==8]+=.03*np.sin(x[labels==8]*1.2)[:,None]
    changed,weight=merge(base,photo,labels,config)
    np.testing.assert_array_equal(changed[weight==0],base[weight==0])
    np.testing.assert_array_equal(changed[labels==9],corrected[labels==9])
    assert np.max(np.abs(changed[labels==8]-base[labels==8]))>.02


def test_low_frequency_witnesses_are_confidence_only_and_exclude_selected_photo():
    from consensus_low_luster_hybrid import confidence
    shape=(40,40);selected=np.full((*shape,3),.2,np.float32);labels=np.full(shape,8,np.int16);valid=np.ones(shape,bool)
    config=dict(low_tolerance=.015,low_witnesses=2,low_fraction=.67,low_patch_sigma=3.)
    records=[(8,selected,valid),(9,selected+.005,valid)]
    assert not confidence(selected,labels,records,config).any()
    records.append((10,selected-.005,valid))
    np.testing.assert_allclose(confidence(selected,labels,records,config),1.)
    records[2]=(10,selected+.1,valid)
    assert not confidence(selected,labels,records,config).any()
    labels[10:20,10:20]=-1
    records[2]=(10,selected-.005,valid)
    weights=confidence(selected,labels,records,config)
    assert not weights[10:20,10:20].any()


def test_cubic_fixed_labels_fall_back_instead_of_switching_sources(tmp_path,monkeypatch):
    import cubic_fixed_luster_hybrid as fixed
    from risk_single_luster_hybrid import DEFAULT
    from luster_hybrid_common import write
    h=w=60;base=np.full((h,w,3),.4,np.float32);parent=tmp_path/'candidates/parent';parent.mkdir(parents=True);(tmp_path/'logs').mkdir()
    monkeypatch.setattr(fixed,'geometry',lambda root,camera:dict(rgb=base))
    config={**DEFAULT,'count':2,'mode':'direct'};write(parent/'config.json',config)
    for target in [7,12,95,150]:
        np.save(parent/f'cam{target:03d}_sources.npy',np.full((h,w),8,np.int16));np.save(parent/f'cam{target:03d}_reasons.npy',np.zeros((h,w),np.uint8))
        write(parent/f'cam{target:03d}_receipt.json',dict(sources=[8,9],config=config,reasons={},texture_source_count_per_pixel=1,lookup_shift=False))
        for cid in [8,9]:
            d=tmp_path/f'variants/cubic4/warps/cam{target:03d}/src{cid:03d}';d.mkdir(parents=True)
            arrays={k:np.ones((h,w),np.float32) for k in ['valid','alpha','resolution_ratio']}
            arrays.update({k:np.zeros((h,w),np.float32) for k in ['depth_error','roundtrip','uncertainty','score']})
            arrays['rgb']=np.full_like(base,.6 if cid==8 else .2)
            if cid==8:arrays['alpha'][25:35]=0
            for key,value in arrays.items():np.save(d/f'{key}.npy',value)
            write(d/'receipt.json',dict(box=[0,0,w,h],arrays=list(arrays)))
    fixed.combine(tmp_path,'parent','checked')
    result=np.load(tmp_path/'candidates/checked/cam007.npy');labels=np.load(tmp_path/'candidates/checked/cam007_sources.npy')
    assert not (labels==9).any() and (labels[25:35]==-1).all()
    np.testing.assert_array_equal(result[25:35],base[25:35])
    np.testing.assert_allclose(result[10,10],.6,atol=1e-7)


def test_fixed_quality_rejects_blur_and_wrong_phase_without_support_growth():
    from fixed_quality_luster_hybrid import detail_confidence
    y,x=np.mgrid[:80,:80];base=np.repeat((.4+.1*np.sin(x/2.))[...,None],3,-1).astype(np.float32)
    valid=np.ones((80,80),bool);valid[30:40,30:40]=False
    settings=dict(energy_floor=.004,sharpness_min=.8,phase_min=.8)
    good,_,_=detail_confidence(base,base,valid,settings)
    assert np.all(good[10:20,10:20]>.999) and not good[~valid].any()
    blurred=np.full_like(base,.4)
    bad,_,_=detail_confidence(base,blurred,valid,settings)
    assert not bad[10:70,10:70].any()
    shifted=.8-base
    bad,_,_=detail_confidence(base,shifted,valid,settings)
    assert not bad[10:70,10:70].any()
    # No supported LC detail to damage: new photo detail is not rejected merely
    # because it has no counterpart in the low-frequency frozen render.
    good,_,_=detail_confidence(blurred,base,valid,settings)
    assert np.all(good[10:20,10:20]>.999) and not good[~valid].any()


def test_multiplicative_illumination_preserves_detail_and_source_isolation():
    from illumination_single_luster_hybrid import merge
    y,x=np.mgrid[:60,:60];base=np.full((60,60,3),.4,np.float32)
    texture=np.repeat((.2+.03*np.sin(x/2.))[...,None],3,-1).astype(np.float32)
    labels=np.full((60,60),8,np.int16);labels[:,30:]=9;labels[:3]=-1
    config=dict(working_gamma=1.,sigma=6.,illumination_floor=.02,min_gain=.5,max_gain=2.,gain=1.)
    result,w=merge(base,texture,labels,config)
    np.testing.assert_array_equal(result[w==0],base[w==0])
    assert result[15:45,10:20].std()>.02
    other=texture.copy();other[:,30:]=.9
    changed,_=merge(base,other,labels,config)
    np.testing.assert_array_equal(changed[labels==8],result[labels==8])
    same,_=merge(base,base,labels,config)
    np.testing.assert_allclose(same,base,atol=2e-7)


def test_fine_phase_prefers_aligned_native_detail_over_lower_angle_wrong_phase(tmp_path,monkeypatch):
    import fine_phase_single_luster_hybrid as single
    from luster_hybrid_common import write
    h=w=80;y,x=np.mgrid[:h,:w];signal=.04*np.sin(x/2.)
    base=np.repeat((.4+signal)[...,None],3,-1).astype(np.float32)
    photos={8:np.repeat((.4-signal)[...,None],3,-1).astype(np.float32),9:np.repeat((.4+2*signal)[...,None],3,-1).astype(np.float32)}
    monkeypatch.setattr(single,'native_rows',lambda:{c:dict(train=True) for c in [7,8,9]})
    monkeypatch.setattr(single,'geometry',lambda root,target:dict(rgb=base))
    monkeypatch.setattr(single,'source_path',lambda cid:tmp_path/f'photo{cid}.jpg')
    (tmp_path/'logs').mkdir();write(tmp_path/'preflight/sources.json',dict(sources={'7':[8,9]}))
    for cid,rgb in photos.items():
        (tmp_path/f'photo{cid}.jpg').write_bytes(bytes([cid]))
        d=tmp_path/f'warps/cam007/src{cid:03d}';d.mkdir(parents=True)
        a={k:np.ones((h,w),np.float32) for k in ['valid','alpha','resolution_ratio']}
        a.update({k:np.zeros((h,w),np.float32) for k in ['depth_error','roundtrip','uncertainty','score']});a['rgb']=rgb
        a['score'][:]=.1 if cid==9 else 0.
        for key,value in a.items():np.save(d/f'{key}.npy',value)
        write(d/'receipt.json',dict(box=[0,0,w,h],arrays=list(a)))
    config={**single.DEFAULT,'count':2,'mode':'direct','match_weight':0.,'match_limit':.2,'fine_phase_weight':1.,'detail_reward':.25}
    single.combine(tmp_path,7,'check',config)
    labels=np.load(tmp_path/'candidates/check/cam007_sources.npy');rgb=np.load(tmp_path/'candidates/check/cam007.npy')
    assert (labels[15:65,15:65]==9).all()
    np.testing.assert_allclose(rgb[20:60,20:60],photos[9][20:60,20:60],atol=1e-7)


def test_view_illumination_uses_model_color_but_only_selected_photo_detail():
    from view_illumination_luster_hybrid import merge
    y,x=np.mgrid[:60,:60];base=np.full((60,60,3),.4,np.float32);source=np.full_like(base,.2)
    texture=np.repeat((.2+.025*np.sin(x/2.))[...,None],3,-1).astype(np.float32)
    labels=np.full((60,60),8,np.int16);labels[:,30:]=9;labels[:3]=-1
    config=dict(correction_sigma=16.,illumination_floor=.02,min_gain=.5,max_gain=2.,transfer_mode='detail',sigma=6.,gain=1.)
    result,w=merge(base,texture,source,labels,config)
    np.testing.assert_array_equal(result[w==0],base[w==0])
    assert result[15:45,10:20].std()>.02
    changed_texture=texture.copy();changed_source=source.copy();changed_texture[:,30:]=.9;changed_source[:,30:]=.8
    changed,_=merge(base,changed_texture,changed_source,labels,config)
    np.testing.assert_array_equal(changed[labels==8],result[labels==8])


def test_adaptive_bands_require_aligned_structure_and_do_not_mix_sources():
    from adaptive_band_luster_hybrid import merge
    y,x=np.mgrid[:120,:120];base=np.repeat((.4+.05*np.sin(x/8.))[...,None],3,-1).astype(np.float32)
    texture=.4+1.5*(base-.4);labels=np.full((120,120),8,np.int16);labels[:,60:]=9;labels[:3]=-1
    config=dict(sigma=6.,extra_sigma=16.,phase_min=.8,energy_floor=.003,gain=1.)
    result,w,gate=merge(base,texture,labels,config)
    assert gate[30:90,15:45].mean()>.8
    np.testing.assert_array_equal(result[w==0],base[w==0])
    bad=.8-texture
    _,_,bad_gate=merge(base,bad,labels,config)
    assert bad_gate[30:90,15:45].max()<1e-5
    changed_texture=texture.copy();changed_texture[:,60:]=.9
    changed,_,_=merge(base,changed_texture,labels,config)
    np.testing.assert_array_equal(changed[labels==8],result[labels==8])


def test_cubic75_constant_grid_and_hidden_footprint():
    from cubic75_luster_hybrid import sample
    photo=np.full((20,20,3),.4,np.float32)
    source={key:np.ones((20,20),np.float32) for key in ['depth_z_q10','depth_z_q50','depth_z_q90','alpha']}
    uv=np.array([[7.,8.],[7.25,8.5]],np.float32);z=np.ones(2,np.float32)
    rgb,valid,alpha,error=sample(photo,source,uv,z)
    np.testing.assert_allclose(rgb,.4,atol=1e-7);assert valid.all() and (error==0).all()
    source['depth_z_q50'][10,9]=2
    rgb,valid,alpha,error=sample(photo,source,uv,z)
    assert error[0]==0 and error[1]==1
    photo[8,7]=0;photo[9,8]=1
    rgb,_,_,_=sample(photo,source,uv,z)
    assert (rgb>=0).all() and (rgb<=1).all()


def test_intrinsic_density_surface_excludes_occluded_peak_and_uses_physical_widths():
    from density_probe_luster_hybrid import packed_density_surface
    rays=Rays(nears=torch.tensor([[0.],[0.]]),fars=torch.tensor([[1.],[1.]]))
    density=np.array([1.,10.,1000.]);left=np.array([.1,.2,.8]);right=np.array([.2,.6,.81])
    optical=density*(right-left);trans=np.exp(-np.r_[0.,np.cumsum(optical)[:-1]])
    weights=trans*(-np.expm1(-optical))
    out=dict(packed_ray_indices=torch.tensor([0,0,0]),packed_weights=torch.tensor(weights[:,None],dtype=torch.float64),
             packed_spacing_starts=torch.tensor(left[:,None]),packed_spacing_ends=torch.tensor(right[:,None]))
    q=packed_quantiles(out,rays,(.1,.9))[0].numpy()
    got=packed_density_surface(out,rays)
    # q10 and q90 both lie in density=10; the much denser occluded bin must not win.
    assert .2<q[0]<q[1]<.6
    np.testing.assert_allclose(got[0].numpy(),np.repeat(q.mean(),2),atol=1e-6)
    assert torch.isnan(got[1]).all()
    # Splitting the constant-density bin cannot alter the density-squared centroid.
    left2=np.array([.1,.2,.3,.8]);right2=np.array([.2,.3,.6,.81]);den2=np.array([1.,10.,10.,1000.])
    tau=den2*(right2-left2);w2=np.exp(-np.r_[0.,np.cumsum(tau)[:-1]])*(-np.expm1(-tau))
    out2=dict(packed_ray_indices=torch.tensor([0,0,0,0]),packed_weights=torch.tensor(w2[:,None]),
              packed_spacing_starts=torch.tensor(left2[:,None]),packed_spacing_ends=torch.tensor(right2[:,None]))
    np.testing.assert_allclose(packed_density_surface(out2,rays)[0,1],got[0,1],atol=1e-6)


def test_pair_phase_requires_visible_structure_and_two_distinct_witnesses():
    from witness_phase_luster_hybrid import pair_phase,update_best
    y,x=np.mgrid[:64,:64];a=(.04*np.sin(x*np.pi/3)).astype(np.float32);support=np.ones((64,64),bool)
    corr,valid=pair_phase(a,a,support);assert corr[8:-8,8:-8].mean()>.99 and valid.all()
    opposite,op_valid=pair_phase(a,-a,support);assert opposite[8:-8,8:-8].mean()<-.99
    best=np.full(a.shape,-2.,np.float32);second=best.copy()
    update_best(best,second,corr,valid);assert (second==-2).all()
    update_best(best,second,opposite,op_valid);assert (second>=-1).all()
    _,flat=pair_phase(a,np.zeros_like(a),support);assert not flat.any()
    _,hidden=pair_phase(a,a,np.zeros_like(support));assert not hidden.any()


def test_screened_poisson_respects_source_boundaries_and_constant_offsets():
    from poisson_single_luster_hybrid import solve_delta
    y,x=np.mgrid[:48,:48];base=np.full((48,48,3),.4,np.float32);texture=base.copy()
    labels=np.full((48,48),-1,np.int16);labels[4:44,4:24]=8;labels[4:44,24:44]=9
    texture[labels==8]+=.02*np.sin(x[labels==8,None]*np.pi/3)
    texture[labels==9]+=.15
    delta,report=solve_delta(base,texture,labels,.05)
    assert np.max(np.abs(delta[labels==8]))>.01
    assert np.max(np.abs(delta[labels==9]))<1e-6
    assert np.array_equal(delta[labels<0],np.zeros_like(delta[labels<0]))
    texture[labels==9]=0.
    other,_=solve_delta(base,texture,labels,.05)
    np.testing.assert_allclose(delta[labels==8],other[labels==8],atol=1e-7)
    assert max(r['relative_residual'] for r in report)<5e-5


def test_gradient_guard_keeps_opposite_structure_and_admits_new_flat_region_detail():
    from gradient_guard_luster_hybrid import solve_delta,gradient_confidence
    y,x=np.mgrid[:48,:48];pattern=np.repeat((.04*np.sin(x*np.pi/3))[...,None],3,-1).astype(np.float32)
    base=.4+pattern;opposite=.4-pattern;labels=np.full((48,48),8,np.int16)
    gate=gradient_confidence(base,opposite,labels);assert gate.max()<1e-6
    delta,_,_=solve_delta(base,opposite,labels,.02);assert np.max(np.abs(delta))<1e-6
    flat=np.full_like(base,.4);delta,_,gate=solve_delta(flat,base,labels,.02)
    assert gate.min()>.99 and np.max(np.abs(delta))>.02


def test_projection_differential_detects_directional_sampling_without_moving_pixels():
    from differential_luster_hybrid import projection_differential,singular_values
    y,x=np.mgrid[:24,:32];z=np.full(x.shape,2.,np.float32);alpha=np.ones(x.shape,np.float32)
    # Rotation/reflection do not change singular values; area alone would miss
    # the fourfold loss of resolution along the compressed direction.
    uv=np.stack([-2*y,.25*x],-1).astype(np.float32);original=uv.copy()
    low,high,valid=projection_differential(uv,z,alpha)
    np.testing.assert_allclose(low[valid],.25);np.testing.assert_allclose(high[valid],2.)
    assert np.array_equal(uv,original)
    z[:,16:]=3.
    _,_,valid=projection_differential(uv,z,alpha)
    assert not valid[2:-2,14:18].any() and valid[4:20,4:12].all()
    lo,hi=singular_values(np.array([[1.,0.]]),np.array([[0.,1.]]))
    np.testing.assert_allclose(lo,1);np.testing.assert_allclose(hi,1)


def test_source_potential_tile_aggregation_counts_masked_pixels_not_tiles():
    from evaluate_source_potential_luster import tile_errors
    values=np.array([[1.,2.,9.,100.],[3.,4.,100.,100.]])
    mask=np.array([[True,True,True,False],[True,True,False,False]])
    errors,pixels=tile_errors(values,mask,2)
    np.testing.assert_allclose(errors,[10.,9.]);np.testing.assert_array_equal(pixels,[4,1])
    assert errors.sum()/pixels.sum()==3.8


def test_log_residual_preserves_model_when_source_matches_and_never_crosses_labels():
    from log_residual_luster_hybrid import merge
    y,x=np.mgrid[:48,:48];base=np.full((48,48,3),.4,np.float32);model=base.copy()
    labels=np.full((48,48),-1,np.int16);labels[4:44,4:24]=8;labels[4:44,24:44]=9
    config=dict(radiance_floor=.02,max_ratio=2.,transfer_mode='detail',sigma=6.,gain=1.)
    same,w=merge(base,model,model,labels,config);assert np.array_equal(same,base)
    texture=model.copy();texture[labels==8]+=.03*np.sin(x[labels==8,None]*np.pi/3)
    result,w=merge(base,texture,model,labels,config)
    assert np.max(np.abs(result[labels==8]-base[labels==8]))>.01
    assert np.array_equal(result[labels<0],base[labels<0])
    texture[labels==9]=.9
    other,_=merge(base,texture,model,labels,config)
    np.testing.assert_array_equal(result[labels==8],other[labels==8])


def test_source_render_phase_ranking_is_gain_invariant_and_detects_wrong_structure():
    from source_phase_quality_luster_hybrid import phase_cost
    y,x=np.mgrid[:64,:64];band=(.04*np.sin(x*np.pi/8)).astype(np.float32)
    support=np.ones(band.shape,bool);region=np.s_[12:-12,12:-12]
    aligned=phase_cost(band,band*.4,support);opposite=phase_cost(band,-band,support)
    assert aligned[region].max()<1e-5 and opposite[region].min()>.999
    assert phase_cost(band,np.zeros_like(band),support)[region].mean()==.5
    assert not phase_cost(np.zeros_like(band),np.zeros_like(band),support).any()
    assert not phase_cost(band,-band,np.zeros_like(support)).any()


def test_bracket_witnesses_must_lie_on_opposite_sides_of_the_target_ray():
    from bracket_witness_luster_hybrid import tangent_cosine
    rays=np.array([[[0.,0.,1.],[0.,0.,-1.]]],np.float32)
    left=np.array([-1.,0.,2.],np.float32);right=np.array([1.,0.,3.],np.float32)
    np.testing.assert_allclose(tangent_cosine(left,right,rays),-1)
    np.testing.assert_allclose(tangent_cosine(left,left*2,rays),1)
    vertical=np.array([0.,1.,2.],np.float32)
    np.testing.assert_allclose(tangent_cosine(left,vertical,rays),0)
    # A camera baseline along the viewing ray gives no directional evidence.
    np.testing.assert_allclose(tangent_cosine(left,np.array([0.,0.,2.]),rays),1)


def test_lanczos3_preserves_bandlimited_detail_and_checks_remote_nonzero_taps():
    from lanczos3_luster_hybrid import sample
    y,x=np.mgrid[:24,:32];gray=(.5+.2*np.sin(x*2*np.pi*.16)).astype(np.float32)
    photo=np.repeat(gray[...,None],3,-1);depth=np.ones(x.shape,np.float32)
    source=dict(alpha=depth.copy(),depth_z_q10=depth.copy(),depth_z_q50=depth.copy(),depth_z_q90=depth.copy())
    xx=np.arange(5,24,dtype=np.float32)+.35
    uv=np.stack([xx,np.full_like(xx,10.25)],-1)[None];z=np.ones(uv.shape[:2],np.float32)
    rgb,valid,_,_=sample(photo,source,uv,z);assert valid.all()
    true=.5+.2*np.sin(xx*2*np.pi*.16)
    bilinear=.65*gray[10,np.floor(xx).astype(int)]+.35*gray[10,np.floor(xx).astype(int)+1]
    assert np.mean((rgb[0,:,0]-true)**2)<np.mean((bilinear-true)**2)
    integers=np.array([[[10.,10.]]],np.float32)
    exact,valid,_,_=sample(photo,source,integers,np.ones((1,1),np.float32))
    np.testing.assert_allclose(exact[0,0],photo[10,10],atol=1e-7)
    source['depth_z_q50'][8,8]=np.nan
    _,valid,_,_=sample(photo,source,np.array([[[10.35,10.25]]],np.float32),np.ones((1,1),np.float32))
    assert not valid.any()


def test_point_view_transfer_has_identity_for_equal_directional_colors():
    from point_view_merge_luster_hybrid import merge
    from transfer_luster_hybrid import merge_details
    y,x=np.mgrid[:48,:48];base=np.full((48,48,3),.4,np.float32)
    texture=base+np.repeat((.03*np.sin(x*np.pi/3))[...,None],3,-1).astype(np.float32)
    labels=np.full((48,48),-1,np.int16);labels[4:44,4:24]=8;labels[4:44,24:44]=9
    config=dict(view_gain=1.,sigma=6.,gain=1.)
    same,w=merge(base,texture,base,base,labels,config)
    expected,ew=merge_details(base,texture,labels,(labels>=0).astype(np.float32),'detail',6.,1.)
    np.testing.assert_array_equal(same,expected);np.testing.assert_array_equal(w,ew)
    # Broad directional offset is removed by detail transfer, not baked into skin.
    target=base+.02;shifted,_=merge(base,texture,target,base,labels,config)
    np.testing.assert_allclose(shifted,expected,atol=5e-7)  # FP32 normalized Gaussian roundoff
    target[labels==9]=.8
    other,_=merge(base,texture,target,base,labels,config)
    np.testing.assert_array_equal(other[labels==8],shifted[labels==8])
    np.testing.assert_array_equal(other[labels<0],base[labels<0])


def test_train_witness_benefit_rewards_added_detail_and_penalizes_wrong_phase():
    from witness_benefit_luster_hybrid import pair_benefit
    y,x=np.mgrid[:64,:64];truth=(.04*np.sin(x*np.pi/8)).astype(np.float32);flat=np.zeros_like(truth);support=np.ones(truth.shape,bool)
    good,valid=pair_benefit(truth,truth,flat,support);bad,_=pair_benefit(-truth,truth,flat,support)
    same,_=pair_benefit(flat,truth,flat,support)
    assert np.median(good[valid])>.99 and np.median(bad[valid])<-.59
    assert not same.any()
    uncertain,_=pair_benefit(flat,flat,flat,support);assert not uncertain.any()
    _,hidden=pair_benefit(truth,truth,flat,np.zeros_like(support));assert not hidden.any()


def test_receipt_audit_follows_transitive_candidate_and_cache_dependencies(tmp_path):
    from dependency_audit_luster_hybrid import verify_dependencies
    from luster_hybrid_common import write, sha
    array=tmp_path/'cost.npy';np.save(array,np.ones(8,np.float32))
    leaf=tmp_path/'quality_receipt.json';parent=tmp_path/'parent_receipt.json';final=tmp_path/'final_receipt.json'
    write(leaf,dict(code_sha256='known',hashes={'cost':sha(array)}))
    write(parent,dict(code_sha256='known',additional_input_hashes={str(leaf):sha(leaf)}))
    write(final,dict(code_sha256='known',additional_input_hashes={str(parent):sha(parent)}))
    bound=set();verify_dependencies(final,{'generator':'known'},bound.add)
    assert {array,leaf,parent,final}<=bound
    np.save(array,np.zeros(8,np.float32))
    with pytest.raises(AssertionError,match='Changed cached array'):
        verify_dependencies(final,{'generator':'known'},set().add)
    with pytest.raises(AssertionError,match='Changed generator'):
        verify_dependencies(final,{'generator':'other'},set().add)


def test_nested_receipt_audit_checks_frozen_state_and_target_specific_witnesses(tmp_path):
    from dependency_audit_luster_hybrid import verify_dependencies
    from luster_hybrid_common import write, sha
    leaf=tmp_path/'witness_receipt.json';parent=tmp_path/'parent_receipt.json'
    state={'model':'frozen','occupancy':'saved','frequency':'saved'}
    def save(**change):
        write(leaf,dict(state_hash=state,source=6,witnesses=[7,8],**change))
        write(parent,dict(inputs={str(leaf):sha(leaf)}))
    save();visited=set()
    verify_dependencies(parent,{},set().add,visited=visited,expected_state_hash=state,forbidden_sources={12,95,150})
    # Shared caches must be rechecked when a different target becomes forbidden.
    with pytest.raises(AssertionError,match='Forbidden witness'):
        verify_dependencies(parent,{},set().add,visited=visited,expected_state_hash=state,forbidden_sources={7,12,95,150})
    with pytest.raises(AssertionError,match='Frozen model/grids changed'):
        verify_dependencies(parent,{},set().add,expected_state_hash={**state,'occupancy':'changed'})
    write(leaf,dict(state_hash=state,source=7,witnesses=[7,8]))
    write(parent,dict(inputs={str(leaf):sha(leaf)}))
    with pytest.raises(AssertionError,match='Self witness'):
        verify_dependencies(parent,{},set().add,expected_state_hash=state)


def test_source_contradiction_does_not_reward_flat_or_missing_evidence():
    from source_contradiction_rank_luster_hybrid import contradiction_cost
    np.testing.assert_array_equal(contradiction_cost(np.array([0.,.25,.5,.75,1.])),
                                  [0.,0.,0.,.25,.5])


def test_conditional_scale_requires_texture_structure_and_precise_projection():
    from conditional_scale_rank_luster_hybrid import scale_trust
    zeros=np.zeros(6,np.float32);source=zeros.copy();source[1]=.2
    view=zeros.copy();view[2]=.2
    rt=zeros.copy();rt[3]=.3
    uncertainty=zeros.copy();uncertainty[4]=.6
    energy=np.full(6,.01,np.float32);energy[5]=0
    np.testing.assert_array_equal(scale_trust(source,view,rt,uncertainty,energy),[1,0,0,0,0,0])
    assert scale_trust(np.array(.075),np.array(0),np.array(0),np.array(0),np.array(.01))==.5


def test_coherent_phase_assignment_preserves_visibility_and_removes_small_cost_island():
    from coherent_phase_rank_luster_hybrid import coherent_labels
    a=np.zeros((16,16),np.float32);b=np.full_like(a,.4);b[8,8]=-.1
    visible=np.ones(a.shape,bool);none=np.zeros_like(visible)
    labels=coherent_labels([a,b],[visible,visible],a.shape,1,.2,8.)
    assert not labels.any()
    hidden=coherent_labels([a,b],[none,none],a.shape,4,.2,8.)
    assert (hidden==2).all()
    partial=visible.copy();partial[3,3]=False
    labels=coherent_labels([a],[partial],a.shape,4,.2,8.)
    assert labels[3,3]==1 and labels[2,2]==0


def test_exact_plate_veto_requires_background_match_and_model_contrast():
    from exact_plate_veto_luster_hybrid import background_veto,footprint_veto
    plate=np.full((3,3),.8,np.float32);photo=plate.copy();photo[1]=.4
    model=np.full_like(plate,.2);model[2]=.8
    np.testing.assert_array_equal(background_veto(photo,plate,model,np.ones(3),.01),[True,False,False])
    assert not background_veto(photo,plate,model,np.zeros(3),.01).any()
    mask=np.zeros((8,8),bool);mask[3,3]=True
    # At integer coordinates neighboring zero-weight cubic taps do not veto.
    uv=np.array([[[2.,3.],[2.5,3.],[3.,3.],[np.nan,3.]]],np.float32)
    np.testing.assert_array_equal(footprint_veto(mask,uv),[[False,True,True,False]])


def test_plate_merge_excludes_rejected_color_from_neighbor_filters(tmp_path,monkeypatch):
    import exact_plate_merge_luster_hybrid as module
    from freeze_luster_hybrid_candidate import verify_plate_merge
    from seam_single_luster_hybrid import merge
    from luster_hybrid_common import read,write
    h=w=32;base=np.full((h,w,3),.4,np.float32)
    monkeypatch.setattr(module,'geometry',lambda *args:dict(rgb=base))
    veto_dir=tmp_path/'veto';veto_dir.mkdir();veto=np.zeros((h,w),bool);veto[:,16:]=True
    np.save(veto_dir/'strict10.npy',veto);write(veto_dir/'receipt.json',{})
    monkeypatch.setattr(module,'projected_veto',lambda *args:veto_dir)
    (tmp_path/'logs').mkdir();parent=tmp_path/'candidates/parent';parent.mkdir(parents=True)
    config=dict(alpha=.99,depth_error=.005,roundtrip=1.,uncertainty=2.,max_score=1.,
        min_resolution=.5,sigma=6.,gain=1.,plate_veto_level='strict10')
    write(parent/'config.json',config)
    for target in [7,12,95,150]:
        np.save(parent/f'cam{target:03d}_sources.npy',np.full((h,w),8,np.int16))
        np.save(parent/f'cam{target:03d}_reasons.npy',np.zeros((h,w),np.uint8))
        write(parent/f'cam{target:03d}_receipt.json',dict(sources=[8],reasons={}))
        d=tmp_path/'variants/cubic75/warps'/f'cam{target:03d}'/'src008';d.mkdir(parents=True)
        arrays={k:np.ones((h,w),np.float32) for k in ['valid','alpha','resolution_ratio']}
        arrays.update({k:np.zeros((h,w),np.float32) for k in ['depth_error','roundtrip','uncertainty','score']})
        arrays['rgb']=np.full_like(base,.6);arrays['rgb'][:,16:]=.95
        for k,v in arrays.items():np.save(d/f'{k}.npy',v)
        write(d/'receipt.json',dict(box=[0,0,w,h],arrays=list(arrays)))
    module.combine(tmp_path,'parent','out')
    for target in [7,12,95,150]:
        d=tmp_path/'candidates/out'
        result=np.load(d/f'cam{target:03d}.npy');labels=np.load(d/f'cam{target:03d}_sources.npy')
        np.testing.assert_array_equal(result[:,16:],base[:,16:])
        np.testing.assert_allclose(result,base,atol=2e-7)
        assert (labels[:,16:]==-1).all() and (labels[:,:16]==8).all()
        weight=np.load(d/f'cam{target:03d}_confidence.npy')
        parent_labels=np.load(parent/f'cam{target:03d}_sources.npy')
        actual_config=read(d/'config.json')
        args=(base,[(8,arrays)],parent_labels,{8:veto})
        verify_plate_merge(*args,labels,result,weight,actual_config)
        # A selected vetoed pixel and a filter that includes background must
        # both be detected, even when output remains finite and in range.
        bad_labels=labels.copy();bad_labels[12,20]=8
        with np.testing.assert_raises(AssertionError):
            verify_plate_merge(*args,bad_labels,result,weight,actual_config)
        polluted,_=merge(base,[(8,arrays)],labels,actual_config)
        assert not np.array_equal(polluted,result)
        with np.testing.assert_raises(AssertionError):
            verify_plate_merge(*args,labels,polluted,weight,actual_config)


def test_band_admission_preserves_unknown_and_rejects_corrobated_harm():
    from band_witness_luster_hybrid import admission,bands,assemble
    support=np.ones((64,64),bool)
    np.testing.assert_allclose(admission(np.zeros(support.shape,np.float32),support,.25),1,atol=1e-7)
    assert not admission(np.full(support.shape,-.5,np.float32),support,.25).any()
    rng=np.random.default_rng(17);base=rng.uniform(.2,.8,(64,64,3)).astype(np.float32)
    texture=rng.uniform(.2,.8,base.shape).astype(np.float32)
    parts=bands(texture,support);reference=bands(base,support)
    delta=sum(a-b for a,b in zip(parts,reference))
    from transfer_luster_hybrid import gaussian
    expected=(texture-gaussian(texture,6))-(base-gaussian(base,6))
    np.testing.assert_allclose(delta,expected,atol=2e-7)
    labels=np.full(support.shape,8,np.int16);labels[:4]=-1
    config=dict(seam_tolerance=.02,seam_fade=3.,gain=1.)
    result,weight=assemble(base,{8:delta},{8:support},labels,config)
    np.testing.assert_array_equal(result[labels<0],base[labels<0])
    zero,_=assemble(base,{8:np.zeros_like(delta)},{8:support},labels,config)
    np.testing.assert_array_equal(zero,base)
    chosen=labels>=0
    np.testing.assert_array_equal(result[chosen],np.clip(base+weight[...,None]*delta,0,1)[chosen])


def test_chroma_detail_preserves_luminance_and_full_rgb_identity():
    from chroma_detail_luster_hybrid import adjust_chroma
    rng=np.random.default_rng(18);delta=rng.uniform(-.2,.2,(32,32,3)).astype(np.float32)
    weights=np.array([.2126,.7152,.0722],np.float32)
    np.testing.assert_array_equal(adjust_chroma(delta,1),delta)
    for gain in [0,.5]:
        actual=adjust_chroma(delta,gain)
        np.testing.assert_allclose(actual@weights,delta@weights,atol=3e-8)
    gray=adjust_chroma(delta,0)
    np.testing.assert_array_equal(gray[...,0],gray[...,1])
    np.testing.assert_array_equal(gray[...,1],gray[...,2])


def test_directional_cost_adds_risk_without_rewarding_anisotropic_oversampling():
    from phase_directional_luster_hybrid import directional_cost
    minimum=np.array([1.,1.,.5,4.,np.nan],np.float32)
    maximum=np.array([1.,4.,.5,4.,np.nan],np.float32)
    cost=directional_cost(minimum,maximum,np.array([1,1,1,1,0],bool))
    np.testing.assert_allclose(cost,[0,1,1,0,1])
    assert np.isfinite(cost).all()


def test_joint_band_gate_requires_adverse_evidence_at_both_scales():
    from joint_band_luster_hybrid import joint_gate
    fine=np.array([0.,0.,1.,.2],np.float32);middle=np.array([1.,0.,0.,.5],np.float32)
    actual=joint_gate(fine,middle)
    np.testing.assert_allclose(actual,[1,0,1,.6],atol=1e-7)
    assert np.all(actual>=fine) and np.all(actual>=middle)


def test_rgb_phase_matches_gray_limit_and_detects_equal_luminance_color_reversal():
    from source_rgb_phase_quality_luster_hybrid import phase_cost as rgb_cost
    from source_phase_quality_luster_hybrid import phase_cost as gray_cost
    rng=np.random.default_rng(19);a=rng.normal(0,.04,(64,64)).astype(np.float32);b=a*.7
    support=np.ones(a.shape,bool)
    ar=np.repeat(a[...,None],3,-1);br=np.repeat(b[...,None],3,-1)
    np.testing.assert_allclose(rgb_cost(ar,br,support),gray_cost(a,b,support),atol=2e-7)
    color=np.zeros_like(ar);color[...,0]=a;color[...,1]=-a*(.2126/.7152)
    assert np.max(np.abs(color@np.array([.2126,.7152,.0722],np.float32)))<1e-8
    assert np.mean(rgb_cost(color,-color,support))>.99
    assert np.mean(rgb_cost(color,color*2,support))<1e-6


def test_edge_cost_constant_preservation_and_invalid_exclusion():
    from edge_cost_luster_hybrid import aggregate
    score=np.full((12,12),.7,np.float32);valid=np.ones(score.shape,bool)
    rgb=np.zeros((*score.shape,3),np.float32);depth=np.ones(score.shape,np.float32)
    score[4,4]=np.inf;valid[4,4]=False;depth[4,4]=np.nan
    actual=aggregate(score,valid,rgb,depth)
    assert np.isinf(actual[4,4])
    np.testing.assert_allclose(actual[valid],.7,atol=2e-7)


def test_edge_cost_respects_color_and_depth_boundaries_without_wraparound():
    from edge_cost_luster_hybrid import aggregate
    score=np.zeros((16,16),np.float32);score[:,8:]=1
    valid=np.ones(score.shape,bool);rgb=np.zeros((*score.shape,3),np.float32);depth=np.ones(score.shape,np.float32)
    plain=aggregate(score,valid,rgb,depth)
    assert plain[8,7]>.2 and plain[8,8]<.8
    rgb[:,8:]=1
    color=aggregate(score,valid,rgb,depth)
    assert color[8,7]<1e-6 and color[8,8]>.99999
    rgb[:]=0;depth[:,8:]=1.1
    geometry=aggregate(score,valid,rgb,depth)
    assert geometry[8,7]<1e-6 and geometry[8,8]>.99999
    assert geometry[8,0]==0 and geometry[8,-1]>.99999


def test_structure_gate_rejects_reversed_middle_band_but_retains_unknown_detail():
    from structure_admission_luster_hybrid import structure_gate
    rng=np.random.default_rng(20);band=rng.normal(0,.05,(64,64,3)).astype(np.float32)
    support=np.ones(band.shape[:2],bool);zero=np.zeros_like(band)
    for start in [.5,.75]:
        np.testing.assert_allclose(structure_gate(band,2*band,support,start),1,atol=1e-6)
        np.testing.assert_allclose(structure_gate(band,zero,support,start),1,atol=1e-6)
        assert np.max(structure_gate(band,-band,support,start))<1e-6
        support[:8]=False
        assert not structure_gate(band,zero,support,start)[:8].any()
        support[:]=True


def test_cubic_eligible_ranking_selects_visible_alternative_before_fallback(tmp_path,monkeypatch):
    import cubic_eligible_phase_luster_hybrid as single
    from luster_hybrid_common import write,sha
    h=w=64;base=np.full((h,w,3),.4,np.float32)
    monkeypatch.setattr(single,'native_rows',lambda:{c:dict(train=True) for c in [7,8,9]})
    monkeypatch.setattr(single,'geometry',lambda root,target:dict(rgb=base))
    monkeypatch.setattr(single,'source_path',lambda cid:tmp_path/f'photo{cid}.jpg')
    (tmp_path/'logs').mkdir();write(tmp_path/'preflight/sources.json',dict(sources={'7':[8,9]}))
    for c in [8,9]:(tmp_path/f'photo{c}.jpg').write_bytes(bytes([c]))
    q=tmp_path/'quality';q.mkdir();dummy=q/'source_receipt.json';write(dummy,{})
    for key in ['source_phase_cost','view_phase_cost']:np.save(q/f'{key}.npy',np.zeros((h,w),np.float32))
    write(q/'receipt.json',dict(source_model_receipt=str(dummy),source_model_receipt_sha256=sha(dummy)))
    monkeypatch.setattr(single,'quality',lambda *args:q)
    for c in [8,9]:
        a={k:np.ones((h,w),np.float32) for k in ['valid','alpha','resolution_ratio']}
        a.update({k:np.zeros((h,w),np.float32) for k in ['depth_error','roundtrip','uncertainty','score']})
        a['rgb']=base+.01*(c-7);a['score'][:]=.1*(c-8)
        for family in ['warps','variants/cubic75/warps']:
            d=tmp_path/family/'cam007'/f'src{c:03d}';d.mkdir(parents=True)
            saved={k:v.copy() for k,v in a.items()}
            if family!='warps':
                saved['alpha'][28:36]=0
                if c==8:saved['depth_error'][:,32:]=1
            for k,v in saved.items():np.save(d/f'{k}.npy',v)
            write(d/'receipt.json',dict(box=[0,0,w,h],arrays=list(saved)))
    single.combine(tmp_path,7,'check',{**single.DEFAULT,'count':2,'mode':'direct','match_weight':0})
    d=tmp_path/'candidates/check';labels=np.load(d/'cam007_sources.npy');result=np.load(d/'cam007.npy');weight=np.load(d/'cam007_confidence.npy')
    assert labels[12,12]==8 and labels[12,52]==9
    assert (labels[28:36]==-1).all()
    np.testing.assert_array_equal(result[weight==0],base[weight==0])
    assert abs(float(result[12,52,0])-.42)<1e-6
