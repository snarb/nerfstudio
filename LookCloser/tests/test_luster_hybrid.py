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


@pytest.mark.parametrize('module',['single_source_luster_hybrid','graph_single_luster_hybrid','scale_single_luster_hybrid','risk_single_luster_hybrid','phase_single_luster_hybrid','quality_risk_luster_hybrid','fine_phase_single_luster_hybrid'])
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
