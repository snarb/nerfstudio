"""TRAIN LPIPS confidence in the exact output pixel frame, with bracketing.

Witness photos supply scalar scores only. Source candidates retain the frozen
q50 lookup and the actual target-space detail bandwidth. There is no target GT.
"""
from pathlib import Path
import time
import numpy as np
import torch
from luster_hybrid_common import assert_source, geometry, native_rows, read, write, sha, unproject, log
from transfer_luster_hybrid import source_score, warp
from cubic75_luster_hybrid import warp as cubic_warp
from supported_luster_hybrid import geometric_support
from risk_single_luster_hybrid import masked_low
from witness_benefit_luster_hybrid import tangent_cosine
from witness_native_restoration_luster_hybrid import median_cost
from train_perceptual_utility_luster_hybrid import utility_from_errors


def confidence_config(config):
    return {k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution','sigma']}


def conservative_broad_utility(utilities, known):
    """Require two witnesses and agreement of every valid witness."""
    assert utilities.shape==known.shape and len(utilities)>=2
    assert np.isfinite(utilities[known]).all()
    assert np.all(np.abs(utilities[known])<=1)
    count=known.sum(axis=0)
    worst=np.min(np.where(known,utilities,np.inf),axis=0)
    return np.where(count>=2,worst,0).astype(np.float32),count.astype(np.uint8)


def compare_modes(baseline, detail, full, witness, support, metric):
    """One common exterior and three fixed LPIPS errors; no fitted parameters."""
    assert all(v.shape==(*support.shape,3) for v in [baseline,detail,full,witness])
    if not support.any():
        zero=np.zeros(support.shape,np.float32)
        return zero,zero,support.copy(),support.copy()
    assert not metric.training and not any(p.requires_grad for p in metric.parameters())
    device=next(metric.parameters()).device
    def tensor(rgb):
        assert np.isfinite(rgb[support]).all()
        array=np.where(support[...,None],np.clip(rgb,0,1),.5).astype(np.float32)
        return torch.from_numpy(array).permute(2,0,1)[None].to(device)*2-1
    with torch.inference_mode():
        truth=tensor(witness)
        errors=[metric(tensor(v),truth)[0,0].cpu().numpy() for v in [baseline,detail,full]]
    detail_utility,kd=utility_from_errors(errors[0],errors[1],support)
    full_vs_detail,kf=utility_from_errors(errors[1],errors[2],support)
    full_vs_base,kb=utility_from_errors(errors[0],errors[2],support)
    # Broad transfer must improve on both frozen baseline and detail-only.
    broad=np.minimum(full_vs_detail,full_vs_base)
    return detail_utility,broad,kd,kf&kb


def selection(root,target,source):
    path=root/'preflight/bracket_witness_all_geometry_coverage.json'
    r=read(path)[str(target)]
    assert r['reads_rgb'] is False and r['uses_masks'] is False
    assert r['code_sha256']==sha(Path(__file__).with_name('probe_bracket_all_geometry_luster.py'))
    witnesses=r['sources'][str(source)]['witnesses']
    assert len(witnesses)==len(set(witnesses)) and len(witnesses)<=4
    rows=native_rows()
    for c in witnesses:
        assert_source(rows,target,c);assert_source(rows,source,c)
        assert c not in [7,12,95,150]
    return witnesses,path


def prepare(root):
    """CPU only; create missing TRAIN confidence warps before acquiring GPU."""
    deadline=read(root/'preflight/manifest.json')['deadline']
    for target in [7,12,95,150]:
        sources=read(root/'candidates/single_photometric_depth25_seam'/f'cam{target:03d}_receipt.json')['sources']
        witnesses=set()
        for source in sources:witnesses.update(selection(root,target,source)[0])
        for witness in sorted(witnesses):
            if time.time()>deadline:raise TimeoutError('Campaign deadline')
            warp(root,target,witness);cubic_warp(root,target,witness)
        log(root,'bracket_perceptual_warps_prepared',target=target,witnesses=sorted(witnesses))


def cached_utility(root,target,source,config):
    assert_source(native_rows(),target,source)
    out=root/'bracket_perceptual_projected'/f'cam{target:03d}'/f'src{source:03d}'
    r=read(out/'receipt.json')
    assert r['code_sha256']==sha(Path(__file__)) and r['target']==target and r['source']==source
    assert r['confidence_config']==confidence_config(config)
    assert all(sha(out/k)==v for k,v in r['hashes'].items())
    return out


def projected_utility(root,target,source,config,metric,metric_receipt):
    rows=native_rows();assert_source(rows,target,source)
    witnesses,dependency=selection(root,target,source)
    out=root/'bracket_perceptual_projected'/f'cam{target:03d}'/f'src{source:03d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json')
        assert r['inputs'][str(dependency)]==sha(dependency)
        assert r['inputs'][str(metric_receipt)]==sha(metric_receipt)
        return cached_utility(root,target,source,config)
    d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
    sr=read(d/'receipt.json');box=sr['box'];x0,y0,x1,y1=box
    arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in sr['arrays']}
    source_support=geometric_support(arrays,config)
    g=geometry(root,target);b=np.asarray(g['rgb'][y0:y1,x0:x1]);s=np.asarray(arrays['rgb'])
    sl,_=masked_low(s,source_support,config['sigma']);bl,_=masked_low(b,source_support,config['sigma'])
    candidate=np.clip(b+(s-sl)-(b-bl),0,1)
    yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
    world=unproject(np.asarray(g['depth_z_q50'][y0:y1,x0:x1]),rows[target],xy)
    center=np.asarray(rows[target]['transform_matrix'],np.float32)[:3,3]
    rays=world-center;rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-8)
    source_offset=np.asarray(rows[source]['transform_matrix'],np.float32)[:3,3]-center
    inputs={str(p):sha(p) for p in [dependency,metric_receipt,d/'receipt.json',root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    for key in sr['arrays']:inputs[str(d/f'{key}.npy')]=sha(d/f'{key}.npy')
    utilities=[];broad_utilities=[];validity=[];broad_validity=[]
    for witness in witnesses:
        wd=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{witness:03d}'
        wr=read(wd/'receipt.json');assert wr['box']==box
        wa={k:np.load(wd/f'{k}.npy',mmap_mode='r') for k in wr['arrays']}
        offset=np.asarray(rows[witness]['transform_matrix'],np.float32)[:3,3]-center
        opposite=tangent_cosine(source_offset,offset,rays)<=-.25
        pair_score,_=source_score(world,rows[source],rows[witness])
        support=source_support&geometric_support(wa,{**config,'max_score':2.})&opposite&(pair_score<=2.)
        utility,broad,known,broad_known=compare_modes(b,candidate,s,np.asarray(wa['rgb']),support,metric)
        utilities.append(utility);broad_utilities.append(broad);validity.append(known);broad_validity.append(broad_known)
        for p in [wd/'receipt.json',*(wd/f'{key}.npy' for key in wr['arrays'])]:inputs[str(p)]=sha(p)
    while len(utilities)<2:
        utilities.append(np.zeros(source_support.shape,np.float32))
        broad_utilities.append(np.zeros(source_support.shape,np.float32))
        validity.append(np.zeros(source_support.shape,bool))
        broad_validity.append(np.zeros(source_support.shape,bool))
    known=np.stack(validity)
    cost,count=median_cost(np.stack(utilities),known)
    broad,bcount=conservative_broad_utility(np.stack(broad_utilities),np.stack(broad_validity))
    for key,value in [('cost',cost),('count',count),('broad_utility',broad),('broad_count',bcount)]:np.save(out/f'{key}.npy',value)
    for module in ['luster_hybrid_common.py','transfer_luster_hybrid.py','cubic75_luster_hybrid.py',
                   'supported_luster_hybrid.py','risk_single_luster_hybrid.py','witness_benefit_luster_hybrid.py',
                   'witness_native_restoration_luster_hybrid.py','train_perceptual_utility_luster_hybrid.py',
                   'probe_bracket_all_geometry_luster.py','render_projection_luster_hybrid.py']:
        path=Path(__file__).with_name(module);inputs[str(path)]=sha(path)
    write(out/'receipt.json',dict(target=target,source=source,witnesses=witnesses,box=box,confidence_config=confidence_config(config),
        code_sha256=sha(Path(__file__)),inputs=inputs,hashes={k:sha(out/k) for k in ['cost.npy','count.npy','broad_utility.npy','broad_count.npy']},
        uses_target_gt=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False,
        geometry='frozen target camera-z q50',confidence_frame='native target pixels, projected TRAIN witnesses',
        bracket_cosine_max=-.25,confidence_max_score=2.,output_photo_max_score=config['max_score'],
        limitation='Projected witness texture is an imperfect perceptual reference; target evaluation remains required.'))
    log(root,'bracket_perceptual_complete',target=target,source=source,known_fraction=float(np.mean(count>=2)))
    return out
