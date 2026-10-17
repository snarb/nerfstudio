"""Coherent patch source labels with current phase/RT2 and cubic eligibility.

Graph cuts choose one TRAIN photograph per patch on the frozen surface.
"""
from pathlib import Path
import argparse
import cv2
import maxflow
from source_phase_quality_luster_hybrid import quality
from supported_luster_hybrid import geometric_support
import numpy as np
from luster_hybrid_common import assert_source,geometry,image,log,native_rows,read,sha,source_path,write
from transfer_luster_hybrid import gaussian,merge_details
cv2.setNumThreads(4)
DEFAULT=dict(count=16,alpha=.99,depth_error=.005,roundtrip=1.,uncertainty=2.,max_score=1.,min_resolution=.5,
             match_sigma=2.,patch_sigma=3.,match_limit=.10,match_weight=4.,mode='detail',sigma=8.,gain=1.,
             boundary_fade=3.,support_mass=.8,scale_compatibility_weight=.3,roundtrip_weight=.1,uncertainty_weight=.0,depth_weight=.0,cost_sigma=0.,source_phase_weight=0.,view_phase_weight=0.)

def coherent_labels(scores,eligibility,shape,step,transition_cost,fallback_cost):
    """Patch-averaged unary costs with a Potts penalty; invalid pixels stay fallback."""
    coarse=(int(np.ceil(shape[1]/step)),int(np.ceil(shape[0]/step)))
    unary=[]
    for score,eligible in zip(scores,eligibility):
        coverage=cv2.resize(eligible.astype(np.float32),coarse,interpolation=cv2.INTER_AREA)
        average=cv2.resize(np.where(eligible,score,0).astype(np.float32),coarse,interpolation=cv2.INTER_AREA)/np.maximum(coverage,1e-8)
        unary.append(np.where(coverage>=.8,average,1e6))
    unary.append(np.full_like(unary[0],fallback_cost))
    costs=np.stack(unary,axis=-1).astype(np.float64)
    binary=np.full((len(scores)+1,len(scores)+1),transition_cost,np.float64)
    np.fill_diagonal(binary,0)
    patch=maxflow.fastmin.aexpansion_grid(costs,binary,max_cycles=5)
    labels=cv2.resize(patch.astype(np.float32),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int16)
    for index,eligible in enumerate(eligibility):
        labels[(labels==index)&~eligible]=len(scores)
    return labels


def masked_low(rgb,valid,sigma):
    mass=gaussian(valid.astype(np.float32),sigma)
    return gaussian(rgb*valid[...,None],sigma)/np.maximum(mass[...,None],1e-8),mass

def combine(root,target,name,config):
    rows=native_rows();g=geometry(root,target);base=np.asarray(g['rgb']);source_ids=read(root/'preflight/sources.json')['sources'][str(target)][:config['count']]
    out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    first=root/'warps'/f'cam{target:03d}'/f'src{source_ids[0]:03d}'
    box=read(first/'receipt.json')['box'];x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];b=base[sl];shape=b.shape[:2]
    best=np.full(shape,np.inf,np.float32);labels=np.full(shape,-1,np.int16)
    texture=b.copy();count=np.zeros(shape,np.uint8);chosen_error=np.zeros(shape,np.float32)
    quality_inputs={};scores=[];eligibility=[]
    for cid in source_ids:
        assert_source(rows,target,cid);d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';receipt=read(d/'receipt.json');assert receipt['box']==box
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in receipt['arrays']}
        valid=(a['valid']>0)&(a['alpha']>=config['alpha'])&(a['depth_error']<=config['depth_error'])&(a['roundtrip']<=config['roundtrip'])&(a['uncertainty']<=config['uncertainty'])&(a['score']<=config['max_score'])&(a['resolution_ratio']>=config['min_resolution'])
        count+=valid
        photo,mass=masked_low(a['rgb'],valid,config['match_sigma']);model,_=masked_low(b,valid,config['match_sigma'])
        difference=np.mean(np.abs(photo-model),axis=-1)
        patch_mass=gaussian(valid.astype(np.float32),config['patch_sigma'])
        error=gaussian(difference*valid,config['patch_sigma'])/np.maximum(patch_mass,1e-8)
        score=a['score']+config['roundtrip_weight']*a['roundtrip']/config['roundtrip']+config['match_weight']*error/config['match_limit']
        score+=config['scale_compatibility_weight']*np.maximum(0,np.log2(np.maximum(a['resolution_ratio'],1e-8)))
        score+=config['uncertainty_weight']*a['uncertainty']/config['uncertainty']+config['depth_weight']*a['depth_error']/config['depth_error']
        qdir=quality(root,target,cid,config);qr=read(qdir/'receipt.json')
        for key,weight_key in [('source_phase_cost','source_phase_weight'),('view_phase_cost','view_phase_weight')]:
            path=qdir/f'{key}.npy';score+=config[weight_key]*np.load(path,mmap_mode='r');quality_inputs[str(path)]=sha(path)
        quality_inputs[str(qdir/'receipt.json')]=sha(qdir/'receipt.json')
        quality_inputs[qr['source_model_receipt']]=qr['source_model_receipt_sha256']
        if config['cost_sigma']>0:
            finite_valid=valid & np.isfinite(score)
            cost_mass=gaussian(finite_valid.astype(np.float32),config['cost_sigma'])
            score=gaussian(np.where(finite_valid,score,0),config['cost_sigma'])/np.maximum(cost_mass,1e-8)
        eligible=valid&(mass>=config['support_mass'])&(patch_mass>=config['support_mass'])&(error<=config['match_limit'])
        cubic=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{cid:03d}'
        cr=read(cubic/'receipt.json');assert cr['box']==box
        ca={k:np.load(cubic/f'{k}.npy',mmap_mode='r') for k in cr['arrays'] if k!='rgb'}
        eligible &= geometric_support(ca,config)
        quality_inputs[str(cubic/'receipt.json')]=sha(cubic/'receipt.json')
        scores.append(np.asarray(score).copy());eligibility.append(eligible)
    patch_labels=coherent_labels(scores,eligibility,shape,config['grid_step'],config['transition_cost'],config['fallback_cost'])
    for index,cid in enumerate(source_ids):
        take=patch_labels==index
        photo=np.load(root/'warps'/f'cam{target:03d}'/f'src{cid:03d}'/'rgb.npy',mmap_mode='r')
        labels[take]=cid;texture[take]=photo[take]
    # A majority filter only removes islands; it cannot invent a source lookup.
    for cid in source_ids:
        mask=(labels==cid).astype(np.uint8);reject=(mask>0)&(cv2.medianBlur(mask,3)==0);labels[reject]=-1
    confidence=(labels>=0).astype(np.float32)
    local,weight=merge_details(b,texture,labels,confidence,config['mode'],config['sigma'],config['gain'])
    result=base.copy();result[sl]=local
    full_labels=np.full(base.shape[:2],-1,np.int16);full_labels[sl]=np.where(weight>0,labels,-1)
    full_weight=np.zeros(base.shape[:2],np.float32);full_weight[sl]=weight
    reasons=np.full(base.shape[:2],4,np.uint8);reasons[sl]=np.where(labels>=0,0,np.where(count==0,1,3))
    assert np.isfinite(result).all() and np.array_equal(result[full_weight==0],base[full_weight==0])
    for suffix,array in [('',result),('_sources',full_labels),('_confidence',full_weight),('_reasons',reasons)]:np.save(out/f'cam{target:03d}{suffix}.npy',array)
    image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(full_weight[...,None],3,-1))
    palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32);colors=palette[np.maximum(full_labels,0)];colors[full_labels<0]=0;image(out/f'cam{target:03d}_sources.png',colors)
    write(out/f'cam{target:03d}_receipt.json',dict(config=config,camera=target,sources=source_ids,source_hashes={str(source_path(c)):sha(source_path(c)) for c in source_ids},code_sha256=sha(Path(__file__)),additional_input_hashes=quality_inputs,uses_target_rgb=False,uses_masks=False,geometry='frozen target camera-z q50',texture_source_count_per_pixel=1,lookup_shift=False,changed_fraction=float(np.any(result!=base,axis=-1).mean()),reasons={'1':'no geometric source','3':'patch mismatch/support/coherence rejected','4':'outside predicted support'}))
    log(root,'coherent_phase_ranked_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--config',type=Path);p.add_argument('--cameras',nargs='+',type=int,default=[7,12,95,150]);a=p.parse_args()
    config={**DEFAULT,**(read(a.config) if a.config else {})}
    for c in a.cameras:combine(a.root,c,a.name,config)
