"""Native TRAIN-only warps, conservative visibility, and selective detail replacement.

This module deliberately does not import the evaluator or read evaluation masks/GT.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from luster_hybrid_common import (TARGETS, allowed_sources, assert_source, geometry,
                                  image, log, native_rows, project, read, sha,
                                  source_path, unproject, write)

cv2.setNumThreads(4)


def source_score(world, target, source):
    ct=np.asarray(target['transform_matrix'],np.float32)[:3,3]
    cs=np.asarray(source['transform_matrix'],np.float32)[:3,3]
    a=world-ct;b=world-cs
    cosine=np.sum(a*b,-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8)
    angle=np.arccos(cosine.clip(-1,1))*180/np.pi
    _,zt=project(world,target);uv,zs=project(world,source)
    ratio=np.sqrt(source['fl_x']*source['fl_y']/(target['fl_x']*target['fl_y']))*zt/np.maximum(zs,1e-8)
    score=angle/30+.3*np.maximum(0,-np.log2(np.maximum(ratio,1e-8)))
    valid=np.isfinite(uv).all(-1)&(uv[...,0]>=1)&(uv[...,1]>=1)&(uv[...,0]<source['w']-2)&(uv[...,1]<source['h']-2)
    return np.where(valid,score,np.inf).astype(np.float32),ratio.astype(np.float32)


def select_sources(root,count=32):
    rows=native_rows();selections={};all_sources=[]
    for target in TARGETS:
        row=rows[target];g=geometry(root,target)
        yy,xx=np.mgrid[0:row['h']:24,0:row['w']:24]
        z=np.asarray(g['depth_z_q50'][yy,xx]);valid=np.isfinite(z)&(g['alpha'][yy,xx]>=.95)
        xy=np.stack([xx,yy],-1)[valid].astype(np.float32)
        world=unproject(z[valid],row,xy)
        candidates=allowed_sources(rows,target)
        scores=np.stack([source_score(world,row,rows[c])[0] for c in candidates])
        order=np.argsort(scores,axis=0,kind='stable')
        # Reward cameras that provide useful coverage anywhere, including close-up views.
        votes=np.zeros(len(candidates))
        for rank in range(8):
            idx=order[rank];value=scores[idx,np.arange(len(world))]
            np.add.at(votes,idx,np.where(np.isfinite(value),np.exp(-value)/(rank+1),0))
        chosen=np.argsort(-votes,kind='stable')[:count]
        ids=[candidates[i] for i in chosen if votes[i]>0]
        selections[str(target)]=ids;all_sources.extend(ids)
        log(root,'sources_selected',camera=target,sources=ids,considered=len(candidates))
    out=dict(usage='prediction_only',selection='coverage-weighted angular and projected-resolution rank; no target RGB',
             sources=selections,all_sources=sorted(set(all_sources)),count=count)
    write(root/'preflight/sources.json',out)
    return out


def footprint_metrics(source_rgb, source, uv, expected_z, world, target, xy, zlo, zhi):
    """Evaluate all nonzero native bilinear taps, without inventing border samples."""
    h,w=source['alpha'].shape
    u,v=uv[...,0],uv[...,1]
    valid=np.isfinite(uv).all(-1)&np.isfinite(expected_z)&(expected_z>0)&(u>=0)&(v>=0)&(u<=w-1)&(v<=h-1)
    u=np.where(valid,u,0);v=np.where(valid,v,0)
    x=np.floor(u).astype(np.int32);y=np.floor(v).astype(np.int32);fx=u-x;fy=v-y
    rgb=np.zeros((*u.shape,3),np.float32);alpha=np.ones(u.shape,np.float32)
    err=np.zeros(u.shape,np.float32);depth=np.zeros_like(err);low=np.zeros_like(err);high=np.zeros_like(err)
    for dy in (0,1):
        for dx in (0,1):
            weight=((fx if dx else 1-fx)*(fy if dy else 1-fy)).astype(np.float32)
            xx=np.clip(x+dx,0,w-1);yy=np.clip(y+dy,0,h-1);tap=weight>1e-7
            dz=np.asarray(source['depth_z_q50'][yy,xx]);lo=source['depth_z_q10'][yy,xx];hi=source['depth_z_q90'][yy,xx]
            valid&=~tap|(np.isfinite(dz)&np.isfinite(lo)&np.isfinite(hi))
            alpha=np.minimum(alpha,np.where(tap,source['alpha'][yy,xx],1))
            err=np.maximum(err,np.where(tap,np.abs(dz-expected_z)/np.maximum(expected_z,1e-8),0))
            depth+=np.nan_to_num(dz)*weight;low+=np.nan_to_num(lo)*weight;high+=np.nan_to_num(hi)*weight
            rgb+=source_rgb[yy,xx]*weight[...,None]
    return rgb,valid,alpha,err,depth,low,high


def warp(root, target, source_id):
    rows=native_rows();assert_source(rows,target,source_id)
    row=rows[target];sr=rows[source_id];g=geometry(root,target);sg=geometry(root,source_id)
    dest=root/'warps'/f'cam{target:03d}'/f'src{source_id:03d}'
    if (dest/'receipt.json').exists():
        receipt=read(dest/'receipt.json')
        assert receipt['source_sha256']==sha(source_path(source_id))
        assert receipt['warp_code_sha256']==warp_fingerprint()
        return
    yy,xx=np.where(np.isfinite(g['depth_z_q50'])&(g['alpha']>=.95))
    # Prediction-derived crop only; no evaluation ROI is used here.
    box=[max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(row['w'],int(xx.max())+33),min(row['h'],int(yy.max())+33)]
    x0,y0,x1,y1=box;shape=(y1-y0,x1-x0)
    dest.mkdir(parents=True,exist_ok=True)
    names=['rgb','alpha','depth_error','roundtrip','uncertainty','score','resolution_ratio','valid']
    arrays={k:np.lib.format.open_memmap(dest/f'{k}.npy',mode='w+',
            dtype=np.uint8 if k=='valid' else np.float32,
            shape=(*shape,3) if k=='rgb' else shape) for k in names}
    manifest=read(root/'preflight/manifest.json')
    assert sha(source_path(source_id))==manifest['source_hashes'][str(source_path(source_id))]
    source_rgb=np.asarray(Image.open(source_path(source_id)).convert('RGB'),np.float32)/255
    cs=np.asarray(sr['transform_matrix'],np.float32)[:3,3]
    started=time.monotonic()
    for y in range(y0,y1,128):
        stop=min(y+128,y1);sl=np.s_[y:stop,x0:x1]
        yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        z=np.asarray(g['depth_z_q50'][sl]);lo=np.asarray(g['depth_z_q10'][sl]);hi=np.asarray(g['depth_z_q90'][sl])
        world=unproject(z,row,xy);uv,zs=project(world,sr)
        colors,valid,alpha,err,sd,slo,shi=footprint_metrics(source_rgb,sg,uv,zs,world,row,xy,lo,hi)
        back=cs+(world-cs)*(sd/np.maximum(zs,1e-8))[...,None]
        buv,_=project(back,row);rt=np.linalg.norm(buv-xy,axis=-1)
        # Project source and target depth intervals into the other image; express both in target pixels.
        b0,_=project(cs+(world-cs)*(slo/np.maximum(zs,1e-8))[...,None],row)
        b1,_=project(cs+(world-cs)*(shi/np.maximum(zs,1e-8))[...,None],row)
        t0,_=project(unproject(lo,row,xy),sr);t1,_=project(unproject(hi,row,xy),sr)
        score,ratio=source_score(world,row,sr)
        uncertainty=np.maximum(np.linalg.norm(b1-b0,axis=-1),np.linalg.norm(t1-t0,axis=-1)/np.maximum(ratio,1e-8))*.5
        valid&=np.isfinite(z)&np.isfinite(rt)&np.isfinite(uncertainty)&np.isfinite(score)
        values=dict(rgb=colors,valid=valid,alpha=np.minimum(alpha,g['alpha'][sl]),depth_error=err,
                    roundtrip=rt,uncertainty=uncertainty,score=score,resolution_ratio=ratio)
        for k,v in values.items():arrays[k][y-y0:stop-y0]=np.nan_to_num(v,nan=0 if k=='rgb' else 1e6,posinf=1e6,neginf=1e6)
    for v in arrays.values():v.flush()
    write(dest/'receipt.json',dict(target=target,source=source_id,box=box,arrays=names,
          source_sha256=sha(source_path(source_id)),code_sha256=sha(Path(__file__)),warp_code_sha256=warp_fingerprint(),
          target_geometry_sha256=sha(root/'geometry'/f'cam{target:03d}'/'receipt.json'),
          source_geometry_sha256=sha(root/'geometry'/f'cam{source_id:03d}'/'receipt.json'),
          seconds=time.monotonic()-started,uses_target_rgb=False,uses_masks=False))
    log(root,'warp_complete',camera=target,source=source_id,seconds=time.monotonic()-started)


def warp_fingerprint():
    return hashlib.sha256(''.join(inspect.getsource(f) for f in
            (warp,footprint_metrics,project,unproject,source_score)).encode()).hexdigest()


def gaussian(value,sigma):
    return cv2.GaussianBlur(value,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)


def merge_details(base,texture,labels,confidence,mode='multiband',sigma=4.,gain=1.):
    """Fallback is an exact copy; filters never cross a source/support boundary."""
    result=base.copy()
    applied=np.zeros(labels.shape,np.float32)
    for cid in np.unique(labels[labels>=0]):
        mask=(labels==cid).astype(np.uint8)
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        # Fade source changes towards the base, not towards another source texture.
        weight=(np.minimum(distance/3,1)*confidence*gain).astype(np.float32)
        chosen=(mask>0)&(weight>0)
        if not chosen.any():continue
        if mode=='direct':delta=texture-base
        else:
            support=mask.astype(np.float32)
            def low(x,s):return gaussian(x*support[...,None],s)/np.maximum(gaussian(support,s)[...,None],1e-8)
            if mode=='detail':
                delta=(texture-low(texture,sigma))-(base-low(base,sigma))
            elif mode=='multiband':
                delta=np.zeros_like(base)
                last_s=texture;last_b=base
                for s in (1.,2.,4.,8.):
                    if s>sigma:break
                    ls=low(texture,s);lb=low(base,s)
                    band=(last_s-ls)-(last_b-lb)
                    # Reliable filter support at each scale, without borrowing background.
                    margin=np.clip((distance-.5)/(s+1),0,1)
                    delta+=band*margin[...,None]
                    last_s=ls;last_b=lb
            else:raise ValueError(mode)
        result[chosen]=np.clip(base[chosen]+weight[chosen,None]*delta[chosen],0,1)
        applied[chosen]=weight[chosen]
    return result,applied


DEFAULT=dict(count=16,alpha=.99,depth_error=.005,roundtrip=1.,uncertainty=2.,max_score=1.,
             min_resolution=.75,agreement=.06,mode='multiband',sigma=4.,gain=1.)


def combine(root,target,name,config):
    rows=native_rows();base=np.asarray(geometry(root,target)['rgb'])
    source_ids=read(root/'preflight/sources.json')['sources'][str(target)][:config['count']]
    out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config_file=out/'config.json'
    if config_file.exists():assert read(config_file)==config
    else:write(config_file,config)
    directory=root/'warps'/f'cam{target:03d}'
    receipt=read(directory/f'src{source_ids[0]:03d}'/'receipt.json');x0,y0,x1,y1=receipt['box']
    shape=(y1-y0,x1-x0);sl=np.s_[y0:y1,x0:x1]
    records=[]
    for cid in source_ids:
        assert_source(rows,target,cid)
        d=directory/f'src{cid:03d}';r=read(d/'receipt.json');assert r['box']==receipt['box']
        records.append((cid,{k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}))
    # Three best geometric candidates; a second independent source must confirm local appearance.
    scores=np.full((3,*shape),np.inf,np.float32);ids=np.full((3,*shape),-1,np.int16)
    colors=np.zeros((3,*shape,3),np.float32)
    reasons=np.zeros(shape,np.uint8);support_count=np.zeros(shape,np.uint8)
    for cid,a in records:
        valid=(a['valid']>0)&(a['alpha']>=config['alpha'])&(a['depth_error']<=config['depth_error'])
        valid&=(a['roundtrip']<=config['roundtrip'])&(a['uncertainty']<=config['uncertainty'])
        valid&=(a['score']<=config['max_score'])&(a['resolution_ratio']>=config['min_resolution'])
        support_count+=valid
        score=np.where(valid,a['score']+.1*a['roundtrip']/max(config['roundtrip'],1e-8),np.inf)
        color=np.asarray(a['rgb']);label=np.full(shape,cid,np.int16)
        for rank in range(3):
            take=score<scores[rank]
            oldscore=scores[rank].copy();oldid=ids[rank].copy();oldcolor=colors[rank].copy()
            scores[rank][take]=score[take];ids[rank][take]=label[take];colors[rank][take]=color[take]
            score=np.where(take,oldscore,score);label=np.where(take,oldid,label);color=np.where(take[...,None],oldcolor,color)
    # Compare aligned local patches; no target image participates in admission.
    chosen=np.full(shape,-1,np.int16);texture=np.asarray(base[sl]).copy();confidence=np.zeros(shape,np.float32)
    for i,j in ((0,1),(0,2),(1,2)):
        pair=(ids[i]>=0)&(ids[j]>=0)
        weight=gaussian(pair.astype(np.float32),2.)
        diff=np.mean(np.abs(colors[i]-colors[j]),axis=-1)*pair
        error=gaussian(diff,2.)/np.maximum(weight,1e-8)
        agree=pair&(weight>.8)&(error<=config['agreement'])&(chosen<0)
        chosen[agree]=ids[i][agree];texture[agree]=colors[i][agree]
        confidence[agree]=np.clip(1-error[agree]/(2*config['agreement']),.5,1)
    # Remove isolated labels and shrink unsupported islands; never create new support.
    for cid in source_ids:
        m=(chosen==cid).astype(np.uint8)
        coherent=cv2.medianBlur(m,3)>0
        reject=(m>0)&~coherent;chosen[reject]=-1;confidence[reject]=0
    reasons[support_count==0]=1;reasons[support_count==1]=2;reasons[(support_count>=2)&(chosen<0)]=3
    local,applied=merge_details(np.asarray(base[sl]),texture,chosen,confidence,config['mode'],config['sigma'],config['gain'])
    result=base.copy();result[sl]=local
    full_labels=np.full(base.shape[:2],-1,np.int16);full_labels[sl]=np.where(applied>0,chosen,-1)
    full_weight=np.zeros(base.shape[:2],np.float32);full_weight[sl]=applied
    full_reasons=np.full(base.shape[:2],4,np.uint8);full_reasons[sl]=reasons
    assert np.array_equal(result[full_weight==0],base[full_weight==0])
    np.save(out/f'cam{target:03d}.npy',result)
    np.save(out/f'cam{target:03d}_sources.npy',full_labels)
    np.save(out/f'cam{target:03d}_confidence.npy',full_weight)
    np.save(out/f'cam{target:03d}_reasons.npy',full_reasons)
    image(out/f'cam{target:03d}.png',result)
    image(out/f'cam{target:03d}_transfer.png',np.repeat(full_weight[...,None],3,-1))
    palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32)
    color_map=palette[np.maximum(full_labels,0)];color_map[full_labels<0]=0
    image(out/f'cam{target:03d}_sources.png',color_map)
    write(out/f'cam{target:03d}_receipt.json',dict(config=config,camera=target,sources=source_ids,
          source_hashes={str(source_path(c)):sha(source_path(c)) for c in source_ids},
          code_sha256=sha(Path(__file__)),uses_target_rgb=False,uses_masks=False,
          changed_fraction=float(np.any(result!=base,axis=-1).mean()),
          reasons={'1':'no geometric source','2':'one source only','3':'appearance/coherence rejected','4':'outside predicted support'}))
    log(root,'hybrid_complete',camera=target,candidate=name,changed_fraction=float(np.any(result!=base,axis=-1).mean()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('sources');q.add_argument('--count',type=int,default=32)
    q=sub.add_parser('warp');q.add_argument('--cameras',nargs='+',type=int,default=list(TARGETS));q.add_argument('--count',type=int,default=32)
    q=sub.add_parser('combine');q.add_argument('--name',required=True);q.add_argument('--config',type=Path);q.add_argument('--cameras',nargs='+',type=int,default=list(TARGETS))
    a=p.parse_args()
    if a.command=='sources':select_sources(a.root,a.count)
    elif a.command=='warp':
        selected=read(a.root/'preflight/sources.json')['sources']
        for c in a.cameras:
            for s in selected[str(c)][:a.count]:warp(a.root,c,s)
    else:
        config={**DEFAULT,**(read(a.config) if a.config else {})}
        for c in a.cameras:combine(a.root,c,a.name,config)
