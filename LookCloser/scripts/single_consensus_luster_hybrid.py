"""Independent TRAIN sources corroborate a hard assignment, never supply its RGB."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from transfer_luster_hybrid import gaussian,merge_details
from single_source_luster_hybrid import masked_low
from coherent_luster_hybrid import corroboration
cv2.setNumThreads(4)

def evidence(root,parent,target,scale=2.,window=3.):
    old=root/'candidates'/parent;config=read(old/'config.json');r=read(old/f'cam{target:03d}_receipt.json')
    cache=root/'single_consensus'/parent/f'cam{target:03d}';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'receipt.json').exists():
        receipt=read(cache/'receipt.json');assert receipt['code_sha256']==sha(Path(__file__))
        return cache
    first=root/'warps'/f'cam{target:03d}'/f"src{r['sources'][0]:03d}";box=read(first/'receipt.json')['box'];x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1]
    labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')[sl];shape=labels.shape
    selected=np.zeros((*shape,3),np.float32);high=selected.copy();mass=np.zeros(shape,np.float32)
    for cid in r['sources']:
        d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';photo=np.load(d/'rgb.npy',mmap_mode='r');mask=labels==cid
        low,m=masked_low(photo,mask,scale);selected[mask]=photo[mask];high[mask]=(photo-low)[mask];mass[mask]=m[mask]
    best=np.full(shape,-1,np.float32);witness=np.full(shape,-1,np.int16)
    for cid in r['sources']:
        d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';receipt=read(d/'receipt.json');a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in receipt['arrays']}
        valid=(a['valid']>0)&(a['alpha']>=config['alpha'])&(a['depth_error']<=config['depth_error'])&(a['roundtrip']<=config['roundtrip'])&(a['uncertainty']<=config['uncertainty'])&(a['score']<=config['max_score'])&(a['resolution_ratio']>=config['min_resolution'])
        low,source_mass=masked_low(a['rgb'],valid,scale)
        support=valid&(labels>=0)&(labels!=cid)&(mass>.9)&(source_mass>.9)
        corr,energy=corroboration(high,a['rgb']-low,support,window)
        patch_mass=gaussian(support.astype(np.float32),window)
        error=gaussian(np.mean(np.abs(selected-a['rgb']),axis=-1)*support,window)/np.maximum(patch_mass,1e-8)
        good=support&(patch_mass>.8)&(energy>.002)&(error<.04)&(corr>best)
        best[good]=corr[good];witness[good]=cid
    np.save(cache/'correlation.npy',best);np.save(cache/'witness.npy',witness)
    write(cache/'receipt.json',dict(box=box,parent=parent,target=target,scale=scale,window=window,code_sha256=sha(Path(__file__)),parent_receipt_sha256=sha(old/f'cam{target:03d}_receipt.json'),correlation_sha256=sha(cache/'correlation.npy'),witness_sha256=sha(cache/'witness.npy')))
    log(root,'single_consensus_evidence',camera=target,parent=parent)
    return cache

def combine(root,parent,name,threshold):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'consensus_parent':parent,'consensus_min':threshold}
    write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        cache=evidence(root,parent,target);cr=read(cache/'receipt.json');x0,y0,x1,y1=cr['box'];sl=np.s_[y0:y1,x0:x1]
        r=read(old/f'cam{target:03d}_receipt.json');base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy')[sl].copy()
        labels[np.load(cache/'correlation.npy')<threshold]=-1
        texture=base[sl].copy()
        for source in r['sources']:
            photo=np.load(root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'rgb.npy',mmap_mode='r');take=labels==source;texture[take]=photo[take]
        result=base.copy();result[sl],weight=merge_details(base[sl],texture,labels,(labels>=0).astype(np.float32),config['mode'],config['sigma'],config['gain'])
        full_labels=np.full(base.shape[:2],-1,np.int16);full_labels[sl]=np.where(weight>0,labels,-1)
        full_weight=np.zeros(base.shape[:2],np.float32);full_weight[sl]=weight
        reason=np.full(base.shape[:2],4,np.uint8);reason[sl]=np.where(labels>=0,0,5)
        assert np.array_equal(result[full_weight==0],base[full_weight==0])
        for suffix,value in [('',result),('_sources',full_labels),('_confidence',full_weight),('_reasons',reason)]:np.save(out/f'cam{target:03d}{suffix}.npy',value)
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(full_weight[...,None],3,-1))
        palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32);colors=palette[np.maximum(full_labels,0)];colors[full_labels<0]=0;image(out/f'cam{target:03d}_sources.png',colors)
        parents={str(p):sha(p) for p in [cache/'receipt.json',cache/'correlation.npy',cache/'witness.npy',old/f'cam{target:03d}_receipt.json',old/f'cam{target:03d}_sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**r,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':parents,'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'reasons':{'4':'outside support','5':'independent phase agreement rejected'}})
        log(root,'single_consensus_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_patch_s8');a=p.parse_args()
    for threshold in [.3,.6,.85]:combine(a.root,a.parent,f'single_consensus_{int(threshold*100)}',threshold)
