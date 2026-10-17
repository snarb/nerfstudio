"""Single-photo merge with explicit source-boundary compatibility checks.

Witness photos affect only seam confidence; one selected photograph supplies RGB.
No target GT, ROI, calibration change or texture-coordinate shift is used.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from transfer_luster_hybrid import gaussian
from supported_luster_hybrid import geometric_support
cv2.setNumThreads(4)


def merge(base,records,labels,config):
    selected_delta=np.zeros_like(base);deltas={};supports={}
    for cid,a in records:
        chosen=labels==cid
        if not chosen.any():continue
        support=geometric_support(a,config) if config['filter_support']=='geometry' else chosen
        sf=support.astype(np.float32);sigma=config['sigma'];mass=gaussian(sf,sigma)
        texture=np.asarray(a['rgb'])
        low_t=gaussian(texture*sf[...,None],sigma)/np.maximum(mass[...,None],1e-8)
        low_b=gaussian(base*sf[...,None],sigma)/np.maximum(mass[...,None],1e-8)
        delta=(texture-low_t)-(base-low_b)
        deltas[cid]=delta;supports[cid]=support;selected_delta[chosen]=delta[chosen]
    full_ok=np.zeros(labels.shape,bool)
    if config['adaptive_seam']:
        # All source labels touching a 3px neighborhood must corroborate the
        # selected detail at the same calibrated surface point. No RGB mixing.
        assert config['filter_support']=='geometry'
        kernel=np.ones((7,7),np.uint8);full_ok=labels>=0
        full_ok&=cv2.erode((labels>=0).astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        for cid,delta in deltas.items():
            near=(cv2.dilate((labels==cid).astype(np.uint8),kernel)>0)&(labels!=cid)&(labels>=0)
            disagreement=np.max(np.abs(delta-selected_delta),axis=-1)
            # Avoid trusting an isolated accidental RGB match on a bad seam.
            disagreement=cv2.dilate(disagreement,np.ones((3,3),np.uint8))
            compatible=supports[cid]&(disagreement<=config['seam_tolerance'])
            full_ok[near&~compatible]=False
    weight=np.zeros(labels.shape,np.float32)
    for cid in deltas:
        chosen=labels==cid;distance=cv2.distanceTransform(chosen.astype(np.uint8),cv2.DIST_L2,5)
        weight[chosen]=np.minimum(distance[chosen]/config['seam_fade'],1)*config['gain']
    weight[full_ok]=config['gain']
    result=base.copy();chosen=weight>0
    result[chosen]=np.clip(base[chosen]+weight[chosen,None]*selected_delta[chosen],0,1)
    return result,weight


def combine(root,parent,name,settings):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_seam','parent':parent,**settings}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');records=[]
        for cid in receipt['sources']:
            d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
            records.append((cid,{k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}))
        x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1];base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        local,local_weight=merge(base[sl],records,labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=local_weight
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_seam_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--filter-support',choices=['labels','geometry'],default='labels');p.add_argument('--seam-fade',type=float,default=3.);p.add_argument('--adaptive-seam',action='store_true');p.add_argument('--seam-tolerance',type=float,default=.02);a=p.parse_args()
    assert a.seam_fade>=1
    combine(a.root,a.parent,a.name,dict(filter_support=a.filter_support,seam_fade=a.seam_fade,adaptive_seam=a.adaptive_seam,seam_tolerance=a.seam_tolerance))
