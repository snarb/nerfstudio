"""One-photo transfer with target-guided local photometric correction.

The frozen target provides an edge-aware color correction, never coordinates.
Selected photo labels and their geometric eligibility are inherited unchanged.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,read,write,sha,log
from transfer_luster_hybrid import gaussian
cv2.setNumThreads(4)

def merge(base,texture,labels,config):
    result=base.copy();weight=np.zeros(labels.shape,np.float32)
    for cid in np.unique(labels[labels>=0]):
        selected=labels==cid;support=selected.astype(np.float32)
        mass=gaussian(support,config['sigma'])
        def mean(x):return gaussian(x*support[...,None],config['sigma'])/np.maximum(mass[...,None],1e-8)
        residual=texture-base;mb=mean(base);mr=mean(residual)
        variance=np.maximum(mean(base*base)-mb*mb,0)
        covariance=mean(base*residual)-mb*mr
        slope=covariance/(variance+config['guided_epsilon'])
        intercept=mr-slope*mb
        correction=mean(slope)*base+mean(intercept)
        delta=residual-correction
        distance=cv2.distanceTransform(selected.astype(np.uint8),cv2.DIST_L2,5)
        w=np.minimum(distance/3,1)*config['gain'];weight[selected]=w[selected]
        result[selected]=np.clip(base[selected]+w[selected,None]*delta[selected],0,1)
    return result,weight

def combine(root,parent,name,epsilon,sigma):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_guided','parent':parent,'guided_epsilon':epsilon,'sigma':sigma}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        first=root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
        local_labels=labels[sl];texture=base[sl].copy()
        for cid in receipt['sources']:
            d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';photo=np.load(d/'rgb.npy',mmap_mode='r');chosen=local_labels==cid;texture[chosen]=photo[chosen]
        local,w=merge(base[sl],texture,local_labels,config);result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=w
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_guided_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--epsilon',type=float,default=.001);p.add_argument('--sigma',type=float,default=8.);a=p.parse_args()
    assert a.epsilon>0 and a.sigma>0
    combine(a.root,a.parent,a.name,a.epsilon,a.sigma)
