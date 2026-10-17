"""Use established automatic source labels with stricter cubic visibility.

Source ranking is kept independent of interpolation. A failed cubic footprint
falls back to LookCloser instead of switching to an unrelated second photo.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,read,write,sha,log
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import merge_details
cv2.setNumThreads(4)

def combine(root,parent,name):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True);variant=root/'variants/cubic75'
    config={**read(old/'config.json'),'parent':parent,'source_selection_sampling':'bilinear','output_sampling':'clamped_cubic75'}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');base=np.asarray(geometry(root,target)['rgb']);old_labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        first=variant/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1];labels=old_labels[sl].copy();texture=base[sl].copy()
        for cid in receipt['sources']:
            d=variant/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
            valid=geometric_support(a,config);labels[(labels==cid)&~valid]=-1;selected=labels==cid;texture[selected]=a['rgb'][selected]
        for cid in receipt['sources']:
            mask=(labels==cid).astype(np.uint8);labels[(mask>0)&(cv2.medianBlur(mask,3)==0)]=-1
        local,w=merge_details(base[sl],texture,labels,(labels>=0).astype(np.float32),config['mode'],config['sigma'],config['gain'])
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=w;full_labels=np.full(base.shape[:2],-1,np.int16);full_labels[sl]=np.where(w>0,labels,-1)
        reasons=np.load(old/f'cam{target:03d}_reasons.npy').copy();reasons[(old_labels>=0)&(full_labels<0)]=5
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        for suffix,array in [('',result),('_sources',full_labels),('_confidence',weight),('_reasons',reasons)]:np.save(out/f'cam{target:03d}{suffix}.npy',array)
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32);colors=palette[np.maximum(full_labels,0)];colors[full_labels<0]=0;image(out/f'cam{target:03d}_sources.png',colors)
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy','reasons.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'warp_root':str(variant),'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'reasons':{**receipt['reasons'],'5':'selected source rejected by cubic footprint'}})
        log(root,'single_cubic75_fixed_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_risk_patch');p.add_argument('--name',default='single_cubic75_fixed');a=p.parse_args();combine(a.root,a.parent,a.name)
