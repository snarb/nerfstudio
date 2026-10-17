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
from seam_single_luster_hybrid import merge
cv2.setNumThreads(4)


def combine(root,parent,name,settings):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_seam','parent':parent,**settings}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');records=[];warp_root=Path(receipt.get('warp_root',root))
        for cid in receipt['sources']:
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
            records.append((cid,{k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}))
        x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1];base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        local,local_weight=merge(base[sl],records,labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=local_weight
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        module=Path(__file__).with_name('seam_single_luster_hybrid.py');inputs[str(module)]=sha(module)
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'fixed_sampling_seam_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--filter-support',choices=['labels','geometry'],default='labels');p.add_argument('--seam-fade',type=float,default=3.);p.add_argument('--adaptive-seam',action='store_true');p.add_argument('--seam-tolerance',type=float,default=.02);a=p.parse_args()
    assert a.seam_fade>=1
    combine(a.root,a.parent,a.name,dict(filter_support=a.filter_support,seam_fade=a.seam_fade,adaptive_seam=a.adaptive_seam,seam_tolerance=a.seam_tolerance))
