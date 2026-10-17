"""Selected-photo detail with multiplicative broad-illumination matching.

Lookup coordinates/source labels are fixed. Only per-photo contrast is scaled
by local low-frequency LookCloser appearance; no evaluation image is read.
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
    gamma=config['working_gamma'];b=np.maximum(base,0)**gamma;t=np.maximum(texture,0)**gamma
    for cid in np.unique(labels[labels>=0]):
        selected=labels==cid;support=selected.astype(np.float32)
        mass=gaussian(support,config['sigma'])
        def mean(x):return gaussian(x*support[...,None],config['sigma'])/np.maximum(mass[...,None],1e-8)
        mb=mean(b);mt=mean(t)
        floor=config['illumination_floor']**gamma
        ratio=np.clip((mb+floor)/(mt+floor),config['min_gain'],config['max_gain'])
        corrected=np.maximum(mb+(t-mt)*ratio,0)**(1/gamma)
        distance=cv2.distanceTransform(selected.astype(np.uint8),cv2.DIST_L2,5)
        w=np.minimum(distance/3,1)*config['gain'];weight[selected]=w[selected]
        result[selected]=np.clip(base[selected]+w[selected,None]*(corrected[selected]-base[selected]),0,1)
    return result,weight

def combine(root,parent,name,gamma,sigma):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_illumination','parent':parent,'working_gamma':gamma,'illumination_floor':.02,'min_gain':.5,'max_gain':2.,'sigma':sigma}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root));base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        first=warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
        local_labels=labels[sl];texture=base[sl].copy()
        for cid in receipt['sources']:
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';photo=np.load(d/'rgb.npy',mmap_mode='r');chosen=local_labels==cid;texture[chosen]=photo[chosen]
        local,w=merge(base[sl],texture,local_labels,config);result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=w
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_illumination_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--gamma',type=float,default=1.);p.add_argument('--sigma',type=float,default=8.);a=p.parse_args()
    assert a.gamma>0 and a.sigma>0
    combine(a.root,a.parent,a.name,a.gamma,a.sigma)
