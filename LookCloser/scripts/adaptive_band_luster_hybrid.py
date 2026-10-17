"""Extend one-photo detail transfer to broader bands only on aligned patches.

All evidence comes from the selected projected TRAIN photo and frozen render.
No alternate photo contributes RGB and no texture-coordinate shift is used.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,read,write,sha,log
from transfer_luster_hybrid import gaussian,merge_details
cv2.setNumThreads(4)

def merge(base,texture,labels,config):
    result,weight=merge_details(base,texture,labels,(labels>=0).astype(np.float32),'detail',config['sigma'],config['gain'])
    gates=np.zeros(labels.shape,np.float32)
    for cid in np.unique(labels[labels>=0]):
        selected=labels==cid;support=selected.astype(np.float32)
        def low(x,sigma):
            mass=gaussian(support,sigma)
            return gaussian(x*support[...,None],sigma)/np.maximum(mass[...,None],1e-8)
        tb=low(texture,config['sigma'])-low(texture,config['extra_sigma'])
        bb=low(base,config['sigma'])-low(base,config['extra_sigma'])
        mt=low(tb,8.);mb=low(bb,8.)
        et2=np.mean(np.maximum(low(tb*tb,8.)-mt*mt,0),axis=-1)
        eb2=np.mean(np.maximum(low(bb*bb,8.)-mb*mb,0),axis=-1)
        cross=np.mean(low(tb*bb,8.)-mt*mb,axis=-1)
        corr=np.clip(cross/np.maximum(np.sqrt(et2*eb2),1e-8),-1,1)
        ratio=np.sqrt(et2)/np.maximum(np.sqrt(eb2),1e-8)
        gate=np.clip((corr-config['phase_min'])/(1-config['phase_min']),0,1)
        gate*=np.clip((ratio-.7)/.3,0,1)*np.clip((3-ratio),0,1)*(eb2>=config['energy_floor']**2)
        gate=low(np.repeat(gate[...,None],3,-1),3.)[...,0]*support
        gates[selected]=gate[selected]
        delta=tb-bb
        result[selected]=np.clip(result[selected]+weight[selected,None]*gate[selected,None]*delta[selected],0,1)
    return result,weight,gates

def combine(root,parent,name,phase,extra_sigma):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_adaptive_band','parent':parent,'phase_min':phase,'extra_sigma':extra_sigma,'energy_floor':.003}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root));base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        first=warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
        local_labels=labels[sl];texture=base[sl].copy()
        for cid in receipt['sources']:
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';photo=np.load(d/'rgb.npy',mmap_mode='r');chosen=local_labels==cid;texture[chosen]=photo[chosen]
        local,w,gate=merge(base[sl],texture,local_labels,config);result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=w
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}_extra_band_gate.npy',gate);np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_adaptive_band_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_cubic_fixed');p.add_argument('--name',required=True);p.add_argument('--phase',type=float,default=.8);p.add_argument('--extra-sigma',type=float,default=16.);a=p.parse_args()
    combine(a.root,a.parent,a.name,a.phase,a.extra_sigma)
