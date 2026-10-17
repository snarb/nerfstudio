"""Transfer one photo's log radiance residual against its frozen source render.

The saved LookCloser source render estimates view-dependent broad illumination;
no image registration, learned model, or additional texture photograph is used.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,read,write,sha,log
from transfer_luster_hybrid import gaussian,merge_details
from residual_luster_hybrid import model_warp
cv2.setNumThreads(4)

def merge(base,texture,source_model,labels,config):
    result=base.copy();weight=np.zeros(labels.shape,np.float32)
    for cid in np.unique(labels[labels>=0]):
        chosen=labels==cid;support=chosen.astype(np.float32)
        residual=np.clip(np.log((texture+config['radiance_floor'])/(source_model+config['radiance_floor'])),-np.log(config['max_ratio']),np.log(config['max_ratio']))
        if config['transfer_mode']=='detail':
            mass=gaussian(support,config['sigma'])
            residual-=gaussian(residual*support[...,None],config['sigma'])/np.maximum(mass[...,None],1e-8)
        distance=cv2.distanceTransform(chosen.astype(np.uint8),cv2.DIST_L2,5)
        w=np.minimum(distance/3,1)*config['gain'];weight[chosen]=w[chosen]
        result[chosen]=np.clip(base[chosen]*np.exp(w[chosen,None]*residual[chosen]),0,1)
    return result,weight


def combine(root,parent,name,mode,sigma):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_log_residual','parent':parent,'transfer_mode':mode,'radiance_floor':.02,'max_ratio':2.,'sigma':sigma}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root));base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        first=warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
        local_labels=labels[sl];texture=base[sl].copy();source_model=base[sl].copy();model_inputs={}
        for cid in receipt['sources']:
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';photo=np.load(d/'rgb.npy',mmap_mode='r');chosen=local_labels==cid;texture[chosen]=photo[chosen]
            if chosen.any():
                predicted=model_warp(root,target,cid);source_model[chosen]=predicted[chosen]
                path=root/'source_model_warps'/f'cam{target:03d}'/f'src{cid:03d}'/'receipt.json';model_inputs[str(path)]=sha(path)

        local,w=merge(base[sl],texture,source_model,local_labels,config);result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=w
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':{**inputs,**model_inputs},'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_log_residual_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_cubic_fixed');p.add_argument('--name',required=True);p.add_argument('--mode',choices=['detail','full'],default='detail');p.add_argument('--sigma',type=float,default=6.);a=p.parse_args()
    combine(a.root,a.parent,a.name,a.mode,a.sigma)
