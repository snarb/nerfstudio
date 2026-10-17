"""One-photo broad appearance transfer corroborated by TRAIN-only witnesses.

Keeps the parent's fine-detail result and source labels. Witness RGB contributes
only an agreement decision, never output color. No target GT or ROI is read.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,read,write,sha,log,assert_source,native_rows
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
cv2.setNumThreads(4)

def confidence(selected,selected_ids,records,config):
    count=np.zeros(selected_ids.shape,np.float32);agree=np.zeros_like(count)
    for cid,low,valid in records:
        witness=valid&(selected_ids>=0)&(selected_ids!=cid)
        count+=witness
        agree+=witness&(np.max(np.abs(low-selected),axis=-1)<=config['low_tolerance'])
    supported=(count>=config['low_witnesses'])&(agree>=config['low_witnesses'])&(agree>=config['low_fraction']*count)
    # A local consensus patch must remain valid; do not grow support into holes.
    mass=gaussian(supported.astype(np.float32),config['low_patch_sigma'])
    return supported*np.clip((mass-.5)/.4,0,1)

def combine(root,parent,name,settings):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_consensus_low','parent':parent,**settings}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    rows=native_rows()
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r');weight=np.load(old/f'cam{target:03d}_confidence.npy',mmap_mode='r')
        first=root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1];b=base[sl];lab=labels[sl]
        selected_low=np.zeros_like(b);low_delta=np.zeros_like(b);records=[]
        for cid in receipt['sources']:
            assert_source(rows,target,cid)
            d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
            valid=geometric_support(a,config);mass=gaussian(valid.astype(np.float32),config['sigma'])
            low=gaussian(a['rgb']*valid[...,None],config['sigma'])/np.maximum(mass[...,None],1e-8)
            witness_valid=valid&(mass>=config['support_mass']);records.append((cid,low,witness_valid))
            chosen=lab==cid;support=chosen.astype(np.float32);smass=gaussian(support,config['sigma'])
            # Exactly the low-frequency difference removed by parent merge_details.
            offset=gaussian((a['rgb']-b)*support[...,None],config['sigma'])/np.maximum(smass[...,None],1e-8)
            selected_low[chosen]=low[chosen];low_delta[chosen]=offset[chosen]
        low_weight=confidence(selected_low,lab,records,config).astype(np.float32)
        result=np.load(old/f'cam{target:03d}.npy').copy();result[sl]=np.clip(result[sl]+weight[sl][...,None]*low_weight[...,None]*low_delta,0,1)
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_low_weight.npy',low_weight)
        for suffix in ['sources.npy','confidence.npy','reasons.npy','sources.png','transfer.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_low_transfer.png',np.repeat(low_weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}{suffix}'):sha(old/f'cam{target:03d}{suffix}') for suffix in ['_receipt.json','_sources.npy','_confidence.npy','.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'construction_box':[x0,y0,x1,y1],'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'witness_rgb_in_output':False,'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'low_changed_fraction':float((low_weight>0).mean())})
        log(root,'single_consensus_low_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',required=True);p.add_argument('--name',required=True);p.add_argument('--tolerance',type=float,default=.015);a=p.parse_args()
    combine(a.root,a.parent,a.name,dict(low_tolerance=a.tolerance,low_witnesses=2,low_fraction=.67,low_patch_sigma=3.))
