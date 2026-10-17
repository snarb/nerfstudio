"""Transfer a selected photo's broad model residual only with TRAIN corroboration.

Witnesses vote on photo-minus-frozen-source-render residuals, never contribute
RGB. Target GT/masks, optical flow, geometry fitting and learned selectors are absent.
"""
from pathlib import Path
import argparse
import numpy as np
from luster_hybrid_common import geometry,image,read,write,sha,log,assert_source,native_rows
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
from residual_luster_hybrid import model_warp
from consensus_low_luster_hybrid import confidence


def combine(root,parent,name,tolerance):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_consensus_model_bias','parent':parent,'residual_sigma':16.,'low_tolerance':tolerance,'low_witnesses':2,'low_fraction':.67,'low_patch_sigma':3.,'max_correction':.10}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    rows=native_rows()
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root));base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r');weight=np.load(old/f'cam{target:03d}_confidence.npy',mmap_mode='r')
        first=warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1];lab=labels[sl]
        selected=np.zeros_like(base[sl]);correction=selected.copy();records=[];inputs={}
        for cid in receipt['sources']:
            assert_source(rows,target,cid);d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};valid=geometric_support(a,config)
            prediction=model_warp(root,target,cid);residual=a['rgb']-prediction
            mr=root/'source_model_warps'/f'cam{target:03d}'/f'src{cid:03d}'/'receipt.json';inputs[str(mr)]=sha(mr)
            mass=gaussian(valid.astype(np.float32),config['residual_sigma']);low=gaussian(residual*valid[...,None],config['residual_sigma'])/np.maximum(mass[...,None],1e-8)
            witness=valid&(mass>=config['support_mass']);records.append((cid,low,witness))
            chosen=lab==cid;selected[chosen]=low[chosen]
            # Only this selected source supplies the correction. The filter
            # cannot pick up RGB from a neighboring selected-source label.
            smass=gaussian(chosen.astype(np.float32),config['residual_sigma']);delta=gaussian(residual*chosen[...,None],config['residual_sigma'])/np.maximum(smass[...,None],1e-8)
            correction[chosen]=delta[chosen]
        gate=confidence(selected,lab,records,config).astype(np.float32)
        gate*=np.max(np.abs(correction),axis=-1)<=config['max_correction']
        result=np.load(old/f'cam{target:03d}.npy').copy();result[sl]=np.clip(result[sl]+weight[sl][...,None]*gate[...,None]*correction,0,1)
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_low_gate.npy',gate);image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_low_gate.png',np.repeat(gate[...,None],3,-1))
        for suffix in ['sources.npy','confidence.npy','reasons.npy','sources.png','transfer.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        inputs.update({str(old/f'cam{target:03d}{suffix}'):sha(old/f'cam{target:03d}{suffix}') for suffix in ['_receipt.json','_sources.npy','_confidence.npy','.npy']})
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'witness_rgb_in_output':False,'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'low_changed_fraction':float((gate>0).mean())})
        log(root,'single_consensus_model_bias_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_cubic_fixed');p.add_argument('--name',required=True);p.add_argument('--tolerance',type=float,default=.015);a=p.parse_args();combine(a.root,a.parent,a.name,a.tolerance)
