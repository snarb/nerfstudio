"""Cubic single-photo merge excluding plate-vetoed pixels from filter support."""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from supported_luster_hybrid import geometric_support
from seam_single_luster_hybrid import merge
from exact_plate_veto_luster_hybrid import projected_veto
cv2.setNumThreads(4)


def combine(root,parent,name):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_plate_seam','parent':parent,
            'filter_support':'geometry','seam_fade':3.,'adaptive_seam':True,'seam_tolerance':.02,
            'output_sampling':'clamped_cubic75'}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    warp_root=root/'variants/cubic75'
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');records=[];inputs={}
        full_labels=np.load(old/f'cam{target:03d}_sources.npy').copy()
        for cid in receipt['sources']:
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
            vd=projected_veto(root,target,cid);veto=np.load(vd/f"{config['plate_veto_level']}.npy",mmap_mode='r')
            a['valid']=(a['valid']>0)&~veto
            x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            labels=full_labels[sl];labels[(labels==cid)&~geometric_support(a,config)]=-1
            records.append((cid,a));inputs[str(vd/'receipt.json')]=sha(vd/'receipt.json')
            inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        base=np.asarray(geometry(root,target)['rgb'])
        local,local_weight=merge(base[sl],records,full_labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=local_weight
        full_labels[weight==0]=-1
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        reasons=np.load(old/f'cam{target:03d}_reasons.npy').copy();reasons[(weight==0)&(reasons==0)]=6
        for suffix,array in [('',result),('_confidence',weight),('_sources',full_labels),('_reasons',reasons)]:np.save(out/f'cam{target:03d}{suffix}.npy',array)
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32)
        colors=palette[np.maximum(full_labels,0)];colors[full_labels<0]=0;image(out/f'cam{target:03d}_sources.png',colors)
        for suffix in ['receipt.json','sources.npy']:
            path=old/f'cam{target:03d}_{suffix}';inputs[str(path)]=sha(path)
        for module in ['seam_single_luster_hybrid.py','supported_luster_hybrid.py','exact_plate_veto_luster_hybrid.py']:
            path=Path(__file__).with_name(module);inputs[str(path)]=sha(path)
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'warp_root':str(warp_root),
            'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,
            'changed_fraction':float(np.any(result!=base,axis=-1).mean()),
            'reasons':{**receipt['reasons'],'6':'plate veto or unsupported cubic/seam sample'},
            'background_rgb_in_output':False})
        log(root,'exact_plate_merge_complete',camera=target,candidate=name)
