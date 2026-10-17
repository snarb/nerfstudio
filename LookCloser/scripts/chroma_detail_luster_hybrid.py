"""Isolate luminance and chromatic detail from the same selected native photo."""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
from band_witness_luster_hybrid import assemble


def adjust_chroma(delta,gain):
    assert 0<=gain<=1
    if gain==1:return delta.copy()
    luminance=np.einsum('...c,c->...',delta,np.array([.2126,.7152,.0722],np.float32))[...,None]
    return luminance+gain*(delta-luminance)


def combine(root,parent,name,gain):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_chroma_detail','parent':parent,'chroma_gain':gain}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt['warp_root'])
        base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        deltas={};supports={}
        for cid in receipt['sources']:
            if not np.any(labels==cid):continue
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};support=geometric_support(a,config);sf=support.astype(np.float32)
            mass=gaussian(sf,config['sigma'])
            def low(x):return gaussian(x*sf[...,None],config['sigma'])/np.maximum(mass[...,None],1e-8)
            texture=np.asarray(a['rgb']);delta=(texture-low(texture))-(base[sl]-low(base[sl]))
            deltas[cid]=adjust_chroma(delta,gain);supports[cid]=support
        local,local_weight=assemble(base[sl],deltas,supports,labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=local_weight
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','sources.png','reasons.npy']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy']}
        module=Path(__file__).with_name('band_witness_luster_hybrid.py');inputs[str(module)]=sha(module)
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'chroma_detail_complete',camera=target,candidate=name)
