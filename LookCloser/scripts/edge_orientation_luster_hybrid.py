"""Limit transverse middle-band structure at strong coherent frozen-render edges."""
from pathlib import Path
import cv2
import numpy as np
from transfer_luster_hybrid import gaussian
from luster_hybrid_common import geometry,image,log,read,sha,write
from supported_luster_hybrid import geometric_support
from band_witness_luster_hybrid import assemble


def orientation_gate(base,photo,support,amount):
    """Only attenuate transverse mid-scale detail at a confident existing edge.

    This computes gradient tensors, never a displacement or shifted lookup.
    Flat, unsupported or noncoherent baseline regions keep the selected photo.
    """
    assert 0<=amount<=1
    sf=support.astype(np.float32);mass2=gaussian(sf,2.);mass4=gaussian(sf,4.)
    luma=np.array([.2126,.7152,.0722],np.float32)
    def tensor(rgb):
        low=gaussian((rgb@luma)*sf,2.)/np.maximum(mass2,1e-8)
        gx=cv2.Sobel(low,cv2.CV_32F,1,0,ksize=3,scale=1/8)
        gy=cv2.Sobel(low,cv2.CV_32F,0,1,ksize=3,scale=1/8)
        def mean(v):return gaussian(v*sf,4.)/np.maximum(mass4,1e-8)
        return mean(gx*gx),mean(gy*gy),mean(gx*gy)
    bx,by,bxy=tensor(base);tx,ty,txy=tensor(photo)
    bt=np.maximum(bx+by,0);tt=np.maximum(tx+ty,0)
    gap=np.sqrt(np.maximum((bx-by)**2+4*bxy*bxy,0))
    coherence=np.clip(gap/np.maximum(bt,1e-12),0,1)
    # Energy perpendicular to the baseline tensor's principal edge normal.
    agreement=((bx-by)*(tx-ty)+4*bxy*txy)/np.maximum(gap*tt,1e-12)
    transverse=np.clip(.5*(1-agreement),0,1)
    known=support&(bt>=.002**2)&(tt>=.002**2)&(mass4>=.9)
    known&=cv2.erode(sf,np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    risk=np.clip((coherence-.8)/.15,0,1)*np.clip((transverse-.15)/.35,0,1)
    return np.where(known,1-amount*risk,1).astype(np.float32)


def correction(base,photo,support,gate):
    sf=support.astype(np.float32)
    mass2=gaussian(sf,2.);mass6=gaussian(sf,6.)
    def low(rgb,sigma,mass):return gaussian(rgb*sf[...,None],sigma)/np.maximum(mass[...,None],1e-8)
    p2=low(photo,2.,mass2);p6=low(photo,6.,mass6)
    b2=low(base,2.,mass2);b6=low(base,6.,mass6)
    primary=(photo-p6)-(base-b6)
    return primary+(gate-1)[...,None]*((p2-p6)-(b2-b6))


def combine(root,parent,name,amount):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'), 'mode':'single_source_edge_orientation','parent':parent,'orientation_attenuation':amount}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt['warp_root'])
        base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        deltas={};supports={};gates={};inputs={str(Path(__file__).with_name(m)):sha(Path(__file__).with_name(m)) for m in ['band_witness_luster_hybrid.py','supported_luster_hybrid.py','transfer_luster_hybrid.py']}
        for cid in receipt['sources']:
            if not np.any(labels==cid):continue
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};support=geometric_support(a,config)
            gs=[support.astype(np.float32),orientation_gate(base[sl],np.asarray(a['rgb']),support,amount)]
            inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
            deltas[cid]=correction(base[sl],np.asarray(a['rgb']),support,gs[1]);supports[cid]=support;gates[cid]=gs
        local,local_weight=assemble(base[sl],deltas,supports,labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=local_weight
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        selected_gates=np.zeros((*labels[sl].shape,2),np.float32)
        for cid,gs in gates.items():
            choose=labels[sl]==cid
            for b in range(2):selected_gates[...,b][choose]=gs[b][choose]
        np.save(out/f'cam{target:03d}_band_gates.npy',selected_gates)
        for suffix in ['sources.npy','sources.png','reasons.npy']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        for suffix in ['receipt.json','sources.npy']:inputs[str(old/f'cam{target:03d}_{suffix}')]=sha(old/f'cam{target:03d}_{suffix}')
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,
              'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'band_gates_sha256':sha(out/f'cam{target:03d}_band_gates.npy')})
        log(root,'edge_orientation_complete',camera=target,candidate=name)
