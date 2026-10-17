"""Keep conservative coarse detail transfer and strengthen only native fine bands."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from transfer_luster_hybrid import gaussian
cv2.setNumThreads(4)


def merge_bands(base,texture,labels,coarse_weight,sigma,fine_sigma,boost=1.):
    result=base.copy();weight=np.zeros(labels.shape,np.float32)
    residual=texture-base
    for label in np.unique(labels[labels>=0]):
        support=(labels==label).astype(np.float32)
        fade=np.minimum(cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)/3,1)
        def high(s):
            mass=gaussian(support,s)
            return residual-gaussian(residual*support[...,None],s)/np.maximum(mass[...,None],1e-8)
        coarse=coarse_weight;extra=np.maximum(fade-coarse,0)*boost
        choose=(support>0)&(coarse>0)
        delta=coarse[...,None]*high(sigma)+extra[...,None]*high(fine_sigma)
        result[choose]=np.clip(base[choose]+delta[choose],0,1)
        weight[choose]=(coarse+extra)[choose]
    return result,weight


def combine(root,parent,name,fine_sigma=1.,boost=1.):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    pc=read(old/'config.json');config={**pc,'mode':'fine_band_boost','parent':parent,'fine_sigma':fine_sigma,'fine_boost':boost}
    write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        r=read(old/f'cam{target:03d}_receipt.json');base=np.asarray(geometry(root,target)['rgb'])
        with np.load(old/f'cam{target:03d}_blend_provenance.npz') as b:
            x0,y0,x1,y1=map(int,b['box']);first=b['first'];second=b['second'];fraction=b['first_weight'];labels=b['pair_labels']
        texture=np.zeros((*labels.shape,3),np.float32)
        for source in r['sources']:
            photo=np.load(root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'rgb.npy',mmap_mode='r')
            w=np.where(first==source,fraction,0)+np.where(second==source,1-fraction,0)
            texture+=photo*w[...,None]
        sl=np.s_[y0:y1,x0:x1];coarse=np.load(old/f'cam{target:03d}_confidence.npy',mmap_mode='r')[sl]
        local,weight=merge_bands(base[sl],texture,labels,coarse,pc['sigma'],fine_sigma,boost)
        result=base.copy();result[sl]=local;fullweight=np.zeros(base.shape[:2],np.float32);fullweight[sl]=weight
        assert np.isfinite(result).all() and np.array_equal(result[fullweight==0],base[fullweight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',fullweight)
        for suffix in ['sources.npy','reasons.npy','blend_provenance.npz','sources.png']:
            p=old/f'cam{target:03d}_{suffix}'
            if p.exists():(out/f'cam{target:03d}_{suffix}').symlink_to(p)
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(fullweight[...,None],3,-1))
        parents={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','confidence.npy','blend_provenance.npz','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**r,'config':config,'code_sha256':sha(Path(__file__)),
             'additional_input_hashes':parents,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'hybrid_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='bracket8_r05_s8')
    p.add_argument('--name',required=True);p.add_argument('--fine-sigma',type=float,default=1.);p.add_argument('--boost',type=float,default=1.)
    a=p.parse_args();combine(a.root,a.parent,a.name,a.fine_sigma,a.boost)
