"""One native TRAIN photo supplies detail missing from its frozen source render."""
from pathlib import Path
import argparse
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from residual_luster_hybrid import model_warp,merge_residual

def combine(root,parent,name,gain=1.):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_model_residual','parent':parent,'residual_gain':gain}
    write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        r=read(old/f'cam{target:03d}_receipt.json');first=root/'warps'/f'cam{target:03d}'/f"src{r['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
        base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')[sl];residual=np.zeros((*labels.shape,3),np.float32);inputs={}
        for source in r['sources']:
            chosen=labels==source
            if not chosen.any():continue
            photo=np.load(root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'rgb.npy',mmap_mode='r');pred=model_warp(root,target,source)
            residual[chosen]=photo[chosen]-pred[chosen]
            path=root/'source_model_warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json';inputs[str(path)]=sha(path)
        weight=np.load(old/f'cam{target:03d}_confidence.npy').copy()*gain
        result=base.copy();result[sl]=merge_residual(base[sl],residual,labels,weight[sl],config['sigma'])
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        for suffix in ['sources.npy','reasons.npy','sources.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(np.minimum(weight,1)[...,None],3,-1))
        for suffix in ['receipt.json','sources.npy','confidence.npy']:
            path=old/f'cam{target:03d}_{suffix}';inputs[str(path)]=sha(path)
        write(out/f'cam{target:03d}_receipt.json',{**r,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_residual_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_patch_s8');p.add_argument('--name',default='single_residual_s8');p.add_argument('--gain',type=float,default=1.);a=p.parse_args();combine(a.root,a.parent,a.name,a.gain)
