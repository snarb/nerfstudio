"""Transfer TRAIN-photo residuals against the same frozen TRAIN-view render.

The target's existing high frequencies are retained. Source residuals are
projected using exactly the already verified native photo footprint, then their
supported high-frequency component is added to the target frozen render.
"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from luster_hybrid_common import geometry,image,log,native_rows,project,read,sha,unproject,write
from transfer_luster_hybrid import footprint_metrics,gaussian
cv2.setNumThreads(4)


def model_warp(root,target,source):
    d=root/'source_model_warps'/f'cam{target:03d}'/f'src{source:03d}'
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(parent/'receipt.json')
    if (d/'receipt.json').exists():
        old=read(d/'receipt.json');assert old['code_sha256']==sha(Path(__file__))
        assert old['parent_warp_receipt_sha256']==sha(parent/'receipt.json')
        return np.load(d/'rgb.npy',mmap_mode='r')
    d.mkdir(parents=True,exist_ok=True);rows=native_rows();g=geometry(root,target);sg=geometry(root,source)
    x0,y0,x1,y1=r['box'];out=np.lib.format.open_memmap(d/'rgb.npy',mode='w+',shape=(y1-y0,x1-x0,3),dtype=np.float32)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        world=unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy);uv,z=project(world,rows[source])
        colors,*_=footprint_metrics(sg['rgb'],sg,uv,z,None,None,None,None,None)
        out[y-y0:stop-y0]=colors
    out.flush()
    write(d/'receipt.json',dict(target=target,source=source,box=r['box'],code_sha256=sha(Path(__file__)),
          rgb_sha256=sha(d/'rgb.npy'),parent_warp_receipt_sha256=sha(parent/'receipt.json'),
          target_geometry_sha256=r['target_geometry_sha256'],source_geometry_sha256=r['source_geometry_sha256']))
    log(root,'source_model_warp',camera=target,source=source)
    return out


def merge_residual(base,residual,labels,weight,sigma):
    result=base.copy()
    for label in np.unique(labels[labels>=0]):
        support=(labels==label).astype(np.float32)
        mass=gaussian(support,sigma)
        low=gaussian(residual*support[...,None],sigma)/np.maximum(mass[...,None],1e-8)
        choose=(support>0)&(weight>0)
        result[choose]=np.clip(base[choose]+weight[choose,None]*(residual-low)[choose],0,1)
    return result


def combine(root,parent,name,full_strength=False,gain=1.):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    parent_config=read(old/'config.json');config={**parent_config,'mode':'source_model_residual','parent':parent,'residual_full_strength':full_strength,'residual_gain':gain}
    write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        deadline=read(root/'preflight/manifest.json')['deadline']
        while not (old/f'cam{target:03d}_receipt.json').exists():
            if time.time()>deadline:raise TimeoutError('Campaign deadline')
            time.sleep(5)
        r=read(old/f'cam{target:03d}_receipt.json');g=geometry(root,target);base=np.asarray(g['rgb'])
        with np.load(old/f'cam{target:03d}_blend_provenance.npz') as b:
            box=b['box'];first=b['first'];second=b['second'];fraction=b['first_weight'];labels=b['pair_labels']
        x0,y0,x1,y1=map(int,box);sl=np.s_[y0:y1,x0:x1]
        residual=np.zeros((*labels.shape,3),np.float32);model_inputs={}
        for source in r['sources']:
            photo=np.load(root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'rgb.npy',mmap_mode='r')
            predicted=model_warp(root,target,source)
            weight=np.where(first==source,fraction,0)+np.where(second==source,1-fraction,0)
            residual+=(photo-predicted)*weight[...,None]
            path=root/'source_model_warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
            model_inputs[str(path)]=sha(path)
        fullweight=np.load(old/f'cam{target:03d}_confidence.npy').copy()
        if full_strength:
            fullweight[:]=0
            for label in np.unique(labels[labels>=0]):
                support=labels==label;fade=np.minimum(cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)/3,1)
                fullweight[sl][support]=fade[support]
        fullweight*=gain
        result=base.copy();result[sl]=merge_residual(base[sl],residual,labels,fullweight[sl],config['sigma'])
        assert np.isfinite(result).all() and np.array_equal(result[fullweight==0],base[fullweight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',fullweight)
        for suffix in ['sources.npy','reasons.npy','blend_provenance.npz','sources.png']:
            p=old/f'cam{target:03d}_{suffix}'
            if p.exists():(out/f'cam{target:03d}_{suffix}').symlink_to(p)
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(fullweight[...,None],3,-1))
        parents={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','confidence.npy','blend_provenance.npz','sources.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**r,'config':config,'code_sha256':sha(Path(__file__)),
             'additional_input_hashes':{**model_inputs,**parents},'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'hybrid_complete',camera=target,candidate=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='bracket8_s8')
    p.add_argument('--name',default='residual_bracket8_s8');p.add_argument('--full-strength',action='store_true');p.add_argument('--gain',type=float,default=1.)
    a=p.parse_args();combine(a.root,a.parent,a.name,a.full_strength,a.gain)
