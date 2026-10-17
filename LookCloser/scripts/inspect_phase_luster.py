"""Evaluation-only structural phase diagnostics on discovered regression tiles."""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,read,write
from risk_single_luster_hybrid import masked_low
from transfer_luster_hybrid import gaussian
from supported_luster_hybrid import geometric_support
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();root=a.root
config=read(root/f'candidates/{a.name}/config.json');tiles=read(root/f'evaluation/{a.name}_spatial/regression_tiles.json')['rows'];result=[]
for target in [7,12,95,150]:
    base=geometry(root,target)['rgb'];labels=np.load(root/f'candidates/{a.name}/cam{target:03d}_sources.npy',mmap_mode='r')
    for tile in [r for r in tiles if r['camera']==target]:
        x0,y0,x1,y1=tile['box'];px0=max(0,x0-48);py0=max(0,y0-48);px1=min(base.shape[1],x1+48);py1=min(base.shape[0],y1+48)
        local_labels=labels[py0:py1,px0:px1];b=np.asarray(base[py0:py1,px0:px1]);inside=np.zeros(local_labels.shape,bool);inside[y0-py0:y1-py0,x0-px0:x1-px0]=True
        values=[]
        for cid in np.unique(local_labels[inside]):
            if cid<0:continue
            d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');wx0,wy0,wx1,wy1=r['box']
            # These diagnostic tiles are inside predicted support; skip any crop crossing its storage boundary.
            if px0<wx0 or py0<wy0 or px1>wx1 or py1>wy1:continue
            sl=np.s_[py0-wy0:py1-wy0,px0-wx0:px1-wx0];record={k:np.load(d/f'{k}.npy',mmap_mode='r')[sl] for k in r['arrays']};valid=geometric_support(record,config)
            photo,_=masked_low(record['rgb'],valid,2.);model,_=masked_low(b,valid,2.);coarse_p,_=masked_low(record['rgb'],valid,8.);coarse_m,_=masked_low(b,valid,8.)
            pband=photo-coarse_p;mband=model-coarse_m;mass=gaussian(valid.astype(np.float32),3.)
            def moment(x):return gaussian(x*valid,3.)/np.maximum(mass,1e-8)
            dot=moment(np.mean(pband*mband,axis=-1));ep=moment(np.mean(pband**2,axis=-1));em=moment(np.mean(mband**2,axis=-1));corr=dot/np.sqrt(np.maximum(ep*em,1e-12));chosen=inside&(local_labels==cid)&valid
            if chosen.any():values.append(np.stack([corr[chosen],np.sqrt(ep[chosen]),np.sqrt(em[chosen])],-1))
        if not values:continue
        values=np.concatenate(values);out={**tile,'correlation_median':float(np.median(values[:,0])),'photo_energy_median':float(np.median(values[:,1])),'model_energy_median':float(np.median(values[:,2])),'selected_pixels':len(values)}
        out['strong_inconsistent_fractions']={str(e):float(((values[:,0]<.3)&(values[:,1]>e)&(values[:,2]>e)).mean()) for e in [.005,.01,.02]};result.append(out)
write(root/f'evaluation/{a.name}_phase_diagnostic.json',dict(usage='evaluation_only_not_construction',rows=result));print(result)
