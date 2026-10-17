"""TRAIN-photo patch agreement, used only to score hard source labels.

Photographs are compared after exact frozen-geometry projection. Witnesses
supply scores, never colors. Smooth patches receive little disagreement cost.
"""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import read,write,sha
from transfer_luster_hybrid import gaussian
from single_source_luster_hybrid import masked_low
cv2.setNumThreads(4)

def agreement(a,b,valid,sigma=3.,floor=.01):
    mass=gaussian(valid.astype(np.float32),sigma)
    def avg(x):return gaussian(np.mean(x,axis=-1)*valid,sigma)/np.maximum(mass,1e-8)
    value=avg((a-b)**2)/(avg(a*a)+avg(b*b)+floor**2)
    return np.where(mass>=.8,value,np.inf).astype(np.float32)

def quality(root,target,source_ids,config):
    keys=['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']
    spec={k:config[k] for k in keys};spec['sources']=source_ids
    import hashlib,json
    token=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'patch_consistency'/f'cam{target:03d}'/token
    if (out/'receipt.json').exists():
        receipt=read(out/'receipt.json');assert receipt['code_sha256']==sha(Path(__file__)) and receipt['spec']==spec
        return out
    out.mkdir(parents=True,exist_ok=True)
    bands=[];valids=[];inputs={}
    for cid in source_ids:
        d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';receipt=read(d/'receipt.json');inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in receipt['arrays']}
        valid=(a['valid']>0)&(a['alpha']>=config['alpha'])&(a['depth_error']<=config['depth_error'])&(a['roundtrip']<=config['roundtrip'])&(a['uncertainty']<=config['uncertainty'])&(a['score']<=config['max_score'])&(a['resolution_ratio']>=config['min_resolution'])
        lo1,m1=masked_low(a['rgb'],valid,1.);lo4,m4=masked_low(a['rgb'],valid,4.)
        valids.append(valid&(m1>=.8)&(m4>=.8));bands.append((lo1-lo4).astype(np.float32))
    shape=valids[0].shape
    best=np.full((len(source_ids),2,*shape),np.inf,np.float32)
    for i in range(len(source_ids)):
        for j in range(i+1,len(source_ids)):
            score=agreement(bands[i],bands[j],valids[i]&valids[j])
            for k in (i,j):
                old=best[k,0].copy();best[k,0]=np.minimum(old,score);best[k,1]=np.minimum(best[k,1],np.maximum(old,score))
    hashes={}
    for i,cid in enumerate(source_ids):
        witnessed=np.isfinite(best[i,1]);cost=np.zeros(shape,np.float32)
        cost[witnessed]=best[i,:,witnessed].mean(axis=-1)
        for key,value in [('cost',cost),('witnessed',witnessed)]:
            f=out/f'src{cid:03d}_{key}.npy';np.save(f,value);hashes[f.name]=sha(f)
    write(out/'receipt.json',dict(spec=spec,code_sha256=sha(Path(__file__)),inputs=inputs,hashes=hashes,uses_target_rgb=False,uses_evaluation_masks=False,colors_from_witnesses=False))
    return out
