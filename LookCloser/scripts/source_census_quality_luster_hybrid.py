"""Contrast-invariant local ordering costs, without a learned feature model.

Compare calibrated neighboring samples only. These scalar costs never move a
texture lookup, alter frozen geometry or read target validation photographs.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,read,write,sha,log
from risk_single_luster_hybrid import masked_low
from residual_luster_hybrid import model_warp
from supported_luster_hybrid import geometric_support


def shifted(value,dy,dx):
    """Neighbor value at (y+dy,x+dx); exterior is zero, never wraps."""
    result=np.zeros_like(value);h,w=value.shape[:2]
    if abs(dy)>=h or abs(dx)>=w:return result
    ya,yb=max(0,-dy),min(h,h-dy);xa,xb=max(0,-dx),min(w,w-dx)
    result[ya:yb,xa:xb]=value[ya+dy:yb+dy,xa+dx:xb+dx]
    return result


def census_cost(first,second,support,radii=(2,4,8),threshold=.004):
    """Ordering disagreements in model-visible structure, robust to gain/DC.

    The second image is frozen-model structure. Weak model differences carry
    no evidence; weak photo differences are neutral, not a false agreement.
    Unknown patches have cost .5. All comparisons require both samples visible.
    """
    total=np.zeros(support.shape,np.float32);mass=total.copy()
    for radius in radii:
        for dy,dx in [(0,radius),(radius,0),(radius,radius),(radius,-radius)]:
            a=shifted(first,dy,dx)-first;b=shifted(second,dy,dx)-second
            known=support&shifted(support,dy,dx)&(np.abs(b)>=threshold)
            value=np.where(np.abs(a)>=threshold,(a*b<0).astype(np.float32),.5)
            total+=np.where(known,value,0);mass+=known
    return np.where(mass>0,total/np.maximum(mass,1),.5).astype(np.float32)


def quality(root,target,source,config):
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(parent/'receipt.json')
    d=root/'source_census_quality'/f'cam{target:03d}'/f'src{source:03d}'
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    if (d/'receipt.json').exists():
        old=read(d/'receipt.json');assert old['code_sha256']==sha(Path(__file__)) and old['gates']==gates
        assert old['inputs'][str(parent/'receipt.json')]==sha(parent/'receipt.json')
        assert all(sha(d/k)==v for k,v in old['hashes'].items());return d
    d.mkdir(parents=True,exist_ok=True);a={k:np.load(parent/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
    valid=geometric_support(a,config);x0,y0,x1,y1=r['box'];base=np.asarray(geometry(root,target)['rgb'][y0:y1,x0:x1])
    model=model_warp(root,target,source);values=[];luma=np.array([.2126,.7152,.0722],np.float32)
    for rgb in [a['rgb'],model,base]:
        smoothed,_=masked_low(rgb,valid,1.);values.append(smoothed@luma)
    costs={'source_census_cost':census_cost(values[0],values[1],valid),
           'view_census_cost':census_cost(values[1],values[2],valid)}
    for key,value in costs.items():np.save(d/f'{key}.npy',value)
    mr=root/'source_model_warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    inputs={str(p):sha(p) for p in [parent/'receipt.json',mr,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    for module in ['risk_single_luster_hybrid.py','residual_luster_hybrid.py','supported_luster_hybrid.py']:
        p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
    write(d/'receipt.json',dict(target=target,source=source,box=r['box'],gates=gates,
        code_sha256=sha(Path(__file__)),inputs=inputs,hashes={f'{k}.npy':sha(d/f'{k}.npy') for k in costs},
        smoothing_sigma=1.,radii=[2,4,8],threshold=.004,uses_target_rgb=False,uses_masks=False,lookup_shift=False))
    log(root,'source_census_quality_cached',camera=target,source=source);return d
