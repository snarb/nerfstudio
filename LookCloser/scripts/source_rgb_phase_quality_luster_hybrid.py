"""RGB-vector structural alignment; one selected photo still supplies all texture."""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,read,write,sha,log
from transfer_luster_hybrid import gaussian
from risk_single_luster_hybrid import masked_low
from residual_luster_hybrid import model_warp
from supported_luster_hybrid import geometric_support


def phase_cost(first,second,support,sigma=4.,floor=.003):
    """Zero for aligned contrast, one for reversed contrast; unknown is neutral.

    Inputs are zero-DC RGB bands at identical calibrated coordinates.
    Flat/flat has no evidence of mismatch; only one textured side gets .5.
    """
    sf=support.astype(np.float32);mass=gaussian(sf,sigma)
    def mean(x):return gaussian(np.where(support,x,0),sigma)/np.maximum(mass,1e-8)
    aa=mean(np.mean(first*first,axis=-1));bb=mean(np.mean(second*second,axis=-1));cross=mean(np.mean(first*second,axis=-1))
    reliable=(aa>=floor*floor)&(bb>=floor*floor)&(mass>=.6)&support
    correlation=np.clip(cross/np.maximum(np.sqrt(aa*bb),1e-12),-1,1)
    cost=np.where(reliable,.5*(1-correlation),np.where((aa>=floor*floor)|(bb>=floor*floor),.5,0))
    return np.where(support,cost,0).astype(np.float32)


def quality(root,target,source,config):
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(parent/'receipt.json')
    d=root/'source_rgb_phase_quality'/f'cam{target:03d}'/f'src{source:03d}'
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    if (d/'receipt.json').exists():
        old=read(d/'receipt.json');assert old['code_sha256']==sha(Path(__file__)) and old['gates']==gates
        assert old['parent_receipt_sha256']==sha(parent/'receipt.json')
        return d
    d.mkdir(parents=True,exist_ok=True);a={k:np.load(parent/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
    valid=geometric_support(a,config);x0,y0,x1,y1=r['box'];base=np.asarray(geometry(root,target)['rgb'][y0:y1,x0:x1])
    model=model_warp(root,target,source);photo=a['rgb']
    bands=[]
    for rgb in [photo,model,base]:
        fine,_=masked_low(rgb,valid,2.);coarse,_=masked_low(rgb,valid,8.)
        bands.append(fine-coarse)
    values={'source_phase_cost':phase_cost(bands[0],bands[1],valid),'view_phase_cost':phase_cost(bands[1],bands[2],valid)}
    for key,value in values.items():np.save(d/f'{key}.npy',value)
    model_receipt=root/'source_model_warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    write(d/'receipt.json',dict(target=target,source=source,box=r['box'],gates=gates,code_sha256=sha(Path(__file__)),
        parent_receipt_sha256=sha(parent/'receipt.json'),source_model_receipt=str(model_receipt),source_model_receipt_sha256=sha(model_receipt),
        hashes={k:sha(d/f'{k}.npy') for k in values},uses_target_rgb=False,uses_masks=False,lookup_shift=False))
    log(root,'source_rgb_phase_quality_cached',camera=target,source=source)
    return d
