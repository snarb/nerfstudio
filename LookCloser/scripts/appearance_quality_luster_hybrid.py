"""Model-only view dependence and TRAIN-photo structure for automatic admission."""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,read,write,sha,log
from transfer_luster_hybrid import gaussian
from single_source_luster_hybrid import masked_low,DEFAULT
from residual_luster_hybrid import model_warp

def quality(root,target,source):
    d=root/'appearance_quality'/f'cam{target:03d}'/f'src{source:03d}';parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(parent/'receipt.json')
    if (d/'receipt.json').exists():
        old=read(d/'receipt.json');assert old['code_sha256']==sha(Path(__file__)) and old['parent_receipt_sha256']==sha(parent/'receipt.json')
        return d
    d.mkdir(parents=True,exist_ok=True);x0,y0,x1,y1=r['box'];base=np.asarray(geometry(root,target)['rgb'][y0:y1,x0:x1]);a={key:np.load(parent/f'{key}.npy',mmap_mode='r') for key in r['arrays']}
    source_model=model_warp(root,target,source);source_model_receipt=root/'source_model_warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    valid=(a['valid']>0)&(a['alpha']>=DEFAULT['alpha'])&(a['depth_error']<=DEFAULT['depth_error'])&(a['roundtrip']<=DEFAULT['roundtrip'])&(a['uncertainty']<=DEFAULT['uncertainty'])&(a['resolution_ratio']>=DEFAULT['min_resolution'])
    photo2,mass=masked_low(a['rgb'],valid,2.);photo8,_=masked_low(a['rgb'],valid,8.);photo1,_=masked_low(a['rgb'],valid,1.)
    source2,_=masked_low(source_model,valid,2.);source8,_=masked_low(source_model,valid,8.)
    base2,_=masked_low(base,valid,2.);base8,_=masked_low(base,valid,8.)
    support=valid.astype(np.float32);window_mass=gaussian(support,3.)
    def rms(value):return np.sqrt(np.maximum(gaussian(np.mean(value*value,axis=-1)*support,3.)/np.maximum(window_mass,1e-8),0))
    source_band=source2-source8;base_band=base2-base8
    view_error=rms(source_band-base_band)/(rms(source_band)+rms(base_band)+.002)
    structure=rms(photo2-photo8);fine=rms(a['rgb']-photo1)
    values={'view_error':np.where(valid&(mass>.8)&(window_mass>.8),view_error,1e6),'structure':structure,'structure_ratio':structure/(fine+.002)}
    for key,value in values.items():np.save(d/f'{key}.npy',value.astype(np.float32))
    write(d/'receipt.json',dict(target=target,source=source,box=r['box'],code_sha256=sha(Path(__file__)),parent_receipt_sha256=sha(parent/'receipt.json'),source_model_receipt_sha256=sha(source_model_receipt),source_model_receipt=str(source_model_receipt),hashes={key:sha(d/f'{key}.npy') for key in values},uses_target_rgb=False,uses_target_model_rgb=True,uses_training_photo=True))
    log(root,'appearance_quality_cached',camera=target,source=source)
    return d
