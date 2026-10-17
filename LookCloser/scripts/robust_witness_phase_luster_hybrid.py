"""Robust TRAIN-only phase consensus at unchanged LookCloser projections.

Every usable witness contributes to a median, rather than selecting the two
most agreeable witnesses. Other photos provide scalar evidence only.
"""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import assert_source,native_rows,read,write,sha,log
from transfer_luster_hybrid import gaussian
from supported_luster_hybrid import geometric_support
from witness_phase_luster_hybrid import pair_phase
cv2.setNumThreads(4)


def robust_cost(correlations,known):
    """Two observations required; absence/flatness is neutral, never agreement."""
    count=known.sum(axis=0)
    ordered=np.sort(np.where(known,correlations,np.inf),axis=0)
    lo=np.maximum((count-1)//2,0);hi=np.maximum(count//2,0)
    a=np.take_along_axis(ordered,lo[None],axis=0)[0]
    b=np.take_along_axis(ordered,hi[None],axis=0)[0]
    median=np.where(count>=2,(a+b)*.5,0)
    return (.5*(1-np.clip(median,-1,1))).astype(np.float32),count.astype(np.uint8)


def evidence(root,target,config):
    ids=read(root/'preflight/sources.json')['sources'][str(target)][:config['count']]
    out=root/'robust_witness_phase'/f'cam{target:03d}';out.mkdir(parents=True,exist_ok=True)
    fingerprint={'count':config['count'],**{k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}}
    if (out/'receipt.json').exists():
        old=read(out/'receipt.json');assert old['code_sha256']==sha(Path(__file__)) and old['settings']==fingerprint
        assert all(sha(Path(k))==v for k,v in old['inputs'].items())
        assert all(sha(out/k)==v for k,v in old['hashes'].items());return out
    rows=native_rows();bands=[];validity=[];inputs={}
    for module in ['witness_phase_luster_hybrid.py','supported_luster_hybrid.py','transfer_luster_hybrid.py','luster_hybrid_common.py']:
        path=Path(__file__).with_name(module);inputs[str(path)]=sha(path)
    for cid in ids:
        assert_source(rows,target,cid)
        d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        valid=geometric_support(a,config);sf=valid.astype(np.float32)
        gray=np.einsum('...c,c->...',a['rgb'],np.array([.2126,.7152,.0722],np.float32))
        def low(sigma):return gaussian(gray*sf,sigma)/np.maximum(gaussian(sf,sigma),1e-8)
        bands.append(low(1.)-low(4.));validity.append(valid)
    hashes={}
    for i,cid in enumerate(ids):
        correlations=[];known=[]
        for j in range(len(ids)):
            if i==j:continue
            corr,valid=pair_phase(bands[i],bands[j],validity[i]&validity[j])
            correlations.append(corr);known.append(valid)
        cost,count=robust_cost(np.stack(correlations),np.stack(known))
        for key,value in [('cost',cost),('count',count)]:
            path=out/f'src{cid:03d}_{key}.npy';np.save(path,value);hashes[path.name]=sha(path)
        log(root,'robust_witness_source_complete',camera=target,source=cid)
    write(out/'receipt.json',dict(target=target,sources=ids,settings=fingerprint,code_sha256=sha(Path(__file__)),inputs=inputs,hashes=hashes,
          aggregation='median of ALL usable distinct TRAIN witnesses',band='G1-G4 luminance',minimum_witnesses=2,
          uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    return out
