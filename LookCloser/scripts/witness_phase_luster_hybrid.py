"""TRAIN-only pairwise phase evidence for hard one-photo source selection.

Other photos supply confidence only; no correspondence shifts or RGB mixtures.
"""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import assert_source,native_rows,read,write,sha,log
from transfer_luster_hybrid import gaussian
from supported_luster_hybrid import geometric_support
cv2.setNumThreads(4)


def pair_phase(first,second,support,sigma=3.,floor=.002):
    mass=gaussian(support.astype(np.float32),sigma)
    def avg(value):return gaussian(np.where(support,value,0.),sigma)/np.maximum(mass,1e-8)
    aa=avg(first*first);bb=avg(second*second);ab=avg(first*second)
    corr=np.clip(ab/np.sqrt(np.maximum(aa*bb,1e-16)),-1,1)
    valid=support&(mass>=.8)&(aa>=floor**2)&(bb>=floor**2)
    return corr,valid


def update_best(best,second,corr,valid):
    value=np.where(valid,corr,-2.)
    second[:]=np.maximum(second,np.minimum(best,value))
    best[:]=np.maximum(best,value)


def evidence(root,target,config):
    source_ids=read(root/'preflight/sources.json')['sources'][str(target)][:config['count']]
    out=root/'witness_phase'/f'cam{target:03d}';out.mkdir(parents=True,exist_ok=True)
    fingerprint={'count':config['count'],**{k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['settings']==fingerprint and r['code_sha256']==sha(Path(__file__))
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    rows=native_rows();bands=[];validity=[];inputs={}
    for cid in source_ids:
        assert_source(rows,target,cid);d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};valid=geometric_support(a,config)
        gray=np.einsum('...c,c->...',a['rgb'],np.array([.2126,.7152,.0722],np.float32))
        def low(s):return gaussian(gray*valid,s)/np.maximum(gaussian(valid.astype(np.float32),s),1e-8)
        bands.append(low(1.)-low(4.));validity.append(valid)
    best=[np.full(bands[0].shape,-2.,np.float32) for _ in source_ids];second=[b.copy() for b in best]
    for i in range(len(source_ids)):
        for j in range(i+1,len(source_ids)):
            corr,valid=pair_phase(bands[i],bands[j],validity[i]&validity[j])
            update_best(best[i],second[i],corr,valid);update_best(best[j],second[j],corr,valid)
    hashes={}
    for cid,first,runnerup in zip(source_ids,best,second):
        # Two distinct witnesses are required to affect rank. Neutral .5 where
        # evidence is absent; no forced fallback and no privilege for flat RGB.
        cost=np.where(runnerup>=-1.,.5-.25*(first+runnerup),.5).astype(np.float32)
        path=out/f'src{cid:03d}_cost.npy';np.save(path,cost);hashes[path.name]=sha(path)
    write(out/'receipt.json',dict(target=target,sources=source_ids,settings=fingerprint,code_sha256=sha(Path(__file__)),inputs=inputs,hashes=hashes,uses_target_rgb=False,witness_rgb_in_output=False))
    log(root,'witness_phase_evidence_complete',camera=target)
    return out
