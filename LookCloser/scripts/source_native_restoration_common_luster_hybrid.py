"""TRAIN-only common-band restoration reliability on each native source surface.

Witness photographs supply scalar errors, never output RGB. Compare candidate
photo detail with each witness photo, relative to that witness's frozen render.
This is not an independent novel-view test or a fitted appearance model.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source,geometry,log,native_rows,project,read,sha,source_path,unproject,write
from source_surface_confidence_luster_hybrid import EXCLUDED,select_witnesses,sample_cost
from residual_luster_hybrid import model_warp
from risk_single_luster_hybrid import masked_low
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian,warp


def restoration_utility(candidate,witness,baseline,support,sigma=3.,floor=.002):
    """Relative supported RGB detail error; flat/unknown is not positive evidence."""
    mass=gaussian(support.astype(np.float32),sigma)
    finite=np.isfinite(candidate).all(-1)&np.isfinite(witness).all(-1)&np.isfinite(baseline).all(-1)
    assert np.all(finite[support])
    def error(a):
        value=np.mean(np.where(support[...,None],a,0)**2,axis=-1)
        return gaussian(value,sigma)/np.maximum(mass,1e-8)
    before=error(baseline-witness);after=error(candidate-witness)
    total=before+after
    known=support&(mass>=.8)&(total>=floor**2)
    utility=(before-after)/(total+floor**2)
    return np.where(known,np.clip(utility,-1,1),0).astype(np.float32),known


def contradiction_median(utilities,known):
    """Require two distinct witnesses and use all available evidence by median."""
    assert utilities.shape==known.shape and utilities.ndim>=2 and len(utilities)>0
    assert np.isfinite(utilities[known]).all()
    count=known.sum(axis=0)
    order=np.sort(np.where(known,utilities,np.inf),axis=0)
    lo=np.maximum((count-1)//2,0);hi=count//2
    lo=np.minimum(lo,len(order)-1);hi=np.minimum(hi,len(order)-1)
    first=np.take_along_axis(order,lo[None],axis=0)[0]
    second=np.take_along_axis(order,hi[None],axis=0)[0]
    median=np.where(count>=2,.5*(first+second),0)
    cost=.5+.5*np.maximum(-median,0)
    return cost.astype(np.float32),count.astype(np.uint8)


def source_quality(root,source,config):
    rows=native_rows();assert rows[source]['train'] and source not in EXCLUDED
    out=root/'source_native_restoration_common'/f'cam{source:03d}';out.mkdir(parents=True,exist_ok=True)
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['gates']==gates
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    witnesses=select_witnesses(root,source);assert len(witnesses)>=2
    path=source_path(source);assert sha(path)==read(root/'preflight/manifest.json')['source_hashes'][str(path)]
    photo=np.asarray(Image.open(path).convert('RGB'),np.float32)/255
    inputs={str(path):sha(path)}
    for mod in ['source_surface_confidence_luster_hybrid.py','residual_luster_hybrid.py','risk_single_luster_hybrid.py',
                'supported_luster_hybrid.py','transfer_luster_hybrid.py','luster_hybrid_common.py']:
        p=Path(__file__).with_name(mod);inputs[str(p)]=sha(p)
    utilities=[];validity=[];box=None
    for witness in witnesses:
        assert_source(rows,source,witness);assert witness not in EXCLUDED
        warp(root,source,witness)
        d=root/'warps'/f'cam{source:03d}'/f'src{witness:03d}';r=read(d/'receipt.json')
        if box is None:box=r['box']
        assert r['box']==box;x0,y0,x1,y1=box
        arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        support=geometric_support(arrays,config)
        witness_model=model_warp(root,source,witness)
        bands=[]
        for rgb in [photo[y0:y1,x0:x1],arrays['rgb'],witness_model]:
            fine,_=masked_low(rgb,support,1.);coarse,_=masked_low(rgb,support,6.);bands.append(fine-coarse)
        utility,known=restoration_utility(*bands,support)
        utilities.append(utility);validity.append(known)
        mr=root/'source_model_warps'/f'cam{source:03d}'/f'src{witness:03d}'/'receipt.json'
        inputs[str(mr)]=sha(mr);inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        for key in r['arrays']:inputs[str(d/f'{key}.npy')]=sha(d/f'{key}.npy')
        for cid in [source,witness]:
            p=root/'geometry'/f'cam{cid:03d}'/'receipt.json';inputs[str(p)]=sha(p)
    cost,count=contradiction_median(np.stack(utilities),np.stack(validity))
    np.save(out/'cost.npy',cost);np.save(out/'count.npy',count)
    write(out/'receipt.json',dict(source=source,witnesses=witnesses,excluded_witnesses=EXCLUDED,box=box,
        gates=gates,inputs=inputs,code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['cost.npy','count.npy']},
        confidence='one-sided negative median TRAIN RGB restoration utility, quorum 2',
        detail='G1-G6 in native source frame; common-band confidence proxy',utility_sigma=3.,utility_floor=.002,
        geometry='source frozen camera-z q50',uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    log(root,'source_native_restoration_common_cached',source=source,witnesses=witnesses,known_fraction=float(np.mean(count>=2)))
    return out


def projected_quality(root,target,source,config):
    rows=native_rows();assert_source(rows,target,source)
    out=root/'source_native_restoration_common_projected'/f'cam{target:03d}'/f'src{source:03d}';out.mkdir(parents=True,exist_ok=True)
    sd=source_quality(root,source,config)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert r['inputs'][str(sd/'receipt.json')]==sha(sd/'receipt.json')
        assert r['hashes']['cost.npy']==sha(out/'cost.npy');return out
    wr=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    box=read(wr)['box'];x0,y0,x1,y1=box;g=geometry(root,target)
    sr=read(sd/'receipt.json');cost=np.load(sd/'cost.npy',mmap_mode='r')
    result=np.empty((y1-y0,x1-x0),np.float32)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[source])
        result[y-y0:stop-y0]=sample_cost(cost,uv,sr['box'])
    np.save(out/'cost.npy',result)
    inputs={str(p):sha(p) for p in [sd/'receipt.json',wr,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    write(out/'receipt.json',dict(target=target,source=source,box=box,inputs=inputs,
        hashes={'cost.npy':sha(out/'cost.npy')},code_sha256=sha(Path(__file__)),
        uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    return out
