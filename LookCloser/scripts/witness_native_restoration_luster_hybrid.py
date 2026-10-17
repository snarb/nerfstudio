"""Measure a source's restoration utility in native TRAIN witness images.

Each source photograph is projected into a witness's frozen q50 surface before
comparison with that witness's native photo and frozen render. Only scalar
utility returns to the source/target surface; witness RGB never enters output.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source,geometry,log,native_rows,project,read,sha,source_path,unproject,write
from source_surface_confidence_luster_hybrid import EXCLUDED,select_witnesses,sample_cost
from source_native_restoration_common_luster_hybrid import restoration_utility
from risk_single_luster_hybrid import masked_low
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import warp
from cubic75_luster_hybrid import warp as cubic_warp


def sample_supported_scalar(values,known,uv,box):
    """Exact bilinear scalar sampling with every nonzero tap supported."""
    h,w=values.shape;assert known.shape==values.shape
    coords=np.asarray(uv,np.float32)-np.array(box[:2],np.float32)
    u,v=coords[...,0],coords[...,1]
    good=np.isfinite(coords).all(-1)&(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    u=np.where(good,u,0);v=np.where(good,v,0)
    x=np.floor(u).astype(np.int32);y=np.floor(v).astype(np.int32);fx=u-x;fy=v-y
    result=np.zeros(u.shape,np.float32)
    for dy in [0,1]:
        for dx in [0,1]:
            weight=(fx if dx else 1-fx)*(fy if dy else 1-fy);tap=weight>1e-7
            xx=(x+dx).clip(0,w-1);yy=(y+dy).clip(0,h-1)
            good&=~tap|(known[yy,xx]&np.isfinite(values[yy,xx]))
            result+=weight*np.nan_to_num(values[yy,xx])
    return np.where(good,result,0).astype(np.float32),good


def median_cost(utilities,known):
    """Signed median utility, quorum two; unknown stays neutral, never rewarded."""
    assert utilities.shape==known.shape and len(utilities)>=2
    assert np.isfinite(utilities[known]).all()
    count=known.sum(axis=0);order=np.sort(np.where(known,utilities,np.inf),axis=0)
    lo=np.clip((count-1)//2,0,len(order)-1);hi=np.minimum(count//2,len(order)-1)
    first=np.take_along_axis(order,lo[None],axis=0)[0]
    second=np.take_along_axis(order,hi[None],axis=0)[0]
    median=np.where(count>=2,.5*(first+second),0)
    return (.5-.5*np.clip(median,-1,1)).astype(np.float32),count.astype(np.uint8)


def pair_quality(root,source,witness,config):
    rows=native_rows();assert_source(rows,witness,source)
    assert source not in EXCLUDED and witness not in EXCLUDED and rows[witness]['train']
    out=root/'witness_native_restoration_pairs'/f'src{source:03d}'/f'cam{witness:03d}';out.mkdir(parents=True,exist_ok=True)
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['gates']==gates
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    warp(root,witness,source);cubic_warp(root,witness,source)
    d=root/'variants/cubic75/warps'/f'cam{witness:03d}'/f'src{source:03d}';r=read(d/'receipt.json')
    box=r['box'];x0,y0,x1,y1=box
    arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
    support=geometric_support(arrays,config)
    path=source_path(witness);assert sha(path)==read(root/'preflight/manifest.json')['source_hashes'][str(path)]
    photo=np.asarray(Image.open(path).convert('RGB'),np.float32)[y0:y1,x0:x1]/255
    baseline=np.asarray(geometry(root,witness)['rgb'][y0:y1,x0:x1])
    bands=[]
    for rgb in [arrays['rgb'],photo,baseline]:
        fine,_=masked_low(rgb,support,1.);coarse,_=masked_low(rgb,support,6.);bands.append(fine-coarse)
    utility,known=restoration_utility(*bands,support)
    np.save(out/'utility.npy',utility);np.save(out/'known.npy',known)
    inputs={str(p):sha(p) for p in [path,d/'receipt.json',root/'geometry'/f'cam{witness:03d}'/'receipt.json',root/'geometry'/f'cam{source:03d}'/'receipt.json']}
    for key in r['arrays']:inputs[str(d/f'{key}.npy')]=sha(d/f'{key}.npy')
    for module in ['source_native_restoration_common_luster_hybrid.py','risk_single_luster_hybrid.py','supported_luster_hybrid.py','transfer_luster_hybrid.py','cubic75_luster_hybrid.py','luster_hybrid_common.py']:
        p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
    write(out/'receipt.json',dict(source=source,witnesses=[witness],box=box,gates=gates,inputs=inputs,
        code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['utility.npy','known.npy']},
        confidence_frame='native TRAIN witness',detail='G1-G6 RGB',geometry='frozen witness camera-z q50',
        utility_sigma=3.,utility_floor=.002,uses_evaluation_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    log(root,'witness_native_restoration_pair_complete',source=source,witness=witness,known_fraction=float(known.mean()))
    return out


def source_quality(root,source,config):
    rows=native_rows();assert rows[source]['train'] and source not in EXCLUDED
    out=root/'witness_native_restoration'/f'cam{source:03d}';out.mkdir(parents=True,exist_ok=True)
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['gates']==gates
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    witnesses=select_witnesses(root,source);assert len(witnesses)>=2
    utilities=[];validity=[];inputs={};box=None;g=geometry(root,source)
    for witness in witnesses:
        wd=pair_quality(root,source,witness,config);wr=read(wd/'receipt.json')
        values=np.load(wd/'utility.npy',mmap_mode='r');known=np.load(wd/'known.npy',mmap_mode='r')
        # Reverse visibility establishes that this source surface point is the
        # surface observed by the witness, not an occluded projection.
        warp(root,source,witness)
        d=root/'warps'/f'cam{source:03d}'/f'src{witness:03d}';r=read(d/'receipt.json')
        if box is None:box=r['box']
        assert r['box']==box;x0,y0,x1,y1=box
        arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        reverse_support=geometric_support(arrays,config)
        utility=np.zeros(reverse_support.shape,np.float32);valid=np.zeros(reverse_support.shape,bool)
        for y in range(y0,y1,128):
            stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
            uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[source],xy),rows[witness])
            sampled,good=sample_supported_scalar(values,known,uv,wr['box']);sl=np.s_[y-y0:stop-y0]
            utility[sl]=sampled;valid[sl]=good&reverse_support[sl]
        utilities.append(utility);validity.append(valid)
        for p in [wd/'receipt.json',d/'receipt.json',root/'geometry'/f'cam{source:03d}'/'receipt.json']:
            inputs[str(p)]=sha(p)
        for key in r['arrays']:inputs[str(d/f'{key}.npy')]=sha(d/f'{key}.npy')
    cost,count=median_cost(np.stack(utilities),np.stack(validity))
    np.save(out/'cost.npy',cost);np.save(out/'count.npy',count)
    p=Path(__file__).with_name('source_surface_confidence_luster_hybrid.py');inputs[str(p)]=sha(p)
    write(out/'receipt.json',dict(source=source,witnesses=witnesses,excluded_witnesses=EXCLUDED,box=box,gates=gates,
        inputs=inputs,code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['cost.npy','count.npy']},
        confidence='median native TRAIN witness restoration utility, quorum 2',geometry='frozen source camera-z q50',
        uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    log(root,'witness_native_restoration_source_complete',source=source,witnesses=witnesses,known_fraction=float(np.mean(count>=2)))
    return out


def projected_quality(root,target,source,config):
    rows=native_rows();assert_source(rows,target,source)
    out=root/'witness_native_restoration_projected'/f'cam{target:03d}'/f'src{source:03d}';out.mkdir(parents=True,exist_ok=True)
    sd=source_quality(root,source,config)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert r['inputs'][str(sd/'receipt.json')]==sha(sd/'receipt.json')
        assert r['hashes']['cost.npy']==sha(out/'cost.npy');return out
    wr=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    box=read(wr)['box'];x0,y0,x1,y1=box;g=geometry(root,target)
    sr=read(sd/'receipt.json');cost=np.load(sd/'cost.npy',mmap_mode='r');result=np.empty((y1-y0,x1-x0),np.float32)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[source])
        result[y-y0:stop-y0]=sample_cost(cost,uv,sr['box'])
    np.save(out/'cost.npy',result)
    inputs={str(p):sha(p) for p in [sd/'receipt.json',wr,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    write(out/'receipt.json',dict(target=target,source=source,box=box,inputs=inputs,hashes={'cost.npy':sha(out/'cost.npy')},
        code_sha256=sha(Path(__file__)),uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    return out
