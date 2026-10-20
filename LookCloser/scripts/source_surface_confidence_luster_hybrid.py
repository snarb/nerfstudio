"""TRAIN texture corroboration on each source's own frozen q50 surface.

Witnesses supply a scalar reliability cost. All four evaluation/demo cameras
are excluded from witnesses, including cam007. No new geometry or RGB mixture.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from luster_hybrid_common import (HELDOUT, assert_source, geometry, log, native_rows,
    project, read, sha, source_path, unproject, write)
from transfer_luster_hybrid import gaussian, source_score, warp
from supported_luster_hybrid import geometric_support
from source_rgb_phase_quality_luster_hybrid import phase_cost
from risk_single_luster_hybrid import masked_low

EXCLUDED = (7, 12, 95, 150)


def frozen_witness_pool(root):
    pool=root/'preflight/witness_pool.json'
    if (root/'context.json').exists() and not pool.exists():
        sources=read(root/'preflight/sources.json')['all_sources']
        write(pool,dict(cameras=sorted(set(sources)-set(EXCLUDED)),
                        policy='Selected source union; fixed before confidence construction; extra caches do not change witnesses'))
    return pool


def witness_candidates(rows, source, available):
    return sorted(c for c in available if c != source and c not in EXCLUDED and rows[c]['train'])


def select_witnesses(root, source, count=4):
    rows = native_rows(); row = rows[source]; g = geometry(root, source)
    assert row['train'] and source not in HELDOUT
    available = [int(p.parent.name[3:]) for p in (root/'geometry').glob('cam*/receipt.json')]
    pool=frozen_witness_pool(root)
    if pool.exists():
        required=set(read(pool)['cameras'])
        missing=required-set(available)
        assert not missing, f'Frozen witness geometry is incomplete: {sorted(missing)}'
        available=sorted(required)
    candidates = witness_candidates(rows, source, available)
    yy, xx = np.mgrid[0:row['h']:24, 0:row['w']:24]
    z = np.asarray(g['depth_z_q50'][yy, xx]); keep = np.isfinite(z) & (g['alpha'][yy, xx] >= .99)
    world = unproject(z[keep], row, np.stack([xx, yy], -1)[keep].astype(np.float32))
    scores = np.stack([source_score(world, row, rows[c])[0] for c in candidates])
    order = np.argsort(scores, axis=0, kind='stable'); votes = np.zeros(len(candidates))
    for rank in range(min(4, len(candidates))):
        idx = order[rank]; value = scores[idx, np.arange(len(world))]
        np.add.at(votes, idx, np.where(np.isfinite(value), np.exp(-value)/(rank+1), 0))
    return [candidates[i] for i in np.argsort(-votes, kind='stable')[:count] if votes[i] > 0]


def two_witness_cost(costs, supports):
    """Two distinct observations required; missing evidence remains neutral."""
    values = np.where(supports, costs, np.inf)
    best = np.partition(values, 1, axis=0)[:2]
    known = np.isfinite(best[1])
    result = np.full(known.shape, .5, np.float32)
    result[known] = best[:, known].mean(axis=0)
    return result, known


def source_confidence(root, source, config):
    out = root/'source_surface_confidence'/f'cam{source:03d}'; out.mkdir(parents=True, exist_ok=True)
    gates = {k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    pool=frozen_witness_pool(root)
    pool_hash=sha(pool) if pool.exists() else None
    if (out/'receipt.json').exists():
        r = read(out/'receipt.json'); assert r['code_sha256'] == sha(Path(__file__)) and r['gates'] == gates
        assert r.get('witness_pool_sha256')==pool_hash
        assert all(sha(out/k) == v for k,v in r['hashes'].items())
        return out
    rows = native_rows(); assert rows[source]['train'] and source not in HELDOUT
    witnesses = select_witnesses(root, source); assert len(witnesses) >= 2
    photo_path = source_path(source); manifest = read(root/'preflight/manifest.json')
    assert sha(photo_path) == manifest['source_hashes'][str(photo_path)]
    photo = np.asarray(Image.open(photo_path).convert('RGB'), np.float32)/255
    inputs = {str(photo_path):sha(photo_path)}
    if pool.exists():inputs[str(pool)]=pool_hash
    for mod in ['transfer_luster_hybrid.py','source_rgb_phase_quality_luster_hybrid.py',
                'risk_single_luster_hybrid.py','supported_luster_hybrid.py','luster_hybrid_common.py']:
        p=Path(__file__).with_name(mod); inputs[str(p)] = sha(p)
    costs=[]; supports=[]; box=None
    for witness in witnesses:
        assert_source(rows, source, witness); assert witness not in EXCLUDED
        warp(root, source, witness)
        d=root/'warps'/f'cam{source:03d}'/f'src{witness:03d}'; r=read(d/'receipt.json')
        if box is None:box=r['box']
        assert box == r['box']; x0,y0,x1,y1=box
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        valid=geometric_support(a,config)
        bands=[]
        for rgb in [photo[y0:y1,x0:x1],a['rgb']]:
            fine,_=masked_low(rgb,valid,2.); coarse,_=masked_low(rgb,valid,8.); bands.append(fine-coarse)
        costs.append(phase_cost(*bands,valid)); supports.append(valid & (gaussian(valid.astype(np.float32),4.) >= .8))
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        for k in r['arrays']:inputs[str(d/f'{k}.npy')]=sha(d/f'{k}.npy')
        for cid in [source,witness]:
            p=root/'geometry'/f'cam{cid:03d}'/'receipt.json';inputs[str(p)]=sha(p)
    cost,known=two_witness_cost(np.stack(costs),np.stack(supports))
    np.save(out/'cost.npy',cost); np.save(out/'known.npy',known)
    write(out/'receipt.json',dict(source=source,witnesses=witnesses,excluded_witnesses=EXCLUDED,
        witness_pool_sha256=pool_hash,
        box=box,gates=gates,code_sha256=sha(Path(__file__)),inputs=inputs,
        hashes={k:sha(out/k) for k in ['cost.npy','known.npy']},
        geometry='source-view frozen camera-z q50',source_rgb_used_for_confidence_only=True,
        uses_target_rgb=False,witness_rgb_in_output=False,lookup_shift=False))
    log(root,'source_surface_confidence_complete',source=source,witnesses=witnesses,known_fraction=float(known.mean()))
    return out


def sample_cost(cost, uv, box):
    """Scalar interpolation; outside the measured source crop is neutral."""
    x0,y0,_,_=box
    coords=np.asarray(uv,np.float32)-np.array([x0,y0],np.float32)
    finite=np.isfinite(coords).all(-1); coords=np.where(finite[...,None],coords,-1e6)
    result=cv2.remap(cost,coords[...,0],coords[...,1],cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=.5)
    return np.where(finite,result,.5).astype(np.float32)


def projected_confidence(root,target,source,config):
    rows=native_rows(); assert_source(rows,target,source)
    out=root/'source_surface_projected'/f'cam{target:03d}'/f'src{source:03d}';out.mkdir(parents=True,exist_ok=True)
    sd=source_confidence(root,source,config)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert r['inputs'][str(sd/'receipt.json')]==sha(sd/'receipt.json')
        assert sha(out/'cost.npy')==r['hashes']['cost.npy'];return out
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
        uses_target_rgb=False,witness_rgb_in_output=False,lookup_shift=False))
    return out
