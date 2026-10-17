"""Analytical detail admission measured only in native TRAIN witness views.

For a fixed projected source, solve the scalar least-squares gain between the
frozen witness render and that photograph's I-G6 detail. The witness photograph
is a confidence reference only. No geometry, image position or model is fitted.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source, geometry, log, native_rows, project, read, sha, source_path, unproject, write
from source_surface_confidence_luster_hybrid import EXCLUDED
from witness_native_restoration_luster_hybrid import sample_supported_scalar
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
from risk_single_luster_hybrid import masked_low


def optimal_gain(candidate, witness, baseline, support, sigma=3., floor=.002):
    """Minimize local RGB MSE of B+g*(S-B), with 0<=g<=1.

    Weak/unsupported evidence is neutral (gain one), not evidence of harm.
    Inputs are already detail bands in the same native witness coordinates.
    """
    assert candidate.shape == witness.shape == baseline.shape
    assert candidate.shape[:2] == support.shape and floor >= 0
    assert all(np.isfinite(a[support]).all() for a in [candidate, witness, baseline])
    sf = support.astype(np.float32)
    mass = gaussian(sf, sigma)
    delta = np.where(support[..., None], candidate-baseline, 0)
    residual = np.where(support[..., None], witness-baseline, 0)
    energy = gaussian(np.mean(delta*delta, axis=-1), sigma) / np.maximum(mass, 1e-8)
    cross = gaussian(np.mean(delta*residual, axis=-1), sigma) / np.maximum(mass, 1e-8)
    known = support & (mass >= .8) & (energy >= max(floor**2, 1e-12))
    gain = np.clip(cross / np.maximum(energy, 1e-12), 0, 1)
    return np.where(known, gain, 1).astype(np.float32), known


def median_gain(gains, known):
    assert gains.shape == known.shape and len(gains) >= 2
    assert np.isfinite(gains[known]).all() and np.all((gains[known]>=0)&(gains[known]<=1))
    count = known.sum(axis=0)
    order = np.sort(np.where(known, gains, np.inf), axis=0)
    lo = np.clip((count-1)//2, 0, len(order)-1)
    hi = np.minimum(count//2, len(order)-1)
    first = np.take_along_axis(order, lo[None], axis=0)[0]
    second = np.take_along_axis(order, hi[None], axis=0)[0]
    result = np.where(count>=2, .5*(first+second), 1)
    return result.astype(np.float32), count.astype(np.uint8)


def source_gain(root, source, config):
    rows = native_rows()
    assert rows[source]['train'] and source not in EXCLUDED
    dependency = root/'witness_native_restoration'/f'cam{source:03d}'/'receipt.json'
    dr = read(dependency)  # Existing producer owns all TRAIN-to-TRAIN warps.
    gates = {k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    assert dr['source']==source and dr['gates']==gates
    witnesses = dr['witnesses']
    assert len(witnesses)>=2 and len(set(witnesses))==len(witnesses)
    assert not set(witnesses)&set(EXCLUDED)
    out = root/'witness_native_gain'/f'cam{source:03d}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json')
        assert r['code_sha256']==sha(Path(__file__)) and r['gates']==gates
        assert r['inputs'][str(dependency)]==sha(dependency)
        assert all(sha(out/k)==v for k,v in r['hashes'].items())
        return out
    inputs={str(dependency):sha(dependency)}
    for module in ['witness_native_restoration_luster_hybrid.py','source_surface_confidence_luster_hybrid.py',
                   'supported_luster_hybrid.py','transfer_luster_hybrid.py','risk_single_luster_hybrid.py','luster_hybrid_common.py']:
        path=Path(__file__).with_name(module);inputs[str(path)]=sha(path)
    gains=[];validity=[];box=None;g=geometry(root,source)
    for witness in witnesses:
        assert_source(rows,witness,source)
        d=root/'variants/cubic75/warps'/f'cam{witness:03d}'/f'src{source:03d}'
        wr=read(d/'receipt.json');assert wr['target']==witness and wr['source']==source
        witness_box=wr['box'];wx0,wy0,wx1,wy1=witness_box
        arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in wr['arrays']}
        support=geometric_support(arrays,config)
        path=source_path(witness)
        assert sha(path)==read(root/'preflight/manifest.json')['source_hashes'][str(path)]
        photo=np.asarray(Image.open(path).convert('RGB'),np.float32)[wy0:wy1,wx0:wx1]/255
        baseline=np.asarray(geometry(root,witness)['rgb'][wy0:wy1,wx0:wx1])
        bands=[]
        for rgb in [arrays['rgb'],photo,baseline]:
            low,_=masked_low(rgb,support,6.)
            bands.append(rgb-low)
        values,known=optimal_gain(*bands,support)
        reverse=root/'warps'/f'cam{source:03d}'/f'src{witness:03d}'
        rr=read(reverse/'receipt.json')
        assert rr['target']==source and rr['source']==witness
        if box is None:box=rr['box']
        assert box==rr['box'];x0,y0,x1,y1=box
        reverse_arrays={k:np.load(reverse/f'{k}.npy',mmap_mode='r') for k in rr['arrays'] if k!='rgb'}
        reverse_support=geometric_support(reverse_arrays,config)
        gain=np.ones(reverse_support.shape,np.float32);valid=np.zeros(reverse_support.shape,bool)
        for y in range(y0,y1,128):
            stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1]
            xy=np.stack([xx,yy],-1).astype(np.float32)
            uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[source],xy),rows[witness])
            sampled,good=sample_supported_scalar(values,known,uv,witness_box)
            sl=np.s_[y-y0:stop-y0]
            valid[sl]=good&reverse_support[sl]
            gain[sl]=np.where(valid[sl],sampled,1)
        gains.append(gain);validity.append(valid)
        for path in [path,d/'receipt.json',reverse/'receipt.json',root/'geometry'/f'cam{source:03d}'/'receipt.json',
                     root/'geometry'/f'cam{witness:03d}'/'receipt.json']:
            inputs[str(path)]=sha(path)
        for folder,receipt in [(d,wr),(reverse,rr)]:
            for key in receipt['arrays']:inputs[str(folder/f'{key}.npy')]=sha(folder/f'{key}.npy')
    gain,count=median_gain(np.stack(gains),np.stack(validity))
    np.save(out/'gain.npy',gain);np.save(out/'count.npy',count)
    write(out/'receipt.json',dict(source=source,witnesses=witnesses,excluded_witnesses=EXCLUDED,box=box,gates=gates,
        inputs=inputs,code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['gain.npy','count.npy']},
        detail='I-G6 RGB in native TRAIN witness',moment_sigma=3.,moment_floor=.002,
        geometry='frozen camera-z q50',confidence='median bounded analytical scalar gain, quorum 2',
        uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    log(root,'witness_native_gain_source_complete',source=source,known_fraction=float(np.mean(count>=2)))
    return out


def projected_gain(root,target,source,config):
    rows=native_rows();assert_source(rows,target,source)
    sd=source_gain(root,source,config)
    out=root/'witness_native_gain_projected'/f'cam{target:03d}'/f'src{source:03d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert r['inputs'][str(sd/'receipt.json')]==sha(sd/'receipt.json')
        assert sha(out/'gain.npy')==r['hashes']['gain.npy'];return out
    wr=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    box=read(wr)['box'];x0,y0,x1,y1=box;g=geometry(root,target)
    sr=read(sd/'receipt.json');gain=np.load(sd/'gain.npy',mmap_mode='r')
    known=np.load(sd/'count.npy',mmap_mode='r')>=2
    result=np.ones((y1-y0,x1-x0),np.float32)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1]
        xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[source])
        sampled,good=sample_supported_scalar(gain,known,uv,sr['box'])
        result[y-y0:stop-y0]=np.where(good,sampled,1)
    np.save(out/'gain.npy',result)
    inputs={str(p):sha(p) for p in [sd/'receipt.json',wr,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    write(out/'receipt.json',dict(target=target,source=source,box=box,inputs=inputs,hashes={'gain.npy':sha(out/'gain.npy')},
        code_sha256=sha(Path(__file__)),unknown_gain=1.,uses_target_rgb=False,uses_masks=False,witness_rgb_in_output=False,lookup_shift=False))
    return out
