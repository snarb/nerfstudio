"""TRAIN witness confidence relative to the actual target frozen render.

The target render is reprojected only to evaluate scalar confidence. Its GT is
never read. Final photo sampling stays at the target's original frozen q50.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source, geometry, log, native_rows, project, read, sha, source_path, unproject, write
from source_surface_confidence_luster_hybrid import EXCLUDED
from witness_native_gain_luster_hybrid import optimal_gain, median_gain
from witness_native_restoration_luster_hybrid import sample_supported_scalar, median_cost
from source_native_restoration_common_luster_hybrid import restoration_utility
from render_projection_luster_hybrid import render_projection
from risk_single_luster_hybrid import masked_low
from supported_luster_hybrid import geometric_support


def projected_gain(root,target,source,config):
    rows=native_rows();assert_source(rows,target,source)
    dependency=root/'witness_native_restoration'/f'cam{source:03d}'/'receipt.json'
    dr=read(dependency)
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    assert dr['source']==source and dr['gates']==gates
    witnesses=dr['witnesses']
    assert len(witnesses)>=2 and len(set(witnesses))==len(witnesses)
    assert not set(witnesses)&set(EXCLUDED) and source not in witnesses
    out=root/'target_witness_gain_projected'/f'cam{target:03d}'/f'src{source:03d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json')
        assert r['code_sha256']==sha(Path(__file__)) and r['gates']==gates
        assert r['inputs'][str(dependency)]==sha(dependency)
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    wr=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    box=read(wr)['box'];x0,y0,x1,y1=box;g=geometry(root,target)
    inputs={str(p):sha(p) for p in [dependency,wr,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    for module in ['luster_hybrid_common.py','source_surface_confidence_luster_hybrid.py',
                   'witness_native_gain_luster_hybrid.py','witness_native_restoration_luster_hybrid.py',
                   'source_native_restoration_common_luster_hybrid.py','render_projection_luster_hybrid.py',
                   'risk_single_luster_hybrid.py','supported_luster_hybrid.py']:
        p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
    gains=[];utilities=[];validity=[]
    for witness in witnesses:
        assert_source(rows,target,witness);assert_source(rows,witness,source)
        d=root/'variants/cubic75/warps'/f'cam{witness:03d}'/f'src{source:03d}'
        sr=read(d/'receipt.json');assert sr['target']==witness and sr['source']==source
        wbox=sr['box'];wx0,wy0,wx1,wy1=wbox
        arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in sr['arrays']}
        forward=render_projection(root,witness,target,wbox,config)
        reverse=render_projection(root,target,witness,box,config)
        support=geometric_support(arrays,config)&np.load(forward/'support.npy',mmap_mode='r')
        native=source_path(witness)
        assert sha(native)==read(root/'preflight/manifest.json')['source_hashes'][str(native)]
        photo=np.asarray(Image.open(native).convert('RGB'),np.float32)[wy0:wy1,wx0:wx1]/255
        baseline=np.load(forward/'rgb.npy',mmap_mode='r')
        bands=[]
        for rgb in [arrays['rgb'],photo,baseline]:
            low,_=masked_low(rgb,support,6.);bands.append(rgb-low)
        gain,kg=optimal_gain(*bands,support)
        utility,ku=restoration_utility(*bands,support)
        known=kg&ku
        reverse_support=np.load(reverse/'support.npy',mmap_mode='r')
        gain_out=np.ones(reverse_support.shape,np.float32)
        utility_out=np.zeros(reverse_support.shape,np.float32)
        valid=np.zeros(reverse_support.shape,bool)
        for y in range(y0,y1,128):
            stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1]
            xy=np.stack([xx,yy],-1).astype(np.float32)
            uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[witness])
            sampled,good=sample_supported_scalar(gain,known,uv,wbox)
            sampled_utility,ugood=sample_supported_scalar(utility,known,uv,wbox)
            assert np.array_equal(good,ugood)
            sl=np.s_[y-y0:stop-y0];valid[sl]=good&reverse_support[sl]
            gain_out[sl]=np.where(valid[sl],sampled,1)
            utility_out[sl]=np.where(valid[sl],sampled_utility,0)
        gains.append(gain_out);utilities.append(utility_out);validity.append(valid)
        for p in [native,d/'receipt.json',forward/'receipt.json',reverse/'receipt.json']:
            inputs[str(p)]=sha(p)
        for key in sr['arrays']:inputs[str(d/f'{key}.npy')]=sha(d/f'{key}.npy')
    valid=np.stack(validity)
    gain,count=median_gain(np.stack(gains),valid)
    cost,cost_count=median_cost(np.stack(utilities),valid)
    np.testing.assert_array_equal(count,cost_count)
    for key,value in [('gain',gain),('cost',cost),('count',count)]:np.save(out/f'{key}.npy',value)
    write(out/'receipt.json',dict(target=target,source=source,witnesses=witnesses,box=box,gates=gates,
        inputs=inputs,code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['gain.npy','cost.npy','count.npy']},
        confidence_frame='native TRAIN witness',baseline='actual target frozen render reprojected for confidence only',
        detail='I-G6 RGB',moment_sigma=3.,moment_floor=.002,unknown_gain=1.,unknown_cost=.5,
        geometry='frozen camera-z q50',uses_target_gt=False,uses_masks=False,witness_rgb_in_output=False,
        target_render_warp_in_output=False,lookup_shift=False))
    log(root,'target_witness_gain_complete',target=target,source=source,known_fraction=float(np.mean(count>=2)))
    return out
