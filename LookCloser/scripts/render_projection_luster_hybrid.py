"""Reproject frozen renders for scalar confidence, never native photo output.

Unlike photo-source warps, render_view can be an evaluation camera: only its
immutable LookCloser RGB/depth/alpha are read. There is no photograph reader.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry, log, native_rows, project, read, sha, unproject, write
from transfer_luster_hybrid import footprint_metrics, source_score
from cubic75_luster_hybrid import sample
from supported_luster_hybrid import geometric_support


def project_block(observer_row, render_row, render_geometry, xy, z, lo, hi, alpha):
    world=unproject(z,observer_row,xy)
    uv,zs=project(world,render_row)
    _,valid,sa,err,sd,slo,shi=footprint_metrics(render_geometry['rgb'],render_geometry,uv,zs,None,None,None,None,None)
    rgb,cvalid,ca,cerr=sample(render_geometry['rgb'],render_geometry,uv,zs)
    center=np.asarray(render_row['transform_matrix'],np.float32)[:3,3]
    def back(depth):return project(center+(world-center)*(depth/np.maximum(zs,1e-8))[...,None],observer_row)[0]
    rt=np.linalg.norm(back(sd)-xy,axis=-1)
    b0,b1=back(slo),back(shi)
    t0,_=project(unproject(lo,observer_row,xy),render_row)
    t1,_=project(unproject(hi,observer_row,xy),render_row)
    score,ratio=source_score(world,observer_row,render_row)
    uncertainty=.5*np.maximum(np.linalg.norm(b1-b0,axis=-1),np.linalg.norm(t1-t0,axis=-1)/np.maximum(ratio,1e-8))
    valid &= cvalid & np.isfinite(z) & np.isfinite(rt) & np.isfinite(uncertainty) & np.isfinite(score)
    arrays=dict(rgb=rgb,valid=valid,alpha=np.minimum(np.minimum(sa,ca),alpha),depth_error=np.maximum(err,cerr),
                roundtrip=rt,uncertainty=uncertainty,score=score,resolution_ratio=ratio)
    return {k:np.nan_to_num(v,nan=0 if k=='rgb' else 1e6,posinf=1e6,neginf=1e6) for k,v in arrays.items()}


def render_projection(root,observer,render_view,box,config):
    out=root/'frozen_render_projections'/f'cam{observer:03d}'/f'render{render_view:03d}'
    out.mkdir(parents=True,exist_ok=True)
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json')
        assert r['code_sha256']==sha(Path(__file__)) and r['box']==box and r['gates']==gates
        assert all(sha(out/k)==v for k,v in r['hashes'].items())
        return out
    rows=native_rows();g=geometry(root,observer);sg=geometry(root,render_view)
    x0,y0,x1,y1=box;shape=(y1-y0,x1-x0)
    rgb=np.lib.format.open_memmap(out/'rgb.npy',mode='w+',dtype=np.float32,shape=(*shape,3))
    support=np.lib.format.open_memmap(out/'support.npy',mode='w+',dtype=np.bool_,shape=shape)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1]
        xy=np.stack([xx,yy],-1).astype(np.float32);sl=np.s_[y:stop,x0:x1]
        a=project_block(rows[observer],rows[render_view],sg,xy,np.asarray(g['depth_z_q50'][sl]),
                        np.asarray(g['depth_z_q10'][sl]),np.asarray(g['depth_z_q90'][sl]),np.asarray(g['alpha'][sl]))
        rgb[y-y0:stop-y0]=a['rgb'];support[y-y0:stop-y0]=geometric_support(a,config)
    rgb.flush();support.flush()
    inputs={}
    for c in {observer,render_view}:
        p=root/'geometry'/f'cam{c:03d}'/'receipt.json';inputs[str(p)]=sha(p)
    for module in ['luster_hybrid_common.py','transfer_luster_hybrid.py','cubic75_luster_hybrid.py','supported_luster_hybrid.py']:
        p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
    write(out/'receipt.json',dict(observer=observer,render_view=render_view,box=box,gates=gates,inputs=inputs,
        code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['rgb.npy','support.npy']},
        reads_photographs=False,uses_target_gt=False,uses_masks=False,geometry='frozen camera-z q50',
        resampling='native_keys_cubic_clamped',cubic_parameter=-.75,lookup_shift=False,
        purpose='scalar confidence only; no reprojected render RGB in output'))
    log(root,'frozen_render_projection_complete',observer=observer,render_view=render_view)
    return out
