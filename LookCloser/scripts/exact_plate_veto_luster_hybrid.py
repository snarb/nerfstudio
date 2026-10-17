"""Reject native samples nearly identical to a measured empty background.

Only reject if the frozen source render predicts a contrasting opaque object.
Ambiguous dark-on-dark samples are retained. No target RGB or mask is used.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source,geometry,log,native_rows,project,read,sha,source_path,unproject,write
from transfer_luster_hybrid import gaussian
from foreground_noise_luster_hybrid import foreground_evidence
from cubic75_luster_hybrid import kernel


def background_veto(photo,plate,model,alpha,tolerance):
    observed=np.max(np.abs(photo-plate),axis=-1)
    predicted=np.max(np.abs(model-plate),axis=-1)
    return (observed<=tolerance)&(predicted>=.12)&(alpha>=.99)


def source_veto(root,source):
    rows=native_rows();assert rows[source]['train'] and source not in (12,95,150)
    d=root/'exact_plate_veto'/f'cam{source:03d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'receipt.json').exists():
        r=read(d/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert all(sha(d/k)==v for k,v in r['hashes'].items());return d
    gr=root/'geometry'/f'cam{source:03d}'/'receipt.json';g=geometry(root,source)
    bg=read(gr)['background'];assert bg and sha(bg['path'])==bg['sha256']
    path=source_path(source);assert sha(path)==read(root/'preflight/manifest.json')['source_hashes'][str(path)]
    photo=np.asarray(Image.open(path).convert('RGB'),np.float32)/255
    row=rows[source]
    plate=np.asarray(Image.open(bg['path']).convert('RGB').resize((row['w'],row['h']),Image.Resampling.LANCZOS),np.float32)/255
    _,fit=foreground_evidence(photo,plate,g['alpha']<.01)
    for channel,(gain,bias) in enumerate(fit['coefficients']):plate[...,channel]=plate[...,channel]*gain+bias
    smooth=[gaussian(rgb,1.) for rgb in [photo,plate,np.asarray(g['rgb'])]]
    for key,tolerance in [('strict10',.01),('strict20',.02)]:
        np.save(d/f'{key}.npy',background_veto(*smooth,g['alpha'],tolerance))
    inputs={str(p):sha(p) for p in [path,Path(bg['path']),gr,
        Path(__file__).with_name('foreground_noise_luster_hybrid.py'),Path(__file__).with_name('transfer_luster_hybrid.py')]}
    write(d/'receipt.json',dict(source=source,code_sha256=sha(Path(__file__)),inputs=inputs,
        hashes={f'{k}.npy':sha(d/f'{k}.npy') for k in ['strict10','strict20']},
        fit=fit,smoothing_sigma=1.,model_contrast_min=.12,alpha_min=.99,
        uses_target_rgb=False,uses_evaluation_masks=False,lookup_shift=False))
    log(root,'exact_plate_source_veto_complete',source=source);return d


def footprint_veto(mask,uv):
    h,w=mask.shape;finite=np.isfinite(uv).all(-1)
    coords=np.where(finite[...,None],uv,0);u,v=coords[...,0],coords[...,1]
    x=np.floor(u).astype(np.int32);y=np.floor(v).astype(np.int32)
    result=np.zeros(u.shape,bool)
    for dy in [-1,0,1,2]:
        for dx in [-1,0,1,2]:
            active=np.abs(kernel(u-(x+dx))*kernel(v-(y+dy)))>1e-7
            xx,yy=x+dx,y+dy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&finite
            result|=active&inside&mask[yy.clip(0,h-1),xx.clip(0,w-1)]
    # Out-of-image/nonfinite projection is separately rejected by geometric support.
    return result


def projected_veto(root,target,source):
    rows=native_rows();assert_source(rows,target,source)
    d=root/'exact_plate_projected'/f'cam{target:03d}'/f'src{source:03d}';d.mkdir(parents=True,exist_ok=True)
    sd=source_veto(root,source)
    if (d/'receipt.json').exists():
        r=read(d/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert r['inputs'][str(sd/'receipt.json')]==sha(sd/'receipt.json')
        assert all(sha(d/k)==v for k,v in r['hashes'].items());return d
    wr=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    box=read(wr)['box'];x0,y0,x1,y1=box;g=geometry(root,target)
    masks={k:np.load(sd/f'{k}.npy',mmap_mode='r') for k in ['strict10','strict20']}
    arrays={k:np.empty((y1-y0,x1-x0),bool) for k in masks}
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[source])
        for k,mask in masks.items():arrays[k][y-y0:stop-y0]=footprint_veto(mask,uv)
    for k,array in arrays.items():np.save(d/f'{k}.npy',array)
    inputs={str(p):sha(p) for p in [sd/'receipt.json',wr,root/'geometry'/f'cam{target:03d}'/'receipt.json',Path(__file__).with_name('cubic75_luster_hybrid.py')]}
    write(d/'receipt.json',dict(target=target,source=source,box=box,code_sha256=sha(Path(__file__)),inputs=inputs,
        hashes={f'{k}.npy':sha(d/f'{k}.npy') for k in masks},
        footprint='all_nonzero_cubic75_taps',uses_target_rgb=False,lookup_shift=False))
    log(root,'exact_plate_projected_veto_complete',camera=target,source=source);return d
