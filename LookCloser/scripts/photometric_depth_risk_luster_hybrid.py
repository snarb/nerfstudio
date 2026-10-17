"""Scalar photo sensitivity to frozen termination-depth uncertainty.

Only q50 supplies output texture coordinates. q10/q90 projections estimate a
directional uncertainty cost, never an alternate lookup or fitted geometry.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source,geometry,log,native_rows,project,read,sha,source_path,unproject,write
from photometric_curvature_luster_hybrid import sample_vector
from transfer_luster_hybrid import gaussian


def directional_cost(gradient,span,valid,scale=.01):
    """RGB RMS first-order change over half the q10-q90 source-image span."""
    assert scale>0 and gradient.shape[-2:]==(3,2)
    known=valid&np.isfinite(gradient).all(axis=(-2,-1))&np.isfinite(span).all(-1)
    safe_gradient=np.where(known[...,None,None],gradient,0)
    safe_span=np.where(known[...,None],span,0)
    change=np.sum(safe_gradient*safe_span[...,None,:],axis=-1)*.5
    rms=np.sqrt(np.mean(change*change,axis=-1))
    # Unknown evidence cannot make a candidate preferable to measured zero risk.
    return np.where(known,np.log1p(rms/scale),1e6).astype(np.float32)


def quality(root,target,source):
    rows=native_rows();assert_source(rows,target,source)
    out=root/'photometric_depth_risk'/f'cam{target:03d}'/f'src{source:03d}';out.mkdir(parents=True,exist_ok=True)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    paths=[parent,root/'geometry'/f'cam{target:03d}'/'receipt.json',source_path(source)]
    paths += [Path(__file__).with_name(m) for m in ['luster_hybrid_common.py','photometric_curvature_luster_hybrid.py','transfer_luster_hybrid.py']]
    inputs={str(p):sha(p) for p in paths}
    assert inputs[str(source_path(source))]==read(parent)['source_sha256']
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['inputs']==inputs
        assert r['hashes']['cost.npy']==sha(out/'cost.npy');return out
    photo=np.asarray(Image.open(source_path(source)).convert('RGB'),np.float32)/255
    low=gaussian(photo,1.)
    gradient=np.stack([cv2.Sobel(low,cv2.CV_32F,1,0,ksize=3,scale=1/8),cv2.Sobel(low,cv2.CV_32F,0,1,ksize=3,scale=1/8)],-1)
    field=gradient.reshape(*gradient.shape[:2],6)
    g=geometry(root,target);tr=rows[target];sr=rows[source]
    box=read(parent)['box'];x0,y0,x1,y1=box;result=np.empty((y1-y0,x1-x0),np.float32)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);sl=np.s_[y:stop,x0:x1]
        yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,z=project(unproject(np.asarray(g['depth_z_q50'][sl]),tr,xy),sr)
        ulo,zlo=project(unproject(np.asarray(g['depth_z_q10'][sl]),tr,xy),sr)
        uhi,zhi=project(unproject(np.asarray(g['depth_z_q90'][sl]),tr,xy),sr)
        grad,inside=sample_vector(field,uv)
        valid=inside&(z>0)&(zlo>0)&(zhi>0)
        result[y-y0:stop-y0]=directional_cost(grad.reshape(*grad.shape[:2],3,2),uhi-ulo,valid)
    np.save(out/'cost.npy',result)
    write(out/'receipt.json',dict(target=target,source=source,box=box,inputs=inputs,
        hashes={'cost.npy':sha(out/'cost.npy')},code_sha256=sha(Path(__file__)),
        uses_target_rgb=False,uses_masks=False,lookup_shift=False,texture_coordinate='q50 unchanged',
        confidence_only_depths=['q10','q90'],native_gradient_sigma=1.,rgb_scale=.01,calibrated_probability=False))
    log(root,'photometric_depth_risk_cached',camera=target,source=source);return out
