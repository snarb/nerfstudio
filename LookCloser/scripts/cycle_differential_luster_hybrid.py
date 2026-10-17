"""Measure local distortion of a frozen target/source/target geometric cycle.

This is a scalar source-ranking cue. Neither depth nor texture coordinates are
optimized. Native photos, evaluation images and masks are never read here.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import assert_source,geometry,log,native_rows,project,read,sha,unproject,write


def sample_depth(depth,uv):
    """Exact bilinear q50 sample, requiring every active tap to be finite."""
    h,w=depth.shape;finite=np.isfinite(uv).all(-1)
    safe=np.where(finite[...,None],uv,0)
    ix=np.floor(safe[...,0]).astype(np.int32);iy=np.floor(safe[...,1]).astype(np.int32)
    fx=safe[...,0]-ix;fy=safe[...,1]-iy
    value=np.zeros(ix.shape,np.float32);valid=finite.copy()
    for dy in (0,1):
        for dx in (0,1):
            weight=(fx if dx else 1-fx)*(fy if dy else 1-fy);active=weight>1e-7
            xx=ix+dx;yy=iy+dy
            z=depth[yy.clip(0,h-1),xx.clip(0,w-1)]
            good=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&np.isfinite(z)&(z>0)
            valid&=~active|good
            value+=np.where(active&good,weight*z,0).astype(np.float32)
    return value,valid


def cycle_cost(residual,valid,step=4):
    """Log cost of the cycle Jacobian minus identity, on a fixed neighborhood.

    Constant displacement is handled by the existing point round-trip term;
    here it has zero differential cost. Unknown neighborhoods get log(2).
    """
    cost=np.full(valid.shape,np.log(2),np.float32);known=np.zeros(valid.shape,bool)
    if min(valid.shape)<=2*step:return cost,known
    core=np.s_[step:-step,step:-step]
    xp=np.s_[step:-step,2*step:];xm=np.s_[step:-step,:-2*step]
    yp=np.s_[2*step:,step:-step];ym=np.s_[:-2*step,step:-step]
    good=valid[core].copy()
    for sl in [core,xp,xm,yp,ym]:good&=valid[sl]&np.isfinite(residual[sl]).all(-1)
    dx=(residual[xp]-residual[xm])/(2*step);dy=(residual[yp]-residual[ym])/(2*step)
    norm=np.sqrt(np.sum(dx*dx+dy*dy,axis=-1))
    cost[core]=np.where(good,np.log1p(np.minimum(norm/.1,20)),np.log(2))
    known[core]=good
    return cost,known


def quality(root,target,source):
    rows=native_rows();assert_source(rows,target,source)
    out=root/'cycle_differential'/f'cam{target:03d}'/f'src{source:03d}';out.mkdir(parents=True,exist_ok=True)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    paths=[parent,root/'geometry'/f'cam{target:03d}'/'receipt.json',
           root/'geometry'/f'cam{source:03d}'/'receipt.json',Path(__file__).with_name('luster_hybrid_common.py')]
    paths += [parent.with_name(f'{k}.npy') for k in ['valid','alpha','roundtrip','uncertainty']]
    inputs={str(p):sha(p) for p in paths}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['inputs']==inputs
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    g=geometry(root,target);sg=geometry(root,source);box=read(parent)['box'];x0,y0,x1,y1=box
    shape=(y1-y0,x1-x0);result=np.empty(shape,np.float32);known=np.empty(shape,bool)
    pa={k:np.load(parent.with_name(f'{k}.npy'),mmap_mode='r') for k in ['valid','alpha','roundtrip','uncertainty']}
    for y in range(y0,y1,128):
        stop=min(y+128,y1);lo=max(y0,y-4);hi=min(y1,stop+4)
        yy,xx=np.mgrid[lo:hi,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][lo:hi,x0:x1]),rows[target],xy),rows[source])
        z,valid=sample_depth(sg['depth_z_q50'],uv)
        back,_=project(unproject(z,rows[source],uv),rows[target]);residual=back-xy
        sl=np.s_[lo-y0:hi-y0]
        valid&=(pa['valid'][sl]>0)&(pa['alpha'][sl]>=.99)&(pa['roundtrip'][sl]<=1)&(pa['uncertainty'][sl]<=2)
        cost,ok=cycle_cost(residual,valid)
        result[y-y0:stop-y0]=cost[y-lo:stop-lo];known[y-y0:stop-y0]=ok[y-lo:stop-lo]
    np.save(out/'cost.npy',result);np.save(out/'known.npy',known)
    write(out/'receipt.json',dict(target=target,source=source,box=box,inputs=inputs,
        hashes={f'{k}.npy':sha(out/f'{k}.npy') for k in ['cost','known']},
        code_sha256=sha(Path(__file__)),uses_rgb=False,uses_masks=False,lookup_shift=False,
        step=4,distortion_scale=.1,unknown_cost=float(np.log(2))))
    log(root,'cycle_differential_complete',camera=target,source=source);return out
