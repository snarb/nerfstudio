"""Measure native projection stretch on frozen q50; never alter the lookup.

Singular values describe source pixels per target pixel along the two local
surface directions. No RGB, target masks, or image registration are used.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,native_rows,project,unproject,read,write,sha,log,assert_source


def singular_values(dx,dy):
    trace=np.sum(dx*dx+dy*dy,axis=-1)
    determinant=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    high=np.sqrt(np.maximum(.5*(trace+np.sqrt(np.maximum(trace*trace-4*determinant**2,0))),0))
    low=np.abs(determinant)/np.maximum(high,1e-12)
    return low,high


def projection_differential(uv,depth,alpha,step=2):
    shape=depth.shape;low=np.ones(shape,np.float32);high=low.copy();valid=np.zeros(shape,bool)
    core=np.s_[step:-step,step:-step]
    xp=np.s_[step:-step,2*step:];xm=np.s_[step:-step,:-2*step]
    yp=np.s_[2*step:,step:-step];ym=np.s_[:-2*step,step:-step]
    dx=(uv[xp]-uv[xm])/(2*step);dy=(uv[yp]-uv[ym])/(2*step)
    a,b=singular_values(dx,dy)
    ok=np.isfinite(a)&np.isfinite(b)&(alpha[core]>=.99)&np.isfinite(depth[core])
    for sl in (xp,xm,yp,ym):
        ok&=(alpha[sl]>=.99)&np.isfinite(depth[sl])&(np.abs(depth[sl]-depth[core])<=.005*np.maximum(depth[core],1e-8))
    low[core]=np.where(ok,a,1);high[core]=np.where(ok,b,1);valid[core]=ok
    return low,high,valid


def cache(root,target,source):
    rows=native_rows();assert_source(rows,target,source)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(parent/'receipt.json')
    out=root/'projection_differential'/f'cam{target:03d}'/f'src{source:03d}'
    if (out/'receipt.json').exists():
        saved=read(out/'receipt.json');assert saved['code_sha256']==sha(Path(__file__))
        assert saved['parent_receipt_sha256']==sha(parent/'receipt.json')
        return out
    out.mkdir(parents=True,exist_ok=True);x0,y0,x1,y1=r['box'];g=geometry(root,target)
    shape=(y1-y0,x1-x0)
    arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.uint8 if k=='valid' else np.float32,shape=shape) for k in ['minimum','maximum','valid']}
    # A two-row halo preserves central differences at chunk boundaries.
    for start in range(y0,y1,128):
        stop=min(start+128,y1);lo=max(y0,start-2);hi=min(y1,stop+2)
        yy,xx=np.mgrid[lo:hi,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        z=np.asarray(g['depth_z_q50'][lo:hi,x0:x1]);alpha=np.asarray(g['alpha'][lo:hi,x0:x1])
        uv,_=project(unproject(z,rows[target],xy),rows[source])
        values=projection_differential(uv,z,alpha)
        for key,value in zip(arrays,values):arrays[key][start-y0:stop-y0]=value[start-lo:stop-lo]
    for value in arrays.values():value.flush()
    write(out/'receipt.json',dict(target=target,source=source,box=r['box'],code_sha256=sha(Path(__file__)),parent_receipt_sha256=sha(parent/'receipt.json'),geometry_receipt_sha256=sha(root/'geometry'/f'cam{target:03d}'/'receipt.json'),hashes={k:sha(out/f'{k}.npy') for k in arrays},uses_rgb=False,lookup_shift=False))
    log(root,'projection_differential_complete',camera=target,source=source)
    return out
