"""Normalize source depth disagreement by frozen termination-depth intervals.

This supplies a source-ranking cost only. Target surface points, source RGB
coordinates, calibration and all frozen field/grids remain unchanged.
"""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,native_rows,project,read,sha,unproject,write,log


def interval_cost(expected,medians,lows,highs,taps,target_spread,pixel_floor):
    """Worst active footprint tap, with asymmetric source and target spreads.

    Footprint depth variation provides a conservative slope tolerance. The
    resulting log1p score is an engineering cost, not a calibrated probability.
    Arrays medians/lows/highs/taps have the footprint-tap axis first.
    """
    good=np.isfinite(medians)&np.isfinite(lows)&np.isfinite(highs)&(highs>=medians)&(medians>=lows)
    finite=np.all(~taps|good,axis=0)&np.any(taps,axis=0)&np.isfinite(expected)&(expected>0)
    zmin=np.min(np.where(taps&good,medians,np.inf),axis=0)
    zmax=np.max(np.where(taps&good,medians,-np.inf),axis=0)
    slope=np.where(finite,.5*np.maximum(zmax-zmin,0),0)
    floor=np.maximum(pixel_floor,slope)
    deviation=expected[None]-medians
    spread=np.maximum(np.where(deviation>=0,highs-medians,medians-lows),0)
    scale=np.sqrt(spread**2+target_spread[None]**2+floor[None]**2)
    normalized=np.abs(deviation)/np.maximum(scale,1e-8)
    worst=np.max(np.where(taps&good,normalized,0),axis=0)
    return np.where(finite,np.log1p(worst),1e6).astype(np.float32)


def quality(root,target,source):
    out=root/'depth_interval_quality'/f'cam{target:03d}'/f'src{source:03d}';out.mkdir(parents=True,exist_ok=True)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    inputs={str(p):sha(p) for p in [parent,root/'geometry'/f'cam{target:03d}'/'receipt.json',root/'geometry'/f'cam{source:03d}'/'receipt.json',Path(__file__).with_name('luster_hybrid_common.py')]}
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['inputs']==inputs
        assert r['hashes']['cost.npy']==sha(out/'cost.npy');return out
    rows=native_rows();tr=rows[target];sr=rows[source];g=geometry(root,target);sg=geometry(root,source)
    box=read(parent)['box'];x0,y0,x1,y1=box;result=np.empty((y1-y0,x1-x0),np.float32)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);sl=np.s_[y:stop,x0:x1];yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,expected=project(unproject(np.asarray(g['depth_z_q50'][sl]),tr,xy),sr)
        _,tlo=project(unproject(np.asarray(g['depth_z_q10'][sl]),tr,xy),sr)
        _,thi=project(unproject(np.asarray(g['depth_z_q90'][sl]),tr,xy),sr)
        finite=np.isfinite(uv).all(-1);safe=np.where(finite[...,None],uv,0)
        ix=np.floor(safe[...,0]).astype(np.int32);iy=np.floor(safe[...,1]).astype(np.int32);fx=safe[...,0]-ix;fy=safe[...,1]-iy
        zs=[];los=[];his=[];taps=[]
        for dy in (0,1):
            for dx in (0,1):
                weight=(fx if dx else 1-fx)*(fy if dy else 1-fy)
                sx=np.clip(ix+dx,0,sr['w']-1);sy=np.clip(iy+dy,0,sr['h']-1)
                zs.append(sg['depth_z_q50'][sy,sx]);los.append(sg['depth_z_q10'][sy,sx]);his.append(sg['depth_z_q90'][sy,sx]);taps.append(weight>1e-7)
        value=interval_cost(expected,np.stack(zs),np.stack(los),np.stack(his),np.stack(taps),
                            .5*np.abs(thi-tlo),.5*np.maximum(expected,0)/np.sqrt(sr['fl_x']*sr['fl_y']))
        inside=finite&(uv[...,0]>=0)&(uv[...,1]>=0)&(uv[...,0]<=sr['w']-1)&(uv[...,1]<=sr['h']-1)
        result[y-y0:stop-y0]=np.where(inside&np.isfinite(value),value,1e6)
    np.save(out/'cost.npy',result)
    write(out/'receipt.json',dict(target=target,source=source,box=box,inputs=inputs,hashes={'cost.npy':sha(out/'cost.npy')},
        code_sha256=sha(Path(__file__)),uses_target_rgb=False,uses_masks=False,lookup_shift=False,
        source_interval='q10/q50/q90 asymmetric spread',target_interval='half q10-q90 source-z range',
        footprint_tolerance='max(half native pixel world size, half active-tap q50 range)',calibrated_probability=False))
    log(root,'depth_interval_quality_cached',camera=target,source=source);return out
