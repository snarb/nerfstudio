"""Alternative projection using frozen termination quantile or mode depths.

Only the surface statistic read from the unchanged volume changes. Model density, weights,
calibration, grids and baseline RGB remain frozen; no photometric fitting.
"""
from pathlib import Path
import argparse,time
import numpy as np
from PIL import Image
from luster_hybrid_common import (assert_source,geometry,log,native_rows,project,read,sha,source_path,unproject,write)
from transfer_luster_hybrid import footprint_metrics,source_score

def surface_geometry(root,cid):
    g=geometry(root,cid);kind=root.name.removeprefix('probe_')
    assert kind in ['density_peak','density_center']
    out=root/'density_probe'/f'cam{cid:03d}';r=read(out/'receipt.json')
    assert r['parent_geometry_sha256']==sha(root/'geometry'/f'cam{cid:03d}'/'receipt.json')
    assert sha(out/f'{kind}.npy')==r['hashes'][kind]
    return {**g,'depth_z_q50':np.load(out/f'{kind}.npy',mmap_mode='r')}

def variant_root(root,kind):
    assert kind in ['density_peak','density_center']
    out=root/'variants'/f'probe_{kind}';out.mkdir(parents=True,exist_ok=True)
    for key in ['preflight','geometry','baseline','evaluation','candidates','logs','runtime','code','density_probe']:
        d=out/key
        if not d.exists():d.symlink_to(root/key,target_is_directory=True)
    return out

def warp_fingerprint():return sha(Path(__file__))

def warp(root, target, source_id):
    rows=native_rows();assert_source(rows,target,source_id)
    row=rows[target];sr=rows[source_id];g=surface_geometry(root,target);sg=surface_geometry(root,source_id)
    dest=root/'warps'/f'cam{target:03d}'/f'src{source_id:03d}'
    if (dest/'receipt.json').exists():
        receipt=read(dest/'receipt.json')
        assert receipt['source_sha256']==sha(source_path(source_id))
        assert receipt['warp_code_sha256']==warp_fingerprint()
        return
    yy,xx=np.where(np.isfinite(g['depth_z_q50'])&(g['alpha']>=.95))
    # Prediction-derived crop only; no evaluation ROI is used here.
    box=[max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(row['w'],int(xx.max())+33),min(row['h'],int(yy.max())+33)]
    x0,y0,x1,y1=box;shape=(y1-y0,x1-x0)
    dest.mkdir(parents=True,exist_ok=True)
    names=['rgb','alpha','depth_error','roundtrip','uncertainty','score','resolution_ratio','valid']
    arrays={k:np.lib.format.open_memmap(dest/f'{k}.npy',mode='w+',
            dtype=np.uint8 if k=='valid' else np.float32,
            shape=(*shape,3) if k=='rgb' else shape) for k in names}
    manifest=read(root/'preflight/manifest.json')
    assert sha(source_path(source_id))==manifest['source_hashes'][str(source_path(source_id))]
    source_rgb=np.asarray(Image.open(source_path(source_id)).convert('RGB'),np.float32)/255
    cs=np.asarray(sr['transform_matrix'],np.float32)[:3,3]
    started=time.monotonic()
    for y in range(y0,y1,128):
        stop=min(y+128,y1);sl=np.s_[y:stop,x0:x1]
        yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        z=np.asarray(g['depth_z_q50'][sl]);lo=np.asarray(g['depth_z_q10'][sl]);hi=np.asarray(g['depth_z_q90'][sl])
        world=unproject(z,row,xy);uv,zs=project(world,sr)
        colors,valid,alpha,err,sd,slo,shi=footprint_metrics(source_rgb,sg,uv,zs,world,row,xy,lo,hi)
        back=cs+(world-cs)*(sd/np.maximum(zs,1e-8))[...,None]
        buv,_=project(back,row);rt=np.linalg.norm(buv-xy,axis=-1)
        # Project source and target depth intervals into the other image; express both in target pixels.
        b0,_=project(cs+(world-cs)*(slo/np.maximum(zs,1e-8))[...,None],row)
        b1,_=project(cs+(world-cs)*(shi/np.maximum(zs,1e-8))[...,None],row)
        t0,_=project(unproject(lo,row,xy),sr);t1,_=project(unproject(hi,row,xy),sr)
        score,ratio=source_score(world,row,sr)
        uncertainty=np.maximum(np.linalg.norm(b1-b0,axis=-1),np.linalg.norm(t1-t0,axis=-1)/np.maximum(ratio,1e-8))*.5
        valid&=np.isfinite(z)&np.isfinite(rt)&np.isfinite(uncertainty)&np.isfinite(score)
        values=dict(rgb=colors,valid=valid,alpha=np.minimum(alpha,g['alpha'][sl]),depth_error=err,
                    roundtrip=rt,uncertainty=uncertainty,score=score,resolution_ratio=ratio)
        for k,v in values.items():arrays[k][y-y0:stop-y0]=np.nan_to_num(v,nan=0 if k=='rgb' else 1e6,posinf=1e6,neginf=1e6)
    for v in arrays.values():v.flush()
    write(dest/'receipt.json',dict(target=target,source=source_id,box=box,arrays=names,
          source_sha256=sha(source_path(source_id)),code_sha256=sha(Path(__file__)),warp_code_sha256=warp_fingerprint(),
          target_geometry_sha256=sha(root/'geometry'/f'cam{target:03d}'/'receipt.json'),
          source_geometry_sha256=sha(root/'geometry'/f'cam{source_id:03d}'/'receipt.json'),
          seconds=time.monotonic()-started,uses_target_rgb=False,uses_masks=False,
          surface='frozen intrinsic density '+root.name.removeprefix('probe_')+' camera-z',
          density_probe_receipts={str(root/'density_probe'/f'cam{c:03d}'/'receipt.json'):sha(root/'density_probe'/f'cam{c:03d}'/'receipt.json') for c in [target,source_id]}))
    log(root,'warp_complete',camera=target,source=source_id,seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--count',type=int,default=4);p.add_argument('--kind',choices=['density_peak','density_center'],required=True);a=p.parse_args()
    root=variant_root(a.root,a.kind);sources=read(root/'preflight/sources.json')['sources']
    for c in [7,12,95,150]:
        for source in sources[str(c)][:a.count]:warp(root,c,source)
