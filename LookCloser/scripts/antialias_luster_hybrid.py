"""Area-footprint ablation using native TRAIN photos and frozen geometric warps.

Adds finite pixel-footprint integration when a source is minified. No target RGB,
ROI or mask is read. All nonzero bilinear taps in the footprint must pass the
same visibility tests as the centre. Original point-sampled caches stay intact.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from luster_hybrid_common import (assert_source,geometry,log,native_rows,project,read,
                                  sha,source_path,unproject,write)
from transfer_luster_hybrid import footprint_metrics


def footprint_transform(jacobian):
    """Remove the native pixel footprint; integrate only the excess minification.

    Returns a source-coordinate integration map and the corresponding target
    offsets. Magnification and unit scaling preserve the original bilinear sample.
    """
    u,s,vh=np.linalg.svd(jacobian)
    reduced=np.sqrt(np.maximum(s*s-1,0))
    source=(u*reduced[...,None,:])@vh
    target=(vh.swapaxes(-1,-2)*(reduced/np.maximum(s,1e-8))[...,None,:])@vh
    return source,target,s


def variant_root(root):
    out=root/'variants/area4';out.mkdir(parents=True,exist_ok=True)
    for name in ['preflight','geometry','baseline','evaluation','candidates','logs','runtime','code']:
        p=out/name
        if not p.exists():p.symlink_to(root/name,target_is_directory=True)
    (out/'warps').mkdir(exist_ok=True)
    return out


def warp_area(root,target,source,count=4):
    rows=native_rows();assert_source(rows,target,source)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    original=read(parent/'receipt.json');dest=variant_root(root)/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    if (dest/'receipt.json').exists():
        saved=read(dest/'receipt.json')
        assert saved['area_code_sha256']==sha(Path(__file__)) and saved['quadrature']==count
        return
    dest.mkdir(parents=True,exist_ok=True)
    row=rows[target];sr=rows[source];g=geometry(root,target);sg=geometry(root,source)
    manifest=read(root/'preflight/manifest.json')
    assert sha(source_path(source))==manifest['source_hashes'][str(source_path(source))]
    photo=np.asarray(Image.open(source_path(source)).convert('RGB'),np.float32)/255
    x0,y0,x1,y1=original['box'];shape=(y1-y0,x1-x0)
    revised=['rgb','alpha','depth_error','valid']
    arrays={}
    for name in original['arrays']:
        if name in revised:
            arrays[name]=np.lib.format.open_memmap(dest/f'{name}.npy',mode='w+',
                       dtype=np.uint8 if name=='valid' else np.float32,
                       shape=(*shape,3) if name=='rgb' else shape)
        elif not (dest/f'{name}.npy').exists():(dest/f'{name}.npy').symlink_to(parent/f'{name}.npy')
    offsets=(np.arange(count,dtype=np.float32)+.5)/count-.5
    started=time.monotonic();filtered=total=0
    for y in range(y0,y1,64):
        stop=min(y+64,y1);ay=max(0,y-1);by=min(row['h'],stop+1)
        ax=max(0,x0-1);bx=min(row['w'],x1+1)
        yy,xx=np.mgrid[ay:by,ax:bx];xy=np.stack([xx,yy],-1).astype(np.float32)
        z=np.asarray(g['depth_z_q50'][ay:by,ax:bx]);world=unproject(z,row,xy);uv,zs=project(world,sr)
        du_y,du_x=np.gradient(uv,axis=(0,1));dz_y,dz_x=np.gradient(zs)
        cut=np.s_[y-ay:stop-ay,x0-ax:x1-ax]
        uv=uv[cut];zs=zs[cut];xy=xy[cut];world=world[cut]
        jac=np.stack([du_x[cut],du_y[cut]],-1)
        finite=np.isfinite(jac).all(axis=(-1,-2))&np.isfinite(zs)
        clean=np.where(finite[...,None,None],jac,np.eye(2,dtype=np.float32))
        matrix,back,singular=footprint_transform(clean)
        finite&=(singular[...,0]<=8)&(singular[...,0]/np.maximum(singular[...,1],1e-8)<=8)
        color=np.zeros((*uv.shape[:2],3),np.float32)
        local=np.s_[y-y0:stop-y0,:]
        alpha=np.asarray(np.load(parent/'alpha.npy',mmap_mode='r')[local]).copy()
        error=np.asarray(np.load(parent/'depth_error.npy',mmap_mode='r')[local]).copy()
        valid=np.asarray(np.load(parent/'valid.npy',mmap_mode='r')[local],bool)&finite
        lo=g['depth_z_q10'][y:stop,x0:x1];hi=g['depth_z_q90'][y:stop,x0:x1]
        for dy in offsets:
            for dx in offsets:
                offset=np.array([dx,dy],np.float32)
                delta=np.einsum('...ij,j->...i',matrix,offset)
                target_delta=np.einsum('...ij,j->...i',back,offset)
                expected=zs+dz_x[cut]*target_delta[...,0]+dz_y[cut]*target_delta[...,1]
                rgb,ok,a,e,*_=footprint_metrics(photo,sg,uv+delta,expected,world,row,xy,lo,hi)
                color+=rgb/(count*count);valid&=ok;alpha=np.minimum(alpha,a);error=np.maximum(error,e)
        arrays['rgb'][local]=np.nan_to_num(color)
        arrays['alpha'][local]=alpha;arrays['depth_error'][local]=np.nan_to_num(error,nan=1e6)
        arrays['valid'][local]=valid
        filtered+=int((valid&(singular[...,0]>1.05)).sum());total+=int(valid.sum())
    for a in arrays.values():a.flush()
    write(dest/'receipt.json',{**original,'resampling':'area_footprint_excess_minification',
          'quadrature':count,'area_code_sha256':sha(Path(__file__)),
          'parent_warp_receipt_sha256':sha(parent/'receipt.json'),
          'filtered_valid_pixels':filtered,'valid_pixels':total,'seconds':time.monotonic()-started})
    log(root,'area_warp_complete',camera=target,source=source,seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cameras',type=int,nargs='+',default=[7,12,95,150]);p.add_argument('--count',type=int,default=8)
    a=p.parse_args();selected=read(a.root/'preflight/sources.json')['sources']
    for c in a.cameras:
        for s in selected[str(c)][:a.count]:warp_area(a.root,c,s)
