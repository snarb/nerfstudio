"""Conservative TRAIN-photo foreground corroboration from measured empty plates.

Useful where a frozen density surface incorrectly fills lace holes. Background
photometric fitting is used only for source admission, not for altering textures.
No target photograph, evaluation mask or semantic segmentation model is used.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from luster_hybrid_common import (assert_source,geometry,log,native_rows,project,read,
                                  sha,source_path,unproject,write)


def foreground_evidence(photo,plate,empty):
    # Sparse robust background fit; both images belong to the TRAIN source.
    yy,xx=np.where(empty[::8,::8]);yy*=8;xx*=8
    if len(yy)<1000:raise ValueError('Insufficient measured background support')
    coefficients=[];fitted=np.empty_like(plate)
    for channel in range(3):
        x=plate[yy,xx,channel];y=photo[yy,xx,channel];a=np.stack([x,np.ones_like(x)],-1)
        weight=np.ones_like(x)
        for _ in range(4):
            coefficient=np.linalg.lstsq(a*weight[:,None],y*weight,rcond=None)[0]
            coefficient[0]=np.clip(coefficient[0],.8,1.2);coefficient[1]=np.clip(coefficient[1],-.1,.1)
            error=np.abs(a@coefficient-y);weight=np.sqrt(np.minimum(1,.025/np.maximum(error,1e-8)))
        coefficients.append(coefficient.tolist());fitted[...,channel]=plate[...,channel]*coefficient[0]+coefficient[1]
    difference=np.max(np.abs(photo-fitted),axis=-1)
    background_errors=difference[yy,xx]
    median=float(np.median(background_errors));mad=float(np.median(np.abs(background_errors-median)))
    # Robust noise bound ignores changed props and missing wisps in the empty set.
    threshold=max(.04,median+6*1.4826*mad)
    return difference>threshold,dict(coefficients=coefficients,threshold=threshold,background_samples=len(yy),median=median,mad=mad)


def source_mask(root,camera):
    row=native_rows()[camera]
    assert row['train'] and camera not in (12,95,150)
    dest=root/'source_foreground_noise'/f'cam{camera:03d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'receipt.json').exists():
        assert read(dest/'receipt.json')['code_sha256']==sha(Path(__file__))
        return np.load(dest/'mask.npy',mmap_mode='r')
    receipt=read(root/'geometry'/f'cam{camera:03d}'/'receipt.json')['background']
    assert receipt and sha(receipt['path'])==receipt['sha256']
    manifest=read(root/'preflight/manifest.json');photo_path=source_path(camera)
    assert sha(photo_path)==manifest['source_hashes'][str(photo_path)]
    photo=np.asarray(Image.open(photo_path).convert('RGB'),np.float32)/255
    plate=np.asarray(Image.open(receipt['path']).convert('RGB').resize((row['w'],row['h']),Image.Resampling.LANCZOS),np.float32)/255
    mask,fit=foreground_evidence(photo,plate,geometry(root,camera)['alpha']<.01)
    np.save(dest/'mask.npy',mask)
    write(dest/'receipt.json',dict(camera=camera,fit=fit,source_sha256=sha(photo_path),
        background=receipt,code_sha256=sha(Path(__file__)),uses_target_gt=False))
    return mask


def variant_root(root):
    out=root/'variants/foreground_noise';out.mkdir(parents=True,exist_ok=True)
    for name in ['preflight','geometry','baseline','evaluation','candidates','logs','runtime','code']:
        p=out/name
        if not p.exists():p.symlink_to(root/name,target_is_directory=True)
    (out/'warps').mkdir(exist_ok=True)
    return out


def guard_warp(root,target,source):
    rows=native_rows();assert_source(rows,target,source)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    original=read(parent/'receipt.json');dest=variant_root(root)/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    if (dest/'receipt.json').exists():
        assert read(dest/'receipt.json')['foreground_code_sha256']==sha(Path(__file__))
        return
    started=time.monotonic();mask=source_mask(root,source);h,w=mask.shape
    row=rows[target];sr=rows[source];g=geometry(root,target)
    x0,y0,x1,y1=original['box'];dest.mkdir(parents=True,exist_ok=True)
    for name in original['arrays']:
        if name!='valid' and not (dest/f'{name}.npy').exists():(dest/f'{name}.npy').symlink_to(parent/f'{name}.npy')
    old=np.load(parent/'valid.npy',mmap_mode='r')
    valid=np.lib.format.open_memmap(dest/'valid.npy',mode='w+',dtype=np.uint8,shape=old.shape)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),row,xy),sr)
        finite=np.isfinite(uv).all(-1);uv=np.nan_to_num(uv)
        x=np.floor(uv[...,0]).astype(np.int32);v=np.floor(uv[...,1]).astype(np.int32)
        fx=uv[...,0]-x;fy=uv[...,1]-v;keep=finite&(x>=0)&(v>=0)&(x<w-1)&(v<h-1)
        for dy in (0,1):
            for dx in (0,1):
                tap=(fx if dx else 1-fx)*(fy if dy else 1-fy)>1e-7
                keep&=~tap|mask[np.clip(v+dy,0,h-1),np.clip(x+dx,0,w-1)]
        valid[y-y0:stop-y0]=(old[y-y0:stop-y0]>0)&keep
    valid.flush()
    write(dest/'receipt.json',{**original,'foreground_code_sha256':sha(Path(__file__)),
          'source_foreground_receipt_sha256':sha(root/'source_foreground_noise'/f'cam{source:03d}'/'receipt.json'),
          'parent_warp_receipt_sha256':sha(parent/'receipt.json'),
          'rejected_pixels':int(((old>0)&(valid==0)).sum()),'seconds':time.monotonic()-started})
    log(root,'foreground_guard_complete',camera=target,source=source,seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cameras',type=int,nargs='+',default=[7,12,95,150]);p.add_argument('--count',type=int,default=8)
    a=p.parse_args();selected=read(a.root/'preflight/sources.json')['sources']
    for c in a.cameras:
        for s in selected[str(c)][:a.count]:guard_warp(a.root,c,s)
