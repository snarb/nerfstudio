"""Bounded classical image-registration ablation, without GT or learned models.

Register projected TRAIN appearance to the frozen render at low frequencies.
Reproject native RGB with the corrected lookup and recompute visibility/roundtrip;
this changes texture lookup only, never model geometry or camera calibration.
"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from luster_hybrid_common import (assert_source,geometry,log,native_rows,project,read,
                                  sha,source_path,unproject,write)
from transfer_luster_hybrid import footprint_metrics,gaussian
cv2.setNumThreads(4)


def registration(base,texture,support,max_shift=2.):
    h,w=support.shape
    def gray(rgb):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        return np.rint(np.clip(cv2.resize(gaussian(gray,.8),(w//2,h//2),interpolation=cv2.INTER_AREA)*255,0,255)).astype(np.uint8)
    target=gray(base);source=gray(np.where(support[...,None],texture,base))
    def flow(a,b):
        solver=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        solver.setUseMeanNormalization(True)
        result=solver.calc(a,b,None)
        return cv2.resize(result,(w,h),interpolation=cv2.INTER_LINEAR)*np.array([w/a.shape[1],h/a.shape[0]],np.float32)
    forward=flow(target,source);backward=flow(source,target)
    yy,xx=np.indices((h,w),np.float32);uv=np.stack([xx,yy],-1)+forward
    reverse=cv2.remap(backward,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=100)
    residual=np.linalg.norm(forward+reverse,axis=-1)
    safe=cv2.erode(support.astype(np.uint8),np.ones((17,17),np.uint8))>0
    safe&=(np.linalg.norm(forward,axis=-1)<=max_shift)&(residual<=.5)
    before=np.mean(np.abs(gaussian(texture,2.)-gaussian(base,2.)),axis=-1)
    moved=cv2.remap(texture,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    after=np.mean(np.abs(gaussian(moved,2.)-gaussian(base,2.)),axis=-1)
    safe&=(after+.001<before)
    forward[~safe]=0
    return forward,safe,residual


def variant_root(root):
    out=root/'variants/align2';out.mkdir(parents=True,exist_ok=True)
    for name in ['preflight','geometry','baseline','evaluation','candidates','logs','runtime','code']:
        p=out/name
        if not p.exists():p.symlink_to(root/name,target_is_directory=True)
    (out/'warps').mkdir(exist_ok=True)
    return out


def warp_aligned(root,target,source):
    rows=native_rows();assert_source(rows,target,source)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    original=read(parent/'receipt.json');dest=variant_root(root)/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    if (dest/'receipt.json').exists():
        assert read(dest/'receipt.json')['alignment_code_sha256']==sha(Path(__file__))
        return
    dest.mkdir(parents=True,exist_ok=True)
    row=rows[target];sr=rows[source];g=geometry(root,target);sg=geometry(root,source)
    manifest=read(root/'preflight/manifest.json')
    assert sha(source_path(source))==manifest['source_hashes'][str(source_path(source))]
    photo=np.asarray(Image.open(source_path(source)).convert('RGB'),np.float32)/255
    x0,y0,x1,y1=original['box'];sl=np.s_[y0:y1,x0:x1]
    old={k:np.load(parent/f'{k}.npy',mmap_mode='r') for k in original['arrays']}
    started=time.monotonic()
    support=(old['valid']>0)&(old['alpha']>=.99)&(old['depth_error']<=.005)&(old['roundtrip']<=2.)
    flow,apply,residual=registration(np.asarray(g['rgb'][sl]),np.asarray(old['rgb']),support)
    arrays={}
    for name in original['arrays']:
        arrays[name]=np.lib.format.open_memmap(dest/f'{name}.npy',mode='w+',dtype=old[name].dtype,shape=old[name].shape)
        arrays[name][:]=old[name]
    np.save(dest/'alignment_applied.npy',apply);np.save(dest/'flow.npy',flow)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);local=np.s_[y-y0:stop-y0,:];chosen=apply[local]
        if not chosen.any():continue
        yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        lookup=xy+flow[local]
        # Interpolate only inside the eroded valid support of the frozen surface.
        z=cv2.remap(np.nan_to_num(np.asarray(g['depth_z_q50'])),lookup[...,0],lookup[...,1],cv2.INTER_LINEAR)
        world=unproject(z,row,lookup);uv,_=project(world,sr)
        centre=unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),row,xy);_,expected=project(centre,sr)
        rgb,valid,alpha,error,sd,lo,hi=footprint_metrics(photo,sg,uv,expected,None,None,None,None,None)
        back,_=project(unproject(sd,sr,uv),row)
        b0,_=project(unproject(lo,sr,uv),row);b1,_=project(unproject(hi,sr,uv),row)
        rt=np.linalg.norm(back-xy,axis=-1)
        uncertainty=np.maximum(old['uncertainty'][local],.5*np.linalg.norm(b1-b0,axis=-1))
        valid&=np.isfinite(rt)&np.isfinite(uncertainty)
        values=dict(rgb=rgb,valid=valid,alpha=np.minimum(alpha,old['alpha'][local]),
                    depth_error=error,roundtrip=rt,uncertainty=uncertainty)
        for name,value in values.items():
            value=np.nan_to_num(value,nan=0 if name=='rgb' else 1e6,posinf=1e6,neginf=1e6)
            arrays[name][local][chosen]=value[chosen]
    for a in arrays.values():a.flush()
    write(dest/'receipt.json',{**original,'registration':'classical_DIS_low_frequency_to_frozen_RGB',
          'alignment_code_sha256':sha(Path(__file__)),'max_lookup_shift_native_pixels':2.,
          'parent_warp_receipt_sha256':sha(parent/'receipt.json'),
          'aligned_pixels':int(apply.sum()),'uses_target_gt':False,'seconds':time.monotonic()-started})
    log(root,'aligned_warp_complete',camera=target,source=source,pixels=int(apply.sum()),seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cameras',type=int,nargs='+',default=[7,12,95,150]);p.add_argument('--count',type=int,default=8)
    a=p.parse_args();selected=read(a.root/'preflight/sources.json')['sources']
    for c in a.cameras:
        for s in selected[str(c)][:a.count]:warp_aligned(a.root,c,s)
