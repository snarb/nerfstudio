"""Projection differential and incidence derived solely from frozen LookCloser depth."""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,native_rows,project,unproject,read,write,sha,log

def differential_quality(world,uv,target_center,source_center):
    wy,wx=np.gradient(world,axis=(0,1));uy,ux=np.gradient(uv,axis=(0,1))
    normal=np.cross(wx,wy);length=np.linalg.norm(normal,axis=-1)
    normal=normal/np.maximum(length[...,None],1e-15)
    target=target_center-world;source=source_center-world
    target/=np.maximum(np.linalg.norm(target,axis=-1)[...,None],1e-12)
    source/=np.maximum(np.linalg.norm(source,axis=-1)[...,None],1e-12)
    orient=np.where(np.sum(normal*target,axis=-1)>=0,1.,-1.)
    normal*=orient[...,None]
    cosine=np.sum(normal*source,axis=-1)
    jac=np.stack([ux,uy],-1);valid=np.isfinite(jac).all(axis=(-1,-2))&np.isfinite(cosine)&(length>1e-15)
    clean=np.where(valid[...,None,None],jac,np.eye(2,dtype=np.float32))
    singular=np.linalg.svd(clean,compute_uv=False)
    return dict(min_axis=np.where(valid,singular[...,1],0),max_axis=np.where(valid,singular[...,0],1e6),incidence=np.where(valid,cosine,-1),valid=valid.astype(np.uint8))

def quality(root,target,source):
    d=root/'surface_quality'/f'cam{target:03d}'/f'src{source:03d}';parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json';pr=read(parent)
    if (d/'receipt.json').exists():
        r=read(d/'receipt.json');assert r['code_sha256']==sha(Path(__file__)) and r['parent_receipt_sha256']==sha(parent)
        return d
    d.mkdir(parents=True,exist_ok=True);rows=native_rows();row=rows[target];sr=rows[source];g=geometry(root,target);x0,y0,x1,y1=pr['box'];shape=(y1-y0,x1-x0)
    arrays={key:np.lib.format.open_memmap(d/f'{key}.npy',mode='w+',dtype=np.uint8 if key=='valid' else np.float32,shape=shape) for key in ['min_axis','max_axis','incidence','valid']}
    ct=np.asarray(row['transform_matrix'],np.float32)[:3,3];cs=np.asarray(sr['transform_matrix'],np.float32)[:3,3]
    for y in range(y0,y1,128):
        end=min(y+128,y1);ay=max(0,y-1);by=min(row['h'],end+1);ax=max(0,x0-1);bx=min(row['w'],x1+1)
        yy,xx=np.mgrid[ay:by,ax:bx];xy=np.stack([xx,yy],-1).astype(np.float32);world=unproject(np.asarray(g['depth_z_q50'][ay:by,ax:bx]),row,xy);uv,_=project(world,sr)
        values=differential_quality(world,uv,ct,cs);cut=np.s_[y-ay:end-ay,x0-ax:x1-ax]
        for key,value in values.items():arrays[key][y-y0:end-y0]=value[cut]
    for value in arrays.values():value.flush()
    write(d/'receipt.json',dict(target=target,source=source,box=pr['box'],code_sha256=sha(Path(__file__)),parent_receipt_sha256=sha(parent),hashes={key:sha(d/f'{key}.npy') for key in arrays},uses_target_rgb=False,geometry_modified=False))
    log(root,'surface_quality_cached',camera=target,source=source)
    return d
