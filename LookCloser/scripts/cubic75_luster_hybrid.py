"""Native one-photo cubic a=-.75 sampling at unchanged LookCloser projection centers.

All nonzero 4x4 interpolation taps must pass source visibility. Clamp to the
central bilinear footprint's RGB range to prevent interpolation overshoot.
"""
import argparse,time
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source,geometry,log,native_rows,project,read,sha,source_path,unproject,write


def kernel(x):
    x=np.abs(x);a=-.75
    return np.where(x<1,(a+2)*x**3-(a+3)*x**2+1,np.where(x<2,a*x**3-5*a*x**2+8*a*x-4*a,0)).astype(np.float32)


def sample(photo,source,uv,expected_z):
    h,w=photo.shape[:2];u,v=uv[...,0],uv[...,1]
    good=np.isfinite(uv).all(-1)&(expected_z>0)&np.isfinite(expected_z)&(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    u=np.where(good,u,0);v=np.where(good,v,0);x=np.floor(u).astype(np.int32);y=np.floor(v).astype(np.int32)
    result=np.zeros((*u.shape,3),np.float32);mass=np.zeros(u.shape,np.float32);alpha=np.ones(u.shape,np.float32);err=np.zeros(u.shape,np.float32)
    lower=np.full_like(result,np.inf);upper=np.full_like(result,-np.inf)
    for dy in [-1,0,1,2]:
        for dx in [-1,0,1,2]:
            weight=kernel(u-(x+dx))*kernel(v-(y+dy));tap=np.abs(weight)>1e-7
            xx=x+dx;yy=y+dy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            xx=xx.clip(0,w-1);yy=yy.clip(0,h-1);z=source['depth_z_q50'][yy,xx]
            good&=~tap|(inside&np.isfinite(z)&np.isfinite(source['depth_z_q10'][yy,xx])&np.isfinite(source['depth_z_q90'][yy,xx]))
            alpha=np.minimum(alpha,np.where(tap,source['alpha'][yy,xx],1))
            err=np.maximum(err,np.where(tap,np.abs(z-expected_z)/np.maximum(expected_z,1e-8),0))
            color=photo[yy,xx];result+=weight[...,None]*color;mass+=weight
            if dy in [0,1] and dx in [0,1]:lower=np.minimum(lower,color);upper=np.maximum(upper,color)
    result/=np.maximum(mass[...,None],1e-8)
    result=np.minimum(np.maximum(result,lower),upper).clip(0,1)
    return result,good,alpha,err


def variant_root(root):
    out=root/'variants/cubic75';out.mkdir(parents=True,exist_ok=True)
    for key in ['preflight','geometry','baseline','evaluation','candidates','logs','runtime','code']:
        path=out/key
        if not path.exists():path.symlink_to(root/key,target_is_directory=True)
    return out


def warp(root,target,source):
    rows=native_rows();assert_source(rows,target,source);variant=variant_root(root)
    parent=root/'warps'/f'cam{target:03d}'/f'src{source:03d}';pr=read(parent/'receipt.json');out=variant/'warps'/f'cam{target:03d}'/f'src{source:03d}'
    if (out/'receipt.json').exists():
        old=read(out/'receipt.json');assert old['cubic_code_sha256']==sha(Path(__file__)) and old['parent_warp_receipt_sha256']==sha(parent/'receipt.json');return
    assert sha(source_path(source))==pr['source_sha256']
    out.mkdir(parents=True,exist_ok=True);x0,y0,x1,y1=pr['box'];shape=(y1-y0,x1-x0)
    arrays={k:np.lib.format.open_memmap(out/f'{k}.npy',mode='w+',dtype=np.uint8 if k=='valid' else np.float32,shape=(*shape,3) if k=='rgb' else shape) for k in ['rgb','valid','alpha','depth_error']}
    for key in pr['arrays']:
        if key not in arrays:(out/f'{key}.npy').symlink_to(parent/f'{key}.npy')
    g=geometry(root,target);sg=geometry(root,source);photo=np.asarray(Image.open(source_path(source)).convert('RGB'),np.float32)/255
    original={k:np.load(parent/f'{k}.npy',mmap_mode='r') for k in ['valid','alpha','depth_error']};started=time.monotonic()
    for y in range(y0,y1,128):
        stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32);z=g['depth_z_q50'][y:stop,x0:x1]
        uv,zs=project(unproject(z,rows[target],xy),rows[source]);rgb,valid,alpha,err=sample(photo,sg,uv,zs);sl=np.s_[y-y0:stop-y0]
        arrays['rgb'][sl]=np.nan_to_num(rgb);arrays['valid'][sl]=valid&(original['valid'][sl]>0)
        arrays['alpha'][sl]=np.minimum(alpha,original['alpha'][sl]);arrays['depth_error'][sl]=np.nan_to_num(np.maximum(err,original['depth_error'][sl]),nan=1e6)
    for value in arrays.values():value.flush()
    write(out/'receipt.json',{**pr,'resampling':'native_keys_cubic_clamped','cubic_code_sha256':sha(Path(__file__)),'parent_warp_receipt_sha256':sha(parent/'receipt.json'),'lookup_shift':False,'tap_count':16,'cubic_parameter':-.75,'seconds':time.monotonic()-started,'array_hashes':{k:sha(out/f'{k}.npy') for k in pr['arrays']}})
    log(root,'cubic75_warp_complete',camera=target,source=source,seconds=time.monotonic()-started)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--count',type=int,default=16);a=p.parse_args()
    for c,sources in read(a.root/'preflight/sources.json')['sources'].items():
        for source in sources[:a.count]:warp(a.root,int(c),source)
