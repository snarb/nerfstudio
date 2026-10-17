"""Photo-gradient sensitivity to curvature of an unchanged geometric projection.

This is a scalar diagnostic, not a corrected lookup, flow or geometry fit.
Curvature parallel to a native isophote has no first-order color effect.
"""
import cv2
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source,geometry,native_rows,project,read,sha,source_path,unproject
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian


def sample_vector(field,uv):
    h,w=field.shape[:2];u=uv[...,0];v=uv[...,1]
    valid=np.isfinite(uv).all(-1)&(u>=0)&(v>=0)&(u<=w-1)&(v<=h-1)
    u=np.where(valid,u,0);v=np.where(valid,v,0)
    x=np.floor(u).astype(np.int32);y=np.floor(v).astype(np.int32);fx=u-x;fy=v-y
    value=np.zeros((*u.shape,field.shape[-1]),np.float32)
    for dx,dy,weight in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        value+=weight[...,None]*field[(y+dy).clip(0,h-1),(x+dx).clip(0,w-1)]
    return value,valid


def sensitivity(uv,photo_gradient,valid):
    value=np.zeros(valid.shape,np.float32);known=np.zeros(valid.shape,bool)
    if min(valid.shape)<3:return value,known
    finite=np.isfinite(uv).all(-1)&np.isfinite(photo_gradient).all(-1)
    ok=cv2.erode((valid&finite).astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    center=uv[1:-1,1:-1]
    hxx=uv[1:-1,2:]+uv[1:-1,:-2]-2*center
    hyy=uv[2:,1:-1]+uv[:-2,1:-1]-2*center
    hxy=(uv[2:,2:]-uv[2:,:-2]-uv[:-2,2:]+uv[:-2,:-2])*.25
    grad=photo_gradient[1:-1,1:-1]
    a=np.sum(grad*hxx,-1);b=np.sum(grad*hyy,-1);c=np.sum(grad*hxy,-1)
    value[1:-1,1:-1]=np.where(ok[1:-1,1:-1],np.sqrt(a*a+b*b+2*c*c),0)
    known[:]=ok;return value,known


def feature(root,target,source,config):
    rows=native_rows();assert_source(rows,target,source)
    d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(d/'receipt.json')
    assert sha(source_path(source))==r['source_sha256']
    photo=np.asarray(Image.open(source_path(source)).convert('RGB'),np.float32)/255
    gray=gaussian(photo@np.array([.2126,.7152,.0722],np.float32),1.)
    gradient=np.stack([cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8),cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8)],-1)
    a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays'] if k!='rgb'}
    support=geometric_support(a,config);x0,y0,x1,y1=r['box'];g=geometry(root,target)
    result=np.zeros(support.shape,np.float32);known=np.zeros(support.shape,bool)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);lo=max(y0,y-1);hi=min(y1,stop+1)
        yy,xx=np.mgrid[lo:hi,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][lo:hi,x0:x1]),rows[target],xy),rows[source])
        grad,inside=sample_vector(gradient,uv);sl=np.s_[lo-y0:hi-y0]
        v,k=sensitivity(uv,grad,support[sl]&inside)
        result[y-y0:stop-y0]=v[y-lo:stop-lo];known[y-y0:stop-y0]=k[y-lo:stop-lo]
    return result,known,r['box']
