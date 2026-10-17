"""Scalar local curvature of the unchanged frozen geometric projection.

No RGB, optical flow, source-coordinate fitting or geometry changes. This is
currently a diagnostic feature; it does not change any prediction artifact.
"""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import assert_source,geometry,native_rows,project,read,unproject
from supported_luster_hybrid import geometric_support


def curvature(uv,valid,ratio):
    value=np.zeros(valid.shape,np.float32);known=np.zeros(valid.shape,bool)
    if min(valid.shape)<3:return value,known
    finite=np.isfinite(uv).all(-1)&np.isfinite(ratio)&(ratio>0)
    ok=cv2.erode((valid&finite).astype(np.uint8),np.ones((3,3),np.uint8),
                 borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    center=uv[1:-1,1:-1]
    xx=uv[1:-1,2:]+uv[1:-1,:-2]-2*center
    yy=uv[2:,1:-1]+uv[:-2,1:-1]-2*center
    xy=(uv[2:,2:]-uv[2:,:-2]-uv[:-2,2:]+uv[:-2,:-2])*.25
    norm=np.sqrt(np.sum(xx*xx+yy*yy+2*xy*xy,axis=-1))
    value[1:-1,1:-1]=np.where(ok[1:-1,1:-1],norm/np.maximum(ratio[1:-1,1:-1],1e-8),0)
    known[:]=ok
    return value,known


def feature(root,target,source,config):
    rows=native_rows();assert_source(rows,target,source)
    d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(d/'receipt.json')
    a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays'] if k!='rgb'}
    support=geometric_support(a,config);x0,y0,x1,y1=r['box'];g=geometry(root,target)
    result=np.zeros(support.shape,np.float32);known=np.zeros(support.shape,bool)
    for y in range(y0,y1,128):
        stop=min(y+128,y1);lo=max(y0,y-1);hi=min(y1,stop+1)
        yy,xx=np.mgrid[lo:hi,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        uv,_=project(unproject(np.asarray(g['depth_z_q50'][lo:hi,x0:x1]),rows[target],xy),rows[source])
        sl=np.s_[lo-y0:hi-y0];v,k=curvature(uv,support[sl],a['resolution_ratio'][sl])
        result[y-y0:stop-y0]=v[y-lo:stop-lo];known[y-y0:stop-y0]=k[y-lo:stop-lo]
    return result,known,r['box']
