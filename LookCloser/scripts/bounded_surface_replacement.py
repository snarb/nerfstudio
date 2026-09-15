"""Identify only faces fully inside a finite, depth-bounded replacement domain."""
import numpy as np


def removable_faces(uv,z,faces,domain,depth,max_distance=.012):
    if max_distance<=0 or domain.shape!=depth.shape:raise ValueError('Invalid replacement bounds')
    uv=np.asarray(uv);z=np.asarray(z);finite=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
    xy=np.zeros(uv.shape,dtype=int);xy[finite]=np.rint(uv[finite]).astype(int)
    good=finite&(xy[:,0]>=0)&(xy[:,0]<domain.shape[1])&(xy[:,1]>=0)&(xy[:,1]<domain.shape[0])
    ids=np.flatnonzero(good);x,y=xy[ids].T;reference=depth[y,x]
    good[ids]=domain[y,x]&np.isfinite(reference)&(reference>0)&(np.abs(z[ids]-reference)<=max_distance)
    return good[faces].all(1)
