"""CPU-only conservative whole-cell cover of a fixed semantic admission grid.

Allowed means positive OR unknown semantic support, never physical occupancy.
A target cell is excluded only when every intersected semantic cell is excluded.
Closed-boundary contact and outside-domain portions are conservatively admitted.
"""
import numpy as np


class WholeCellAdmission:
    def __init__(self,allowed,bounds,epsilon=2e-7):
        self.allowed=np.asarray(allowed)
        self.bounds=np.asarray(bounds,dtype=np.float64)
        if self.allowed.dtype!=np.bool_ or self.allowed.ndim!=3 or min(self.allowed.shape)<1:raise ValueError('Boolean XYZ semantic grid required')
        if self.bounds.shape!=(2,3)or not np.isfinite(self.bounds).all()or (self.bounds[1]<=self.bounds[0]).any():raise ValueError('Increasing finite floor-frame bounds required')
        if not 0<=epsilon<.01:raise ValueError('Explicit small coordinate epsilon required')
        self.epsilon=float(epsilon);self.resolution=np.array(self.allowed.shape);self.pitch=(self.bounds[1]-self.bounds[0])/self.resolution
        self.summed=np.pad(self.allowed.astype(np.int32).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32).cumsum(2,dtype=np.int32),((1,0),(1,0),(1,0)))

    def boxes_admitted(self,lower,upper):
        lo,hi=np.asarray(lower,dtype=np.float64),np.asarray(upper,dtype=np.float64)
        if lo.ndim!=2 or lo.shape!=hi.shape or lo.shape[1]!=3 or not np.isfinite([lo,hi]).all()or (hi<lo).any():raise ValueError('Finite ordered [N,3] boxes required')
        outside=((lo<self.bounds[0]-self.epsilon)|(hi>self.bounds[1]+self.epsilon)).any(1)
        # Including both sides of exact contact prevents edge-only false rejection.
        first=np.floor(np.nextafter((lo-self.bounds[0]-self.epsilon)/self.pitch,-np.inf)).astype(np.int64)
        last=np.floor(np.nextafter((hi-self.bounds[0]+self.epsilon)/self.pitch,np.inf)).astype(np.int64)
        first=np.clip(first,0,self.resolution-1);last=np.clip(last,0,self.resolution-1)+1
        a,b,c=first.T;x,y,z=last.T;s=self.summed
        count=s[x,y,z]-s[a,y,z]-s[x,b,z]-s[x,y,c]+s[a,b,z]+s[a,y,c]+s[x,b,c]-s[a,b,c]
        return outside|(count>0)

    def point_admitted(self,points):
        p=np.asarray(points,dtype=np.float64);return self.boxes_admitted(p,p)

    def map_target_grid(self,resolution,bounds,chunk=262144):
        r=np.asarray(resolution,dtype=np.int64);b=np.asarray(bounds,dtype=np.float64)
        if r.shape!=(3,)or (r<1).any()or b.shape!=(2,3)or not np.isfinite(b).all()or (b[1]<=b[0]).any():raise ValueError('Valid target XYZ resolution and bounds required')
        total=int(np.prod(r));out=np.empty(total,dtype=bool);pitch=(b[1]-b[0])/r
        for start in range(0,total,chunk):
            ids=np.arange(start,min(total,start+chunk),dtype=np.int64);ijk=np.stack([ids//(r[1]*r[2]),ids//r[2]%r[1],ids%r[2]],axis=1)
            lo=b[0]+ijk*pitch;out[start:start+len(ids)]=self.boxes_admitted(lo,lo+pitch)
        return out.reshape(tuple(r))
