"""Exact train-camera line-of-sight visibility against the unchanged TSDF mesh."""
from __future__ import annotations
import numpy as np


def bilinear_depth_footprint_support(depth,u,v,projected_z,*,log_tolerance=.005):
    """Require every contributing native RGB tap to lie on the target depth layer.

    A visible sample centre does not guarantee an unoccluded bilinear footprint.
    Use unfilled source-camera raycast depths, never interpolated/averaged depths.
    This conservative test may reject a source; it cannot invent surface support.
    """
    import torch
    if (depth.ndim!=2 or u.shape!=v.shape or u.shape!=projected_z.shape
            or not np.isfinite(log_tolerance) or log_tolerance<=0):
        raise ValueError('Invalid RGB footprint inputs')
    h,w=depth.shape
    finite=torch.isfinite(u)&torch.isfinite(v)&torch.isfinite(projected_z)
    valid=finite&(projected_z>0)&(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    su=torch.where(finite,u,0);sv=torch.where(finite,v,0)
    x=su.floor().long();y=sv.floor().long();fx=su-x;fy=sv-y
    for dy in (0,1):
        for dx in (0,1):
            weight=(fx if dx else 1-fx)*(fy if dy else 1-fy)
            xx=x+dx;yy=y+dy
            z=depth[yy.clamp(0,h-1),xx.clamp(0,w-1)]
            agrees=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&torch.isfinite(z)&(z>0)
            agrees&=((z.clamp_min(1e-7)/projected_z.clamp_min(1e-7)).log().abs()<=log_tolerance)
            valid&=(weight<=1e-6)|agrees
    return valid


def observed_depth_support(depth,u,v,projected_z,*,log_tolerance=.005,radius=2):
    """Count nearby measured depths without bilinearly mixing zeros/depth layers."""
    import torch
    if depth.ndim!=2 or u.shape!=v.shape or u.shape!=projected_z.shape or radius<0 or log_tolerance<=0:
        raise ValueError('Invalid observed-depth support inputs')
    h,w=depth.shape
    x=u.round().long();y=v.round().long()
    count=torch.zeros_like(u,dtype=torch.int16);matches=torch.zeros_like(count)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            xx=x+dx;yy=y+dy
            sample=depth[yy.clamp(0,h-1),xx.clamp(0,w-1)]
            good=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&torch.isfinite(sample)&(sample>0)&(projected_z>0)
            agrees=good&((sample.clamp_min(1e-7)/projected_z.clamp_min(1e-7)).log().abs()<=log_tolerance)
            count+=good.to(torch.int16);matches+=agrees.to(torch.int16)
    fraction=matches.float()/count.clamp_min(1)
    return (count>=3)&(fraction>=.5),fraction,count


def sample_rgb_depth_aware(rgb,depth,u,v,projected_z,*,log_tolerance=.005):
    """Renormalized bilinear interpolation of ONE camera's same-layer RGB taps.

    This does not blend cameras or expand centre-point mesh visibility. It only
    removes wrong-layer taps from the ordinary subpixel interpolation footprint.
    A sample with no matching native taps has no valid color support.
    """
    import torch
    if (rgb.ndim!=3 or rgb.shape[0]!=3 or tuple(rgb.shape[1:])!=tuple(depth.shape)
            or depth.ndim!=2 or u.shape!=v.shape or u.shape!=projected_z.shape
            or not np.isfinite(log_tolerance) or log_tolerance<=0):
        raise ValueError('Invalid depth-aware RGB sample inputs')
    h,w=depth.shape
    finite=torch.isfinite(u)&torch.isfinite(v)&torch.isfinite(projected_z)
    bounds=finite&(projected_z>0)&(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    su=torch.where(finite,u,0);sv=torch.where(finite,v,0)
    x=su.floor().long();y=sv.floor().long();fx=su-x;fy=sv-y
    color=torch.zeros((3,*u.shape),dtype=rgb.dtype,device=rgb.device)
    mass=torch.zeros_like(u)
    for dy in (0,1):
        for dx in (0,1):
            xx=(x+dx).clamp(0,w-1);yy=(y+dy).clamp(0,h-1)
            z=depth[yy,xx]
            agrees=bounds&torch.isfinite(z)&(z>0)
            agrees&=((z.clamp_min(1e-7)/projected_z.clamp_min(1e-7)).log().abs()<=log_tolerance)
            weight=(fx if dx else 1-fx)*(fy if dy else 1-fy)
            weight=torch.where(agrees,weight,0)
            mass+=weight
            color+=rgb[:,yy,xx]*weight[None]
    valid=mass>1e-6
    return torch.where(valid[None],color/mass.clamp_min(1e-6)[None],0),valid,mass


class MeshVisibility:
    def __init__(self,mesh_path):
        import open3d as o3d
        self.o3d=o3d
        mesh=o3d.io.read_triangle_mesh(str(mesh_path))
        if not len(mesh.triangles):raise ValueError('Empty visibility mesh')
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=8)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def visible(self,world,center,support,*,tolerance=0.000025):
        """A first hit in front of the point vetoes RGB from this camera.

        Points in explicitly permitted small plane-filled target holes can lack
        their own triangle, so no-hit rays mean unoccluded, not new depth evidence.
        """
        if tolerance<=0 or not np.isfinite(tolerance):raise ValueError('Invalid visibility tolerance')
        points=np.asarray(world,np.float32)[support];center=np.asarray(center,np.float32)
        if not len(points):
            return np.zeros(support.shape,bool),{'rays':0,'occluded':0,'no_hit':0,'hit_near_target':0,
                                                'tolerance_normalized':tolerance}
        directions=points-center;length=np.linalg.norm(directions,axis=-1)
        if (length<=0).any():raise ValueError('Camera lies on a target point')
        rays=np.concatenate((np.broadcast_to(center,points.shape),directions),axis=-1)
        hit=self.scene.cast_rays(self.o3d.core.Tensor(rays))['t_hit'].numpy()
        difference=(hit-1)*length
        allowed=difference>=-tolerance
        valid=np.zeros(support.shape,bool);valid[support]=allowed
        stats={'rays':len(points),'occluded':int((~allowed).sum()),'no_hit':int((~np.isfinite(hit)).sum()),
               'hit_near_target':int((np.abs(difference)<=tolerance).sum()),'tolerance_normalized':tolerance}
        return valid,stats
