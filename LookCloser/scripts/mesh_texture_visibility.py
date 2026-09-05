"""Exact train-camera line-of-sight visibility against the unchanged TSDF mesh."""
from __future__ import annotations
import numpy as np


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
