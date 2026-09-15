"""Anatomical correction domain and all-pair discrete contact guard."""
import numpy as np
import open3d as o3d
from fit_mhr_constrained_correction import ConstraintGuard


def anatomical_domain(neutral):
    neutral=np.asarray(neutral)
    if neutral.ndim!=2 or neutral.shape[1]!=3 or not np.isfinite(neutral).all():
        raise ValueError('Invalid neutral model coordinates')
    # Height from correction band; width from the existing anchor protocol.
    return (neutral[:,1]>135)&(neutral[:,1]<153)&(abs(neutral[:,0])<12)


def all_pairs(v,t):
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    pairs=np.asarray(mesh.get_self_intersecting_triangles(),int).reshape(-1,2)
    return set(map(tuple,np.sort(pairs,axis=1)))


class AllContactGuard(ConstraintGuard):
    def __init__(self,*args):
        super().__init__(*args);self.allowed_all=all_pairs(self.base,self.t)

    def check(self,trial):
        ok,detail=super().check(trial)
        if not ok:return ok,detail
        pairs=all_pairs(trial,self.t);new=pairs-self.allowed_all
        return not new,dict(detail,all_intersection_pairs=len(pairs),new_all_intersection_pairs=len(new),
            reason='new_contact_or_overlap' if new else 'accepted')
