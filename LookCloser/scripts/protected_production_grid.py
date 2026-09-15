"""Compose a constrained prior only over holes in a protected production mesh."""
import numpy as np
from scipy.ndimage import binary_dilation
from study_confidence_depth_prior import unproject,project_integer,raycast_integer
from confidence_boundary_completion import grid_faces
from bounded_surface_replacement import removable_faces
from diffusion_mesh_repair import scene_for


def grid_plan(domain,solved,protected_depth):
    if not (domain.shape==solved.shape==protected_depth.shape):raise ValueError('Grid shape mismatch')
    if not np.isfinite(protected_depth).all() or (protected_depth<0).any():raise ValueError('Protected depth must use zero misses')
    active=domain&(protected_depth==0)
    if not np.isfinite(solved[active]).all() or (solved[active]<=0).any():raise ValueError('Invalid prior depth')
    grid=binary_dilation(active)&((protected_depth>0)|active)
    depth=np.where(active,solved,protected_depth)
    return active,grid,depth


def protected_removal(removed,production_count):
    if not 0<=production_count<=len(removed):raise ValueError('Invalid production prefix count')
    result=np.asarray(removed,bool).copy();result[:production_count]=False
    return result


def assemble(ov,ot,pv,pt,camera,domain,solved):
    if not np.array_equal(ov[:len(pv)],pv) or not np.array_equal(ot[:len(pt)],pt):raise ValueError('Production mesh is not an exact prefix')
    md=raycast_integer(scene_for(pv,pt),camera)
    active,grid,depth=grid_plan(domain,solved,md)
    uv,z=project_integer(camera,ov)
    removed=protected_removal(removable_faces(uv,z,ot,domain,solved,.012),len(pt));retained=ot[~removed]
    y,x=np.nonzero(grid);vertices=np.concatenate([ov,unproject(camera,x,y,depth[y,x])])
    index=np.full(grid.shape,-1,int);index[y,x]=np.arange(len(x))+len(ov)
    faces=grid_faces(grid,active,index)
    evidence=dict(active=active,grid=grid,grid_depth=depth,protected_depth=md)
    stats=dict(protected_production_vertices=len(pv),protected_production_triangles=len(pt),
        active_missing_pixels=int(active.sum()),grid_vertices=len(x),sampled_boundary_vertices=int((grid&~active).sum()),
        removed_production_triangles=0,removed_previous_prior_triangles=int(removed.sum()),
        original_surface_prefix_exact=True,ring_depth_uses_production_mesh=True)
    return vertices,retained,faces,removed,evidence,stats
