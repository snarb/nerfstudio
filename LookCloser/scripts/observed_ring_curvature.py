"""Blend curvature only near actual old-depth boundary samples, not unknown edges."""
import numpy as np
from scipy.ndimage import binary_dilation,distance_transform_edt
from curve_forearm_delta import curve_vertices
from study_confidence_depth_prior import project_integer,unproject


def boundary_curve_vertices(vertices,base_count,reference,fit,vertex_ids,accepted,mesh_depth,feather_px=10):
    if feather_px<=0 or accepted.shape!=mesh_depth.shape:raise ValueError('Invalid observed-ring grid')
    selected=np.asarray(vertex_ids,dtype=int)
    if (selected<base_count).any() or (selected>=len(vertices)).any():raise ValueError('Invalid appended indices')
    uv,z=project_integer(reference,vertices[selected]);xy=np.rint(uv).astype(int)
    if not np.isfinite(uv).all() or (xy<0).any() or (xy[:,0]>=accepted.shape[1]).any() or (xy[:,1]>=accepted.shape[0]).any():
        raise ValueError('Boundary vertex outside reference domain')
    interior=accepted[xy[:,1],xy[:,0]]
    curved,receipt=curve_vertices(vertices,base_count,reference,fit,selected[interior])
    _,curved_z=project_integer(reference,curved[selected[interior]])
    ring=binary_dilation(accepted)&~accepted&np.isfinite(mesh_depth)&(mesh_depth>0)
    distance=distance_transform_edt(~ring) if ring.any() else np.full(accepted.shape,np.inf)
    d=distance[xy[interior,1],xy[interior,0]]
    u=np.clip(d/feather_px,0,1);weight=u*u*(3-2*u)
    newz=z[interior]+weight*(curved_z-z[interior])
    result=vertices.copy();result[selected[interior]]=unproject(reference,uv[interior,0],uv[interior,1],newz)
    if not np.array_equal(result[selected[~interior]],vertices[selected[~interior]]):raise ValueError('Moved old-depth ring')
    receipt.update(boundary_ring_vertices=int((~interior).sum()),boundary_ring_exact=True,
        feather_px=feather_px,interior_vertices=int(interior.sum()),observed_ring_pixels=int(ring.sum()),
        feather_domain='actual_old_depth_ring_only',unknown_patch_boundary_forces_plane=False,
        applied_depth_displacement_quantiles=np.quantile(newz-z[interior],[0,.5,1]).tolist())
    return result,receipt
