"""Refit only appended forearm vertices to a train-observed inverse-depth quadric."""
import numpy as np
from study_confidence_depth_prior import project_integer,unproject
from joint_temporal_texture import project


def curve_vertices(vertices,base_count,reference,fit,vertex_ids=None):
    # Input PLYs may retain unused proposal vertices outside the accepted patch.
    # They must neither move nor veto a correction to referenced geometry.
    selected=np.arange(base_count,len(vertices)) if vertex_ids is None else np.asarray(vertex_ids,dtype=int)
    if (selected<base_count).any() or (selected>=len(vertices)).any():raise ValueError('Attempt to curve original/out-of-range vertex')
    uv,z=project_integer(reference,vertices[selected]);q=(uv-np.asarray(fit['reference_center']))/100
    design=np.column_stack([q,np.ones(len(q)),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2])
    inverse=design@np.asarray(fit['all_camera_coefficients'])
    if not np.isfinite(inverse).all() or (inverse<=0).any():raise ValueError('Invalid curved inverse depth')
    newz=1/inverse
    if np.max(np.abs(newz-z))>.01:raise ValueError('Curvature exceeds frozen normalized displacement bound')
    result=vertices.copy();result[selected]=unproject(reference,uv[:,0],uv[:,1],newz)
    return result,dict(displacement_depth_quantiles=np.quantile(newz-z,[0,.5,1]).tolist(),original_vertices_unchanged=True)


def boundary_curve_vertices(vertices,base_count,reference,fit,vertex_ids,accepted,feather_px=10):
    """Keep the existing-depth ring exact; smoothly introduce curvature inside it."""
    from scipy.ndimage import distance_transform_edt
    if feather_px<=0:raise ValueError('Positive boundary feather required')
    selected=np.asarray(vertex_ids,dtype=int)
    if (selected<base_count).any() or (selected>=len(vertices)).any():raise ValueError('Invalid appended indices')
    uv,z=project_integer(reference,vertices[selected]);xy=np.rint(uv).astype(int)
    if not np.isfinite(uv).all() or (xy<0).any() or (xy[:,0]>=accepted.shape[1]).any() or (xy[:,1]>=accepted.shape[0]).any():
        raise ValueError('Boundary vertex outside reference domain')
    interior=accepted[xy[:,1],xy[:,0]]
    curved,receipt=curve_vertices(vertices,base_count,reference,fit,selected[interior])
    _,curved_z=project_integer(reference,curved[selected[interior]])
    distance=distance_transform_edt(accepted)[xy[interior,1],xy[interior,0]]
    u=np.clip(distance/feather_px,0,1);weight=u*u*(3-2*u)
    newz=z[interior]+weight*(curved_z-z[interior])
    result=vertices.copy();result[selected[interior]]=unproject(reference,uv[interior,0],uv[interior,1],newz)
    if not np.array_equal(result[selected[~interior]],vertices[selected[~interior]]):raise ValueError('Moved old-depth ring')
    receipt.update(boundary_ring_vertices=int((~interior).sum()),boundary_ring_exact=True,
        feather_px=feather_px,interior_vertices=int(interior.sum()),
        applied_depth_displacement_quantiles=np.quantile(newz-z[interior],[0,.5,1]).tolist())
    return result,receipt


def semantic_faces(vertices,faces,rows,masks,axis_extent=False):
    used=np.unique(faces);points=vertices[used];support=np.zeros(len(points),np.uint8);outside=np.zeros(len(points),bool)
    for camera in rows:
        name=camera['physical_camera']
        if name not in masks:continue
        uv,z=project(points,[camera]);uv=uv[0];z=z[0];xy=np.rint(uv).astype(int)
        available=(z>0)&(uv[:,0]>2)&(uv[:,0]<1917)&(uv[:,1]>2)&(uv[:,1]<1077)
        inside=np.zeros(len(points),bool);ids=np.flatnonzero(available)
        inside[ids]=masks[name][xy[ids,1],xy[ids,0]]
        support+=inside;outside|=available&~inside
    valid=np.zeros(len(vertices),bool);valid[used]=(support>=2)&~outside
    edges=vertices[faces[:,[1,2,0]]]-vertices[faces]
    extent_ok=(np.ptp(vertices[faces],axis=1).max(1)<.002) if axis_extent else (np.linalg.norm(edges,axis=2).max(1)<=.002)
    keep=valid[faces].all(1)&extent_ok
    return faces[keep],dict(proposed_triangles=len(faces),semantic_or_extent_rejected=int((~keep).sum()),
        unchanged_skin_masks=True,minimum_skin_cameras=2,maximum_triangle_extent=.002,
        extent_metric='axis_extent_strict' if axis_extent else 'euclidean_edge')
