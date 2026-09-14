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


def semantic_faces(vertices,faces,rows,masks):
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
    keep=valid[faces].all(1)&(np.linalg.norm(edges,axis=2).max(1)<=.002)
    return faces[keep],dict(proposed_triangles=len(faces),semantic_or_extent_rejected=int((~keep).sum()),
        unchanged_skin_masks=True,minimum_skin_cameras=2,maximum_triangle_extent=.002)
