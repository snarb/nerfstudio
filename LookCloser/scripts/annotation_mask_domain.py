"""Separate known annotation pixels from renderer interpolation footprint.

The frozen forearm polygons stop three pixels from the image boundary. A false
mask value in that unannotated border is unknown, not negative skin evidence.
No polygon is dilated; all disagreements inside the known domain still veto.
"""
import numpy as np
from joint_temporal_texture import project

def known_domain(uv,z,width,height,margin=3):
    xy=np.rint(uv).astype(int)
    footprint=(z>0)&(uv[:,0]>2)&(uv[:,0]<width-3)&(uv[:,1]>2)&(uv[:,1]<height-3)
    return footprint&(xy[:,0]>=margin)&(xy[:,0]<width-margin)&(xy[:,1]>=margin)&(xy[:,1]<height-margin)

def semantic_faces(vertices,faces,rows,masks,axis_extent=False,positive_only_annotations=False):
    used=np.unique(faces);points=vertices[used];support=np.zeros(len(points),np.uint8);outside=np.zeros(len(points),bool)
    unknown_border=[]
    for camera in rows:
        name=camera['physical_camera']
        if name not in masks:continue
        uv,z=project(points,[camera]);uv,z=uv[0],z[0];xy=np.rint(uv).astype(int)
        available=known_domain(uv,z,camera['w'],camera['h']);ids=np.flatnonzero(available)
        inside=np.zeros(len(points),bool);inside[ids]=masks[name][xy[ids,1],xy[ids,0]]
        support+=inside
        if not positive_only_annotations:outside|=available&~inside
        footprint=(z>0)&(uv[:,0]>2)&(uv[:,0]<camera['w']-3)&(uv[:,1]>2)&(uv[:,1]<camera['h']-3)
        unknown_border.append(dict(camera=name,footprint_but_unannotated=int((footprint&~available).sum())))
    valid=np.zeros(len(vertices),bool);valid[used]=(support>=2)&~outside
    edges=vertices[faces[:,[1,2,0]]]-vertices[faces]
    extent=(np.ptp(vertices[faces],axis=1).max(1)<.002) if axis_extent else (np.linalg.norm(edges,axis=2).max(1)<=.002)
    keep=valid[faces].all(1)&extent
    return faces[keep],dict(proposed_triangles=len(faces),semantic_or_extent_rejected=int((~keep).sum()),
        unchanged_skin_masks=True,minimum_skin_cameras=2,maximum_triangle_extent=.002,
        extent_metric='axis_extent_strict' if axis_extent else 'euclidean_edge',known_annotation_margin=3,
        unknown_border=unknown_border,all_known_domain_disagreements_veto=not positive_only_annotations,
        positive_only_annotations=positive_only_annotations)
