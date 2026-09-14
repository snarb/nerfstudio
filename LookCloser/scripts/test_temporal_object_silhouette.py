"""Conservative material-specific silhouette test, not a certified cylinder fit.

Automatically selects a strong elongated neutral-material cluster in the reviewed
lower-face/hand search area of real H/C RGB. The scene-specific search rectangle
is disclosed. All carving evidence comes from actual train RGB, never eval RGB.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import cameras,read,sha,atomic_json,ROOT,display,exr,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

PARENT=Path('/mnt/data/dec5_dynamic_grid_150_guard_v2')
DIAG=Path('/mnt/data/dec5_dynamic_metal_diagnosis')


def select_cluster(clusters,rows):
    reference=next(r for r in rows if r['physical_camera']=='H004_C005_1210SZ');candidates=[]
    for c in clusters:
        uv,z=project(np.array(c['center'])[None],[reference]);px,py=uv[0,0,1],1919-uv[0,0,0]
        if z[0,0]<=0 or not (200<px<650 and 850<py<1380):continue
        if len(c['faces'])<80 or not .004<c['length']<.016 or c['neutral_view_count_median']<30:continue
        if c['radial_median_p90'][1]>.003:continue
        candidates.append(c)
    if len(candidates)!=1:return None
    return candidates[0]


def prepare(frame,output):
    root=output/frame;root.mkdir(parents=True,exist_ok=True)
    rows,original_path,_=cameras(frame);diagnostic=read(DIAG/frame/'clusters.json')
    if diagnostic['mesh_sha256']!=sha(original_path):raise ValueError('Material evidence mesh mismatch')
    cluster=select_cluster(diagnostic['clusters'],rows)
    if cluster is None:
        atomic_json(root/'result.json',dict(frame_id=frame,status='no_unambiguous_object_cluster',production_modified=False));return
    guard=PARENT/'foreground_guard'/frame;body=read(guard/'result.json')
    mesh=o3d.io.read_triangle_mesh(str(guard/'mesh.ply'));v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);centroid=v[t].mean(1)
    original=o3d.io.read_triangle_mesh(str(original_path));ov=np.asarray(original.vertices);ot=np.asarray(original.triangles)
    points=ov[ot[cluster['faces']]].mean(1);center=np.array(cluster['center']);axis=np.array(cluster['axis'])
    delta=centroid-center;along=delta@axis;radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
    observed_along=(points-center)@axis;lo,hi=np.quantile(observed_along,[.10,.95])
    # Avoid both object endpoints / finger contact. Strong neutral consensus
    # protects the visible barrel; nonmetallic fingers cannot enter proposals.
    near=(along>lo)&(along<hi)&(radial<cluster['radial_median_p90'][1]*2.5)
    mapping=np.flatnonzero(~np.load(guard/'evidence.npz')['removed'])
    if len(mapping)!=len(t):raise ValueError('Body/parent triangle order mismatch')
    evidence=np.load(DIAG/frame/'evidence.npz');neutral=evidence['neutral_count'][mapping];visible=evidence['visible_count'][mapping]
    candidate=near&(neutral>=4)&(neutral<.8*visible)
    used_vertices=np.unique(t[candidate]);vp=v[used_vertices];uv,z=project(vp,rows);outside=np.zeros(len(vp),np.uint16)
    response=read(ROOT/'camera_profiles.json');gains=dict(zip(response['physical_cameras'],response['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain'];scene=scene_for(v,t)
    masks=[];observations=[]
    for i,row in enumerate(rows):
        rgb=np.rint(display(exr(row['file_path'])*np.array(gains[row['physical_camera']]),exposure)*255).clip(0,255).astype(np.uint8)
        projected,_=project(points,[row]);q=projected[0];mn=np.floor(q.min(0)-8).astype(int);mx=np.ceil(q.max(0)+8).astype(int)
        x0,y0=np.maximum(mn,[0,0]);x1,y1=np.minimum(mx,[1920,1080]);crop=rgb[y0:y1,x0:x1]
        if crop.size==0:continue
        maximum=crop.max(2).astype(float);gray=(maximum>60)&(np.ptp(crop.astype(float),axis=2)<.23*maximum)
        gray=cv2.morphologyEx(gray.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(gray,8)
        if n<2 or stats[1:,cv2.CC_STAT_AREA].max()<30:continue
        mask=(labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
        mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        sd,_,_=camera_depth(scene,row);sd=np.where(np.isfinite(sd),sd,0)
        if len(vp):
            qv=uv[i].reshape(1,-1,2);d=cv2.remap(sd,qv,None,cv2.INTER_LINEAR)[0]
            valid=(z[i]>0)&(d>0)&(np.abs(d-z[i])<.0015*z[i])
            value=cv2.remap(mask,(uv[i]-[x0,y0]).astype(np.float32).reshape(1,-1,2),None,cv2.INTER_NEAREST)[0]
            outside+=valid&(value==0)
        overlay=crop.copy();edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;overlay[edge]=[255,0,255]
        Image.fromarray(np.rot90(overlay)).resize((240,320)).save(root/f'mask_{i:02d}.png')
        observations.append(dict(camera=row['physical_camera'],crop_native=[int(x0),int(y0),int(x1),int(y1)],source_sha256=sha(row['file_path'])))
    votes=np.zeros(len(v),np.uint16);votes[used_vertices]=outside
    removed=candidate&(votes[t]>=6).all(1)
    # Semantic/material evidence never overrules independently supported fingers.
    evroot=Path('/mnt/data/lookcloser_dec5_5a3_surface_repair/supported_shell_control/mesh')
    protected=0
    if read(evroot/'carving_request.json')['mesh_sha256']==sha(original_path):
        near_depth=np.load(evroot/'triangle_evidence.npz')['near_counts'][mapping]
        protected=int((removed&(near_depth>=2)).sum());removed&=near_depth<2
    if removed.sum()>1500:raise ValueError('Unexpectedly large local object deletion')
    result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[~removed]));result.remove_unreferenced_vertices();result.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(root/'mesh.ply'),result)
    np.savez_compressed(root/'evidence.npz',removed=removed,vertex_outside_votes=votes,candidate=candidate)
    atomic_json(root/'result.json',dict(frame_id=frame,status='candidate_requires_render_and_review',
        cluster=cluster,search_rectangle_HC_portrait=[200,850,650,1380],parent_mesh_sha256=body['mesh_sha256'],
        material_evidence_sha256=sha(DIAG/frame/'evidence.npz'),mask_views=observations,
        original_triangles=len(t),candidate_faces=int(candidate.sum()),removed_faces=int(removed.sum()),depth_protected=protected,
        mesh_sha256=sha(root/'mesh.ply'),script_sha256=sha(__file__),rgb_averaging=False,geometry_completion=False,
        actual_object_shape_not_certified=True))
    print(frame,'object',cluster['cluster'],'proposed',candidate.sum(),'removed',removed.sum(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_dynamic_object_silhouette'))
    p.add_argument('--frames',nargs='+',default=['000973','000975','000983']);a=p.parse_args();cv2.setNumThreads(2)
    for frame in a.frames:prepare(frame,a.output)
