#!/usr/bin/env python3
"""Read-only all-train native-depth evidence for already traced surface points.

Missing/zero taps remain unknown; neither bilinear depths nor self-mesh visibility
are independent evidence. The JSON records all native taps and triangle identity.
No RGB is read, no mesh is changed, and target pixels are diagnostic queries only.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from carve_patchmatch_mesh_free_space import free_space_evidence,train_frames
from colmap_patchmatch_tsdf_campaign_common import atomic_json,sha256
from render_patchmatch_camera_path import normalize_frame


def native_evidence(depth,u,v,z,*,radius=2,minimum_gap=.005):
    """Classify a native footprint conservatively; also expose stricter support."""
    if depth.ndim!=2 or radius<0 or not np.isfinite(minimum_gap) or minimum_gap<=0:
        raise ValueError('Invalid native-depth audit parameters')
    if not np.isfinite([u,v,z]).all() or z<=0:
        return {'status':'behind_camera_or_nonfinite','native_taps':[]}
    x,y=int(np.rint(u)),int(np.rint(v));h,w=depth.shape
    if x-radius<0 or y-radius<0 or x+radius>=w or y+radius>=h:
        return {'status':'outside_native_footprint','native_taps':[]}
    taps=np.asarray(depth[y-radius:y+radius+1,x-radius:x+radius+1],dtype=np.float64)
    positive=np.isfinite(taps)&(taps>0)
    free,near=free_space_evidence(depth,np.array([u]),np.array([v]),np.array([z]),
                                 minimum_gap=minimum_gap,radius=radius)
    gap=max(minimum_gap,.01*z)
    required=int(np.ceil(.8*taps.size))
    front=positive&(taps<z-gap)
    status='measured_free_space' if free[0] else 'near_surface' if near[0] else (
        'foreground_occlusion' if front.sum()>=required else 'unknown_or_mixed')
    return {'status':status,'center_xy':[x,y],'native_taps':[
                [float(a) if np.isfinite(a) else None for a in row] for row in taps],
            'positive_taps':int(positive.sum()),'total_taps':int(taps.size),
            'near_within_0_001':int((positive&(np.abs(taps-z)<=.001)).sum()),
            'near_within_gap':int((positive&(np.abs(taps-z)<=minimum_gap)).sum()),
            'farther_taps':int((positive&(taps>z+gap)).sum()),
            'foreground_taps':int(front.sum()),
            'positive_median':float(np.median(taps[positive])) if positive.any() else None,
            'positive_min':float(taps[positive].min()) if positive.any() else None,
            'positive_max':float(taps[positive].max()) if positive.any() else None}


def trace_points(trace):
    valid=[];missing=[]
    for index,row in enumerate(trace['pixels']):
        z=row.get('target_depth')
        if z is None or not np.isfinite(z) or z<=0:
            missing.append({'index':index,'pixel':row['pixel'],'status':'no_target_surface'})
            continue
        world=np.asarray(row.get('world'),dtype=np.float64)
        if world.shape!=(3,) or not np.isfinite(world).all():
            raise ValueError('Positive target depth requires a finite 3D point')
        valid.append({'index':index,'pixel':row['pixel'],'world':world.tolist(),'sources':[]})
    return valid,missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trace','depth-data','mesh','mesh-metadata','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    trace=json.loads(a.trace.read_text());payload=json.loads((a.depth_data/'transforms.json').read_text())
    meta=json.loads(a.mesh_metadata.read_text())
    if sha256(a.mesh)!=meta['output_sha256']:raise ValueError('Mesh metadata/hash mismatch')
    if trace['pixel_center_offset']!=.5:raise ValueError('Audit requires native pixel centres')
    expected=trace['input_hashes'].get(str(a.mesh))
    if expected!=sha256(a.mesh):raise ValueError('Trace and audit mesh differ')
    frames=train_frames(payload)
    rows,missing=trace_points(trace)
    if not rows:raise ValueError('No actual surface points to audit')
    world=np.array([row['world'] for row in rows])
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));triangles=np.asarray(mesh.triangles);vertices=np.asarray(mesh.vertices)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    closest=scene.compute_closest_points(o3d.core.Tensor(world.astype(np.float32)))
    for row,pid,point in zip(rows,closest['primitive_ids'].numpy(),closest['points'].numpy()):
        row['triangle_id']=int(pid);row['triangle_vertices']=vertices[triangles[pid]].tolist()
        row['distance_to_mesh']=float(np.linalg.norm(np.array(row['world'])-point))
    hashes={str(path):sha256(path) for path in [a.trace,a.mesh,a.mesh_metadata,a.depth_data/'transforms.json',Path(__file__),
        Path(__file__).with_name('carve_patchmatch_mesh_free_space.py'),Path(__file__).with_name('render_patchmatch_camera_path.py')]}
    for frame in frames:
        path=a.depth_data/frame['depth_file_path'];hashes[str(path)]=sha256(path)
        with gzip.open(path,'rb') as stream:depth=np.load(stream,allow_pickle=False)
        if depth.shape!=(1080,1920) or not np.isfinite(depth).all() or (depth<0).any():
            raise ValueError('Invalid raw geometric depth')
        depth=depth*float(meta['dataparser_scale'])
        f=normalize_frame(frame,payload,meta);pose=np.asarray(f['transform_matrix'])
        q=(world-pose[:3,3])@pose[:3,:3];z=-q[:,2];safe=np.where(z>0,z,1.)
        u=f['fl_x']*q[:,0]/safe+f['cx']-.5;v=-f['fl_y']*q[:,1]/safe+f['cy']-.5
        for row,uu,vv,zz in zip(rows,u,v,z):
            evidence=native_evidence(depth,uu,vv,zz)
            row['sources'].append({'physical_camera':f['physical_camera'],'uv':[float(uu),float(vv)],
                                   'projected_z':float(zz),**evidence})
    for row in rows:
        statuses=[s['status'] for s in row['sources']]
        row['counts']={s:statuses.count(s) for s in sorted(set(statuses))}
        row['strict_near_views']=sum(s.get('near_within_0_001',0)>=20 for s in row['sources'])
    result={'uses_rgb':False,'changes_prediction':False,'train_camera_count':len(frames),
            'pixel_center_offset':.5,'radius':2,'minimum_gap_normalized':.005,
            'relative_free_space_gap':.01,'native_tap_fraction':.8,'input_hashes':hashes,
            'points':rows,'missing_target_points':missing,
            'caveat':'Near counts are agreement with measured stereo, not proof of physical truth; invalid samples never vote.'}
    atomic_json(a.output,result)
    print(json.dumps({'points':[{k:r[k] for k in ('index','pixel','triangle_id','counts','strict_near_views')} for r in rows],
                      'missing_target_points':missing}),flush=True)


if __name__=='__main__':main()
