#!/usr/bin/env python3
"""Opt-in removal of mesh faces contradicted by reliable train-depth free space.

Raw depth behind a mesh point measures empty space in front of that observation.
Several independent train cameras must supply a locally supported farther layer.
No RGB, semantic masks, held-out images, pose fitting or new surface is used.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,canonical_sha256,sha256
from render_patchmatch_camera_path import normalize_frame

FORBIDDEN={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}


def free_space_evidence(depth,u,v,z,*,minimum_gap=.005,radius=2):
    """At least 80% native taps support a compact farther depth layer; no averaging."""
    if depth.ndim!=2 or u.shape!=v.shape or u.shape!=z.shape or minimum_gap<=0:
        raise ValueError('Invalid free-space sampling inputs')
    h,w=depth.shape
    finite=np.isfinite(u)&np.isfinite(v)&np.isfinite(z)&(z>0)
    x=np.rint(np.where(finite,u,0)).astype(np.int64);y=np.rint(np.where(finite,v,0)).astype(np.int64)
    inside=finite&(x>=radius)&(x<w-radius)&(y>=radius)&(y<h-radius)
    free=np.zeros(u.shape,bool);near=np.zeros(u.shape,bool)
    selected=np.flatnonzero(inside)
    if not len(selected):return free,near
    xx=x.flat[selected];yy=y.flat[selected];zz=z.flat[selected]
    values=np.stack([depth[yy+dy,xx+dx] for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)])
    positive=np.isfinite(values)&(values>0)
    gap=np.maximum(minimum_gap,.01*zz)
    farther=positive&(values>zz+gap)
    required=int(np.ceil(.8*values.shape[0]))
    # A robust middle quantile avoids treating a single far outlier as free space.
    ordered=np.sort(np.where(positive,values,np.inf),axis=0)
    lo=ordered[int(.2*(len(values)-1))];hi=ordered[int(.8*(len(values)-1))]
    spread=np.full_like(hi,np.inf)
    np.subtract(hi,lo,out=spread,where=np.isfinite(hi)&np.isfinite(lo))
    stable=np.isfinite(hi)&(lo>0)&(spread<=.005*lo)
    free.flat[selected]=(farther.sum(0)>=required)&stable
    near.flat[selected]=(positive&(np.abs(values-zz)<=minimum_gap)).sum(0)>=required
    return free,near


def train_frames(payload):
    train=set(payload.get('train_filenames',[]))
    frames=[f for f in payload['frames'] if f['file_path'] in train]
    if len(train)!=62 or len(frames)!=62 or len({f['physical_camera'] for f in frames})!=62:
        raise ValueError('Require exactly 62 unique explicit train cameras')
    if any(f['physical_camera'] in FORBIDDEN or not Path(f['file_path']).name.startswith('frame_train_') or not f.get('depth_file_path') for f in frames):
        raise ValueError('Unexpected held-out/non-train/missing-depth camera')
    return frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('depth-data','mesh','mesh-metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minimum-free-views',type=int,default=3)
    p.add_argument('--minimum-gap',type=float,default=.005)
    a=p.parse_args()
    if a.minimum_free_views<2 or not np.isfinite(a.minimum_gap) or a.minimum_gap<=0:p.error('Invalid carving evidence thresholds')
    a.output.mkdir(parents=True,exist_ok=False)
    payload=json.loads((a.depth_data/'transforms.json').read_text());meta=json.loads(a.mesh_metadata.read_text())
    if sha256(a.mesh)!=meta['output_sha256']:raise ValueError('Input mesh receipt hash mismatch')
    frames=train_frames(payload)
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(str(a.mesh));vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    if not len(triangles):raise ValueError('Empty input mesh')
    centers=vertices[triangles].mean(1)
    free_counts=np.zeros(len(triangles),np.uint16);near_counts=np.zeros_like(free_counts);rows=[]
    request={'mesh_sha256':sha256(a.mesh),'mesh_metadata_sha256':sha256(a.mesh_metadata),
             'source_transforms_sha256':sha256(a.depth_data/'transforms.json'),'minimum_free_views':a.minimum_free_views,
             'minimum_gap_normalized':a.minimum_gap,'relative_minimum_gap':.01,'radius':2,
             'minimum_native_tap_fraction':.8,'maximum_middle_depth_spread_fraction':.005,
             'pixel_center_offset':.5,'sample_location':'triangle_centroid','train_camera_count':62,
             'uses_rgb':False,'uses_semantic_masks':False,'script_sha256':sha256(Path(__file__)),
             'normalization_helper_sha256':sha256(Path(__file__).with_name('render_patchmatch_camera_path.py')),
             'source_depth_hashes':{f['physical_camera']:sha256(a.depth_data/f['depth_file_path']) for f in frames}}
    request['sha256']=canonical_sha256(request);atomic_json(a.output/'carving_request.json',request)
    for frame in frames:
        f=normalize_frame(frame,payload,meta);pose=np.asarray(f['transform_matrix'])
        q=(centers-pose[:3,3])@pose[:3,:3];z=-q[:,2]
        safe_z=np.where(z>0,z,1)
        u=f['fl_x']*q[:,0]/safe_z+f['cx']-.5;v=-f['fl_y']*q[:,1]/safe_z+f['cy']-.5
        with gzip.open(a.depth_data/frame['depth_file_path'],'rb') as stream:depth=np.load(stream,allow_pickle=False)
        if depth.shape!=(1080,1920) or not np.isfinite(depth).all() or (depth<0).any():raise ValueError('Invalid full-resolution raw depth')
        free,near=free_space_evidence(depth*float(meta['dataparser_scale']),u,v,z,minimum_gap=a.minimum_gap)
        free_counts+=free;near_counts+=near
        rows.append({'physical_camera':f['physical_camera'],'free_space_votes':int(free.sum()),'surface_votes':int(near.sum())})
        print(json.dumps(rows[-1]),flush=True)
    remove=free_counts>=a.minimum_free_views
    np.savez_compressed(a.output/'triangle_evidence.npz',centers=centers,free_counts=free_counts,near_counts=near_counts,removed=remove)
    mesh.remove_triangles_by_mask(remove);mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):raise RuntimeError('Carving removed the complete mesh')
    labels,counts,_=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    threshold=max(100,int(np.ceil(.002*counts.max())))
    small=counts[np.asarray(labels)]<threshold;small_removed=int(small.sum())
    mesh.remove_triangles_by_mask(small);mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    if not len(mesh.triangles):raise RuntimeError('Component filter removed the complete mesh')
    output=a.output/'carved.ply'
    if not o3d.io.write_triangle_mesh(str(output),mesh,write_ascii=False):raise OSError('Mesh write failed')
    _,counts,_=mesh.cluster_connected_triangles();counts=sorted(map(int,counts),reverse=True)
    result=deepcopy(meta)
    result.update(output=str(output),output_sha256=sha256(output),vertices=len(mesh.vertices),triangles=len(mesh.triangles),
                  connected_components=len(counts),component_triangles=counts,
                  artifact_type='TSDF-derived mesh with train-depth free-space carving; not raw TSDF volume',
                  free_space_carving={'request_sha256':request['sha256'],'input_mesh':str(a.mesh),'input_mesh_sha256':request['mesh_sha256'],
                      'triangles_before':len(triangles),'triangles_carved':int(remove.sum()),'small_component_triangles_removed':small_removed,
                      'effective_component_threshold':threshold,'per_camera':rows,'free_view_histogram':np.bincount(free_counts).tolist(),
                      'evidence_path':str(a.output/'triangle_evidence.npz'),'evidence_sha256':sha256(a.output/'triangle_evidence.npz')})
    atomic_json(a.output/'carved.json',result)
    print(json.dumps({'complete':True,'mesh':str(output),'vertices':len(mesh.vertices),'triangles':len(mesh.triangles),'carved':int(remove.sum())}),flush=True)


if __name__=='__main__':main()
