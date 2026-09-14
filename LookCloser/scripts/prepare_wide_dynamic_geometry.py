"""Revalidate old silhouette removals at new target poses; never add new removals.

Cached per-time masks and removal proposals are reused, not the old view-specific
hole safeguard. Monotonic restoration uses the original surface, not invented RGB.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes
from joint_temporal_texture import read,sha,atomic_json
from render_smooth_temporal_mesh_video import verify_request
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def prepare(parent,output):
    previous=verify_request(parent);parent_sha=sha(parent/'request.json')
    output.mkdir(parents=True,exist_ok=True)
    def one(record):
        frame=record['frame_id'];root=output/'geometry'/frame;root.mkdir(parents=True,exist_ok=True)
        guard=Path(record['source_masks']['root']);receipt=read(guard/'complete.json')
        if sha(guard/'complete.json')!=record['source_masks']['complete_sha256']:raise ValueError('Changed cached guard')
        if sha(guard/'evidence.npz')!=receipt['hashes']['evidence.npz']:raise ValueError('Changed cached proposal')
        binding=dict(parent_request_sha256=parent_sha,source_guard_complete_sha256=sha(guard/'complete.json'),
                     original_mesh_sha256=record['untrimmed_mesh_sha256'],script_sha256=sha(__file__),camera=record['camera'])
        if (root/'complete.json').exists():
            done=read(root/'complete.json')
            if done['request']!=binding or sha(root/'mesh.ply')!=done['mesh_sha256']:raise ValueError('Restoration resume mismatch')
        else:
            if sha(record['untrimmed_mesh'])!=record['untrimmed_mesh_sha256']:raise ValueError('Changed original mesh')
            mesh=o3d.io.read_triangle_mesh(record['untrimmed_mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
            remove=np.load(guard/'evidence.npz')['removed'].copy();initial=remove.copy()
            if len(remove)!=len(t):raise ValueError('Wrong triangle inventory')
            before,ids,_=camera_depth(scene_for(v,t),record['camera']);hit=np.isfinite(before);rounds=0
            while remove.any():
                after,_,_=camera_depth(scene_for(v,t[~remove]),record['camera']);after_hit=np.isfinite(after)
                removed_hit=hit&remove[np.minimum(ids,len(t)-1)]
                enclosed=binary_fill_holes(after_hit)&~after_hit
                restore=np.unique(ids[removed_hit&(after_hit|enclosed)])
                if not len(restore):break
                remove[restore]=False;rounds+=1
                if rounds>100:raise ValueError('Nonconverging conservative restoration')
            if np.any(remove&~initial):raise ValueError('Restoration cannot introduce removals')
            result=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[~remove]))
            result.remove_unreferenced_vertices();result.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(root/'mesh.ply'),result)
            done=dict(request=binding,mesh_sha256=sha(root/'mesh.ply'),restored_faces=int((initial&~remove).sum()),
                retained_removals=int(remove.sum()),rounds=rounds,source_geometry_unchanged=True,
                originally_missing_geometry_reconstructed=False)
            atomic_json(root/'complete.json',done)
        row=deepcopy(record);row['previous_guard_mesh']=row['mesh'];row['previous_guard_mesh_sha256']=row['mesh_sha256']
        row['mesh']=str(root/'mesh.ply');row['mesh_sha256']=done['mesh_sha256']
        row['target_restoration_receipt']=str(root/'complete.json');row['target_restoration_receipt_sha256']=sha(root/'complete.json')
        print(f'restored={frame} faces={done["restored_faces"]}',flush=True)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:inventory=list(pool.map(one,previous['inventory']))
    request=deepcopy(previous);request['inventory']=inventory
    request['target_geometry_parent']=dict(path=str(parent/'request.json'),sha256=parent_sha)
    request['recipe']['new_pose_monotonic_restoration']=True
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    request['script_hashes']['run_wide_dynamic_workers.py']=sha(Path(__file__).with_name('run_wide_dynamic_workers.py'))
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable restored request mismatch')
    atomic_json(output/'request.json',request);(output/'frames').mkdir(exist_ok=True)
    atomic_json(output/'progress.json',dict(stage='initialized',complete=0,total=150))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,default=Path('/mnt/data/dec5_wide_dynamic_flight_150'))
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_wide_dynamic_flight_150_v2'));a=p.parse_args();prepare(a.parent,a.output)
