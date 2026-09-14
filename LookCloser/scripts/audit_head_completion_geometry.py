"""Independent preservation/locality checks for the selected 150-frame repair."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from render_smooth_temporal_mesh_video import verify_request

def arrays(path):
    mesh=o3d.io.read_triangle_mesh(str(path));return np.asarray(mesh.vertices),np.asarray(mesh.triangles)

def audit(output):
    request=verify_request(output);records=[]
    fixed=request['recipe'].get('geometry_identical_to_camera_workaround_parent',False)
    boundary_only=request['recipe'].get('camera_workaround_geometry_stage')=='pre_notch_mesh'
    parent={}
    if fixed or boundary_only:
        spec=request['camera_workaround_parent']
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Changed fixed-geometry parent')
        parent={row['frame_id']:row for row in read(spec['path'])['inventory']}
    if request.get('partial_diagnostic_only'):raise ValueError('Expected full campaign')
    for row in request['inventory']:
        head=read(row['head_repair_receipt']);notch=read(row['notch_receipt'])
        source=head['request']['source_mesh'];v,t=arrays(source);vv,tt=arrays(row['mesh'])
        if sha(source)!=head['request']['source_sha256'] or sha(row['mesh'])!=row['mesh_sha256']:raise ValueError('Changed geometry')
        if not np.array_equal(v[t],vv[tt[:len(t)]]):raise ValueError('Original triangles moved, removed, or reordered')
        additions=vv[tt[len(t):]]
        if len(additions) and (not np.isfinite(additions).all() or additions[...,0].min()<=-.03):raise ValueError('Patch outside bounded head')
        construction_camera=parent[row['frame_id']]['camera'] if fixed or boundary_only else row['camera']
        expected_mesh=head['mesh_sha256'] if boundary_only else notch['mesh_sha256']
        if notch['request']['camera']!=construction_camera or expected_mesh!=row['mesh_sha256']:raise ValueError('Wrong patch provenance')
        if fixed and parent[row['frame_id']]['mesh_sha256']!=row['mesh_sha256']:raise ValueError('Shot workaround changed geometry')
        if boundary_only and parent[row['frame_id']]['pre_notch_mesh_sha256']!=row['mesh_sha256']:raise ValueError('Changed fixed boundary-only stage')
        records.append(dict(frame=row['frame_id'],original_triangles=len(t),added_triangles=len(tt)-len(t),
            original_triangle_coordinates_unchanged=True,additions_only_in_bounded_head=True))
    result=dict(status='preservation_and_locality_pass',frames=len(records),records=records,
        request_sha256=sha(output/'request.json'),script_sha256=sha(__file__),upstream_root_cause_eliminated=False,
        full_watertightness_certified=False,global_self_intersections_certified=False,
        camera_workaround_geometry_identical_to_parent=fixed,
        camera_workaround_geometry_is_parent_boundary_only_stage=boundary_only)
    if len(records)!=150:raise ValueError('Wrong frame inventory')
    atomic_json(output/'head_geometry_audit.json',result);print('head_geometry_audit=150 passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_expanded_head_dynamic_150_v3'));args=parser.parse_args();audit(args.output)
