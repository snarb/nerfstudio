"""Opt-in dynamic camera flight strictly inset two physical rows from rig edges.

The five-level rig leaves only C vertically. This explicit constraint supersedes
the earlier wide two-axis path; it must not silently relax the user's edge margin.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame

PARENT=Path('/mnt/data/dec5_wide_dynamic_flight_150')
OUTPUT=Path('/mnt/data/dec5_interior_dynamic_flight_150')


def interior_path(rows,target,count=150,fps=24):
    if count<32 or fps<=0:raise ValueError('Invalid sampling')
    lookup={r['physical_camera'][:9]:r for r in rows}
    anchors=[lookup[n] for n in ['F004_C005','J004_C005']]
    if set(r['physical_camera'] for r in anchors)&HELD_CAMERAS:raise ValueError('Held anchor')
    corners=np.asarray([r['transform_matrix'] for r in anchors])[:,:3,3]
    reference=lookup['H004_C005'];up=np.array(reference['transform_matrix'])[:3,0]
    # Periodic cubic eased out-and-back, zero velocity at the reversals. Unlike
    # arc-length normalization, this preserves a smooth physical reversal.
    spline=CubicSpline([0,.25,.5,.75,1],[-2,0,2,0,-2],bc_type='periodic')
    x=spline(np.arange(count)/count);u=(x+2)/4;weights=np.column_stack([1-u,u])
    path=[]
    for i,pos in enumerate(weights@corners):
        z=pos-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y)
        pose=np.eye(4);pose[:3,:3]=np.column_stack([np.cross(y,z),y,z]);pose[:3,3]=pos
        row=deepcopy(reference);row.pop('file_path',None)
        row.update(transform_matrix=pose.tolist(),physical_camera=f'interior_dynamic_{i:05d}',
                   convex_weights=weights[i].tolist(),rig_offset_xy=[float(x[i]),0.])
        path.append(row)
    poses=np.array([p['transform_matrix'] for p in path]);directions=poses[:,:3,2]
    angle=np.rad2deg(np.arccos(np.clip(directions@directions.T,-1,1))).max()
    if weights.min()<-1e-12 or np.ptp(x)<3.99:raise ValueError('Missing interior motion')
    return path,dict(path_kind='interior_inset_two',anchors=[r['physical_camera'] for r in anchors],
        center_camera=reference['physical_camera'],rig_columns='ABCDEFGHIJKLMN',rig_vertical_rows='ABCDE',
        edge_margin_intervals=2,allowed_absolute_column_range=[2,11],allowed_absolute_row_range=[2,2],
        horizontal_requested_offsets=[-2,2],horizontal_achieved_offsets=[float(x.min()),float(x.max())],
        vertical_available_offsets=[0,0],vertical_achieved_offsets=[0.,0.],vertical_limit_disclosed=True,
        extrema_indices=dict(left=0,right=int(x.argmax())),
        trajectory='periodic_cubic_horizontal_eased_out_and_back_on_C',continuous_periodic_camera=True,
        fixed_target=target.tolist(),fixed_intrinsics=True,actor_clip_periodic=False,frames=count,fps=fps,
        duration_seconds=count/fps,maximum_pairwise_view_angle_degrees=float(angle),
        minimum_convex_weight=float(weights.min()),portrait_up_axis='reference native camera X',
        prior_path_not_identical_reason='Prior 4x4 uses forbidden outer/penultimate vertical rows; edge constraint takes precedence')


def initialize(output):
    old=verify_request(PARENT);cal=read(CALIBRATION);rows,_,meta=cameras('000973')
    target=np.asarray(old['camera_path_report']['fixed_target'])
    path,report=interior_path(rows,target)
    request=deepcopy(old)
    request['recipe'].update(camera_path_kind='interior_inset_two',horizontal_radius_rows=2,
        vertical_radius_rows=0,camera_periodic=True,fps=24)
    request['camera_path_report']=report
    request['superseded_wide_request']=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json'))
    for record,p in zip(request['inventory'],path):
        record['camera']=normalize_frame(calibration_pose(p,cal,read(meta)),cal,read(record['metadata']))
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable request mismatch')
    atomic_json(output/'request.json',request)
    atomic_json(output/'progress.json',dict(stage='initialized',complete=0,total=150))
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    initialize(p.parse_args().output)
