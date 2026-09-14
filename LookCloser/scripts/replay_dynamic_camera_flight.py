"""Replay the verified static 4x4 camera loop over 150 real scene instants.

Reuses saved positions AND rotations; it never rebuilds a look-at camera. The
full 720-pose loop is resampled, not truncated to its first 150 poses. Geometry,
RGB time and camera normalization remain separately bound in the request.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame

PILOT=Path('/mnt/data/dec5_camera_grid_diagnosis_v2/4x4')
PARENT=Path('/mnt/data/dec5_wide_dynamic_flight_150')
OUTPUT=Path('/mnt/data/dec5_replayed_4x4_dynamic_150')


def replay_path(pilot,count=150):
    source=pilot['path'];n=len(source)
    if n<4 or count<4:raise ValueError('Insufficient loop samples')
    poses=np.asarray([r['transform_matrix'] for r in source])
    weights=np.asarray([r['convex_weights'] for r in source])
    phase=np.arange(count)*n/count
    ww=CubicSpline(np.arange(n+1),np.vstack([weights,weights[:1]]),bc_type='periodic')(phase)
    xyz=CubicSpline(np.arange(n+1),np.vstack([poses[:,:3,3],poses[:1,:3,3]]),bc_type='periodic')(phase)
    rr=RotationSpline(np.arange(n+1),Rotation.from_matrix(np.concatenate([poses[:,:3,:3],poses[:1,:3,:3]])))(phase).as_matrix()
    if ww.min()<0 or not np.allclose(ww.sum(1),1):raise ValueError('Replay escaped saved camera hull')
    result=[]
    for i in range(count):
        row=deepcopy(source[0]);p=np.eye(4);p[:3,:3]=rr[i];p[:3,3]=xyz[i]
        row.update(transform_matrix=p.tolist(),convex_weights=ww[i].tolist(),
            physical_camera=f'replayed_4x4_{i:05d}',pilot_sample_phase=float(phase[i]),
            rig_offset_xy=[float(-2+3*(ww[i,1]+ww[i,2])),float(2-3*(ww[i,2]+ww[i,3]))])
        result.append(row)
    return result


def initialize(output):
    pilot=read(PILOT/'request.json');receipt=read(PILOT/'result.json')
    if sha(PILOT/'video.mp4')!=receipt['video_sha256']:raise ValueError('Changed chosen pilot video')
    old=verify_request(PARENT);cal=read(CALIBRATION);_,_,meta=cameras('000973')
    path=replay_path(pilot);request=deepcopy(old)
    request['recipe'].update(camera_path_kind='replay_static_4x4',camera_periodic=True,fps=24)
    request['recipe'].pop('horizontal_radius_rows',None);request['recipe'].pop('vertical_radius_rows',None)
    angles=np.asarray([p['transform_matrix'] for p in path])[:,:3,2]
    request['camera_path_report']=dict(path_kind='replay_static_4x4',anchors=pilot['report']['anchors'],
        extrema_indices=dict(right=0,bottom=37,left=75,top=112),
        trajectory='same_saved_static_4x4_loop_positions_and_orientations',frames=150,fps=24,
        duration_seconds=6.25,pilot_duration_seconds=30,pilot_sample_count=720,
        camera_speed_vs_pilot=4.8,actor_clip_periodic=False,continuous_periodic_camera=True,
        maximum_pairwise_view_angle_degrees=float(np.rad2deg(np.arccos(np.clip(angles@angles.T,-1,1))).max()),
        edge_margin_note='Exact requested earlier 4x4 supersedes interim two-row inset restriction; spans F..I / A..D',
        pilot_request=dict(path=str(PILOT/'request.json'),sha256=sha(PILOT/'request.json')),
        pilot_video=dict(path=str(PILOT/'video.mp4'),sha256=sha(PILOT/'video.mp4')),
        pilot_metadata=dict(path=str(meta),sha256=sha(meta)))
    for record,p in zip(request['inventory'],path):
        record['camera']=normalize_frame(calibration_pose(p,cal,read(meta)),cal,read(record['metadata']))
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable replay mismatch')
    atomic_json(output/'request.json',request)
    atomic_json(output/'progress.json',dict(stage='initialized',complete=0,total=150))
    print(request['camera_path_report'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    initialize(p.parse_args().output)
