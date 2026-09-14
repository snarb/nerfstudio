"""Opt-in smooth camera-envelope workaround, not a reconstruction improvement.

Keep the dynamic source-time sequence and saved loop phase. Compress only the
camera envelope away from the upper-left views exposing known head defects.
No per-frame snapping, stabilization, RGB replacement, or held-out fitting.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read, sha, atomic_json, cameras, CALIBRATION
from expanded_head_camera_flight import expanded_path, polygon_weights
from screen_travel_camera_flight import portrait_projection
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from render_patchmatch_camera_path import normalize_frame

PARENT=Path('/mnt/data/dec5_expanded_head_dynamic_150_v3')
OUTPUT=Path('/mnt/data/dec5_camera_avoidance_dynamic_150')
LIMITS=dict(left=-.8,right=3.82,bottom=-1.92,top=.95)

def avoidance_path(rows,pilot,count=150):
    path,report=expanded_path(rows,pilot,count)
    lookup={r['physical_camera']:r for r in rows}
    corners=np.array([lookup[name]['transform_matrix'] for name in report['anchors']])[:,:3,3]
    target=np.array(report['fixed_target'])
    center=next(r for r in rows if r['physical_camera'].startswith('H004_C005'))
    up=np.array(center['transform_matrix'])[:3,0]
    for row in path:
        x,y=row['rig_offset_xy']
        x=LIMITS['left']+(x+4.82)/8.64*(LIMITS['right']-LIMITS['left'])
        y=LIMITS['bottom']+(y+1.92)/3.84*(LIMITS['top']-LIMITS['bottom'])
        weights=polygon_weights(np.array([x,y]));position=weights@corners
        z=position-target;z/=np.linalg.norm(z);axis_y=np.cross(z,up);axis_y/=np.linalg.norm(axis_y)
        look=np.column_stack((np.cross(axis_y,z),axis_y,z))
        desired=np.array(row['scene_landmark_portrait_xy'])
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1.]);ray/=np.linalg.norm(ray)
        rotation=Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        pose=np.eye(4);pose[:3,:3]=look@rotation;pose[:3,3]=position
        row.update(transform_matrix=pose.tolist(),rig_offset_xy=[x,y],convex_weights=weights.tolist(),
                   physical_camera=row['physical_camera'].replace('expanded_head','artifact_avoidance'))
        if not np.allclose(portrait_projection(target,row),desired,atol=1e-7):raise ValueError('Wrong composition')
    report.update(path_kind='artifact_avoidance',camera_envelope=LIMITS,
        trajectory='smooth_affine_envelope_of_existing_loop',geometry_fixed_by_camera_path=False,
        rationale='User-authorized shot workaround for upper-left exposed head defects; not geometric recovery')
    return path,report

def initialize(output,frames=None):
    request=deepcopy(verify_request(PARENT));cal=read(CALIBRATION);rows,_,meta=cameras('000973')
    path,report=avoidance_path(rows,read(request['reference_camera_pilot']['path']))
    request['recipe'].update(camera_path_kind='artifact_avoidance',camera_envelope_workaround=LIMITS,
                            geometry_identical_to_camera_workaround_parent=False,
                            camera_workaround_geometry_stage='pre_notch_mesh')
    request['camera_path_report']=report
    for record,p in zip(request['inventory'],path):
        record['camera']=normalize_frame(calibration_pose(p,cal,read(meta)),cal,read(record['metadata']))
        # Provenance controls show view-conditioned patches become stretched
        # membranes from another camera. Use the fixed boundary-only stage,
        # identical for every pose; never refit patches to hide the new shot.
        record['mesh']=record['pre_notch_mesh'];record['mesh_sha256']=record['pre_notch_mesh_sha256']
        if sha(record['mesh'])!=record['mesh_sha256']:raise ValueError('Changed fixed parent mesh')
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    request['camera_workaround_parent']=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json'))
    request['partial_diagnostic_only']=frames is not None
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable request mismatch')
    atomic_json(output/'request.json',request);print('initialized',output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--frames',nargs='+')
    args=parser.parse_args();initialize(args.output,args.frames)
