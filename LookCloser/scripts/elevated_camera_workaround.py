"""Elevated shot envelope to avoid exposed under-chin reconstruction boundaries."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from artifact_aware_camera_flight import avoidance_path
from expanded_head_camera_flight import polygon_weights
from screen_travel_camera_flight import portrait_projection
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame

PARENT=Path('/mnt/data/dec5_camera_avoidance_dynamic_150')
OUTPUT=Path('/mnt/data/dec5_elevated_camera_dynamic_150')

def elevated_path(rows,pilot,count=150):
    path,report=avoidance_path(rows,pilot,count);lookup={r['physical_camera']:r for r in rows}
    corners=np.array([lookup[n]['transform_matrix'] for n in report['anchors']])[:,:3,3]
    target=np.array(report['fixed_target']);up=np.array(next(r for r in rows if r['physical_camera'].startswith('H004_C005'))['transform_matrix'])[:3,0]
    # Flattening the vertical envelope without retiming causes slow turns and
    # fast straights. Resample the periodic curve by actual 3D camera distance.
    dense,_=avoidance_path(rows,pilot,1200);xy=np.array([r['rig_offset_xy'] for r in dense]);xy[:,1]=.2+(xy[:,1]+1.92)/2.87*.75
    positions=np.array([polygon_weights(p)@corners for p in xy])
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.vstack([positions,positions[:1]]),axis=0),axis=1))]
    sampled=CubicSpline(length,np.vstack([xy,xy[:1]]),axis=0,bc_type='periodic')(np.arange(count)/count*length[-1])
    for index,(row,(x,y)) in enumerate(zip(path,sampled)):
        weights=polygon_weights(np.array([x,y]));position=weights@corners
        z=position-target;z/=np.linalg.norm(z);axis_y=np.cross(z,up);axis_y/=np.linalg.norm(axis_y)
        look=np.column_stack((np.cross(axis_y,z),axis_y,z));desired=np.array(row['scene_landmark_portrait_xy'])
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1.]);ray/=np.linalg.norm(ray)
        pose=np.eye(4);pose[:3,:3]=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix();pose[:3,3]=position
        row.update(transform_matrix=pose.tolist(),rig_offset_xy=[x,y],convex_weights=weights.tolist(),physical_camera=row['physical_camera'].replace('artifact_avoidance','elevated_workaround'),arc_length_fraction=index/count)
        if not np.allclose(portrait_projection(target,row),desired,atol=1e-7):raise ValueError('Wrong composition')
    report.update(camera_envelope=dict(left=-.8,right=3.82,bottom=.2,top=.95),
        trajectory='periodic_cubic_spline_elevated_loop_resampled_by_3d_arc_length',vertical_motion_reduced_for_artifact_avoidance=True,
        extrema_indices=dict(left=int(sampled[:,0].argmin()),right=int(sampled[:,0].argmax()),
                             bottom=int(sampled[:,1].argmin()),top=int(sampled[:,1].argmax())))
    return path,report

def initialize(output,frames=None):
    request=deepcopy(verify_request(PARENT));cal=read(CALIBRATION);rows,_,meta=cameras('000973')
    path,report=elevated_path(rows,read(request['reference_camera_pilot']['path']))
    request['recipe'].update(camera_path_variant='elevated',camera_envelope_workaround=report['camera_envelope'])
    request['camera_path_report']=report
    for record,p in zip(request['inventory'],path):
        record['camera']=normalize_frame(calibration_pose(p,cal,read(meta)),cal,read(record['metadata']))
        if sha(record['mesh'])!=record['mesh_sha256']:raise ValueError('Changed fixed geometry')
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    request['elevated_camera_parent']=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json'))
    request['partial_diagnostic_only']=frames is not None
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable request mismatch')
    atomic_json(output/'request.json',request);print('initialized',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--frames',nargs='+');a=p.parse_args();initialize(a.output,a.frames)
