"""Wider real 3D camera translation with explicitly non-centered composition.

No crop, stabilization, image translation or changing focal length. A fixed
virtual lens is wider than the physical source lens. Camera orientation leaves
the fixed scene landmark traveling across the image, instead of centering it.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,HELD_CAMERAS
from replay_dynamic_camera_flight import replay_path,PILOT,PARENT
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame

OUTPUT=Path('/mnt/data/dec5_screen_travel_dynamic_150')


def optical_target(poses):
    p=np.asarray(poses);axis=p[:,:3,2];a=np.eye(3)-axis[:,:,None]*axis[:,None,:]
    return np.linalg.solve(a.sum(0),np.einsum('nij,nj->i',a,p[:,:3,3]))


def portrait_projection(point,row):
    p=np.asarray(row['transform_matrix']);q=p[:3,:3].T@(point-p[:3,3]);depth=-q[2]
    u=row['cx']+row['fl_x']*q[0]/depth;v=row['cy']-row['fl_y']*q[1]/depth
    return np.array([v,row['w']-1-u])


def screen_path(rows,pilot,count=150):
    old=replay_path(pilot,count);target=optical_target([p['transform_matrix'] for p in pilot['path']])
    lookup={r['physical_camera'][:9]:r for r in rows}
    anchors=[lookup[n] for n in ['D004_A005','K004_A005','K004_D005','D004_D005']]
    if set(r['physical_camera'] for r in anchors)&HELD_CAMERAS:raise ValueError('Held camera anchor')
    positions=np.array([r['transform_matrix'] for r in anchors])[:,:3,3]
    ref=lookup['H004_C005'];up=np.array(ref['transform_matrix'])[:3,0];result=[]
    for i,previous in enumerate(old):
        weights=np.array(previous['convex_weights']);u=weights[1]+weights[2];v=weights[2]+weights[3]
        pos=weights@positions;z=pos-target;z/=np.linalg.norm(z)
        y=np.cross(z,up);y/=np.linalg.norm(y);look=np.column_stack([np.cross(y,z),y,z])
        row=deepcopy(ref);row.pop('file_path',None);row['fl_x']*=.70;row['fl_y']*=.70
        desired=np.array([540-200*(2*u-1),959+80*(2*v-1)])
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1.])
        ray/=np.linalg.norm(ray)
        rotation=Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        pose=np.eye(4);pose[:3,:3]=look@rotation;pose[:3,3]=pos
        row.update(transform_matrix=pose.tolist(),physical_camera=f'screen_travel_{i:05d}',
            convex_weights=weights.tolist(),rig_offset_xy=[float(-4+7*u),float(2-3*v)],
            scene_landmark_portrait_xy=desired.tolist(),pilot_sample_phase=previous['pilot_sample_phase'])
        if not np.allclose(portrait_projection(target,row),desired,atol=1e-7):raise ValueError('Wrong camera composition')
        result.append(row)
    return result,dict(path_kind='screen_travel',anchors=[r['physical_camera'] for r in anchors],
        fixed_target=target.tolist(),trajectory='saved_loop_phase_widened_D_to_K_with_screen_travel',
        extrema_indices=dict(right=0,bottom=37,left=75,top=112),frames=count,fps=24,
        fixed_virtual_focal_scale=.70,image_crop=False,image_stabilization=False,post_render_translation=False,
        actor_clip_periodic=False,continuous_periodic_camera=True,
        requested_horizontal_expansion_columns_each_side=2,previous_columns='F..I',columns='D..K',vertical_rows='A..D',
        projected_landmark_extent_pixels=np.ptp([portrait_projection(target,r) for r in result],axis=0).tolist())


def initialize(output):
    old=verify_request(PARENT);pilot=read(PILOT/'request.json');cal=read(CALIBRATION);rows,_,meta=cameras('000973')
    path,report=screen_path(rows,pilot);request=deepcopy(old)
    request['recipe'].update(camera_path_kind='screen_travel',camera_periodic=True,fps=24)
    request['camera_path_report']=report
    request['reference_camera_pilot']=dict(path=str(PILOT/'request.json'),sha256=sha(PILOT/'request.json'))
    for record,p in zip(request['inventory'],path):
        record['camera']=normalize_frame(calibration_pose(p,cal,read(meta)),cal,read(record['metadata']))
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    request['script_hashes']['replay_dynamic_camera_flight.py']=sha(Path(__file__).with_name('replay_dynamic_camera_flight.py'))
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable request mismatch')
    atomic_json(output/'request.json',request);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    initialize(p.parse_args().output)
