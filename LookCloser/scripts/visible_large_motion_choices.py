"""Time-scheduled camera tilt to exclude the broken lowering forearm.

Same physical 51–54-degree arcs and fixed 0.85x lens as framed v2. A smooth
analytic lower composition during hand descent is a real camera orientation,
not a tracked ROI/crop. Failed v1/v2 geometry gates remain separate artifacts.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import large_motion_camera_choices as core
import framed_large_motion_choices as framed
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,cameras
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

BASE=Path('/mnt/data/dec5_large_motion_choices_v3')
PARENT=core.PARENT;VARIANTS=core.VARIANTS;CANARIES=core.CANARIES


def path_for(variant,parent):
    raw,report=framed.path_for(variant,parent);cal=read(CALIBRATION);rows,_,metadata=cameras('000973');meta=read(metadata)
    target=np.array(report['fixed_target']);up=np.array(next(r for r in rows if r['physical_camera'].startswith('H004_C005'))['transform_matrix'])[:3,0]
    result=[]
    for i,row in enumerate(raw):
        row=normalize_frame(row,cal,meta);pose=np.array(row['transform_matrix']);position=pose[:3,3]
        desired=np.array(row['scene_landmark_portrait_xy']);desired[1]+=400*np.exp(-((i-70)/19)**4)
        z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y);look=np.column_stack((np.cross(y,z),y,z))
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1]);ray/=np.linalg.norm(ray)
        pose[:3,:3]=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        row.update(transform_matrix=pose.tolist(),scene_landmark_portrait_xy=desired.tolist(),physical_camera=f'visible_{variant}_{i:05d}')
        np.testing.assert_allclose(portrait_projection(target,row),desired,atol=1e-6);result.append(calibration_pose(row,cal,meta))
    poses=np.array([r['transform_matrix'] for r in result]);angles=np.degrees(np.arccos(np.clip(poses[:,:3,2]@poses[:,:3,2].T,-1,1)))
    report.update(view_angle_extent_degrees=float(angles.max()),projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in result],axis=0).tolist(),
        descent_composition_rule='add 400*exp(-((index-70)/19)^4) portrait y pixels by camera orientation; no post-render transform',
        framing_gate_revision='v1 crown clipping and v2 exposed forearm rejected in clay; v3 analytic descent composition requires actual RGB review',
        previous_geometry_gate_root=str(framed.BASE))
    return result,report


def initialize():
    parent=verify_request(PARENT);cal=read(CALIBRATION)
    for variant in VARIANTS:
        path,report=path_for(variant,parent);request=deepcopy(parent)
        for record,row in zip(request['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        request['camera_path_report']=report;request['recipe'].update(camera_path_variant='visible_large_motion_'+variant,camera_periodic=False,fixed_focal_multiplier=.85)
        request.update(partial_diagnostic_only=False,full_video_candidate=True,artifact_free_approval=False,inherited_actor_inventory_unchanged=True,
            inherited_camera_and_actor_inventory_unchanged=False,required_initial_rgb_gate=CANARIES,
            initial_gate=dict(status='requires_actual_clay_rgb_and_motion_review'),
            camera_workaround_parent=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json')))
        for name in ['large_motion_camera_choices.py','framed_large_motion_choices.py',Path(__file__).name]:request['script_hashes'][name]=sha(Path(__file__).with_name(name))
        out=BASE/variant;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
        if (out/'request.json').exists():assert read(out/'request.json')==request,'Immutable request mismatch'
        atomic_json(out/'request.json',request);print(variant,report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','screen','canary','supervise','worker']);p.add_argument('--index',type=int);p.add_argument('--assignment',type=Path);a=p.parse_args()
    if a.action=='init':initialize()
    elif a.action in ['screen','canary','supervise']:
        core.BASE=BASE;core.__file__=__file__
        if a.action=='screen':core.screen()
        else:core.supervise(a.action=='canary')
    else:
        import torch
        from run_view_consistent_dynamic_video import worker
        torch.set_num_threads(2);assignment=read(a.assignment)
        for variant in VARIANTS:
            frames=[f for v,f in assignment if v==variant]
            if frames:(BASE/variant/'workers').mkdir(exist_ok=True);worker(BASE/variant,a.index,frames)
