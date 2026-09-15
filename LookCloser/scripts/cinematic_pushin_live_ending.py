"""Opt-in v4: C2 camera rest at118, disclosed live-action ending after125.

Raw mesh frames0..125 remain separate from the parent's presentation compositor.
The physical curve is v3's curve, evaluated continuously at i*126/118.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import cinematic_pushin_beauty as previous
from cinematic_pushin_choices import ease
from cinematic_pushin_framed import timing
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

BASE=Path('/mnt/data/dec5_cinematic_pushin_v4');PARENT=previous.PARENT;VARIANTS=previous.VARIANTS
CANARY_INDICES=[0,20,48,54,65,69,73,100,112,118,122,125]
CANARIES=[f'{899+2*i:06d}' for i in CANARY_INDICES]
RAW_IDS=[f'{899+2*i:06d}' for i in range(126)]

def path_for(variant,parent):
    _,report=previous.path_for(variant,parent)
    rows,_,metapath=cameras('000973');meta=read(metapath);cal=read(CALIBRATION)
    lookup={r['physical_camera'][:9]:r for r in rows};end=lookup['H004_C005'];endpose=np.array(end['transform_matrix'])
    target=np.array(report['fixed_target']);up=endpose[:3,0]
    control=np.array(report['reference_control_centers'])
    rig=np.array([sum(w*np.array([ord(n[0])-ord('H'),ord('C')-ord(n[5])]) for n,w in spec) for spec in report['control_mixtures']])
    q=np.minimum(np.arange(150)*126/118,126);u=timing(q/126)
    weights=np.column_stack(((1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3));positions=weights@control;xy=weights@rig
    beauty=variant in ['free_arc','soft_diagonal'];result=[]
    for i,position in enumerate(positions):
        row=deepcopy(end);base=.85+.6*u[i];late=float(ease((q[i]-80)/46));focal=.85+.6*ease(q[i]/95)+(.45 if beauty else .20)*late
        z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y);rotation=np.column_stack((np.cross(y,z),y,z))
        if variant!='locked_arc':
            amp={'free_arc':(100.,90.),'rising_arc':(-70.,100.),'soft_diagonal':(75.,-60.)}[variant];gesture=np.sin(np.pi*u[i])**2
            ray=np.array([-amp[1]*gesture/(row['fl_x']*base),-amp[0]*gesture/(row['fl_y']*base),-1.]);ray/=np.linalg.norm(ray)
            rotation=rotation@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=position
        if i>=118:pose=endpose.copy()
        row.update(transform_matrix=pose.tolist(),physical_camera=f'cinematic_live_{variant}_{i:05d}',rig_offset_xy=xy[i].tolist())
        row['fl_x']*=float(focal);row['fl_y']*=float(focal);row['cx']+=400*late if beauty else 0
        row['scene_landmark_portrait_xy']=portrait_projection(target,row).tolist();result.append(calibration_pose(row,cal,meta))
    report.update(hold_start_index=118,hold_count=32,hold_actual_times=[f'{899+2*i:06d}' for i in range(118,150)],
        easing='Continuous v3 curve evaluated at i*126/118; C2 rest at118. Acceleration9.365 frames, cruise78.667, deceleration29.968.',
        focal_easing='Both v3 optical ramps evaluated at i*126/118; all intrinsics fixed118..149',
        virtual_sensor_shift_easing='v3 quintic evaluated at i*126/118; fixed118..149',physical_path_identical_to_v2=True,
        presentation=dict(raw_mesh_indices=[0,125],dissolve_indices=[118,125],pure_train_indices=[126,149],
            endpoint_source='H004_C005_1210SZ',native_background_preserved=True,final_second_is_3d=False),
        projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in result],axis=0).tolist())
    return result,report

def initialize():
    parent=verify_request(PARENT);cal=read(CALIBRATION)
    for variant in VARIANTS:
        rows,report=path_for(variant,parent);req=deepcopy(verify_request(previous.BASE/variant))
        for record,row in zip(req['inventory'],rows):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        req['camera_path_report']=report;req['recipe']['camera_path_variant']='cinematic_live_v4_'+variant
        req.update(required_initial_rgb_gate=CANARIES,raw_render_frame_ids=RAW_IDS,presentation_final_second='real train RGB, not mesh prediction')
        req['script_hashes'][Path(__file__).name]=sha(__file__)
        root=BASE/variant;root.mkdir(parents=True,exist_ok=True);(root/'frames').mkdir(exist_ok=True)
        if (root/'request.json').exists():assert read(root/'request.json')==req
        atomic_json(root/'request.json',req);print(variant,'hold118, blend118..125, pure train126..149',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','screen']);a=p.parse_args()
    if a.action=='init':initialize()
    else:
        import large_motion_camera_choices as renderer
        renderer.BASE=BASE;renderer.VARIANTS=VARIANTS;renderer.CANARY_INDICES=CANARY_INDICES;renderer.screen()
