"""Fixed-lens closer portrait camera workaround; never a post-render crop.

Retains S-curve camera translation and all production actor meshes/times, while
the dropping lower arm naturally leaves the narrower field of view.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,cameras
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection
from artifact_aware_video_variants import PARENT,BASE,path_for

OUTPUT=BASE/'closer_portrait'
CANARIES=['000899','000995','001029','001037','001193']


def portrait_path(parent):
    old,report=path_for('s_curve',parent);cal=read(CALIBRATION);rows,_,metadata=cameras('000973');meta=read(metadata)
    target=np.array(report['fixed_target']);up=np.array(next(r for r in rows if r['physical_camera'].startswith('H004_C005'))['transform_matrix'])[:3,0]
    result=[]
    for i,raw in enumerate(old):
        row=normalize_frame(raw,cal,meta);pose=np.array(row['transform_matrix']);position=pose[:3,3]
        row['fl_x']*=1.7;row['fl_y']*=1.7
        phase=2*np.pi*i/149
        desired=np.array([520+90*np.sin(phase),1110+20*np.cos(phase)])
        z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y)
        look=np.column_stack((np.cross(y,z),y,z))
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1]);ray/=np.linalg.norm(ray)
        pose[:3,:3]=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        row.update(transform_matrix=pose.tolist(),scene_landmark_portrait_xy=desired.tolist(),physical_camera=f'closer_portrait_{i:05d}')
        np.testing.assert_allclose(portrait_projection(target,row),desired,atol=1e-6)
        result.append(calibration_pose(row,cal,meta))
    poses=np.array([r['transform_matrix'] for r in result]);angles=np.degrees(np.arccos(np.clip(poses[:,:3,2]@poses[:,:3,2].T,-1,1)))
    report.update(variant='closer_portrait',trajectory='same_physical_S_curve_with_fixed_closer_lens_and_analytic_composition',
        fixed_focal_multiplier_from_production=1.7,composition_rule='x=520+90*sin(2*pi*i/149), y=1110+20*cos(2*pi*i/149)',
        projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in result],axis=0).tolist(),
        view_angle_extent_degrees=float(angles.max()),image_crop=False,tracked_ROI=False,
        intended_tradeoff='Keep crown and painting hand; naturally exclude lower forearm as hand drops. Requires actual RGB verification.')
    return result,report


def initialize():
    parent=verify_request(PARENT);request=deepcopy(parent);cal=read(CALIBRATION);path,report=portrait_path(parent)
    for record,row in zip(request['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
    request['camera_path_report']=report
    request['recipe'].update(camera_path_variant='closer_portrait',camera_periodic=False,fixed_portrait_focal_multiplier=1.7)
    request.update(partial_diagnostic_only=False,full_video_candidate=True,artifact_free_approval=False,
        inherited_camera_and_actor_inventory_unchanged=False,inherited_actor_inventory_unchanged=True,
        required_initial_rgb_gate=CANARIES,initial_gate=dict(status='requires_actual_crown_hand_and_forearm_canary_review'),
        camera_workaround_parent=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json')))
    for name in ['artifact_aware_video_variants.py',Path(__file__).name]:request['script_hashes'][name]=sha(Path(__file__).with_name(name))
    OUTPUT.mkdir(exist_ok=True);(OUTPUT/'frames').mkdir(exist_ok=True);(OUTPUT/'workers').mkdir(exist_ok=True)
    if (OUTPUT/'request.json').exists() and read(OUTPUT/'request.json')!=request:raise ValueError('Immutable portrait request mismatch')
    atomic_json(OUTPUT/'request.json',request);print(report,flush=True)


def panels():
    from PIL import Image,ImageDraw
    root=OUTPUT/'review';root.mkdir(exist_ok=True)
    for frame in CANARIES:
        paths=[BASE/'s_curve'/'frames'/frame/'frame.png',OUTPUT/'frames'/frame/'frame.png']
        if not all(p.exists() for p in paths):continue
        panel=Image.new('RGB',(1080,984));draw=ImageDraw.Draw(panel)
        for j,(path,label) in enumerate(zip(paths,['S curve','closer portrait / fixed lens'])):
            panel.paste(Image.open(path).resize((540,960)),(j*540,24));draw.text((j*540+3,3),label+' '+frame,fill='white')
        panel.save(root/(frame+'_comparison.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','canary','panels']);a=p.parse_args()
    if a.action=='init':initialize()
    elif a.action=='panels':panels()
    else:
        import torch
        from run_view_consistent_dynamic_video import worker
        torch.set_num_threads(2);worker(OUTPUT,0,CANARIES)
