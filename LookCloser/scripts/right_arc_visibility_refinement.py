"""One bounded refinement of the failed right-arc RGB visibility gate."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import visible_large_motion_choices as previous
from large_motion_camera_choices import smoothstep
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

BASE=previous.BASE;PARENT=previous.PARENT;CANARIES=previous.CANARIES
VARIANTS=('diagonal_sweep','wide_oval','left_high_arc','right_high_arc_refined')


def path_for(variant,parent):
    if variant!='right_high_arc_refined':return previous.path_for(variant,parent)
    raw,report=previous.path_for('right_high_arc',parent);safe,_=previous.path_for('diagonal_sweep',parent)
    cal=read(CALIBRATION);rows,_,metadata=cameras('000973');meta=read(metadata)
    target=np.array(report['fixed_target']);up=np.array(next(r for r in rows if r['physical_camera'].startswith('H004_C005'))['transform_matrix'])[:3,0]
    result=[];positions=[];rig=[]
    for i,(row,safer) in enumerate(zip(raw,safe)):
        row=normalize_frame(row,cal,meta);safer=normalize_frame(safer,cal,meta);pose=np.array(row['transform_matrix'])
        w=smoothstep(i/60);position=w*pose[:3,3]+(1-w)*np.array(safer['transform_matrix'])[:3,3];pose[:3,3]=position;positions.append(position)
        rig.append(w*np.array(row['rig_offset_xy'])+(1-w)*np.array(safer['rig_offset_xy']))
        desired=np.array(row['scene_landmark_portrait_xy']);desired[1]+=150*np.exp(-((i-70)/19)**4)
        z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y);look=np.column_stack((np.cross(y,z),y,z))
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1]);ray/=np.linalg.norm(ray)
        pose[:3,:3]=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        row.update(transform_matrix=pose.tolist(),scene_landmark_portrait_xy=desired.tolist(),physical_camera=f'right_refined_{i:05d}',rig_offset_xy=rig[-1].tolist())
        np.testing.assert_allclose(portrait_projection(target,row),desired,atol=1e-6);result.append(calibration_pose(row,cal,meta))
    poses=np.array([r['transform_matrix'] for r in result]);rays=np.array(positions)-target;rays/=np.linalg.norm(rays,axis=1)[:,None]
    report.update(variant=variant,center_ray_angle_extent_degrees=float(np.degrees(np.arccos(np.clip(rays@rays.T,-1,1))).max()),
        view_angle_extent_degrees=float(np.degrees(np.arccos(np.clip(poses[:,:3,2]@poses[:,:3,2].T,-1,1))).max()),
        position_extent=np.ptp(poses[:,:3,3],axis=0).tolist(),projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in result],axis=0).tolist(),
        rig_parameter_min=np.min(rig,axis=0).tolist(),rig_parameter_max=np.max(rig,axis=0).tolist(),
        refinement='Early real centers blend from diagonal corridor to right arc with quintic smoothstep(index/60); add 150 descent composition pixels. No wrap, lens change, crop or geometry edit.',
        refinement_parent=str(BASE/'right_high_arc'/'request.json'),refinement_parent_sha256=sha(BASE/'right_high_arc'/'request.json'))
    assert report['center_ray_angle_extent_degrees']>45
    return result,report


def initialize():
    parent=verify_request(PARENT);cal=read(CALIBRATION);path,report=path_for('right_high_arc_refined',parent)
    request=deepcopy(verify_request(BASE/'right_high_arc'))
    for record,row in zip(request['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
    request['camera_path_report']=report;request['recipe']['camera_path_variant']='right_high_arc_refined'
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    out=BASE/'right_high_arc_refined';out.mkdir(exist_ok=True);(out/'frames').mkdir(exist_ok=True)
    if (out/'request.json').exists():assert read(out/'request.json')==request
    atomic_json(out/'request.json',request);print(report,flush=True)


if __name__=='__main__':initialize()
