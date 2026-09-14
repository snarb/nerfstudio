"""C..L / A..E moving-actor loop and audited local head boundary completion."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,HELD_CAMERAS
from screen_travel_camera_flight import screen_path,portrait_projection
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from repair_temporal_head_boundaries import repair
from complete_head_depth_notches import complete

PARENT=Path('/mnt/data/dec5_screen_travel_dynamic_150_v2')
OUTPUT=Path('/mnt/data/dec5_expanded_head_dynamic_150_v3')
REPAIRS=Path('/mnt/data/dec5_expanded_head_repairs_preserve_carving')
RIG_POLYGON=np.array([[-5.,1.],[-4.,2.],[4.,2.],[4.,-2.],[-5.,-2.]])

def polygon_weights(point,polygon=RIG_POLYGON):
    delta=polygon-point;length=np.linalg.norm(delta,axis=1)
    unit=delta/length[:,None];nxt=np.roll(unit,-1,axis=0)
    cross=unit[:,0]*nxt[:,1]-unit[:,1]*nxt[:,0]
    tangent=np.abs(cross)/(1+(unit*nxt).sum(1)).clip(1e-12)
    weights=(tangent+np.roll(tangent,1))/length;weights/=weights.sum()
    if not np.allclose(weights@polygon,point,atol=1e-8):raise ValueError('Outside calibrated polygon')
    return weights

def expanded_path(rows,pilot,count=150):
    old,report=screen_path(rows,pilot,count);lookup={r['physical_camera'][:9]:r for r in rows}
    anchors=[lookup[n] for n in ['C004_B005','D004_A005','L004_A005','L004_E005','C004_E005']]
    if set(r['physical_camera'] for r in anchors)&HELD_CAMERAS:raise ValueError('Held-out anchor')
    corners=np.array([r['transform_matrix'] for r in anchors])[:,:3,3]
    target=np.array(report['fixed_target']);up=np.array(lookup['H004_C005']['transform_matrix'])[:3,0]
    result=[]
    for index,previous in enumerate(old):
        row=deepcopy(previous);oldweights=np.array(row['convex_weights']);u=oldweights[1]+oldweights[2];v=oldweights[2]+oldweights[3]
        xy=np.array([-5+9*u,2-4*v]);weights=polygon_weights(xy)
        position=weights@corners;z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y)
        look=np.column_stack((np.cross(y,z),y,z));desired=np.array(row['scene_landmark_portrait_xy'])
        native=np.array([row['w']-1-desired[1],desired[0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1.]);ray/=np.linalg.norm(ray)
        rotation=Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        pose=np.eye(4);pose[:3,:3]=look@rotation;pose[:3,3]=position
        row.update(transform_matrix=pose.tolist(),physical_camera=f'expanded_head_{index:05d}',rig_offset_xy=xy.tolist(),convex_weights=weights.tolist())
        if not np.allclose(portrait_projection(target,row),desired,atol=1e-7):raise ValueError('Wrong screen composition')
        result.append(row)
    report.update(path_kind='expanded_head',anchors=[r['physical_camera'] for r in anchors],columns='C..L',vertical_rows='A..E',
        previous_columns='D..K',requested_horizontal_expansion_columns_each_side=1,vertical_top_expansion_unavailable=True,
        reason='A is already the highest of five real rows; no extrapolation beyond calibrated camera hull',
        trajectory='saved_loop_phase_widened_C_to_L_A_to_E_with_screen_travel',rig_polygon=RIG_POLYGON.tolist(),
        missing_corner='C/A does not exist; smooth mean-value coordinates over five real camera anchors')
    return result,report

def initialize(output,frames=None,matched=False):
    previous=verify_request(PARENT);request=deepcopy(previous);cal=read(CALIBRATION);rows,_,meta=cameras('000973')
    pilot=read(previous['reference_camera_pilot']['path']);path,report=expanded_path(rows,pilot)
    if not matched:
        request['recipe']['camera_path_kind']='expanded_head';request['camera_path_report']=report
    request['recipe'].update(local_head_boundary_completion=True,preserve_existing_semantic_carving=True)
    selected=[r for r in request['inventory'] if frames is None or r['frame_id'] in frames]
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda r:repair(r,REPAIRS),selected))
    done=dict(zip([r['frame_id'] for r in selected],results))
    for record,p in zip(request['inventory'],path):
        frame=record['frame_id']
        if not matched:record['camera']=normalize_frame(calibration_pose(p,cal,read(meta)),cal,read(record['metadata']))
        if frame in done:
            record['mesh']=str(REPAIRS/frame/'mesh.ply');record['mesh_sha256']=done[frame]['mesh_sha256']
            record['head_repair_receipt']=str(REPAIRS/frame/'complete.json');record['head_repair_receipt_sha256']=sha(REPAIRS/frame/'complete.json')
            record.pop('target_restoration_receipt',None);record.pop('target_restoration_receipt_sha256',None)
    def notch(record):
        if record['frame_id'] not in done:return record
        result=complete(record,output/'geometry');record['pre_notch_mesh']=record['mesh'];record['pre_notch_mesh_sha256']=record['mesh_sha256']
        record['mesh']=str(output/'geometry'/record['frame_id']/'mesh.ply');record['mesh_sha256']=result['mesh_sha256']
        record['notch_receipt']=str(output/'geometry'/record['frame_id']/'complete.json');record['notch_receipt_sha256']=sha(record['notch_receipt']);return record
    with ThreadPoolExecutor(max_workers=4) as pool:request['inventory']=list(pool.map(notch,request['inventory']))
    for name in ['expanded_head_camera_flight.py','repair_temporal_head_boundaries.py','local_mesh_repair.py','complete_head_depth_notches.py']:
        request['script_hashes'][name]=sha(Path(__file__).with_name(name))
    request['repair_parent']=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json'))
    request['partial_diagnostic_only']=frames is not None
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable request mismatch')
    atomic_json(output/'request.json',request);print('initialized',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--frames',nargs='+');p.add_argument('--matched-control',action='store_true');a=p.parse_args()
    initialize(a.output,a.frames,a.matched_control)
