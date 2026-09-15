"""Four larger open camera shots; unchanged production actor and radiometry.

Real train-camera convex interpolation spans C..K and interior B..D. The
five-row rig cannot provide large vertical travel while avoiding both outer
and penultimate rows entirely: we remain strictly between B and D, never A/E.
No time repetition, lens animation, image tracking/crop or geometry repair.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read, sha, atomic_json, cameras, CALIBRATION, HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

PARENT=Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
BASE=Path('/mnt/data/dec5_large_motion_choices_v1')
VARIANTS=('diagonal_sweep','wide_oval','left_high_arc','right_high_arc')
KNOTS=np.array([0,22,48,72,100,125,147,149],float)
CONTROLS={
 'diagonal_sweep': [[2,.2],[2.5,.7],[0,.75],[-4.3,.05],[-3.5,-.75],[-.8,-.5],[-1,0],[-1,0]],
 'wide_oval': [[2,.2],[1.4,.75],[-2.5,.7],[-4.3,0],[-2.2,-.75],[2.5,-.4],[-1,0],[-1,0]],
 'left_high_arc': [[2,.2],[2.5,.7],[-1.5,.75],[-4.3,.45],[-3,-.75],[.5,-.5],[-1,0],[-1,0]],
 'right_high_arc': [[2,.2],[-1.5,.75],[-4.3,.4],[-2,-.6],[2.5,-.5],[1.5,.75],[-1,0],[-1,0]],
}
CANARY_INDICES=[0,22,48,53,65,69,73,100,112,125,147,149]
CANARIES=[f'{899+2*i:06d}' for i in CANARY_INDICES]


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*x*(10-15*x+6*x*x)


def path_for(variant,parent):
    cal=read(CALIBRATION);rows,_,metadata=cameras('000973');meta=read(metadata)
    lookup={r['physical_camera'][:9]:r for r in rows}
    anchors=[lookup[n] for n in ['C004_B005','K004_B005','K004_D005','C004_D005']]
    assert not set(r['physical_camera'] for r in anchors)&HELD_CAMERAS
    corners=np.array([r['transform_matrix'] for r in anchors])[:,:3,3]
    index=np.arange(150);xy=CubicSpline(KNOTS,CONTROLS[variant],axis=0,bc_type='clamped')(index)
    assert xy[:,0].min()>-4.8 and xy[:,0].max()<2.95
    assert xy[:,1].min()>-.95 and xy[:,1].max()<.95
    u=(xy[:,0]+5)/8;v=(1-xy[:,1])/2
    weights=np.column_stack(((1-u)*(1-v),u*(1-v),u*v,(1-u)*v))
    assert weights.min()>0 and np.allclose(weights.sum(1),1)
    positions=weights@corners
    old=[normalize_frame(calibration_pose(r['camera'],cal,read(r['metadata'])),cal,meta) for r in parent['inventory']]
    # Convex blends, not a periodic displacement or extrapolated translation.
    early=smoothstep(index/22);late=smoothstep((index-125)/22)
    positions=(1-early[:,None])*np.array(old[0]['transform_matrix'])[:3,3]+early[:,None]*positions
    positions=(1-late[:,None])*positions+late[:,None]*np.array(lookup['G004_C005']['transform_matrix'])[:3,3]
    target=np.array(parent['camera_path_report']['fixed_target']);up=np.array(lookup['H004_C005']['transform_matrix'])[:3,0]
    # Broad, explicit non-centered composition, coupled to the open path.
    # 540 horizontal and 500 vertical pixels, with exact original first pose.
    desired=np.column_stack((540+270*(xy[:,0]-(xy[:,0].min()+xy[:,0].max())/2)/(np.ptp(xy[:,0])/2),
                             960+250*(xy[:,1]-(xy[:,1].min()+xy[:,1].max())/2)/(np.ptp(xy[:,1])/2)))
    desired=(1-early[:,None])*old[0]['scene_landmark_portrait_xy']+early[:,None]*desired
    end=np.array(parent['inventory'][-1]['camera']['scene_landmark_portrait_xy'])
    desired=(1-late[:,None])*desired+late[:,None]*end
    result=[]
    for i,position in enumerate(positions):
        row=deepcopy(old[i]);z=position-target;z/=np.linalg.norm(z);y=np.cross(z,up);y/=np.linalg.norm(y)
        look=np.column_stack((np.cross(y,z),y,z));native=np.array([row['w']-1-desired[i,1],desired[i,0]])
        ray=np.array([(native[0]-row['cx'])/row['fl_x'],-(native[1]-row['cy'])/row['fl_y'],-1]);ray/=np.linalg.norm(ray)
        pose=np.eye(4);pose[:3,3]=position;pose[:3,:3]=look@Rotation.align_vectors(np.array([[0.,0.,-1.]]),ray[None])[0].as_matrix()
        row.update(transform_matrix=pose.tolist(),physical_camera=f'{variant}_{i:05d}',rig_offset_xy=xy[i].tolist(),scene_landmark_portrait_xy=desired[i].tolist())
        for key in ['convex_weights','arc_length_fraction','pilot_sample_phase']:row.pop(key,None)
        np.testing.assert_allclose(portrait_projection(target,row),desired[i],atol=1e-6)
        result.append(calibration_pose(row,cal,meta))
    poses=np.array([r['transform_matrix'] for r in result]);view=poses[:,:3,2];angles=np.degrees(np.arccos(np.clip(view@view.T,-1,1)))
    rays=positions-target;rays/=np.linalg.norm(rays,axis=1)[:,None]
    center_angle=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1))).max()
    step=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
    report=dict(variant=variant,trajectory='open_clamped_cubic_large_horizontal_vertical_arc',frames=150,fps=24,
        anchors=[r['physical_camera'] for r in anchors],anchor_positions=corners.tolist(),knot_indices=KNOTS.tolist(),controls_xy=CONTROLS[variant],
        fixed_target=target.tolist(),view_angle_extent_degrees=float(angles.max()),center_ray_angle_extent_degrees=float(center_angle),
        projected_landmark_extent_pixels=np.ptp(desired,axis=0).tolist(),position_extent=np.ptp(poses[:,:3,3],axis=0).tolist(),
        rig_parameter_min=xy.min(0).tolist(),rig_parameter_max=xy.max(0).tolist(),
        step_max=float(step.max()),step_median=float(np.median(step)),max_second_difference=float(np.linalg.norm(np.diff(poses[:,:3,3],n=2,axis=0),axis=1).max()),
        minimum_interior_anchor_weight=float(weights.min()),camera_periodic=False,no_periodic_wrap=True,
        early_pose_identical_to_production=True,endpoint_train_camera=lookup['G004_C005']['physical_camera'],
        boundary_tradeoff='C..K columns avoid outermost A/N and penultimate B/M. Five vertical rows A..E: strictly interior between B/D; cannot avoid their neighborhood and obtain large vertical travel. No A/E, no extrapolation.',
        image_crop=False,image_stabilization=False,focal_length_unchanged_from_production=True,actor_clip_periodic=False)
    assert center_angle>30,report
    assert np.ptp(desired,axis=0)[0]>450 and np.ptp(desired,axis=0)[1]>400,report
    np.testing.assert_allclose(result[0]['transform_matrix'],calibration_pose(old[0],cal,meta)['transform_matrix'],atol=1e-8)
    return result,report


def initialize(dry=False):
    parent=verify_request(PARENT);cal=read(CALIBRATION)
    for variant in VARIANTS:
        path,report=path_for(variant,parent);print(variant,json.dumps(report),flush=True)
        if dry:continue
        request=deepcopy(parent)
        for record,row in zip(request['inventory'],path):record['camera']=normalize_frame(row,cal,read(record['metadata']))
        request['camera_path_report']=report;request['recipe'].update(camera_path_variant='large_motion_'+variant,camera_periodic=False)
        request.update(partial_diagnostic_only=False,full_video_candidate=True,artifact_free_approval=False,inherited_actor_inventory_unchanged=True,
            inherited_camera_and_actor_inventory_unchanged=False,required_initial_rgb_gate=CANARIES,
            initial_gate=dict(status='requires_actual_clay_rgb_and_motion_review'),
            camera_workaround_parent=dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json')))
        request['script_hashes'][Path(__file__).name]=sha(__file__)
        out=BASE/variant;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
        if (out/'request.json').exists():assert read(out/'request.json')==request,'Immutable request mismatch'
        atomic_json(out/'request.json',request)


def screen():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    for variant in VARIANTS:
        out=BASE/variant;request=verify_request(out);root=out/'geometry_screen';root.mkdir(exist_ok=True)
        panel=Image.new('RGB',(1080,1632));draw=ImageDraw.Draw(panel);checks=[]
        for j,i in enumerate(CANARY_INDICES):
            record=request['inventory'][i];mesh=o3d.io.read_triangle_mesh(record['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
            camera=deepcopy(record['camera'])
            for key in ['w','h','fl_x','fl_y','cx','cy']:camera[key]/=2
            camera['w']=int(camera['w']);camera['h']=int(camera['h']);depth=camera_depth(scene,camera)[0];hit=np.isfinite(depth)
            shade=np.zeros((*depth.shape,3),np.uint8);lo,hi=np.quantile(depth[hit],[.02,.98]);shade[hit]=np.clip(220-130*(depth[hit]-lo)/(hi-lo),40,240)[:,None]
            im=Image.fromarray(np.rot90(shade));path=root/(record['frame_id']+'.png');im.save(path)
            x,y=j%4*270,j//4*544;panel.paste(im.resize((270,480)),(x,y+24));draw.text((x+3,y+3),record['frame_id'],fill='white')
            checks.append(dict(frame=record['frame_id'],sha256=sha(path),hit_fraction=float(hit.mean())))
        panel.save(root/'contact.png');atomic_json(root/'receipt.json',dict(checks=checks,visual_status='pending'))


def supervise(canary=False):
    work=[(v,f) for v in VARIANTS for f in (CANARIES if canary else verify_request(BASE/v)['ordered_frame_ids']) if not (BASE/v/'frames'/f/'complete.json').exists()]
    root=BASE/('canary_workers' if canary else 'full_workers');root.mkdir(exist_ok=True);jobs=[]
    # Exactly six processes; each handles its deterministic cross-variant shard.
    for i in range(6):
        shard=work[i::6]
        if not shard:continue
        assignment=root/f'{i}_assignment.json';atomic_json(assignment,shard);log=(root/f'{i}.log').open('a')
        command=[sys.executable,__file__,'worker','--index',str(i),'--assignment',str(assignment)]
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        jobs.append((p,log,command))
    atomic_json(root/'execution.json',dict(pid=os.getpid(),jobs=[dict(pid=p.pid,command=c) for p,_,c in jobs]))
    while True:
        rec=dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),jobs=[dict(pid=p.pid,exit_code=p.poll()) for p,_,_ in jobs],
            complete={v:len(list((BASE/v/'frames').glob('*/complete.json'))) for v in VARIANTS},free_bytes=shutil.disk_usage(BASE).free,
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip())
        with (root/'checks.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        atomic_json(root/'progress.json',rec);print(json.dumps(rec),flush=True)
        if all(p.poll() is not None for p,_,_ in jobs):break
        time.sleep(30)
    for _,log,_ in jobs:log.close()
    assert all(p.returncode==0 for p,_,_ in jobs),'Worker failed; outputs retained'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['describe','init','screen','canary','supervise','worker']);p.add_argument('--index',type=int);p.add_argument('--assignment',type=Path);a=p.parse_args()
    if a.action in ['describe','init']:initialize(a.action=='describe')
    elif a.action=='screen':screen()
    elif a.action in ['canary','supervise']:supervise(a.action=='canary')
    else:
        import torch
        from run_view_consistent_dynamic_video import worker
        torch.set_num_threads(2);assignment=read(a.assignment)
        for variant in VARIANTS:
            frames=[f for v,f in assignment if v==variant]
            if frames:(BASE/variant/'workers').mkdir(exist_ok=True);worker(BASE/variant,a.index,frames)
