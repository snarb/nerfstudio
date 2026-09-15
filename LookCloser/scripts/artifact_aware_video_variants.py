"""Three open camera shots over the unchanged production 150-time actor.

The camera is constrained independently at the start and end; there is no
periodic displacement wrapping a late waypoint into the early hand pose.
Geometry, exposure, source profiles and hard unwarped RGB remain production.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read, sha, atomic_json, cameras, CALIBRATION
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection

PARENT = Path('/mnt/data/dec5_incidence2_unwarped_dynamic_150')
BASE = Path('/mnt/data/dec5_artifact_aware_variants')
VARIANTS = ('sweep', 'high_arc', 's_curve')
CANARIES = ['000899', '000929', '000995', '001005', '001029', '001059', '001123', '001193', '001197']
# These are OPEN curves, with an exact real G/C camera center at the end.
# All middle coordinates stay between central G..J and rows C..B.
KNOTS = np.array([0, 30, 60, 90, 120, 147, 149], float)
CONTROLS = {
    'sweep': [[1.9,.3], [1.6,.4], [.8,.5], [.05,.45], [-.7,.2], [-1.,0], [-1.,0]],
    'high_arc': [[1.9,.3], [2.,.65], [1.2,.8], [.3,.8], [-.5,.6], [-1.,0], [-1.,0]],
    's_curve': [[1.9,.3], [.75,.5], [-.6,.6], [.9,.45], [.3,.2], [-1.,0], [-1.,0]],
}


def path_for(variant, parent):
    cal = read(CALIBRATION); rows, _, metadata = cameras('000973'); meta = read(metadata)
    lookup = {r['physical_camera'][:9]: r for r in rows}
    coords = np.array([[lookup[f'{col}004_{row}005']['transform_matrix'] for row in 'CB'] for col in 'GHIJ'])[:, :, :3, 3]
    interpolators = [RectBivariateSpline([-1,0,1,2], [0,1], coords[:,:,k], kx=3, ky=1, s=0) for k in range(3)]
    samples = np.arange(150)
    xy = CubicSpline(KNOTS, CONTROLS[variant], axis=0, bc_type='clamped')(samples)
    if xy[:,0].min() < -1.02 or xy[:,0].max() > 2.15 or xy[:,1].min() < -.02 or xy[:,1].max() > .9:
        raise ValueError('Central-envelope overshoot')
    # Fix the endpoint segment without periodic wrapping. A short natural stop
    # in camera motion is allowed; the actor continues at every actual instant.
    positions = np.column_stack([s.ev(xy[:,0], xy[:,1]) for s in interpolators])
    old_ref = [normalize_frame(calibration_pose(r['camera'], cal, read(r['metadata'])), cal, meta) for r in parent['inventory']]
    old_position = np.array(old_ref[0]['transform_matrix'])[:3,3]
    u = np.clip(samples/30, 0, 1); smooth = u*u*u*(10-15*u+6*u*u)
    positions += (1-smooth[:,None])*(old_position-positions[0])
    target = np.array(parent['camera_path_report']['fixed_target'])
    up = np.array(lookup['H004_C005']['transform_matrix'])[:3,0]
    paths = []
    for index, (position, previous) in enumerate(zip(positions, old_ref)):
        row = deepcopy(previous)
        # Retain the published 384 x 154 pixel on-image composition travel.
        # This is a camera orientation, not a moving crop or stabilization.
        desired = np.array(row['scene_landmark_portrait_xy'])
        z = position-target; z /= np.linalg.norm(z)
        axis_y = np.cross(z, up); axis_y /= np.linalg.norm(axis_y)
        look = np.column_stack((np.cross(axis_y,z), axis_y,z))
        native = np.array([row['w']-1-desired[1], desired[0]])
        ray = np.array([(native[0]-row['cx'])/row['fl_x'], -(native[1]-row['cy'])/row['fl_y'], -1.]); ray /= np.linalg.norm(ray)
        pose = np.eye(4); pose[:3,3] = position
        pose[:3,:3] = look @ Rotation.align_vectors(np.array([[0.,0.,-1.]]), ray[None])[0].as_matrix()
        row.update(transform_matrix=pose.tolist(), physical_camera=f'{variant}_{index:05d}', rig_offset_xy=xy[index].tolist())
        for key in ['convex_weights', 'arc_length_fraction', 'pilot_sample_phase']:
            row.pop(key, None)
        np.testing.assert_allclose(portrait_projection(target,row), desired, atol=1e-6)
        paths.append(calibration_pose(row, cal, meta))
    poses = np.array([r['transform_matrix'] for r in paths]); step = np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
    angles = np.degrees(np.arccos(np.clip(poses[:,:3,2]@poses[:,:3,2].T,-1,1)))
    report = dict(variant=variant, trajectory='open_clamped_cubic_central_rig_shot', camera_periodic=False,
        continuous_periodic_camera=False, actor_clip_periodic=False, knot_indices=KNOTS.tolist(),
        controls_xy=CONTROLS[variant], rig_coordinates_note='central grid spline plus initial 30-frame smooth translation to preserve production start',
        fixed_target=target.tolist(), endpoint_train_camera=lookup['G004_C005']['physical_camera'],
        early_position_identical_to_production=True, no_periodic_wrap=True, camera_motion_stops_at_endpoint=True,
        view_angle_extent_degrees=float(angles.max()), position_extent=np.ptp(poses[:,:3,3],axis=0).tolist(),
        step_min=float(step.min()), step_max=float(step.max()), step_median=float(np.median(step)),
        max_second_difference=float(np.linalg.norm(np.diff(poses[:,:3,3],n=2,axis=0),axis=1).max()),
        actual_rig_parameter_extent=np.ptp(xy,axis=0).tolist(),
        projected_landmark_extent_pixels=np.ptp([r['scene_landmark_portrait_xy'] for r in paths],axis=0).tolist(),
        fps=24, frames=150, image_crop=False, image_stabilization=False, geometry_fixed_by_camera_path=False)
    return paths, report


def initialize():
    parent = verify_request(PARENT); cal = read(CALIBRATION)
    for variant in VARIANTS:
        paths, report = path_for(variant, parent); request = deepcopy(parent)
        for record, camera in zip(request['inventory'], paths):
            record['camera'] = normalize_frame(camera, cal, read(record['metadata']))
        request['camera_path_report'] = report
        request['recipe'].update(camera_path_variant='artifact_aware_'+variant, camera_periodic=False)
        request.update(partial_diagnostic_only=False, full_video_candidate=True, artifact_free_approval=False,
            inherited_camera_and_actor_inventory_unchanged=False, inherited_actor_inventory_unchanged=True,
            required_initial_rgb_gate=CANARIES, initial_gate=dict(status='requires_geometry_and_rgb_canary_review',
            note='Historical parent texture gates do not approve these camera changes.'),
            camera_workaround_parent=dict(path=str(PARENT/'request.json'), sha256=sha(PARENT/'request.json')))
        request['script_hashes'][Path(__file__).name] = sha(__file__)
        out = BASE/variant; out.mkdir(parents=True, exist_ok=True); (out/'frames').mkdir(exist_ok=True)
        if (out/'request.json').exists() and read(out/'request.json') != request:
            raise ValueError('Immutable camera request mismatch')
        atomic_json(out/'request.json',request)
        print(variant, json.dumps(report), flush=True)


def geometry_screen():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    for variant in VARIANTS:
        out = BASE/variant; request = verify_request(out); review = out/'geometry_screen'; review.mkdir(exist_ok=True)
        panel = Image.new('RGB',(300*3,450*3)); draw = ImageDraw.Draw(panel); checks = []
        for j, record in enumerate(r for r in request['inventory'] if r['frame_id'] in CANARIES):
            mesh = o3d.io.read_triangle_mesh(record['mesh']); scene = scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
            camera = deepcopy(record['camera'])
            for key in ['w','h','fl_x','fl_y','cx','cy']: camera[key] /= 2
            camera['w'] = int(camera['w']); camera['h'] = int(camera['h'])
            depth = camera_depth(scene,camera)[0]; hit = np.isfinite(depth)
            shade = np.zeros((*depth.shape,3),np.uint8)
            lo,hi = np.quantile(depth[hit],[.02,.98]); value = np.clip(220-130*(depth[hit]-lo)/max(hi-lo,1e-6),40,240)
            shade[hit] = value[:,None]
            image = Image.fromarray(np.rot90(shade)); image.save(review/(record['frame_id']+'.png'))
            x,y = j%3*300,j//3*450; panel.paste(image.resize((240,427)),(x,y+23));draw.text((x+3,y+3),record['frame_id'],fill='white')
            checks.append(dict(frame=record['frame_id'],hit_fraction=float(hit.mean()),sha256=sha(review/(record['frame_id']+'.png'))))
        panel.save(review/'contact.png'); atomic_json(review/'receipt.json',dict(checks=checks,visual_status='pending'))


def supervise(canary=False):
    from concurrent.futures import ThreadPoolExecutor
    work = [(v, f) for v in VARIANTS for f in (CANARIES if canary else verify_request(BASE/v)['ordered_frame_ids'])
            if not (BASE/v/'frames'/f/'complete.json').exists()]
    # Six long-lived processes, grouped by variant, keep exactly six workers.
    jobs = []; root = BASE/('canary_workers' if canary else 'full_workers');root.mkdir(exist_ok=True)
    for vi,variant in enumerate(VARIANTS):
        ids = [f for v,f in work if v==variant]
        for part in range(2):
            frames = ids[part::2]
            if not frames: continue
            index = 2*vi+part; log = (root/f'{index}.log').open('a')
            command = [sys.executable,__file__,'worker','--variant',variant,'--index',str(index),'--frames',*frames]
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            jobs.append((variant,process,log,command))
    atomic_json(root/'execution.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),jobs=[dict(variant=v,pid=p.pid,command=c) for v,p,_,c in jobs]))
    while True:
        record = dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),
            jobs=[dict(variant=v,pid=p.pid,exit_code=p.poll()) for v,p,_,_ in jobs],
            complete={v:len(list((BASE/v/'frames').glob('*/complete.json'))) for v in VARIANTS},
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip(),
            free_bytes=shutil.disk_usage(BASE).free)
        with (root/'checks.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
        atomic_json(root/'progress.json',record);print(json.dumps(record),flush=True)
        if all(p.poll() is not None for _,p,_,_ in jobs): break
        time.sleep(30)
    for _,_,handle,_ in jobs: handle.close()
    if any(p.returncode for _,p,_,_ in jobs): raise RuntimeError('Worker failed; retained log and outputs')


def panels():
    for frame in CANARIES:
        paths = [PARENT/'frames'/frame/'frame.png']+[BASE/v/'frames'/frame/'frame.png' for v in VARIANTS]
        if not all(p.exists() for p in paths):continue
        out = BASE/'review';out.mkdir(exist_ok=True)
        images = [Image.open(p).convert('RGB') for p in paths]
        overview = Image.new('RGB',(1080,504));draw=ImageDraw.Draw(overview)
        for j,(im,label) in enumerate(zip(images,['published',*VARIANTS])):
            overview.paste(im.resize((270,480)),(j*270,24));draw.text((j*270+3,3),label+' '+frame,fill='white')
        overview.save(out/(frame+'_overview.png'))
        for name,box in [('head',(120,420,980,1250)),('jaw',(240,800,920,1220))]:
            w,h=box[2]-box[0],box[3]-box[1];panel=Image.new('RGB',(w*2,(h+24)*2));draw=ImageDraw.Draw(panel)
            for j,(im,label) in enumerate(zip(images,['published',*VARIANTS])):
                x,y=j%2*w,j//2*(h+24);panel.paste(im.crop(box),(x,y+24));draw.text((x+4,y+4),label+' '+frame,fill='white')
            panel.save(out/(frame+'_'+name+'.png'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','screen','canary','supervise','worker','panels'])
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--index',type=int);parser.add_argument('--frames',nargs='+')
    args=parser.parse_args()
    if args.action=='init':initialize()
    elif args.action=='screen':geometry_screen()
    elif args.action in ['canary','supervise']:supervise(args.action=='canary')
    elif args.action=='panels':panels()
    else:
        from run_view_consistent_dynamic_video import worker
        import torch
        torch.set_num_threads(2);(BASE/args.variant/'workers').mkdir(exist_ok=True)
        worker(BASE/args.variant,args.index,args.frames)


if __name__=='__main__':main()
