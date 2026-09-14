"""Wide moving-actor camera control: horizontal sweep, vertical sweep, closed return.

Rig columns D..L are H +/-4. There are only FIVE vertical levels A..E;
the vertical sweep uses all of them (C +/-2), never fictional +/-4 levels.
Existing per-time surfaces and source eligibility are immutable inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read, sha, atomic_json, cameras, CALIBRATION, HELD_CAMERAS
from render_patchmatch_camera_path import normalize_frame
from render_smooth_temporal_mesh_video import calibration_pose, verify_request

PARENT=Path('/mnt/data/dec5_dynamic_grid_150_guard_v2')
OUTPUT=Path('/mnt/data/dec5_wide_dynamic_flight_150')


def wide_path(rows, target, count=150, fps=24):
    if count<32 or fps<=0:raise ValueError('Invalid sampling')
    prefixes=['D004_A005','L004_A005','L004_E005','D004_E005']
    by_prefix={r['physical_camera'][:9]:r for r in rows}
    anchors=[by_prefix[n] for n in prefixes]
    if set(r['physical_camera'] for r in anchors)&HELD_CAMERAS:raise ValueError('Held anchor')
    corners=np.asarray([r['transform_matrix'] for r in anchors])[:,:3,3]
    # Start left, traverse right; turn smoothly to top, descend to bottom,
    # and return left. Normalize the entire periodic spline by its true
    # extrema instead of clipping samples (which creates flat/cusped motion).
    control=np.array([[-4,0],[-2,.1],[0,.15],[2,.1],[4,0],
        [3.6,1],[2,1.8],[0,2],[-.12,1],[-.15,0],[-.12,-1],[0,-2],
        [-2,-1.8],[-3.6,-1],[-4,0]],float)
    spline=CubicSpline(np.arange(len(control)),control,bc_type='periodic')
    parameter=np.linspace(0,len(control)-1,48001)
    xy=spline(parameter);lo=xy.min(0);hi=xy.max(0)
    def rig_xy(t):return (spline(t)-(lo+hi)/2)/(hi-lo)*[8,4]
    def convex(q):
        u=(q[:,0]+4)/8;v=(2-q[:,1])/4
        return np.column_stack(((1-u)*(1-v),u*(1-v),u*v,(1-u)*v))
    dense=convex(rig_xy(parameter))@corners
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    t=np.interp(np.arange(count)*distance[-1]/count,distance,parameter)
    xy=rig_xy(t);ww=convex(xy)
    if ww.min() < -1e-10 or not np.allclose(ww.sum(1),1):raise ValueError('Outside real rig hull')
    reference=next(r for r in rows if r['physical_camera']=='H004_C005_1210SZ')
    ref=np.array(reference['transform_matrix']);portrait_up=ref[:3,0]
    frames=[]
    for i,pos in enumerate(ww@corners):
        z=pos-target;z/=np.linalg.norm(z)
        y=np.cross(z,portrait_up);y/=np.linalg.norm(y);x=np.cross(y,z)
        pose=np.eye(4);pose[:3,:3]=np.column_stack((x,y,z));pose[:3,3]=pos
        row=deepcopy(reference);row.pop('file_path',None)
        row.update(transform_matrix=pose.tolist(),physical_camera=f'wide_dynamic_{i:05d}',
                   convex_weights=ww[i].tolist(),rig_offset_xy=xy[i].tolist())
        frames.append(row)
    poses=np.asarray([r['transform_matrix'] for r in frames]);nxt=np.roll(poses,-1,axis=0)
    speed=np.linalg.norm(nxt[:,:3,3]-poses[:,:3,3],axis=1)*fps
    angular=Rotation.from_matrix(poses[:,:3,:3].transpose(0,2,1)@nxt[:,:3,:3]).magnitude()*fps*180/np.pi
    view=poses[:,:3,2];pair_angles=np.rad2deg(np.arccos(np.clip(view@view.T,-1,1)))
    extrema=[int(xy[:,0].argmin()),int(xy[:,0].argmax()),int(xy[:,1].argmax()),int(xy[:,1].argmin())]
    if np.ptp(xy[:,0])<7.9 or np.ptp(xy[:,1])<3.95:raise ValueError('Missing requested sweep')
    if not extrema[1]<extrema[2]<extrema[3]:raise ValueError('Wrong horizontal/vertical sweep order')
    if speed.max()/speed.min()>1.04:raise ValueError('Nonuniform sampled camera speed')
    return frames,dict(anchors=[r['physical_camera'] for r in anchors],center_camera='H004_C005_1210SZ',
        horizontal_requested_offsets=[-4,4],horizontal_achieved_offsets=[float(xy[:,0].min()),float(xy[:,0].max())],
        vertical_requested_offsets=[-4,4],vertical_available_offsets=[-2,2],vertical_limit_disclosed=True,
        vertical_achieved_offsets=[float(xy[:,1].min()),float(xy[:,1].max())],
        extrema_indices=dict(zip(['left','right','top','bottom'],extrema)),
        trajectory='periodic_cubic_horizontal_then_vertical_then_return',continuous_periodic_camera=True,
        actor_clip_periodic=False,frames=count,fps=fps,duration_seconds=count/fps,
        fixed_target=target.tolist(),fixed_intrinsics=True,portrait_up_axis='reference native camera X',
        speed_max_min_ratio=float(speed.max()/speed.min()),
        angular_speed_degrees_per_second_min_median_max=np.quantile(angular,[0,.5,1]).tolist(),
        maximum_pairwise_view_angle_degrees=float(pair_angles.max()),minimum_convex_weight=float(ww.min()))


def initialize(output):
    old=verify_request(PARENT);cal=read(CALIBRATION);rows,_,meta=cameras('000973')
    # Preserve the old optical focus for a matched framing comparison.
    target=np.asarray(old['camera_path_report']['fixed_optical_target'])
    path,report=wide_path(rows,target);raw=[calibration_pose(p,cal,read(meta)) for p in path]
    request=deepcopy(old);request.pop('composition',None)
    request['recipe'].update(camera_path_kind='wide_horizontal_vertical_closed',camera_periodic=True,
        train_foreground_guard=False,reuse_verified_foreground_geometry=True,horizontal_radius_rows=4,
        vertical_radius_rows=2,fps=24)
    request['camera_path_report']=report
    request['reused_guard_parent']=dict(path=str(PARENT),request_sha256=sha(PARENT/'request.json'))
    for i,record in enumerate(request['inventory']):
        frame=record['frame_id'];guard=PARENT/'foreground_guard'/frame;receipt=read(guard/'complete.json')
        if receipt['request']['parent_request_sha256']!=sha(PARENT/'request.json'):raise ValueError('Unbound guard parent')
        for name in ['mesh.ply','masks.npz','cameras.json']:
            if sha(guard/name)!=receipt['hashes'][name]:raise ValueError('Changed reusable guard data')
        record['untrimmed_mesh']=record['mesh'];record['untrimmed_mesh_sha256']=record['mesh_sha256']
        record['mesh']=str(guard/'mesh.ply');record['mesh_sha256']=sha(guard/'mesh.ply')
        record['source_masks']=dict(root=str(guard),complete_sha256=sha(guard/'complete.json'),
            masks_sha256=sha(guard/'masks.npz'),cameras_sha256=sha(guard/'cameras.json'))
        record['camera']=normalize_frame(raw[i],cal,read(record['metadata']))
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable request mismatch')
    atomic_json(output/'request.json',request);atomic_json(output/'progress.json',dict(stage='initialized',complete=0,total=150))
    print(report,flush=True)


def probe(output):
    """Actual raycasts/atlas RGB with ONE mesh; diagnostic only, not the movie."""
    from diffusion_mesh_repair import BASE,scene_for,render_atlas
    from bake_joint_temporal_mesh import camera_depth
    request=verify_request(output);old=verify_request(PARENT);cal=read(CALIBRATION)
    _,_,meta=cameras('000973');metadata=read(meta)
    atlas=dict(np.load(BASE/'atlas_geometry.npz'));texture=np.array(Image.open(BASE/'texture_joint.png').convert('RGB'))
    scene=scene_for(atlas['vertices'],atlas['triangles']);root=output/'camera_probe';root.mkdir(exist_ok=True)
    variants={
        'previous_actual_poses':[old['inventory'][i] for i in [0,49,99,149]],
        'new_extreme_poses':[request['inventory'][request['camera_path_report']['extrema_indices'][key]] for key in ['left','right','top','bottom']]}
    result={}
    for name,records in variants.items():
        panel=Image.new('RGB',(2160,984));draw=ImageDraw.Draw(panel);depths=[];paths=[]
        for j,record in enumerate(records):
            row=normalize_frame(calibration_pose(record['camera'],cal,read(record['metadata'])),cal,metadata)
            rgb,depth,_,_=render_atlas(scene,atlas,texture,row);depths.append(depth)
            portrait=Image.fromarray(np.rot90(rgb));p=root/f'{name}_{j}.png';portrait.save(p);paths.append(dict(path=str(p),sha256=sha(p)))
            panel.paste(portrait.resize((540,960)),(j*540,24));draw.text((j*540+5,5),f'{name} pose-index={record["index"]}',fill='white')
        panel.save(root/f'{name}.png')
        silhouettes=[np.isfinite(d) for d in depths]
        result[name]=dict(images=paths,changed_silhouette_pixels_vs_first=[int(np.count_nonzero(h!=silhouettes[0])) for h in silhouettes],
            depth_absolute_difference_mean_vs_first=[float(np.abs(d[np.isfinite(d)&np.isfinite(depths[0])]-depths[0][np.isfinite(d)&np.isfinite(depths[0])]).mean()) for d in depths])
    # Check the SAVED old renderer output against an independent fresh raycast,
    # not just a JSON pose copied by the worker.
    import open3d as o3d
    checks=[]
    for i in [0,75,149]:
        record=old['inventory'][i];frame=record['frame_id'];mp=PARENT/'foreground_guard'/frame/'mesh.ply'
        mesh=o3d.io.read_triangle_mesh(str(mp));s=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
        fresh=camera_depth(s,record['camera'])[0];saved=np.load(PARENT/'frames'/frame/'target_depth.npz')['depth']
        match=np.allclose(np.where(np.isfinite(fresh),fresh,0),saved,atol=1e-6)
        if not match:raise ValueError('Old rendered depth does not match requested pose')
        checks.append(dict(frame_id=frame,independent_raycast_matches_saved_depth=bool(match)))
    atomic_json(root/'result.json',dict(static_diagnostic_only=True,source_frame='000973',
        atlas_sha256=sha(BASE/'atlas_geometry.npz'),texture_sha256=sha(BASE/'texture_joint.png'),
        actual_saved_depth_checks=checks,variants=result,visual_status='pending'))


def install_source_masks(module):
    original=module.render_one;original_depth=module.camera_depth
    def render(output,record,source):
        spec=record['source_masks'];root=Path(spec['root'])
        for name,key in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:
            if sha(root/name)!=spec[key]:raise ValueError('Changed reusable source masks')
        masks=np.load(root/'masks.npz')['masks'];names=read(root/'cameras.json')
        if len(names)!=62 or set(names)&HELD_CAMERAS:raise ValueError('Invalid foreground camera list')
        lookup=dict(zip(names,masks))
        def depth(scene,row):
            d,ids,b=original_depth(scene,row)
            if row['physical_camera'] in lookup:
                d=d.copy();d[lookup[row['physical_camera']]==0]=np.inf
            return d,ids,b
        module.camera_depth=depth
        try:return original(output,record,source)
        finally:module.camera_depth=original_depth
    module.render_one=render


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','probe','render'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--frames',nargs='+');a=p.parse_args()
    if a.action=='init':initialize(a.output)
    elif a.action=='probe':probe(a.output)
    else:
        import render_smooth_temporal_mesh_video as renderer
        from temporal_texture_view_prior import install
        renderer.torch.set_num_threads(4);install(renderer);install_source_masks(renderer);renderer.render(a.output,a.frames)
