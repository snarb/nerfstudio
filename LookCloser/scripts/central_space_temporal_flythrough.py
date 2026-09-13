"""A slow open arc strictly inside the central train-camera tetrahedron.

Reuses the corrected 150-time geometry and frozen native hard-source texturing.
Only target poses change. No temporal duplication, morphing, or actor slow-down.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from joint_temporal_texture import cameras,read,sha,atomic_json,CALIBRATION
from smooth_mesh_flythrough import central_path,ANCHORS
from render_smooth_temporal_mesh_video import calibration_pose,verify_request
from render_patchmatch_camera_path import normalize_frame

PARENT=Path('/mnt/data/lookcloser_dec5_5a3_smooth_temporal_150_repaired_v3')
OUTPUT=Path('/mnt/data/lookcloser_dec5_5a3_central_space_flight_150')


def motion_report(poses,fps=30):
    poses=np.asarray(poses);position=poses[:,:3,3];delta=np.diff(position,axis=0)
    speed=np.linalg.norm(delta,axis=1)*fps
    angular=Rotation.from_matrix(poses[:-1,:3,:3].transpose(0,2,1)@poses[1:,:3,:3]).magnitude()*fps*180/np.pi
    acceleration=np.linalg.norm(np.diff(delta,axis=0),axis=1)*fps**2
    direction=delta/np.linalg.norm(delta,axis=1)[:,None]
    bend=np.arccos(np.clip((direction[:-1]*direction[1:]).sum(1),-1,1))*180/np.pi
    return {'angular_speed_degrees_per_second_min_median_max':np.quantile(angular,[0,.5,1]).tolist(),
            'linear_speed_min_median_max':np.quantile(speed,[0,.5,1]).tolist(),
            'speed_max_min_ratio':float(speed.max()/speed.min()),'acceleration_max':float(acceleration.max()),
            'consecutive_velocity_direction_change_max_degrees':float(bend.max()),
            'endpoint_distance':float(np.linalg.norm(position[-1]-position[0]))}


def initialize(output,parent=PARENT):
    previous=verify_request(parent);rows,mp,meta=cameras('000973')
    mesh=o3d.io.read_triangle_mesh(str(mp));cal=read(CALIBRATION)
    # Traverse only 150 samples of a 360-sample loop: larger extent without a
    # faster orbit. The last and first video frames are intentionally not joined.
    full,loop_report=central_path(rows,np.asarray(mesh.vertices).mean(0),count=360,fps=30,radius=.65)
    selected=full[:150]
    raw=[calibration_pose(r,cal,read(meta)) for r in selected]
    poses=np.array([r['transform_matrix'] for r in raw]);motion=motion_report(poses)
    weights=np.array([r['convex_weights'] for r in raw])
    if weights.min()<.015 or not np.allclose(weights.sum(1),1):raise ValueError('Insufficient central-hull safety margin')
    if motion['angular_speed_degrees_per_second_min_median_max'][-1]>3.5 or motion['speed_max_min_ratio']>1.01:
        raise ValueError('Fast or nonuniform central camera flight')
    if motion['consecutive_velocity_direction_change_max_degrees']>2.5:raise ValueError('Abrupt camera direction change')
    request=deepcopy(previous);request.pop('composition',None)
    request['parent_geometry_request']={'path':str(parent/'request.json'),'sha256':sha(parent/'request.json'),
                                      'all_150_meshes_reused':True,'all_target_views_rerendered':True}
    request['recipe'].update(radius=.65,camera_path_kind='central_open_arc',camera_periodic=False)
    for i,record in enumerate(request['inventory']):
        if sha(record['mesh'])!=record['mesh_sha256'] or sha(record['metadata'])!=record['metadata_sha256']:raise ValueError('Corrected reusable geometry changed')
        record['camera']=normalize_frame(raw[i],cal,read(record['metadata']))
    request['camera_path_report']={'frames':150,'fps':30,'duration_seconds':5.,'anchors':ANCHORS,
         'continuous_periodic_loop':False,'inside_train_hull':True,'minimum_convex_weight':float(weights.min()),
         'central_radius':.65,'reference_loop_samples':360,'traversed_samples':150,
         'camera_teleport_at_loop_restart_not_hidden':True,'no_loop_restart_in_delivered_video':True,
         'fixed_intrinsics':True,'fixed_optical_target':loop_report['target_world'],'raw_calibration_motion':motion}
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Immutable flight request mismatch')
    atomic_json(output/'request.json',request)
    atomic_json(output/'progress.json',{'stage':'initialized','complete':0,'total':150})
    diagram(output,raw,cal)
    print(request['camera_path_report'],flush=True)


def diagram(output,path,cal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    physical=[r for r in cal['frames'] if r['physical_camera'] not in {'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}]
    centers=np.array([r['transform_matrix'] for r in physical])[:,:3,3]
    by_name={r['physical_camera']:i for i,r in enumerate(physical)}
    corners=centers[[by_name[n] for n in ANCHORS]];origin=corners.mean(0)
    _,_,vt=np.linalg.svd(corners-origin);basis=vt[:2]
    projected=(centers-origin)@basis.T;corners2=(corners-origin)@basis.T
    flight=(np.array([r['transform_matrix'] for r in path])[:,:3,3]-origin)@basis.T
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax in axes:
        ax.scatter(*projected.T,s=18,c='#777777',label='62 real train cameras')
        ax.plot(*np.vstack([corners2,corners2[:1]]).T,c='#2288aa',label='central train anchors')
        ax.plot(*flight.T,c='#dd5533',lw=2,label='5-second open flight')
        ax.scatter(*flight[[0,-1]].T,c=['green','red'],s=45)
        for name,xy in zip(ANCHORS,corners2):ax.annotate('/'.join(name.split('_')[:2]),xy,fontsize=8)
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('rig PCA axis 1');ax.set_ylabel('rig PCA axis 2')
    lo=corners2.min(0);hi=corners2.max(0);margin=(hi-lo)*.2
    axes[1].set_xlim(lo[0]-margin[0],hi[0]+margin[0]);axes[1].set_ylim(lo[1]-margin[1],hi[1]+margin[1])
    axes[0].legend(fontsize=8);axes[0].set_title('Train rig');axes[1].set_title('Central space; green=start, red=end')
    fig.tight_layout();fig.savefig(output/'camera_path.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','canary'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--parent',type=Path,default=PARENT)
    p.add_argument('--frames',nargs='+',default=['000899','000971','000973','001047','001197']);a=p.parse_args()
    if a.action=='init':initialize(a.output,a.parent)
    else:
        import render_smooth_temporal_mesh_video as renderer
        from temporal_texture_view_prior import install
        renderer.torch.set_num_threads(4);install(renderer);renderer.render(a.output,a.frames)
