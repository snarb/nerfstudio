"""Independent physical/lens/hold controls, including frozen-actor depth views."""
import argparse
from copy import deepcopy
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,cameras,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection
from cinematic_pushin_live_ending import BASE,VARIANTS,path_for,PARENT,timing

def audit(variant):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=BASE/variant;request=verify_request(root);cal=read(CALIBRATION)
    expected,report=path_for(variant,verify_request(PARENT));meta=read(report['reference_metadata'])
    rows=[normalize_frame(r,cal,meta) for r in expected];poses=np.array([r['transform_matrix'] for r in rows]);centers=poses[:,:3,3]
    target=np.array(report['fixed_target']);radius=np.linalg.norm(centers-target,axis=1);speed=np.r_[0,np.linalg.norm(np.diff(centers,axis=0),axis=1)]*24
    acceleration=np.r_[0,np.diff(speed)]*24;focal=np.array([r['fl_x'] for r in rows]);rays=(centers-target)/radius[:,None]
    angle=np.degrees(np.arccos(np.clip(rays@rays[0],-1,1)))
    source=next(r for r in cameras('000973')[0] if r['physical_camera']==report['endpoint_train_camera'])
    hold=report['hold_start_index']
    np.testing.assert_allclose(poses[hold:],np.repeat(np.array(source['transform_matrix'])[None],150-hold,axis=0),atol=1e-10)
    assert np.ptp(focal[hold:])==0 and np.max(np.diff(radius))<1e-9
    for key in ['fl_x','fl_y','cx','cy']:assert np.ptp([r[key] for r in rows[hold:]])==0
    assert np.all(np.diff(timing(np.arange(150)/126))>=-1e-12)
    if variant=='locked_arc':assert report['look_at_angular_error_max_degrees']<.001
    motion=[];probe=root/'motion_probe';probe.mkdir(exist_ok=True)
    reference=request['inventory'][75];mesh=o3d.io.read_triangle_mesh(reference['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    fixedmeta=read(reference['metadata']);panel=Image.new('RGB',(1080,816));draw=ImageDraw.Draw(panel);projections=[];centroids=[]
    # True motion isolated from actor time AND focal ramp: identical mesh and
    # identical intrinsics for all six probes; only camera extrinsics change.
    for j,i in enumerate([0,12,24,48,100,126]):
        cam=normalize_frame(expected[i],cal,fixedmeta)
        for key in ['fl_x','fl_y','cx','cy']:cam[key]=expected[0][key]
        depth=camera_depth(scene,cam)[0];hit=np.isfinite(depth);yy,xx=np.nonzero(np.rot90(hit));centroids.append([float(xx.mean()),float(yy.mean())])
        pixels=np.zeros((*depth.shape,3),np.uint8);lo,hi=np.quantile(depth[hit],[.02,.98]);pixels[hit]=np.clip(220-130*(depth[hit]-lo)/(hi-lo),40,240)[:,None]
        im=Image.fromarray(np.rot90(pixels));dest=probe/f'{i:03d}_fixed_actor_fixed_lens.png';im.save(dest);x,y=j%3*360,j//3*408;panel.paste(im.resize((216,384)),(x,y+24));draw.text((x,y),str(i),fill='white')
        motion.append(dict(index=i,depth_sha256=__import__('hashlib').sha256(np.where(hit,depth,0).tobytes()).hexdigest(),image=str(dest),sha256=sha(dest)))
    panel.save(probe/'contact.png')
    # Reference-space landmarks with different depths isolate perspective
    # parallax from pure image-plane translation or lens scaling.
    landmarks=np.array([target,target+.03*poses[0,:3,2],target-.03*poses[0,:3,2]])
    for i,row in enumerate(rows):
        cam=deepcopy(row)
        for key in ['fl_x','fl_y','cx','cy']:cam[key]=rows[0][key]
        projections.append([portrait_projection(p,cam) for p in landmarks])
    projections=np.array(projections);relative=projections[:,1]-projections[:,2]
    fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True)
    for ax,val,label in zip(axes,[radius,speed,focal/focal[0]],['Physical radius (reference units)','Physical speed (reference units/s)','Virtual focal / initial focal']):
        ax.plot(np.arange(150)/24,val);ax.axvline(hold/24,color='gray',ls='--');ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[-1].set_xlabel('Seconds; camera fixed118..149, real train RGB126..149');fig.tight_layout();fig.savefig(root/'motion_plot.png');plt.close(fig)
    data=dict(status='physical_lens_hold_gate_pass',request_sha256=sha(root/'request.json'),radius=radius.tolist(),speed_per_second=speed.tolist(),acceleration_per_second_squared=acceleration.tolist(),
        physical_angle_from_start=angle.tolist(),focal_pixels=focal.tolist(),first24_physical_distance=float(np.linalg.norm(np.diff(centers[:25],axis=0),axis=1).sum()),
        first48_physical_distance=float(np.linalg.norm(np.diff(centers[:49],axis=0),axis=1).sum()),
        first24_radial_approach=float(radius[0]-radius[24]),first48_radial_approach=float(radius[0]-radius[48]),
        fixed_lens_depth_separated_relative_parallax_extent_pixels=np.ptp(relative,axis=0).tolist(),
        early24_fixed_lens_relative_parallax_pixels=(relative[24]-relative[0]).tolist(),early48_fixed_lens_relative_parallax_pixels=(relative[48]-relative[0]).tolist(),
        frozen_actor_fixed_lens_controls=motion,frozen_actor_fixed_lens_centroid_extent_pixels=np.ptp(centroids,axis=0).tolist(),
        exact_endpoint_hold_24=True,exact_endpoint_hold_count=150-hold,hold_start_index=hold,endpoint_train_camera=source['physical_camera'],actor_hold=False,physical_report=report)
    atomic_json(root/'motion_audit.json',data);print(variant,report['radial_approach_fraction'],report['center_ray_angle_extent_degrees'],data['first24_radial_approach'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    for v in ([a.variant] if a.variant else VARIANTS):audit(v)
