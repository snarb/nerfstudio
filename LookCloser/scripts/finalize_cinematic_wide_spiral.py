"""Reuse cinematic integrity checks with the new broad spiral and its report."""
from pathlib import Path
from copy import deepcopy
import argparse
import shutil
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from joint_temporal_texture import read,sha,atomic_json,cameras,CALIBRATION,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from render_patchmatch_camera_path import normalize_frame
from screen_travel_camera_flight import portrait_projection
from cinematic_wide_spiral import BASE,PARENT,PREVIOUS,VARIANTS,CANARIES,RAW_IDS,path_for
import finalize_cinematic_live as shared


def configure():
    shared.BASE=BASE;shared.PARENT=PARENT;shared.VARIANTS=VARIANTS;shared.CANARIES=CANARIES;shared.RAW_IDS=RAW_IDS;shared.path_for=path_for


def motion(variant):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=BASE/variant;q=verify_request(root);cal=read(CALIBRATION);report=q['camera_path_report'];meta=read(report['reference_metadata'])
    # Audit the actual saved per-time matrices, not just the path generator's
    # nominal controls. Transfer each through its own normalization first.
    raw=[calibration_pose(r['camera'],cal,read(r['metadata'])) for r in q['inventory']]
    rows=[normalize_frame(r,cal,meta) for r in raw];poses=np.array([r['transform_matrix'] for r in rows]);pos=poses[:,:3,3]
    anchors=np.array(report['reference_anchor_centers']);weights=np.array([r['convex_weights'] for r in rows])
    assert weights.min()>-1e-12;np.testing.assert_allclose(weights.sum(1),1,atol=1e-12)
    np.testing.assert_allclose(pos,weights@anchors,atol=1e-10,rtol=0)
    train=next(r for r in cameras('000973')[0] if r['physical_camera']==report['endpoint_train_camera'])
    np.testing.assert_allclose(poses[118:],np.repeat(np.array(train['transform_matrix'])[None],32,axis=0),atol=1e-10,rtol=0)
    for key in ['fl_x','fl_y','cx','cy']:assert np.ptp([r[key] for r in rows[118:]])==0
    assert [r['frame_id'] for r in q['inventory']]==[f'{899+2*i:06d}' for i in range(150)]
    rawpos=np.array([r['transform_matrix'] for r in raw])[:,:3,3]
    hull=ConvexHull(np.array([r['transform_matrix'] for r in cal['frames'] if r['physical_camera'] not in HELD_CAMERAS])[:,:3,3])
    assert (rawpos@hull.equations[:,:3].T+hull.equations[:,3]).max()<1e-7
    phase=np.unwrap([r['spiral_phase_radians'] for r in rows]);end_turn=report['first_turn_end_index']
    assert phase[0]-phase[end_turn]>=2*np.pi and rows[end_turn]['spiral_radius_factor']>.7
    target=np.array(report['fixed_target']);radius=np.linalg.norm(pos-target,axis=1)
    speed=np.r_[0,np.linalg.norm(np.diff(pos,axis=0),axis=1)]*24
    probe=root/'motion_probe';probe.mkdir(exist_ok=True);reference=q['inventory'][48]
    mesh=o3d.io.read_triangle_mesh(reference['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    images=[];records=[];projections=[];times=[0,12,24,36,48,60,80,94]
    canvas=Image.new('RGB',(1080,864));draw=ImageDraw.Draw(canvas)
    for j,i in enumerate(times):
        cam=normalize_frame(raw[i],cal,read(reference['metadata']))
        for key in ['fl_x','fl_y','cx','cy']:cam[key]=raw[0][key]
        d=camera_depth(scene,cam)[0];hit=np.isfinite(d);rgb=np.zeros((*d.shape,3),np.uint8)
        lo,hi=np.quantile(d[hit],[.02,.98]);rgb[hit]=np.clip(220-130*(d[hit]-lo)/(hi-lo),40,240)[:,None]
        image=Image.fromarray(np.rot90(rgb));path=probe/f'{i:03d}_fixed_actor_fixed_lens.png';image.save(path)
        x,y=j%4*270,j//4*432;canvas.paste(image.resize((216,384)),(x,y+24));draw.text((x,y),f'pose{i}; same mesh/lens',fill='white')
        records.append(dict(index=i,image=str(path),sha256=sha(path)))
    canvas.save(probe/'contact.png')
    landmarks=np.array([target,target+.03*poses[0,:3,2],target-.03*poses[0,:3,2]])
    for row in rows:
        cam=deepcopy(row)
        for key in ['fl_x','fl_y','cx','cy']:cam[key]=rows[0][key]
        projections.append([portrait_projection(p,cam) for p in landmarks])
    projections=np.array(projections);relative=projections[:,1]-projections[:,2]
    # Full physical trajectory projected onto endpoint transverse axes.
    ep=np.array(train['transform_matrix']);axes=ep[:3,[1,0]]
    xy=(pos-ep[:3,3])@axes
    old=read(PREVIOUS/'request.json');oldpos=np.array([normalize_frame(calibration_pose(r['camera'],cal,read(r['metadata'])),cal,meta)['transform_matrix'] for r in old['inventory']])[:,:3,3]
    oldxy=(oldpos-ep[:3,3])@axes
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].plot(oldxy[:,0],oldxy[:,1],label='previous free arc');ax[0].plot(xy[:,0],xy[:,1],label='new physical spiral')
    ax[0].scatter(xy[0,0],xy[0,1],c='green');ax[0].scatter(xy[118,0],xy[118,1],c='red');ax[0].axis('equal');ax[0].legend();ax[0].set_title('Actual centers; reference units')
    for i in [0,24,48,72,94,118]:ax[0].annotate(str(i),xy[i])
    ax[1].plot(np.arange(150)/24,speed);ax[1].set_title('Physical speed; reference units/s')
    ax[2].plot(np.arange(150)/24,[r['fl_x']/train['fl_x'] for r in rows]);ax[2].set_title('Separate virtual focal multiplier')
    for a in ax:a.grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'motion_plot.png');plt.close(fig)
    atomic_json(root/'motion_audit.json',dict(status='physical_lens_hold_gate_pass',request_sha256=sha(root/'request.json'),
        saved_camera_matrices_checked=True,convex_camera_mixture_checked=True,first_turn_end_index=end_turn,
        phase_turns_before_contraction=float((phase[0]-phase[end_turn])/(2*np.pi)),
        actual_first95_transverse_extent=np.ptp(xy[:95],axis=0).tolist(),
        previous_first95_transverse_extent=np.ptp(oldxy[:95],axis=0).tolist(),
        fixed_lens_relative_parallax_extent_pixels=np.ptp(relative,axis=0).tolist(),
        hold_start_index=118,hold_count=32,actor_hold=False,geometry_unchanged=True,
        fixed_actor_fixed_lens_controls=records,physical_radius=radius.tolist(),physical_speed=speed.tolist(),
        monotonic_radial_dolly=False,script_sha256=sha(__file__)))
    print(variant,'actual saved spiral verified, firstturn',end_turn,flush=True)


def publish(variant):
    root=BASE/variant;out=root/'presentation';q=verify_request(root);notes=read(root/'manual_visual_review.json');m=read(out/'video_manifest.json')
    assert notes['status']=='reviewed_hybrid_choice_with_known_residuals'
    assert notes['overview_groups_inspected']==list(range(0,150,10))
    for name,digest in notes['inspected_image_hashes'].items():assert sha(root/name)==digest
    assert sha(out/'video.mp4')==m['video_sha256'] and sha(out/'frames.zip')==m['frames_zip_sha256']
    report=Path(__file__).parents[1]/'experiments/dec5_cinematic_wide_spiral.md';shutil.copyfile(report,root/'report.md')
    snapshot=root/'script_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in q['script_hashes'].items():assert sha(Path(__file__).with_name(name))==digest;shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    for name in [Path(__file__).name,'finalize_cinematic_live.py','compose_cinematic_train_ending.py']:shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    files=['request.json','motion_audit.json','raw_integrity_audit.json','manual_visual_review.json','report.md',
        'presentation/integrity_audit.json','presentation/video_manifest.json','presentation/video.mp4','presentation/frames.zip']
    atomic_json(root/'publication.json',dict(status='reviewed_wide_spiral_hybrid_with_known_residuals',bindings={n:sha(root/n) for n in files},
        video=str(out/'video.mp4'),frames_zip=str(out/'frames.zip'),all_frames_are_3d_renders=False,
        last24_actual_train_RGB=True,artifact_free_approval=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['motion','panels','sheets','raw_audit','audit','encode','publish','record'])
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--groups',nargs='*',type=int,default=[]);p.add_argument('--images',nargs='*',default=[]);p.add_argument('--note');a=p.parse_args();configure()
    for variant in ([a.variant] if a.variant else VARIANTS):
        if a.action in ['motion','publish']:globals()[a.action](variant)
        elif a.action=='record':shared.record_review(variant,a.groups,a.images.copy(),a.note)
        else:getattr(shared,a.action)(variant)
