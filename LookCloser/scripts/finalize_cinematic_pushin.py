"""Full temporal/depth integrity and normal-speed encode for cinematic shots."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,ROOT,HELD_CAMERAS,SOURCE
from render_smooth_temporal_mesh_video import verify_request
from finalize_local_mesh_repair import verify_hashes
from render_patchmatch_camera_path import normalize_frame
from cinematic_pushin_beauty import BASE,PARENT,VARIANTS,CANARIES,path_for
import finalize_large_motion_choices as shared

def panels(variant):
    root=BASE/variant;dest=root/'canary_review';dest.mkdir(exist_ok=True);canvas=Image.new('RGB',(1080,1632));d=ImageDraw.Draw(canvas);bindings=[]
    for j,frame in enumerate(CANARIES):
        path=root/'frames'/frame/'frame.png'
        if not path.exists():continue
        im=Image.open(path);rgb=np.asarray(im);yy,xx=np.nonzero(rgb.max(2)>0);y0=int(yy.min());sel=xx[yy<y0+650];cx=int(np.median(sel));x0=max(0,min(360,cx-360))
        box=(x0,y0,x0+720,min(y0+1000,1920));im.crop(box).save(dest/f'{frame}_head_native.png')
        x,y=j%4*270,j//4*544;canvas.paste(im.resize((270,480)),(x,y+24));d.text((x,y),frame,fill='white');bindings.append(dict(frame=frame,sha256=sha(path),box=box))
    canvas.save(dest/'contact.png');atomic_json(dest/'receipt.json',dict(bindings=bindings))

def audit(variant):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    root=BASE/variant;req=verify_request(root);parent=verify_request(PARENT);raw,report=path_for(variant,parent);cal=read(CALIBRATION)
    motion=read(root/'motion_audit.json');assert motion['status']=='physical_lens_hold_gate_pass' and motion['request_sha256']==sha(root/'request.json')
    ids=[f'{899+2*i:06d}' for i in range(150)];assert req['ordered_frame_ids']==ids
    assert {p.parent.name for p in (root/'frames').glob('*/complete.json')}==set(ids)
    hashes={};meshes=[];exposures=[];centroids=[];casts=[];hold=[]
    for i,(rec,original,camera) in enumerate(zip(req['inventory'],parent['inventory'],raw)):
        frame=ids[i];assert rec['frame_id']==frame;folder=root/'frames'/frame;receipt=read(folder/'complete.json');result=read(folder/'result.json')
        assert receipt['request_sha256']==sha(root/'request.json');verify_hashes(folder,receipt['hashes'])
        assert result['camera']==rec['camera'] and result['source_time_frame_count']==1 and not result['target_rgb_read'] and not result['rgb_averaging']
        assert len(set(result['source_cameras']))==62 and not(set(result['source_cameras'])&HELD_CAMERAS)
        for key in ['mesh','mesh_sha256','metadata','metadata_sha256','source_masks','source_transforms_sha256']:assert rec[key]==original[key]
        for key in ['mesh','metadata']:assert sha(rec[key])==rec[key+'_sha256']
        assert sha(SOURCE/frame/'transforms.json')==rec['source_transforms_sha256']
        for name,key in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:assert sha(Path(rec['source_masks']['root'])/name)==rec['source_masks'][key]
        expected=normalize_frame(camera,cal,read(rec['metadata']));np.testing.assert_allclose(expected['transform_matrix'],rec['camera']['transform_matrix'],atol=1e-9)
        for k in ['fl_x','fl_y','cx','cy','w','h']:assert expected[k]==rec['camera'][k]
        im=np.asarray(Image.open(folder/'frame.png'));native=np.asarray(Image.open(folder/'prediction_native.png'))
        assert im.shape==(1920,1080,3) and np.array_equal(im,np.rot90(native));yy,xx=np.nonzero(im.max(2)>0);assert len(xx)>20000
        centroids.append([float(xx.mean()),float(yy.mean())]);hashes[frame]=sha(folder/'frame.png');meshes.append(rec['mesh_sha256']);exposures.append(result['fixed_exposure'])
        depth=np.load(folder/'target_depth.npz')['depth'];assert np.isfinite(depth).all() and (depth>0).mean()>.01
        if i in [0,24,48,69,112,126,149]:
            mesh=o3d.io.read_triangle_mesh(rec['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));fresh=camera_depth(scene,rec['camera'])[0]
            np.testing.assert_allclose(np.where(np.isfinite(fresh),fresh,0),depth,atol=1e-6);casts.append(frame)
        if i>=126:hold.append(dict(frame=frame,mesh_sha256=rec['mesh_sha256'],render_sha256=hashes[frame]))
    assert len(set(hashes.values()))==len(set(meshes))==150 and len(set(exposures))==1
    assert exposures[0]==read(ROOT/'exposure.json')['fixed_exposure_gain'] and req['recipe']['fps']==24
    assert req['source_rows']==parent['source_rows'] and req['recipe']['source_incidence_power']==2 and not req['recipe']['static_registration']
    assert len({r['mesh_sha256'] for r in hold})==len({r['render_sha256'] for r in hold})==24
    atomic_json(root/'integrity_audit.json',dict(status='integrity_pass_visual_residuals_separate',request_sha256=sha(root/'request.json'),render_hashes=hashes,
        unique_times=150,unique_meshes=150,unique_renders=150,normal_fps=24,production_geometry_unchanged=True,production_radiometry_unchanged=True,
        native_rotation_only=True,source_EXR_hashes_verified_by_frozen_producer=True,all_depths_finite=True,fresh_depth_cast_times=casts,
        endpoint_hold_dynamic_actor=hold,actual_rgb_foreground_centroid_extent_pixels=np.ptp(centroids,axis=0).tolist(),motion_audit_sha256=sha(root/'motion_audit.json'),
        artifact_free_claimed=False,script_sha256=sha(__file__),utc=datetime.now(timezone.utc).isoformat()))
    print(variant,'integrity_pass',flush=True)

def publish(variant):
    import shutil
    root=BASE/variant;req=verify_request(root);m=read(root/'video_manifest.json');notes=read(root/'manual_visual_review.json')
    assert notes['overview_groups_inspected']==list(range(0,150,10)) and notes['known_residuals']
    assert sha(root/'video.mp4')==m['video_sha256'] and sha(root/'frames.zip')==m['frames_zip_sha256']
    report=Path(__file__).parents[1]/'experiments'/'dec5_cinematic_pushin_choices.md';shutil.copyfile(report,root/'report.md')
    scripts=root/'script_snapshot';scripts.mkdir(exist_ok=True)
    for name,h in req['script_hashes'].items():assert sha(Path(__file__).with_name(name))==h;shutil.copyfile(Path(__file__).with_name(name),scripts/name)
    for name in ['finalize_cinematic_pushin.py','audit_cinematic_pushin_motion.py','supervise_cinematic_pushin.py']:shutil.copyfile(Path(__file__).with_name(name),scripts/name)
    evidence=[dict(path=str(root/p),sha256=sha(root/p)) for p in notes['inspected_image_paths']]
    files=['request.json','integrity_audit.json','motion_audit.json','video_manifest.json','manual_visual_review.json','report.md','video.mp4','frames.zip']
    atomic_json(root/'publication.json',dict(status='reviewed_cinematic_choice_with_known_residuals',bindings={n:sha(root/n) for n in files},evidence=evidence,
        video=str(root/'video.mp4'),frames_zip=str(root/'frames.zip'),artifact_free_approval=False,production_replacement=False,utc=datetime.now(timezone.utc).isoformat()))
    print(variant,'published',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['panels','sheets','audit','encode','publish']);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    shared.BASE=BASE
    for variant in ([a.variant] if a.variant else VARIANTS):
        if a.action in ['sheets','encode']:getattr(shared,a.action)(variant)
        else:globals()[a.action](variant)
