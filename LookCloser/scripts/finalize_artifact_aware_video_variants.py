"""Audit, review and encode three 24 fps open camera alternatives.

Checks camera execution against fresh depth casts, complete temporal inventory,
unchanged production geometry/radiometry and native-rotation-only output.
Visual findings are recorded separately from integrity checks.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,ROOT,SOURCE,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from finalize_local_mesh_repair import verify_hashes
from artifact_aware_video_variants import BASE,PARENT,VARIANTS,path_for


def sheets(variant):
    out=BASE/variant;request=verify_request(out);root=out/'review';root.mkdir(exist_ok=True)
    for start in range(0,150,10):
        selected=request['inventory'][start:start+10]
        if not all((out/'frames'/r['frame_id']/'complete.json').exists() for r in selected):continue
        expected_bindings=[dict(frame=r['frame_id'],render_sha256=sha(out/'frames'/r['frame_id']/'frame.png')) for r in selected]
        for name,box,size in [('overview',None,(216,384)),('head',(100,400,1000,1250),(450,425)),
                              ('hand',(220,930,880,1550),(330,310))]:
            target=root/f'{start:03d}_{start+9:03d}_{name}.png'
            if target.exists() and target.with_suffix('.json').exists():
                previous=read(target.with_suffix('.json'))
                if previous['bindings']==expected_bindings and previous['sha256']==sha(target):continue
            w,h=size;panel=Image.new('RGB',(w*5,(h+24)*2));draw=ImageDraw.Draw(panel);bindings=[]
            for j,r in enumerate(selected):
                path=out/'frames'/r['frame_id']/'frame.png';im=Image.open(path)
                if box: im=im.crop(box)
                x,y=j%5*w,j//5*(h+24);panel.paste(im.resize(size),(x,y+24));draw.text((x+3,y+4),r['frame_id'],fill='white')
                bindings.append(dict(frame=r['frame_id'],render_sha256=sha(path)))
            panel.save(target)
            atomic_json(target.with_suffix('.json'),dict(bindings=bindings,sha256=sha(target),visual_status='pending'))


def audit(variant):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    from render_patchmatch_camera_path import normalize_frame
    out=BASE/variant;request=verify_request(out);parent=verify_request(PARENT);digest=sha(out/'request.json')
    ids=[f'{899+2*i:06d}' for i in range(150)]
    assert request['ordered_frame_ids']==ids and [r['frame_id'] for r in request['inventory']]==ids
    assert set(p.parent.name for p in (out/'frames').glob('*/complete.json'))==set(ids)
    assert request['camera_workaround_parent']==dict(path=str(PARENT/'request.json'),sha256=sha(PARENT/'request.json'))
    if variant=='closer_portrait':
        from closer_portrait_camera_variant import portrait_path
        expected,report=portrait_path(parent)
    else:expected,report=path_for(variant,parent)
    cal=read(CALIBRATION);renders={};meshes=[];centroids=[];poses=[];depth_checks=[];exposures=[]
    from scipy.spatial import ConvexHull
    train_centers=np.array([r['transform_matrix'] for r in cal['frames'] if r['physical_camera'] not in HELD_CAMERAS])[:,:3,3]
    hull=ConvexHull(train_centers);positions=np.array([r['transform_matrix'] for r in expected])[:,:3,3]
    hull_residual=positions@hull.equations[:,:3].T+hull.equations[:,3]
    assert hull_residual.max()<1e-5, 'Camera left calibrated train-center convex hull'
    source_manifests={Path(r['source_dataset']).name:r for r in request['source_rows']}
    assert source_manifests.keys()==set(ids)
    for record,original,camera in zip(request['inventory'],parent['inventory'],expected):
        frame=record['frame_id'];folder=out/'frames'/frame;receipt=read(folder/'complete.json');result=read(folder/'result.json')
        assert receipt['request_sha256']==digest;verify_hashes(folder,receipt['hashes'])
        assert result['camera']==record['camera'] and result['frame_id']==frame and result['index']==record['index']
        assert result['source_time_frame_count']==1 and not result['target_rgb_read'] and not result['rgb_averaging']
        assert len(set(result['source_cameras']))==62 and not (set(result['source_cameras'])&HELD_CAMERAS)
        for key in ['mesh','mesh_sha256','metadata','metadata_sha256','source_masks','source_transforms_sha256']:
            assert record[key]==original[key],key
        for key in ['mesh','metadata']: assert sha(record[key])==record[key+'_sha256']
        assert result['mesh_sha256']==record['mesh_sha256']
        assert sha(SOURCE/frame/'transforms.json')==record['source_transforms_sha256']
        spec=record['source_masks']
        for name,key in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:
            assert sha(Path(spec['root'])/name)==spec[key]
        # The frozen producer verifies every train EXR hash while loading. The
        # unchanged source manifest records all image hashes for this time.
        assert source_manifests[frame]==next(r for r in parent['source_rows'] if Path(r['source_dataset']).name==frame)
        fresh_camera=normalize_frame(camera,cal,read(record['metadata']))
        np.testing.assert_allclose(fresh_camera['transform_matrix'],record['camera']['transform_matrix'],atol=1e-8)
        for key in ['fl_x','fl_y','cx','cy','w','h']:
            multiplier=1.7 if variant=='closer_portrait' and key in ['fl_x','fl_y'] else 1
            assert record['camera'][key]==original['camera'][key]*multiplier
        im=np.asarray(Image.open(folder/'frame.png'));native=np.asarray(Image.open(folder/'prediction_native.png'))
        assert im.shape==(1920,1080,3) and np.array_equal(im,np.rot90(native)) and np.isfinite(im).all()
        yy,xx=np.nonzero(im.max(2)>0);assert len(xx)>1080*1920*.01
        centroids.append([float(xx.mean()),float(yy.mean())]);renders[frame]=sha(folder/'frame.png');meshes.append(record['mesh_sha256'])
        exposures.append(result['fixed_exposure']);poses.append(camera['transform_matrix'])
        saved=np.load(folder/'target_depth.npz')['depth'];assert np.isfinite(saved).all() and (saved>0).mean()>.01
        if record['index'] in [0,30,60,90,112,147,149]:
            mesh=o3d.io.read_triangle_mesh(record['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
            fresh=camera_depth(scene,record['camera'])[0];np.testing.assert_allclose(np.where(np.isfinite(fresh),fresh,0),saved,atol=1e-6)
            depth_checks.append(dict(frame=frame,actual_saved_depth_matches_fresh_cast=True))
    assert len(set(renders.values()))==150 and len(set(meshes))==150 and len(set(exposures))==1
    assert exposures[0]==read(ROOT/'exposure.json')['fixed_exposure_gain']
    assert request['recipe']['fps']==24 and not request['recipe']['static_registration'] and request['recipe']['source_incidence_power']==2
    assert not request['recipe']['averages_rgb'] and not request['uses_heldout_rgb']
    # A frozen actor rendered from separated movie poses proves spatial camera
    # execution independently of actor motion and of pose metadata.
    original=request['inventory'][75];mesh=o3d.io.read_triangle_mesh(original['mesh'])
    scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));control=[]
    for index in [0,60,112,149]:
        row=normalize_frame(expected[index],cal,read(original['metadata']));d=camera_depth(scene,row)[0]
        hit=np.isfinite(d);yy,xx=np.nonzero(np.rot90(hit));control.append(dict(index=index,depth_sha256=__import__('hashlib').sha256(np.where(hit,d,0).tobytes()).hexdigest(),
            foreground_centroid_portrait=[float(xx.mean()),float(yy.mean())]))
    assert len(set(r['depth_sha256'] for r in control))==4
    extent=np.ptp(centroids,axis=0);assert extent[0]>(100 if variant=='closer_portrait' else 200)
    assert report['view_angle_extent_degrees']>15
    result=dict(status='integrity_pass_visual_residuals_separate',request_sha256=digest,unique_times=150,unique_meshes=150,unique_renders=150,
        times=ids,render_hashes=renders,production_geometry_unchanged=True,production_fixed_radiometry_unchanged=True,
        native_rotation_only=True,all_saved_depths_finite=True,fresh_depth_checks=depth_checks,identical_actor_camera_controls=control,
        actual_rgb_foreground_centroid_extent_pixels=extent.tolist(),path=report,artifact_free_claimed=False,
        train_center_convex_hull_max_residual=float(hull_residual.max()),
        source_EXR_hashes_verified_by_frozen_producer=True,script_sha256=sha(__file__),utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'integrity_audit.json',result);print(variant,'audit_pass',extent.tolist(),flush=True)


def encode(variant):
    out=BASE/variant;request=verify_request(out);audit=read(out/'integrity_audit.json')
    assert audit['request_sha256']==sha(out/'request.json')
    seq=out/'encode_sequence';seq.mkdir(exist_ok=True)
    for i,frame in enumerate(request['ordered_frame_ids']):
        source=out/'frames'/frame/'frame.png';assert sha(source)==audit['render_hashes'][frame]
        target=seq/f'{i:05d}.png'
        if target.exists():assert sha(target)==sha(source)
        else:os.link(source,target)
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    video=out/'video.mp4';partial=out/'video.partial.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate','24','-i',str(seq/'%05d.png'),'-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-threads','6','-movflags','+faststart',str(partial)],env=env,check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(partial)],env=env,text=True))['streams'][0]
    assert int(probe['nb_read_frames'])==150 and probe['r_frame_rate']=='24/1' and abs(float(probe['duration'])-6.25)<1e-5
    assert probe['width']==1080 and probe['height']==1920
    # Decode EVERY encoded frame, preserving native dimensions and order.
    decoded=out/'decoded';decoded.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(partial),'-vsync','0',str(decoded/'%05d.png')],env=env,check=True)
    files=sorted(decoded.glob('[0-9][0-9][0-9][0-9][0-9].png'));assert len(files)==150
    error=[];hashes=[]
    for i,path in enumerate(files):
        a=np.asarray(Image.open(path));b=np.asarray(Image.open(seq/f'{i:05d}.png'))
        assert a.shape==b.shape;mae=float(np.abs(a.astype(float)-b).mean());assert mae<8
        error.append(mae);hashes.append(sha(path))
    assert len(set(hashes))==150
    os.replace(partial,video)
    overview=Image.new('RGB',(1080,1224));draw=ImageDraw.Draw(overview)
    for j,index in enumerate(range(0,150,10)):
        x,y=j%5*216,j//5*408;overview.paste(Image.open(files[index]).resize((216,384)),(x,y+24))
        draw.text((x+3,y+4),request['ordered_frame_ids'][index],fill='white')
    overview.save(out/'decoded_overview.png')
    archive=out/'frames.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for frame in request['ordered_frame_ids']:z.write(out/'frames'/frame/'frame.png',f'{frame}.png')
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None and z.namelist()==[f+'.png' for f in request['ordered_frame_ids']]
    atomic_json(out/'video_manifest.json',dict(video=str(video),video_sha256=sha(video),frames_zip_sha256=sha(archive),
        ffprobe=probe,decoded_all_150_frames=True,decoded_unique_frames=150,decoded_mean_absolute_error_max=max(error),
        decoded_overview_sha256=sha(out/'decoded_overview.png'),integrity_audit_sha256=sha(out/'integrity_audit.json'),
        normal_speed=True,source_time_repetition=False,frame_interpolation=False,artifact_free_claimed=False,
        visual_status='requires_explicit_review_record',request_sha256=sha(out/'request.json')))
    print(variant,'encoded_150_normal_speed',flush=True)


def publish(variant):
    out=BASE/variant;request=verify_request(out);manifest=read(out/'video_manifest.json');notes=read(out/'manual_visual_review.json')
    assert manifest['request_sha256']==sha(out/'request.json') and sha(out/'video.mp4')==manifest['video_sha256']
    assert sha(out/'frames.zip')==manifest['frames_zip_sha256']
    assert notes['status']=='reviewed_choice_with_known_residuals' and notes['known_residuals']
    assert notes['overview_groups_inspected']==list(range(0,150,10))
    evidence=[]
    for relative in notes['inspected_image_paths']:
        path=out/relative;assert path.is_file();evidence.append(dict(path=str(path),sha256=sha(path)))
    canary=read(BASE/'canary_visual_review.json')
    for relative in canary['inspected_comparison_panels']:
        path=BASE/relative;evidence.append(dict(path=str(path),sha256=sha(path)))
    script_root=out/'script_snapshot';script_root.mkdir(exist_ok=True)
    import shutil
    for name,digest in request['script_hashes'].items():
        source=Path(__file__).with_name(name);assert sha(source)==digest;shutil.copyfile(source,script_root/name)
    shutil.copyfile(__file__,script_root/Path(__file__).name)
    report=Path(__file__).parents[1]/'experiments'/'dec5_artifact_aware_video_variants.md'
    shutil.copyfile(report,out/'report.md')
    atomic_json(out/'publication.json',dict(status='reviewed_camera_choice_with_known_residuals',variant=variant,
        video=str(out/'video.mp4'),video_sha256=sha(out/'video.mp4'),frames_zip=str(out/'frames.zip'),
        frames_zip_sha256=sha(out/'frames.zip'),request_sha256=sha(out/'request.json'),
        audit_sha256=sha(out/'integrity_audit.json'),manifest_sha256=sha(out/'video_manifest.json'),
        visual_review_sha256=sha(out/'manual_visual_review.json'),canary_visual_review_sha256=sha(BASE/'canary_visual_review.json'),
        evidence=evidence,report_sha256=sha(out/'report.md'),artifact_free_approval=False,production_replacement=False,
        utc=datetime.now(timezone.utc).isoformat()))
    print(variant,'published_choice_with_residuals',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['sheets','audit','encode','publish']);p.add_argument('--variant',choices=(*VARIANTS,'closer_portrait'))
    a=p.parse_args()
    for variant in ([a.variant] if a.variant else VARIANTS):globals()[a.action](variant)
