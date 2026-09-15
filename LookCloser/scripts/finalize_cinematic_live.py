"""Audit and encode the explicitly hybrid 118+8+24 cinematic presentation.

Raw mesh renders are immutable inputs. Presentation frames are independently
checked against the exact render/train/dissolve provenance before encoding.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,ROOT,SOURCE,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request
from render_patchmatch_camera_path import normalize_frame
from finalize_local_mesh_repair import verify_hashes
from cinematic_pushin_live_ending import BASE,PARENT,VARIANTS,CANARIES,RAW_IDS,path_for
import finalize_cinematic_pushin as previews
import finalize_large_motion_choices as encoder

def panels(variant):
    previews.BASE=BASE;previews.CANARIES=CANARIES;previews.panels(variant)

def sheets(variant):
    root=BASE/variant;request=verify_request(root);dest=root/'review';dest.mkdir(exist_ok=True)
    for start in range(0,150,10):
        rows=request['inventory'][start:start+10];paths=[root/'presentation'/'frames'/r['frame_id']/'frame.png' for r in rows]
        if not all(p.exists() for p in paths):
            if start>=110:continue
            paths=[root/'frames'/r['frame_id']/'frame.png' for r in rows]
        if not all(p.exists() for p in paths):continue
        bindings=[dict(path=str(p),sha256=sha(p)) for p in paths];target=dest/f'{start:03d}_{start+9:03d}_overview.png'
        if target.exists() and read(target.with_suffix('.json'))['bindings']==bindings:continue
        im=Image.new('RGB',(1080,816));draw=ImageDraw.Draw(im)
        for j,(p,row) in enumerate(zip(paths,rows)):
            x,y=j%5*216,j//5*408;im.paste(Image.open(p).resize((216,384)),(x,y+24));draw.text((x,y),row['frame_id'],fill='white')
        im.save(target);atomic_json(target.with_suffix('.json'),dict(bindings=bindings,sha256=sha(target)))

def raw_audit(variant):
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    root=BASE/variant;req=verify_request(root);parent=verify_request(PARENT);expected,_=path_for(variant,parent);cal=read(CALIBRATION)
    assert req['ordered_frame_ids']==[f'{899+2*i:06d}' for i in range(150)] and req['raw_render_frame_ids']==RAW_IDS
    assert {p.parent.name for p in (root/'frames').glob('*/complete.json')}==set(RAW_IDS)
    hashes={};exposures=[];casts=[];meshes=[]
    for i,(rec,original,raw) in enumerate(zip(req['inventory'],parent['inventory'],expected)):
        for key in ['mesh','mesh_sha256','metadata','metadata_sha256','source_masks','source_transforms_sha256']:assert rec[key]==original[key]
        for key in ['mesh','metadata']:assert sha(rec[key])==rec[key+'_sha256']
        assert sha(SOURCE/rec['frame_id']/'transforms.json')==rec['source_transforms_sha256']
        for name,key in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:assert sha(Path(rec['source_masks']['root'])/name)==rec['source_masks'][key]
        camera=normalize_frame(raw,cal,read(rec['metadata']));np.testing.assert_allclose(camera['transform_matrix'],rec['camera']['transform_matrix'],atol=1e-9)
        for key in ['fl_x','fl_y','cx','cy','w','h']:assert camera[key]==rec['camera'][key]
        if i>=126:continue
        frame=rec['frame_id'];folder=root/'frames'/frame;receipt=read(folder/'complete.json');result=read(folder/'result.json')
        assert receipt['request_sha256']==sha(root/'request.json');verify_hashes(folder,receipt['hashes'])
        assert result['camera']==camera and result['mesh_sha256']==rec['mesh_sha256']
        assert result['source_time_frame_count']==1 and not result['target_rgb_read'] and not result['rgb_averaging']
        assert len(set(result['source_cameras']))==62 and not(set(result['source_cameras'])&HELD_CAMERAS)
        im=np.asarray(Image.open(folder/'frame.png'));native=np.asarray(Image.open(folder/'prediction_native.png'))
        assert im.shape==(1920,1080,3) and np.array_equal(im,np.rot90(native))
        depth=np.load(folder/'target_depth.npz')['depth'];assert np.isfinite(depth).all() and (depth>0).mean()>.01
        if i in [0,24,48,69,112,118,125]:
            mesh=o3d.io.read_triangle_mesh(rec['mesh']);scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles));fresh=camera_depth(scene,camera)[0]
            np.testing.assert_allclose(np.where(np.isfinite(fresh),fresh,0),depth,atol=1e-6);casts.append(frame)
        hashes[frame]=sha(folder/'frame.png');meshes.append(rec['mesh_sha256']);exposures.append(result['fixed_exposure'])
    assert len(set(hashes.values()))==len(set(meshes))==126 and len(set(exposures))==1
    assert exposures[0]==read(ROOT/'exposure.json')['fixed_exposure_gain']
    assert req['source_rows']==parent['source_rows'] and req['recipe']['source_incidence_power']==2 and not req['recipe']['static_registration']
    motion=read(root/'motion_audit.json');assert motion['request_sha256']==sha(root/'request.json') and motion['hold_start_index']==118
    atomic_json(root/'raw_integrity_audit.json',dict(status='raw126_integrity_pass',request_sha256=sha(root/'request.json'),render_hashes=hashes,
        raw_3d_count=126,unchanged150_actor_inventory=True,unique_raw_renders=126,unique_raw_meshes=126,fresh_depth_cast_times=casts,
        all_depths_finite=True,production_geometry_and_radiometry_unchanged=True,heldout_used=False,artifact_free_claimed=False,
        source_EXR_hashes_verified_by_frozen_producer=True,motion_audit_sha256=sha(root/'motion_audit.json'),script_sha256=sha(__file__)))
    print(variant,'raw126_audit_pass',flush=True)

def audit(variant):
    root=BASE/variant;req=verify_request(root);out=root/'presentation';raw=read(root/'raw_integrity_audit.json');complete=read(out/'complete.json')
    assert raw['request_sha256']==sha(root/'request.json')
    ending_config=read(root/'train_ending'/'request.json')
    assert ending_config==read(out/'request.json') and ending_config['raw_request_sha256']==sha(root/'request.json')
    assert ending_config['script_sha256']==sha(Path(__file__).with_name('compose_cinematic_train_ending.py'))
    assert complete['ordered_frame_ids']==req['ordered_frame_ids'] and complete['pure_train_count']==24 and complete['dissolve_count']==8
    hashes={};kinds=[]
    for i,frame in enumerate(req['ordered_frame_ids']):
        rec=read(out/'frames'/frame/'complete.json');assert rec==complete['records'][i]
        assert rec['request_sha256']==sha(out/'request.json') and rec['frame_id']==frame and rec['index']==i
        image=np.asarray(Image.open(out/'frames'/frame/'frame.png'));assert image.shape==(1920,1080,3)
        t=np.clip((i-117)/9,0,1);alpha=float(t**3*(10-15*t+6*t*t));assert rec['train_alpha']==alpha
        a=b=None
        if i<126:
            a=np.asarray(Image.open(root/'frames'/frame/'frame.png'));assert sha(root/'frames'/frame/'frame.png')==raw['render_hashes'][frame]
        if i>=118:
            source=root/'train_ending'/'frames'/frame;sr=read(source/'complete.json');b=np.asarray(Image.open(source/'frame.png'))
            assert sr['request_sha256']==sha(root/'train_ending'/'request.json') and sr['camera']==req['inventory'][i]['camera']
            assert sha(source/'frame.png')==sr['image_sha256'] and sha(sr['source_exr'])==sr['source_sha256']
            assert sr['source_physical_camera']=='H004_C005_1210SZ' and not sr['mesh_used'] and not sr['generated_pixels']
            frozen=next(r for r in req['source_rows'] if Path(r['source_dataset']).name==frame)
            witness=next(r for r in frozen['source_images'] if r['physical_camera']==sr['source_physical_camera'])
            assert witness['sha256']==sr['source_sha256'] and 'frame_train_' in witness['file_path']
            assert Path(sr['source_exr'])==(Path(frozen['source_dataset'])/witness['file_path']).resolve()
        expected=a if i<118 else b if i>=126 else np.rint(a.astype(np.float32)*(1-alpha)+b.astype(np.float32)*alpha).clip(0,255).astype(np.uint8)
        np.testing.assert_array_equal(image,expected);hashes[frame]=sha(out/'frames'/frame/'frame.png');assert hashes[frame]==rec['image_sha256'];kinds.append(rec['kind'])
    assert len(set(hashes.values()))==150 and kinds.count('3d_render')==118 and kinds.count('real_train_rgb')==24
    atomic_json(out/'integrity_audit.json',dict(status='hybrid150_integrity_pass',request_sha256=sha(out/'request.json'),render_hashes=hashes,
        raw_request_sha256=sha(root/'request.json'),raw_audit_sha256=sha(root/'raw_integrity_audit.json'),presentation_complete_sha256=sha(out/'complete.json'),
        raw_3d_only_count=118,explicit_display_dissolve_count=8,pure_real_train_count=24,all150_unique=True,fps=24,duration_seconds=6.25,
        all_frames_are_3d_renders=False,final24_pixels_equal_prepared_real_train=True,original_background_preserved=True,artifact_free_claimed=False))
    print(variant,'hybrid150_audit_pass',flush=True)

def encode(variant):
    # Encoder's checked sequence is presentation/frames, never raw frames/.
    root=BASE/variant;encoder.BASE=BASE
    encoder.verify_request=lambda out: verify_request(Path(out).parent)
    encoder.encode(variant+'/presentation')
    manifest=read(root/'presentation'/'video_manifest.json')
    manifest.update(all_frames_are_3d_renders=False,last_second='24 real train RGB frames',transition='8-frame display dissolve; real background appears',
        raw_request_sha256=sha(root/'request.json'))
    atomic_json(root/'presentation'/'video_manifest.json',manifest)

def publish(variant):
    root=BASE/variant;out=root/'presentation';req=verify_request(root);notes=read(root/'manual_visual_review.json');m=read(out/'video_manifest.json')
    assert notes['overview_groups_inspected']==list(range(0,150,10)) and notes['known_residuals']
    assert notes['status']=='reviewed_hybrid_choice_with_known_residuals'
    for relative,digest in notes['inspected_image_hashes'].items():assert sha(root/relative)==digest
    assert 'presentation/decoded_overview.png' in notes['inspected_image_paths']
    assert sha(out/'video.mp4')==m['video_sha256'] and sha(out/'frames.zip')==m['frames_zip_sha256']
    assert read(root/'train_ending'/'request.json')['script_sha256']==sha(Path(__file__).with_name('compose_cinematic_train_ending.py'))
    shutil.copyfile(Path(__file__).parents[1]/'experiments'/'dec5_cinematic_pushin_choices.md',root/'report.md')
    snapshot=root/'script_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in req['script_hashes'].items():assert sha(Path(__file__).with_name(name))==digest;shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    for name in ['finalize_cinematic_live.py','audit_cinematic_pushin_motion.py','supervise_cinematic_live.py','compose_cinematic_train_ending.py']:shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    files=['request.json','raw_integrity_audit.json','motion_audit.json','manual_visual_review.json','report.md','train_ending/request.json',
        'presentation/request.json','presentation/complete.json','presentation/integrity_audit.json','presentation/video_manifest.json','presentation/video.mp4','presentation/frames.zip']
    evidence=[dict(path=str(root/p),sha256=sha(root/p)) for p in notes['inspected_image_paths']]
    atomic_json(root/'publication.json',dict(status='reviewed_hybrid_cinematic_choice_with_known_residuals',bindings={n:sha(root/n) for n in files},evidence=evidence,
        video=str(out/'video.mp4'),frames_zip=str(out/'frames.zip'),all_frames_are_3d_renders=False,pure_train_ending_count=24,
        artifact_free_approval=False,production_replacement=False,utc=datetime.now(timezone.utc).isoformat()))
    print(variant,'hybrid_published',flush=True)

def record_review(variant,groups,images,note):
    """Call only AFTER the operator has actually viewed the specified images."""
    root=BASE/variant;target=root/'manual_visual_review.json'
    record=read(target) if target.exists() else dict(status='review_in_progress',overview_groups_inspected=[],inspected_image_paths=[],
        known_residuals=read(BASE/'canary_visual_review.json')['known_residuals'],notes=[],inspected_image_hashes={})
    for start in groups:
        assert start in range(0,150,10)
        images.append(f'review/{start:03d}_{start+9:03d}_overview.png')
        record['overview_groups_inspected']=sorted(set(record['overview_groups_inspected']+[start]))
    for relative in images:
        digest=sha(root/relative)
        if relative in record['inspected_image_hashes']:assert record['inspected_image_hashes'][relative]==digest
        record['inspected_image_hashes'][relative]=digest
    record['inspected_image_paths']=sorted(record['inspected_image_hashes'])
    if record['overview_groups_inspected']==list(range(0,150,10)) and 'presentation/decoded_overview.png' in record['inspected_image_paths']:
        record['status']='reviewed_hybrid_choice_with_known_residuals'
    if note:record['notes'].append(note)
    atomic_json(target,record)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['panels','sheets','raw_audit','audit','encode','publish','record']);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--groups',nargs='*',type=int,default=[]);p.add_argument('--images',nargs='*',default=[]);p.add_argument('--note');a=p.parse_args()
    for variant in ([a.variant] if a.variant else VARIANTS):
        if a.action=='record':record_review(variant,a.groups,a.images.copy(),a.note)
        else:globals()[a.action](variant)
