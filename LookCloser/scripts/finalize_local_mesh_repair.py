"""Publish and audit the explicitly reviewed, single-frame mesh repair pilot.

No diffusion RGB becomes real evidence. No source EXR, rig parameter, existing
published mesh, or model default is changed. See dec5_diffusion_mesh_repair.md.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import os
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read,sha,atomic_json,HELD_CAMERAS,CALIBRATION
from diffusion_mesh_repair import OUTPUT,BASE
from bake_joint_temporal_mesh import export_asset


def verify_hashes(root,hashes):
    root=Path(root).resolve()
    for name,expected in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def publish(output):
    source=output/'neutral_supported_asset/frames/000973';target=output/'final_supported/frames/000973'
    verify_hashes(source,read(source/'complete.json')['hashes'])
    request={'source':str(source),'complete_sha256':sha(source/'complete.json'),
             'component_rule':'max(100, 0.002 * largest_component_triangles)',
             'script_sha256':sha(__file__)}
    if (target/'publication_request.json').exists():
        if read(target/'publication_request.json')!=request:raise ValueError('Publication request mismatch')
        if (target/'complete.json').exists():
            verify_hashes(target,read(target/'complete.json')['hashes']);return
    target.mkdir(parents=True,exist_ok=True);atomic_json(target/'publication_request.json',request)
    atlas=dict(np.load(source/'atlas_geometry.npz'));mesh=o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(atlas['vertices']),o3d.utility.Vector3iVector(atlas['triangles']))
    labels,counts,_=mesh.cluster_connected_triangles();counts=np.array(counts)
    threshold=max(100.,.002*counts.max());keep=counts[np.array(labels)]>=threshold
    atlas['triangles']=atlas['triangles'][keep];atlas['indices']=atlas['indices'][keep]
    if not np.array_equal(atlas['mapping'][atlas['indices']],atlas['triangles']):raise ValueError('UV mapping changed')
    np.savez_compressed(target/'atlas_geometry.npz',**atlas)
    np.save(target/'base_face_ids.npy',np.load(source/'base_face_ids.npy')[keep])
    for name in ['texture_joint.png','patch_texture_evidence.npz','repair_bake_result.json']:
        shutil.copyfile(source/name,target/name)
    mesh.triangles=o3d.utility.Vector3iVector(atlas['triangles']);mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(target/'mesh.ply'),mesh)
    export_asset(target,atlas,'000973',None,tag='repaired')
    manifest=read(target/'asset_manifest.json');manifest.update(geometry_changed_vs_original=True,
        geometry_changed_during_export=False,input_is_repaired_mesh=True,raw_tsdf_volume_saved=False)
    atomic_json(target/'asset_manifest.json',manifest)
    ops=read(source/'operations.json');ops.update(output_mesh_sha256=sha(target/'mesh.ply'),
        previous_operations_sha256=sha(source/'operations.json'),publication_component_cleanup={
            'threshold':threshold,'removed_triangles':int((~keep).sum()),
            'component_triangles_before':sorted(counts.tolist(),reverse=True),
            'component_triangles_after':sorted(counts[counts>=threshold].tolist(),reverse=True)},
        status='improved_pilot_with_known_residual_defects_not_artifact_free')
    atomic_json(target/'operations.json',ops)
    atomic_json(target/'complete.json',{'status':'integrity_complete_quality_review_separate',
        'hashes':{str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file() and p.name!='complete.json'}})
    print(f'published mesh={target} removed_detached_triangles={int((~keep).sum())}',flush=True)


def video_crops(output):
    video=output/'flythrough_supported';request=read(video/'request.json');frames=request['camera_path']
    from joint_temporal_texture import project
    tube=np.array(read(output/'request.json')['tube_seed']);target=video/'detail_review';target.mkdir(exist_ok=True)
    sampled=np.linspace(0,len(frames)-1,12,dtype=int);records=[]
    for index in sampled:
        image=Image.open(video/'frames'/f'{index:05d}.png');row=frames[index]
        uv,_=project(tube[None],[row]);x,y=uv[0,0];cx=float(y);cy=float(1919-x)
        # Portrait 1:1 detail: includes tube, fingers, chin and lower face.
        box=[int(cx-110),int(cy-180),int(cx+450),int(cy+270)]
        image.crop(box).save(target/f'{index:05d}.png')
        records.append({'index':int(index),'seconds':int(index)/request['report']['fps'],
                        'portrait_crop_xyxy':box,'sha256':sha(target/f'{index:05d}.png')})
    for batch in range(3):
        sheet=Image.new('RGB',(560*2,474*2));draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(records[batch*4:batch*4+4]):
            x=(k%2)*560;y=(k//2)*474;sheet.paste(Image.open(target/f'{r["index"]:05d}.png'),(x,y+24))
            draw.text((x+4,y+4),f'{r["seconds"]:.2f}s',fill='white')
        sheet.save(target/f'sheet_{batch}.png')
    atomic_json(target/'manifest.json',{'native_scale':True,'samples':records,'not_all_video_frames_visually_reviewed':True})


def audit(output):
    target=output/'final_supported/frames/000973';video=output/'flythrough_supported';request=read(output/'request.json')
    verify_hashes(target,read(target/'complete.json')['hashes'])
    if sha(request['base_mesh'])!=request['base_mesh_sha256']:raise ValueError('Original mesh changed')
    if sha(CALIBRATION)!=request['calibration_sha256'] or sha(request['base_metadata'])!=request['base_metadata_sha256']:
        raise ValueError('Original calibration or geometry normalization changed')
    atlas=dict(np.load(target/'atlas_geometry.npz'));base=dict(np.load(BASE/'atlas_geometry.npz'));ids=np.load(target/'base_face_ids.npy')
    if not np.array_equal(atlas['vertices'][:len(base['vertices'])],base['vertices']):raise ValueError('Original vertices moved')
    if not np.array_equal(atlas['triangles'][ids>=0],base['triangles'][ids[ids>=0]]):raise ValueError('Retained original triangles changed')
    old=np.asarray(Image.open(BASE/'texture_joint.png'));new=np.asarray(Image.open(target/'texture_joint.png'))
    if not np.array_equal(new[:old.shape[0],:old.shape[1]],old):raise ValueError('Original texture changed')
    for i in range(3):
        p=output/'views'/f'{i:02d}';vr=read(p/'request.json');x0,y0,x1,y1=vr['native_crop_xyxy']
        original=np.asarray(Image.open(p/'render_native.png'));edited=np.asarray(Image.open(p/'synthetic_masked_native.png'))
        mask=np.zeros(original.shape[:2],bool);mask[y0:y1,x0:x1]=np.asarray(Image.open(p/'mask_native_crop.png'))>0
        if not np.array_equal(original[~mask],edited[~mask]):raise ValueError('Synthetic mask leakage')
        receipt=read(p/'generation_receipt.json')
        if sha(p/'generated_raw.png')!=receipt['generated_raw_sha256'] or sha(p/'synthetic_masked_native.png')!=receipt['synthetic_masked_sha256']:
            raise ValueError('Generated image changed after stereo')
    stereo=read(output/'mvs/input_manifest.json');sources=stereo['sources']
    real=[r for r in sources if r['kind']=='real_train_rgb']
    if len(real)!=8 or len(sources)!=11 or set(r['physical_camera'] for r in real)&HELD_CAMERAS:
        raise ValueError('Synthetic/real/heldout split mismatch')
    for row in sources:
        if sha(output/'mvs/data'/row['file'])!=row['image_sha256']:raise ValueError('Stereo input checksum mismatch')
    metrics=read(output/'heldout_review_supported_final/face_metrics.json')
    for row in metrics['variants'].values():
        if not all(np.isfinite(row[k]) for k in ['face_psnr','face_ssim','face_lpips']):raise ValueError('Nonfinite face metric')
        if any(k in row for k in ['psnr','ssim','lpips','loss']):raise ValueError('Unexpected non-face metrics')
    manifest=read(video/'frame_manifest.json');records=manifest['frames'];paths=sorted((video/'frames').glob('*.png'))
    if len(paths)!=360 or [r['index'] for r in records]!=list(range(360)):raise ValueError('Video inventory mismatch')
    for record,path in zip(records,paths):
        if sha(path)!=record['sha256']:raise ValueError('Video frame checksum mismatch')
        with Image.open(path) as im:
            if im.size!=(1080,1920) or not np.asarray(im).any():raise ValueError('Invalid video frame')
    env=dict(os.environ);env['LD_PRELOAD']='/lib/x86_64-linux-gnu/libmpg123.so.0'
    import json
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(video/'smooth_000973.mp4')],env=env,text=True))['streams'][0]
    if int(probe['nb_read_frames'])!=360 or abs(float(probe['duration'])-12)>1e-3:raise ValueError('Encoded video inventory mismatch')
    result=read(video/'result.json')
    if sha(video/'smooth_000973.mp4')!=result['video_sha256']:raise ValueError('Video checksum mismatch')
    verdict=read(output/'visual_review.json')
    if verdict['status'] in ['pending','uncertain']:raise ValueError('Missing completed pilot visual review')
    result.update(status='reviewed_improved_pilot_with_known_visual_defects',visual_review_sha256=sha(output/'visual_review.json'))
    atomic_json(video/'result.json',result)
    atomic_json(output/'result.json',{'status':'completed_pilot_with_known_visual_failures','source_frame':'000973',
        'mesh':str(target/'mesh.ply'),'mesh_sha256':sha(target/'mesh.ply'),
        'textured_glb':str(target/'dec5_000973_repaired.glb'),'glb_sha256':sha(target/'dec5_000973_repaired.glb'),
        'video':str(video/'smooth_000973.mp4'),'video_sha256':result['video_sha256'],
        'face_metrics':metrics['variants']['cylinder'],'visual_status':'fail','camera_motion_status':'pass',
        'source_time_frame_count':1,'rendered_camera_frame_count':360,'temporal_150_frame_campaign':False,
        'raw_tsdf_volume_saved':False})
    scripts=output/'replay_scripts';scripts.mkdir(exist_ok=True)
    for name in ['diffusion_mesh_repair.py','local_mesh_repair.py','bake_local_mesh_repair.py',
                 'review_local_mesh_repair.py','smooth_mesh_flythrough.py','finalize_local_mesh_repair.py',
                 'joint_temporal_texture.py','bake_joint_temporal_mesh.py','hard_surface_texture.py',
                 'render_patchmatch_camera_path.py','import_colmap_mvs_depth_dataset.py',
                 'export_nerfstudio_colmap_model.py','score_colmap_patchmatch_tsdf_face.py']:
        shutil.copyfile(Path(__file__).with_name(name),scripts/name)
    atomic_json(scripts/'manifest.json',{'description':'Final replay code snapshot, not a claim that rejected iterations used identical code',
        'hashes':{p.name:sha(p) for p in sorted(scripts.glob('*.py'))}})
    atomic_json(output/'audit.json',{'status':'pass_integrity_not_artifact_free_quality',
        'original_vertices_and_retained_faces_unchanged':True,'original_texture_pixels_unchanged':True,
        'generated_rgb_outside_masks_unchanged':True,'heldout_excluded_from_repair':True,
        'finite_face_only_metrics':True,'frame_count':360,'source_time_frame_count':1,'ffprobe':probe,
        'final_complete_sha256':sha(target/'complete.json'),'video_sha256':sha(video/'smooth_000973.mp4'),
        'visual_review_sha256':sha(output/'visual_review.json'),'face_metrics_sha256':sha(output/'heldout_review_supported_final/face_metrics.json')})
    print('audit=pass_integrity quality=known_residual_defects',flush=True)


def check(output):
    """Compact on-demand supervision; expected worker absence after completion."""
    from datetime import datetime,timezone
    import json
    lines=subprocess.check_output(['ps','-eo','pid,etime,rss,args'],text=True).splitlines()
    workers=[line for line in lines if 'python scripts/' in line and any(name in line for name in
        ['bake_local_mesh_repair.py','smooth_mesh_flythrough.py','review_local_mesh_repair.py'])
        and '/bin/bash' not in line]
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    record={'utc':datetime.now(timezone.utc).isoformat(),'stage':'worker_active' if workers else 'post_render_audit',
            'worker_processes':workers,'gpu_memory_and_utilization':memory,
            'free_bytes':shutil.disk_usage(output).free,'final_video_exists':(output/'flythrough_supported/smooth_000973.mp4').is_file()}
    with (output/'checks.jsonl').open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
    print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['publish','video-crops','audit','check'])
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    {'publish':publish,'video-crops':video_crops,'audit':audit,'check':check}[a.action](a.output)


if __name__=='__main__':main()
