"""Independent wide-path raycast audit, ordered encoding, and explicit visual review."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from joint_temporal_texture import read,sha,atomic_json,CALIBRATION,SOURCE,HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request,calibration_pose
from finalize_local_mesh_repair import verify_hashes

OUTPUT=Path('/mnt/data/dec5_wide_dynamic_flight_150_v2')


def diagram(output):
    request=verify_request(output);xy=np.array([r['camera']['rig_offset_xy'] for r in request['inventory']])
    replay=request['recipe']['camera_path_kind']=='replay_static_4x4'
    screen=request['recipe']['camera_path_kind'] in {'screen_travel','expanded_head','artifact_avoidance'}
    image=Image.new('RGB',(1100,700),(20,20,25));draw=ImageDraw.Draw(image)
    def pixel(q):return (int(550+110*q[0]),int(340-120*q[1]))
    for x in range(-4,5):
        for y in range(-2,3):
            px,py=pixel((x,y));draw.ellipse((px-3,py-3,px+3,py+3),fill=(100,100,100))
            draw.text((px+5,py+5),f'{chr(ord("H")+x)}/{chr(ord("C")-y)}',fill=(130,130,130))
    draw.line([pixel(q) for q in np.vstack([xy,xy[:1]])],fill=(255,150,40),width=3)
    for name,index in request['camera_path_report']['extrema_indices'].items():
        px,py=pixel(xy[index]);draw.ellipse((px-6,py-6,px+6,py+6),fill='cyan')
        draw.text((px+8,py-20),f'{name} n={index}',fill='cyan')
    for i in [10,30,60,90,110,135]:
        p=np.array(pixel(xy[i]));q=np.array(pixel(xy[i+1]));delta=q-p;delta=delta/max(np.linalg.norm(delta),1)
        side=np.array([-delta[1],delta[0]])
        draw.polygon([tuple(p+delta*9),tuple(p-delta*7+side*5),tuple(p-delta*7-side*5)],fill='white')
    draw.text((20,15),'EXACT earlier static 4x4 loop, now with 150 moving-actor times' if replay else 'Actual requested rig coordinates: LEFT -> RIGHT -> TOP -> BOTTOM -> LEFT',fill='white')
    draw.text((20,40),'150 DIFFERENT source times; source video is not periodic. Camera path is periodic.',fill='white')
    draw.text((20,650),'Saved pilot path: F..I / A..D. Full loop resampled, never only the first 150 of 720 poses.' if replay else 'Horizontal H +/-4 columns; vertical C +/-2 = ALL FIVE existing rows (not fictional +/-4).',fill='white')
    if screen:
        draw.rectangle((0,0,1100,70),fill=(20,20,25));draw.rectangle((0,640,1100,700),fill=(20,20,25))
        draw.text((20,15),'D..K / A..D: expanded two columns each side, with REAL screen-space object travel',fill='white')
        draw.text((20,40),'150 changing source times; fixed virtual lens; no image crop, stabilization or post-render translation',fill='white')
        draw.text((20,650),'The camera no longer keeps the scene landmark centered. Camera loop is not an actor-time loop.',fill='white')
        if request['recipe']['camera_path_kind']=='expanded_head':
            draw.rectangle((0,0,1100,35),fill=(20,20,25))
            draw.text((20,15),'C..L / A..E: five real anchors, no invented C/A corner or row above A',fill='white')
        if request['recipe']['camera_path_kind']=='artifact_avoidance':
            draw.rectangle((0,0,1100,35),fill=(20,20,25))
            draw.text((20,15),'Shot workaround: smooth smaller upper-left envelope; still real camera AND actor movement',fill='white')
    image.save(output/'camera_path.png')


def audit(output,require_reviews=False):
    request=verify_request(output);digest=sha(output/'request.json');inventory=request['inventory']
    for field in ['repair_parent','camera_workaround_parent','elevated_camera_parent']:
        if field in request:
            spec=request[field]
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Changed parent request provenance')
    ids=request['ordered_frame_ids']
    if ids!=[f'{899+2*i:06d}' for i in range(150)] or [r['frame_id'] for r in inventory]!=ids:
        raise ValueError('Wrong dynamic source-time inventory')
    if set(p.parent.name for p in (output/'frames').glob('*/complete.json'))!=set(ids):raise ValueError('Incomplete renders')
    cal=read(CALIBRATION);lookup={r['physical_camera']:r for r in cal['frames']}
    corners=np.asarray([lookup[n]['transform_matrix'] for n in request['camera_path_report']['anchors']])[:,:3,3]
    poses=[];weights=[];render_hashes={};mesh_hashes=[];reviews=[];image_centroids=[]
    workaround=request['recipe']['camera_path_kind']=='artifact_avoidance'
    expanded=request['recipe']['camera_path_kind'] in {'expanded_head','artifact_avoidance'}
    screen=request['recipe']['camera_path_kind'] in {'screen_travel','expanded_head','artifact_avoidance'}
    if expanded and request.get('partial_diagnostic_only'):raise ValueError('Cannot publish partial diagnostic')
    for record in inventory:
        frame=record['frame_id'];root=output/'frames'/frame;receipt=read(root/'complete.json')
        if receipt['request_sha256']!=digest:raise ValueError('Wrong frame request')
        verify_hashes(root,receipt['hashes']);result=read(root/'result.json')
        if result['frame_id']!=frame or result['camera']!=record['camera'] or result['index']!=record['index']:
            raise ValueError('Camera/time substitution')
        if result['rgb_averaging'] or result['target_rgb_read']:raise ValueError('Unexpected RGB construction')
        if len(set(result['source_cameras']))!=62 or set(result['source_cameras'])&HELD_CAMERAS:raise ValueError('Invalid source cameras')
        for field in ['mesh','metadata']:
            if sha(record[field])!=record[field+'_sha256']:raise ValueError('Changed geometry input')
        if result['mesh_sha256']!=record['mesh_sha256']:raise ValueError('Wrong rendered mesh')
        spec=record['source_masks'];source_root=Path(spec['root'])
        for name,key in [('complete.json','complete_sha256'),('masks.npz','masks_sha256'),('cameras.json','cameras_sha256')]:
            if sha(source_root/name)!=spec[key]:raise ValueError('Changed mask input')
        if expanded:
            for field in ['head_repair_receipt','notch_receipt']:
                if sha(record[field])!=record[field+'_sha256']:raise ValueError('Changed head repair receipt')
                repaired=read(record[field]);folder=Path(record[field]).parent
                if sha(folder/'mesh.ply')!=repaired['mesh_sha256'] or sha(folder/'operations.json')!=repaired['operations_sha256']:raise ValueError('Changed repair output')
                if 'evidence_sha256' in repaired and sha(folder/'evidence.npz')!=repaired['evidence_sha256']:raise ValueError('Changed repair evidence')
        elif sha(record['target_restoration_receipt'])!=record['target_restoration_receipt_sha256']:raise ValueError('Changed restoration receipt')
        if sha(SOURCE/frame/'transforms.json')!=record['source_transforms_sha256']:raise ValueError('Source time changed')
        image=np.array(Image.open(root/'frame.png'))
        if image.shape!=(1920,1080,3) or not np.isfinite(image).all() or (image.max(2)>0).mean()<.01:raise ValueError('Bad image')
        if screen:
            native=np.asarray(Image.open(root/'prediction_native.png'))
            if not np.array_equal(image,np.rot90(native)):raise ValueError('Unexpected output crop/stabilization')
            yy,xx=np.nonzero(image.max(2)>0);image_centroids.append([float(xx.mean()),float(yy.mean())])
        poses.append(calibration_pose(record['camera'],cal,read(record['metadata']))['transform_matrix'])
        weights.append(record['camera']['convex_weights']);render_hashes[frame]=sha(root/'frame.png');mesh_hashes.append(record['mesh_sha256'])
        review=output/'visual_reviews'/f'{frame}.json'
        if require_reviews:
            value=read(review)
            if value['render_sha256']!=render_hashes[frame] or not value['notes'] or value['status']=='pending':raise ValueError('Invalid actual review')
            for e in value['evidence']:
                if sha(e['path'])!=e['sha256']:raise ValueError('Changed review sheet')
            reviews.append(value)
    if len(set(render_hashes.values()))!=150 or len(set(mesh_hashes))!=150:raise ValueError('Frozen/repeated dynamic render')
    poses=np.asarray(poses);weights=np.asarray(weights)
    if weights.min()<-1e-10 or not np.allclose(weights.sum(1),1) or not np.allclose(poses[:,:3,3],weights@corners,atol=1e-6):
        raise ValueError('Incorrect actual camera translation')
    u=weights[:,1]+weights[:,2];v=weights[:,2]+weights[:,3]
    replay=request['recipe']['camera_path_kind']=='replay_static_4x4'
    xy=np.column_stack((3*u-2,2-3*v)) if replay else np.column_stack((8*u-4,2-4*v))
    if screen:xy=np.column_stack((7*u-4,2-3*v))
    if expanded:
        from expanded_head_camera_flight import RIG_POLYGON
        xy=weights@RIG_POLYGON
    # The periodic turn passes the exact left extremum just before phase zero;
    # phase zero must start within 0.2 interval of the left edge, not at center.
    if screen:
        from screen_travel_camera_flight import screen_path,portrait_projection
        if expanded:
            from expanded_head_camera_flight import expanded_path as screen_path
        if workaround:
            from artifact_aware_camera_flight import avoidance_path as screen_path
        elevated=request['recipe'].get('camera_path_variant')=='elevated'
        if elevated:
            from elevated_camera_workaround import elevated_path as screen_path
        from joint_temporal_texture import cameras
        reference=request['reference_camera_pilot']
        if sha(reference['path'])!=reference['sha256']:raise ValueError('Changed source loop')
        rows,_,meta=cameras('000973');expected,report=screen_path(rows,read(reference['path']))
        raw_expected=np.asarray([calibration_pose(p,cal,read(meta))['transform_matrix'] for p in expected])
        if not np.allclose(poses,raw_expected,atol=1e-8):raise ValueError('Wrong translated composition')
        for record,p in zip(inventory,expected):
            for key in ['fl_x','fl_y','cx','cy','w','h']:
                if record['camera'][key]!=p[key]:raise ValueError('Unexpected animated lens or image dimensions')
        projected=np.array([portrait_projection(np.array(report['fixed_target']),p) for p in expected])
        if (np.ptp(projected,axis=0)<[380,150]).any():raise ValueError('Scene still centered')
        if np.ptp(np.array(image_centroids)[:,0])<200:raise ValueError('Insufficient visible RGB displacement')
        minimum_extent=[4.5,.7] if elevated else ([4.5,2.8] if workaround else ([8.6,3.8] if expanded else [6.7,2.87]))
        if (np.ptp(xy,axis=0)<minimum_extent).any():raise ValueError('Missing requested rig motion')
    elif replay:
        from replay_dynamic_camera_flight import replay_path
        report=request['camera_path_report']
        for field in ['pilot_request','pilot_video','pilot_metadata']:
            if sha(report[field]['path'])!=report[field]['sha256']:raise ValueError('Changed chosen static pilot')
        expected=replay_path(read(report['pilot_request']['path']))
        raw_expected=np.asarray([calibration_pose(p,cal,read(report['pilot_metadata']['path']))['transform_matrix'] for p in expected])
        if not np.allclose(poses,raw_expected,atol=1e-8):raise ValueError('Actual pose is not the selected earlier loop')
        if (np.ptp(xy,axis=0)<2.87).any():raise ValueError('Truncated replay loop')
    else:
        if np.ptp(xy[:,0])<7.9 or np.ptp(xy[:,1])<3.95 or xy[0,0]>-3.8:raise ValueError('Missing wide sweep')
        if not xy[:,0].argmax()<xy[:,1].argmax()<xy[:,1].argmin():raise ValueError('Wrong sweep sequence')
    direction=poses[:,:3,2];angular=np.rad2deg(np.arccos(np.clip(direction@direction.T,-1,1)))
    if angular.max()<(20 if replay or screen else 40):raise ValueError('Insufficient actual viewing-angle change')
    step=np.linalg.norm(np.roll(poses[:,:3,3],-1,axis=0)-poses[:,:3,3],axis=1)
    if screen:
        # Widening one axis changes speed smoothly along the old phase curve;
        # test local continuity, not a false constant-speed assertion.
        ratio=step/np.roll(step,-1)
        if max(ratio.max(),1/ratio.min())>1.07:raise ValueError('Abrupt camera speed change')
    elif step.max()/step.min()>1.04:raise ValueError('Camera loop join/speed discontinuity')
    # Independent fresh casts verify the requested camera affected actual saved
    # depth, rather than accepting a worker copying pose metadata to its result.
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from bake_joint_temporal_mesh import camera_depth
    raychecks=[]
    for index in sorted({0,149,*request['camera_path_report']['extrema_indices'].values()}):
        record=inventory[index];mesh=o3d.io.read_triangle_mesh(record['mesh'])
        scene=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
        fresh=camera_depth(scene,record['camera'])[0]
        saved=np.load(output/'frames'/record['frame_id']/'target_depth.npz')['depth']
        if not np.allclose(np.where(np.isfinite(fresh),fresh,0),saved,atol=1e-6):raise ValueError('Saved raycast uses wrong camera')
        raychecks.append(dict(index=index,frame_id=record['frame_id'],fresh_raycast_matches=True))
    result=dict(status='integrity_and_actual_camera_motion_pass',unique_source_times=150,unique_meshes=150,unique_renders=150,
        render_hashes=render_hashes,request_sha256=digest,actual_grid_extent_xy=np.ptp(xy,axis=0).tolist(),
        maximum_pairwise_view_angle_degrees=float(angular.max()),camera_loop_speed_ratio=float(step.max()/step.min()),
        independent_saved_depth_checks=raychecks,visual_reviews=len(reviews),artifact_free_claimed=False,
        full_frame_quality_metrics_computed=False,script_sha256=sha(__file__))
    if screen:
        result.update(projected_scene_landmark_extent_pixels=np.ptp(projected,axis=0).tolist(),
            actual_render_foreground_centroid_extent_pixels=np.ptp(image_centroids,axis=0).tolist(),
            all_rgb_verified_as_native_rotation_only=True)
    atomic_json(output/'integrity_audit.json',result);return result


def sheets(output):
    request=verify_request(output)
    for first in range(0,150,10):
        records=request['inventory'][first:first+10]
        if not all((output/'frames'/r['frame_id']/'complete.json').exists() for r in records):continue
        root=output/'contact_sheets'/f'{first:03d}_{first+9:03d}';root.mkdir(parents=True,exist_ok=True)
        panel=Image.new('RGB',(1350,1020));draw=ImageDraw.Draw(panel);hashes={}
        for j,record in enumerate(records):
            path=output/'frames'/record['frame_id']/'frame.png';hashes[record['frame_id']]=sha(path)
            x,y=j%5*270,j//5*510;panel.paste(Image.open(path).resize((270,480)),(x,y+30))
            xy=record['camera']['rig_offset_xy'];draw.text((x+3,y+3),f'{record["frame_id"]} x={xy[0]:+.2f} y={xy[1]:+.2f}',fill='white')
        panel.save(root/'overview.png');atomic_json(root/'manifest.json',dict(render_hashes=hashes))
    replay=request['recipe']['camera_path_kind']=='replay_static_4x4'
    loop=replay or request['recipe']['camera_path_kind'] in {'screen_travel','expanded_head','artifact_avoidance'}
    chosen=[0,37,75,112] if loop else [0,49,78,118]
    if all((output/'frames'/request['inventory'][i]['frame_id']/'complete.json').exists() for i in chosen):
        panel=Image.new('RGB',(2160,984));draw=ImageDraw.Draw(panel)
        for j,i in enumerate(chosen):
            rec=request['inventory'][i];panel.paste(Image.open(output/'frames'/rec['frame_id']/'frame.png').resize((540,960)),(j*540,24))
            labels=['phase 0','phase 1/4','phase 1/2','phase 3/4'] if loop else ['LEFT','RIGHT','TOP','BOTTOM']
            draw.text((j*540+3,3),f'{labels[j]} real-time {rec["frame_id"]}',fill='white')
        panel.save(output/'dynamic_extremes.png')
        if replay:
            pilot=Path(request['camera_path_report']['pilot_request']['path']).parent
            comparison=Image.new('RGB',(1080,1036));draw=ImageDraw.Draw(comparison);evidence=[]
            for j,i in enumerate(chosen):
                rec=request['inventory'][i];phase=rec['camera']['pilot_sample_phase']
                old=pilot/'frames'/f'{round(phase):05d}.png';new=output/'frames'/rec['frame_id']/'frame.png'
                for y,p in [(28,old),(544,new)]:
                    comparison.paste(Image.open(p).resize((270,480)),(j*270,y))
                draw.text((j*270+3,5),f'OLD static / pose {round(phase)}',fill='white')
                draw.text((j*270+3,520),f'NEW moving / {rec["frame_id"]}',fill='white')
                evidence.append(dict(dynamic_frame=rec['frame_id'],pilot_fractional_pose=phase,
                    nearest_saved_pilot_pose=round(phase),old_sha256=sha(old),new_sha256=sha(new)))
            comparison.save(output/'reference_comparison.png')
            atomic_json(output/'reference_comparison.json',dict(evidence=evidence,
                note='Static pilot top, dynamic actor bottom. Reference PNG uses nearest saved pilot phase, within 0.5/720 of loop.'))


def record_review(output,group,notes,status):
    if not notes or status not in {'reviewed_known_artifacts','fail'}:raise ValueError('Explicit findings required')
    root=output/'contact_sheets'/group;manifest=read(root/'manifest.json')
    evidence=[dict(path=str(root/'overview.png'),sha256=sha(root/'overview.png'))]
    for frame,digest in manifest['render_hashes'].items():
        if sha(output/'frames'/frame/'frame.png')!=digest:raise ValueError('Stale review')
        atomic_json(output/'visual_reviews'/f'{frame}.json',dict(frame_id=frame,render_sha256=digest,
            status=status,notes=notes,evidence=evidence,review_scope='Camera/actor motion and gross artifacts at overview resolution; native extrema reviewed separately',
            artifact_free=False,reviewer='LLM_actual_image_inspection'))


def record_native_review(output,findings):
    if len(findings)<4 or any(len(n.strip())<30 for n in findings.values()):
        raise ValueError('Four explicit native image observations required')
    records=[]
    for frame,notes in findings.items():
        root=output/'frames'/frame;receipt=read(root/'complete.json');verify_hashes(root,receipt['hashes'])
        records.append(dict(frame_id=frame,path=str(root/'frame.png'),sha256=sha(root/'frame.png'),notes=notes))
    atomic_json(output/'native_visual_review.json',dict(reviewer='LLM_actual_native_image_inspection',
        artifact_free=False,scope='Four distributed camera phases; not exhaustive native inspection of all 150 frames',records=records))


def encode(output):
    checked=audit(output);request=read(output/'request.json');sequence=output/'encode_sequence';sequence.mkdir(exist_ok=True)
    for i,frame in enumerate(request['ordered_frame_ids']):
        source=output/'frames'/frame/'frame.png';dest=sequence/f'{i:05d}.png'
        if dest.exists():
            if sha(dest)!=sha(source):raise ValueError('Changed encode sequence')
        else:os.link(source,dest)
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0');videos={}
    normal_only=request['recipe']['camera_path_kind'] in {'expanded_head','artifact_avoidance'}
    versions=[('video.mp4',24)] if normal_only else [('video.mp4',24),('video_slow_12fps.mp4',12)]
    for name,fps in versions:
        temp=output/f'{name}.partial.mp4'
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(fps),'-i',str(sequence/'%05d.png'),
            '-frames:v','150','-c:v','libx264','-crf','16','-preset','slow','-threads','8','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True,env=env)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
            'stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(temp)],text=True,env=env))['streams'][0]
        if int(probe['nb_read_frames'])!=150 or abs(float(probe['duration'])-150/fps)>.001:raise ValueError('Wrong movie inventory')
        os.replace(temp,output/name);videos[name]=dict(sha256=sha(output/name),probe=probe)
    decoded=output/'decoded';decoded.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output/'video.mp4'),'-vf','scale=270:480',str(decoded/'%03d.png')],check=True,env=env)
    for first in range(0,150,10):
        panel=Image.new('RGB',(1350,1008));draw=ImageDraw.Draw(panel)
        for j,index in enumerate(range(first,first+10)):
            x,y=j%5*270,j//5*504;panel.paste(Image.open(decoded/f'{index+1:03d}.png'),(x,y+24))
            draw.text((x+3,y+3),f'{request["ordered_frame_ids"][index]} / {index/24:.2f}s',fill='white')
        panel.save(decoded/f'sheet_{first:03d}.png')
    atomic_json(output/'video_manifest.json',dict(videos=videos,render_hashes=checked['render_hashes'],
        request_sha256=sha(output/'request.json'),unique_source_times=150,source_time_repetition=False,
        slow_version='Not requested; not created' if normal_only else 'Same 150 unique frames at 12fps: half-speed actor and camera, lower cadence; no optical-flow interpolation',
        camera_periodic=True,actor_clip_periodic=False,encoded_visual_status='pending'))


def record_encoded_review(output,notes):
    if not notes or len(notes.strip())<30:raise ValueError('Explicit actual MP4 findings required')
    video=read(output/'video_manifest.json')
    for name,spec in video['videos'].items():
        if sha(output/name)!=spec['sha256']:raise ValueError('Changed encoded video')
    evidence=[dict(path=str(output/'decoded'/f'sheet_{i:03d}.png'),
                   sha256=sha(output/'decoded'/f'sheet_{i:03d}.png')) for i in range(0,150,10)]
    video.update(encoded_visual_status='reviewed_with_known_failures',reviewer='LLM_actual_decoded_image_inspection',
                 review_evidence=evidence,review_notes=notes)
    atomic_json(output/'video_manifest.json',video)


def publish(output):
    checked=audit(output,True);video=read(output/'video_manifest.json')
    native=read(output/'native_visual_review.json')
    if len(native['records'])<4:raise ValueError('Native distributed-phase review required')
    for evidence in native['records']:
        if sha(evidence['path'])!=evidence['sha256']:raise ValueError('Changed native review')
    if video['encoded_visual_status']!='reviewed_with_known_failures':raise ValueError('Encoded review required')
    if len(video.get('review_evidence',[]))!=15:raise ValueError('Review all actual decoded frame sheets')
    for evidence in video['review_evidence']:
        if sha(evidence['path'])!=evidence['sha256']:raise ValueError('Changed encoded review')
    for name,spec in video['videos'].items():
        if sha(output/name)!=spec['sha256']:raise ValueError('Changed video')
    path=output/'frames.zip';temporary=output/'frames.partial.zip'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as z:
        for frame,digest in checked['render_hashes'].items():z.write(output/'frames'/frame/'frame.png',f'frames/{frame}.png')
    import hashlib
    with zipfile.ZipFile(temporary) as z:
        if len(z.namelist())!=150:raise ValueError('Wrong archive inventory')
        for frame,digest in checked['render_hashes'].items():
            if hashlib.sha256(z.read(f'frames/{frame}.png')).hexdigest()!=digest:raise ValueError('Archive mismatch')
    os.replace(temporary,path)
    scripts=Path(__file__).resolve().parent;snapshot=output/'script_snapshot';snapshot.mkdir(exist_ok=True)
    request=verify_request(output)
    for name,digest in request['script_hashes'].items():
        if Path(name).name!=name or sha(scripts/name)!=digest:raise ValueError('Invalid frozen script snapshot')
        shutil.copyfile(scripts/name,snapshot/name)
    shutil.copyfile(__file__,snapshot/Path(__file__).name)
    report='dec5_replayed_4x4_dynamic.md' if request['recipe']['camera_path_kind']=='replay_static_4x4' else 'dec5_wide_dynamic_camera_flight.md'
    if request['recipe']['camera_path_kind']=='screen_travel':report='dec5_screen_travel_camera_flight.md'
    if request['recipe']['camera_path_kind'] in {'expanded_head','artifact_avoidance'}:report='dec5_expanded_head_camera_flight.md'
    if request['recipe']['camera_path_kind'] in {'expanded_head','artifact_avoidance'}:
        geometry=read(output/'head_geometry_audit.json')
        if (geometry['status']!='preservation_and_locality_pass' or geometry['frames']!=150
                or geometry['request_sha256']!=sha(output/'request.json')
                or geometry['script_sha256']!=sha(scripts/'audit_head_completion_geometry.py')):
            raise ValueError('Independent head preservation/locality audit required')
        shutil.copyfile(scripts/'audit_head_completion_geometry.py',snapshot/'audit_head_completion_geometry.py')
        shutil.copyfile(scripts/'diagnose_camera_patch_provenance.py',snapshot/'diagnose_camera_patch_provenance.py')
    if request['recipe']['camera_path_kind']=='screen_travel':
        diagnostic=read(output/'framing_diagnosis/result.json')
        if diagnostic['visual_status']!='reviewed_screen_displacement_confirmed':raise ValueError('Review identical-mesh framing diagnostic')
        if sha(scripts/'diagnose_screen_travel.py')!=diagnostic['script_sha256']:raise ValueError('Changed diagnostic script')
        shutil.copyfile(scripts/'diagnose_screen_travel.py',snapshot/'diagnose_screen_travel.py')
    shutil.copyfile(scripts.parent/'experiments'/report,output/'report.md')
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()
           and p.name!='publication.json' and '.partial.' not in p.name and p.name!='supervisor.lock'}
    atomic_json(output/'publication.json',dict(status='camera_and_actor_dynamic_artifacts_remain',hashes=files,
        source_time_count=150,artifact_free=False,repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print(f'published={output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','sheets','review','encode','encoded-review','diagram','publish'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--require-reviews',action='store_true')
    p.add_argument('--group');p.add_argument('--notes');p.add_argument('--status',default='reviewed_known_artifacts');a=p.parse_args()
    if a.action=='audit':print(audit(a.output,a.require_reviews))
    elif a.action=='sheets':sheets(a.output)
    elif a.action=='review':record_review(a.output,a.group,a.notes,a.status)
    elif a.action=='encode':encode(a.output)
    elif a.action=='encoded-review':record_encoded_review(a.output,a.notes)
    elif a.action=='diagram':diagram(a.output)
    else:publish(a.output)
