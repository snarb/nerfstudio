"""Audit, review, and encode 150 changing source times at the requested FPS.

Separate integrity from visual quality: a hash-valid video can contain diagnosed
artifacts. No novel-view full-frame or candidate-mask image metrics are computed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, CALIBRATION, SOURCE, HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request, calibration_pose
from finalize_local_mesh_repair import verify_hashes
from central_space_temporal_flythrough import motion_report


def audit(output, require_reviews=False):
    request=verify_request(output);digest=sha(output/'request.json');ids=request['ordered_frame_ids']
    inventory=request['inventory'];cal=read(CALIBRATION);by_name={r['physical_camera']:r for r in cal['frames']}
    if len(set(ids))!=150 or ids!=sorted(ids) or [r['frame_id'] for r in inventory]!=ids:
        raise ValueError('Not 150 different chronological source times')
    if set(p.parent.name for p in (output/'frames').glob('*/complete.json'))!=set(ids):
        raise ValueError('Incomplete or extra published times')
    anchors=request['camera_path_report']['anchors']
    if set(anchors)&HELD_CAMERAS:raise ValueError('Held-out camera anchor')
    corners=np.array([by_name[n]['transform_matrix'] for n in anchors])[:,:3,3]
    poses=[];weights=[];hashes={};mesh_hashes=[];reviews=[]
    for record in inventory:
        frame=record['frame_id'];root=output/'frames'/frame
        receipt=read(root/'complete.json')
        if receipt['request_sha256']!=digest:raise ValueError('Frame request mismatch')
        verify_hashes(root,receipt['hashes']);result=read(root/'result.json')
        if result['frame_id']!=frame or result['index']!=record['index'] or result['camera']!=record['camera']:
            raise ValueError('Source-time or target-camera substitution')
        if result['target_rgb_read'] or result['rgb_averaging']:raise ValueError('Unexpected RGB recipe')
        if len(set(result['source_cameras']))!=62 or set(result['source_cameras'])&HELD_CAMERAS:raise ValueError('Invalid source cameras')
        if sha(SOURCE/frame/'transforms.json')!=record['source_transforms_sha256']:raise ValueError('Source inventory changed')
        if sha(record['mesh'])!=record['mesh_sha256'] or sha(record['metadata'])!=record['metadata_sha256']:raise ValueError('Geometry input changed')
        if request['recipe'].get('train_foreground_guard'):
            guard=output/'foreground_guard'/frame;gr=read(guard/'complete.json')
            verify_hashes(guard,gr['hashes'])
            if gr['request']['parent_request_sha256']!=digest or gr['request']['mesh_sha256']!=record['mesh_sha256']:
                raise ValueError('Guard parent mismatch')
            expected_mesh=sha(guard/'mesh.ply')
        else:expected_mesh=record['mesh_sha256']
        if result['mesh_sha256']!=expected_mesh or sha(result['mesh_path'])!=expected_mesh:raise ValueError('Rendered geometry mismatch')
        im=np.array(Image.open(root/'frame.png'))
        if im.shape!=(1920,1080,3) or not np.isfinite(im).all() or not (im.max(2)>0).mean()>.01:raise ValueError('Invalid image')
        raw=calibration_pose(record['camera'],cal,read(record['metadata']));poses.append(raw['transform_matrix'])
        weights.append(record['camera']['convex_weights']);hashes[frame]=sha(root/'frame.png');mesh_hashes.append(expected_mesh)
        review_path=output/'visual_reviews'/f'{frame}.json'
        if review_path.exists():
            review=read(review_path)
            if review['render_sha256']!=hashes[frame] or review['frame_id']!=frame:raise ValueError('Stale visual review')
            if require_reviews and review['status'] not in {'pass','accepted_known_artifacts','fail'}:
                raise ValueError('Unresolved visual review')
            if not review.get('notes') or not review.get('evidence'):
                raise ValueError('Review lacks findings or evidence')
            for evidence in review['evidence']:
                if sha(evidence['path'])!=evidence['sha256']:raise ValueError('Visual evidence changed')
            reviews.append(review)
        elif require_reviews:raise ValueError('Missing actual visual review')
    if len(set(mesh_hashes))!=150 or len(set(hashes.values()))!=150:raise ValueError('Frozen/repeated source geometry or render')
    weights=np.asarray(weights);positions=np.asarray(poses)[:,:3,3]
    if weights.min()<0 or not np.allclose(weights.sum(1),1) or not np.allclose(positions,weights@corners,atol=5e-7):
        raise ValueError('Camera containment or normalization error')
    uv=np.column_stack((weights[:,1]+weights[:,2],weights[:,2]+weights[:,3]))
    extent=np.ptp(uv,axis=0)*(request['recipe']['grid_size']-1)
    if (extent<.95*(request['recipe']['grid_size']-1)).any():raise ValueError('Static/contracted camera path')
    motion=motion_report(poses,request['recipe']['fps'])
    if motion['speed_max_min_ratio']>1.002 or motion['angular_speed_degrees_per_second_min_median_max'][-1]>6:
        raise ValueError('Camera speed gate failed')
    result=dict(status='integrity_pass',unique_source_times=150,unique_meshes=150,unique_renders=150,
        frame_interpolation=False,temporal_frame_repetition=False,request_sha256=digest,render_hashes=hashes,
        grid_extent_xy=extent.tolist(),camera_motion=motion,visual_reviews=len(reviews),
        visual_status_counts={s:sum(r['status']==s for r in reviews) for s in sorted({r['status'] for r in reviews})},
        artifact_free_claimed=False,full_frame_metrics_computed=False,script_sha256=sha(__file__))
    atomic_json(output/'integrity_audit.json',result);return result


def encode(output):
    checked=audit(output);request=read(output/'request.json');fps=request['recipe']['fps']
    sequence=output/'video_frames';sequence.mkdir(exist_ok=True)
    for i,frame in enumerate(request['ordered_frame_ids']):
        src=output/'frames'/frame/'frame.png';dst=sequence/f'{i:05d}_{frame}.png'
        if dst.exists():
            if sha(dst)!=sha(src):raise ValueError('Video frame collision')
        else:os.link(src,dst)
    # Explicit ordered concat avoids relying on filesystem/glob frame ordering.
    concat=output/'video_inputs.txt'
    lines=[]
    for i,frame in enumerate(request['ordered_frame_ids']):
        path=(sequence/f'{i:05d}_{frame}.png').resolve()
        if "'" in str(path):raise ValueError('Unsafe concat filename')
        lines.extend([f"file '{path}'",f'duration {1/fps:.12f}'])
    # concat demuxer timestamps are quantized at input image rate; use image2
    # hard links for exact requested CFR instead of these audit-only timestamps.
    concat.write_text('\n'.join(lines)+'\n')
    cfr=output/'encode_sequence';cfr.mkdir(exist_ok=True)
    for i,frame in enumerate(request['ordered_frame_ids']):
        dst=cfr/f'{i:05d}.png';src=output/'frames'/frame/'frame.png'
        if dst.exists():
            if sha(dst)!=sha(src):raise ValueError('CFR sequence mismatch')
        else:os.link(src,dst)
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    temp=output/'video.partial.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(fps),'-i',str(cfr/'%05d.png'),
        '-frames:v','150','-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-threads','8','-movflags','+faststart',str(temp)],env=env,check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(temp)],env=env,text=True))['streams'][0]
    if int(probe['nb_read_frames'])!=150 or abs(float(probe['duration'])-150/fps)>.001 or (probe['width'],probe['height'])!=(1080,1920):
        raise ValueError('Incorrect movie inventory / duration')
    os.replace(temp,output/'video.mp4')
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output/'video.mp4'),'-vf',
        'select=not(mod(n\\,10)),scale=270:480','-vsync','vfr',str(output/'decoded_%03d.png')],env=env,check=True)
    panel=Image.new('RGB',(1350,1512));draw=ImageDraw.Draw(panel)
    for j in range(15):
        x,y=j%5*270,j//5*504;panel.paste(Image.open(output/f'decoded_{j+1:03d}.png'),(x,y+24))
        draw.text((x+3,y+3),f'{request["ordered_frame_ids"][j*10]} / {j*10/fps:.2f}s',fill='white')
    panel.save(output/'encoded_overview.png')
    atomic_json(output/'video_manifest.json',dict(video_sha256=sha(output/'video.mp4'),ffprobe=probe,
        request_sha256=sha(output/'request.json'),source_frame_ids=request['ordered_frame_ids'],render_hashes=checked['render_hashes'],
        unique_source_times=150,review_status='pending_actual_encoded_review',encoded_overview_sha256=sha(output/'encoded_overview.png')))
    print(f'encoded={output/"video.mp4"} distinct_times=150 fps={fps}',flush=True)


def camera_diagram(output):
    request=verify_request(output);cal=read(CALIBRATION)
    rows=[r for r in cal['frames'] if r['physical_camera'] not in HELD_CAMERAS]
    reference=next(r for r in rows if r['physical_camera']=='H004_C005_1210SZ')
    ref=np.array(reference['transform_matrix']);basis=ref[:3,[1,0]];origin=ref[:3,3]
    centers=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
    poses=[calibration_pose(r['camera'],cal,read(r['metadata']))['transform_matrix'] for r in request['inventory']]
    xy=(centers-origin)@basis;flight=(np.asarray(poses)[:,:3,3]-origin)@basis
    lo=xy.min(0);span=np.ptp(xy,axis=0);scale=min(880/span[0],660/span[1])
    def pixel(q):return tuple(np.rint([60+(q[0]-lo[0])*scale,740-(q[1]-lo[1])*scale]).astype(int))
    image=Image.new('RGB',(1040,800),(18,18,22));draw=ImageDraw.Draw(image)
    draw.text((20,20),'150 actual source times + open cubic 4x4 camera traversal; no repeated actor frame',fill='white')
    anchors=request['camera_path_report']['anchors'];lookup={r['physical_camera']:q for r,q in zip(rows,xy)}
    draw.line([pixel(lookup[n]) for n in anchors+[anchors[0]]],fill=(70,150,180),width=2)
    for row,q in zip(rows,xy):
        x,y=pixel(q);draw.ellipse((x-3,y-3,x+3,y+3),fill=(130,130,130))
        if row['physical_camera'][0] in 'FGHI':draw.text((x+4,y+4),row['physical_camera'][:9],fill=(180,180,180))
    draw.line([pixel(q) for q in flight],fill=(255,140,30),width=3)
    for q,label,color in [(flight[0],'000899 start','lime'),(flight[-1],'001197 end','red')]:
        x,y=pixel(q);draw.ellipse((x-6,y-6,x+6,y+6),fill=color);draw.text((x+8,y-16),label,fill=color)
    draw.text((20,775),'Actual calibration camera centers projected onto reference image-plane axes; gray = train cameras',fill='white')
    image.save(output/'camera_path.png')


def encoded_crops(output):
    """Decode consecutive native-resolution patches, not pre-encode PNGs."""
    request=read(output/'request.json');video=read(output/'video_manifest.json')
    if sha(output/'video.mp4')!=video['video_sha256']:raise ValueError('Changed movie')
    root=output/'encoded_review';root.mkdir(exist_ok=True)
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    evidence=[]
    for first in [24,40,52,146]:
        panel=Image.new('RGB',(900,948));draw=ImageDraw.Draw(panel)
        for j,index in enumerate(range(first,first+4)):
            path=root/f'{index:03d}.png'
            subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output/'video.mp4'),
                '-vf',f'select=eq(n\\,{index}),crop=450:450:280:970','-frames:v','1',str(path)],env=env,check=True)
            x,y=j%2*450,j//2*474;panel.paste(Image.open(path),(x,y+24))
            draw.text((x+3,y+3),f'{request["ordered_frame_ids"][index]} / encoded frame {index}',fill='white')
        path=root/f'{first:03d}_{first+3:03d}.png';panel.save(path)
        evidence.append(dict(path=str(path),sha256=sha(path)))
    atomic_json(root/'manifest.json',dict(video_sha256=video['video_sha256'],
        native_crop_xyxy=[280,970,730,1420],evidence=evidence,status='pending_actual_visual_review'))


def verify_video_review(output):
    video=read(output/'video_manifest.json')
    if sha(output/'video.mp4')!=video['video_sha256']:raise ValueError('Changed movie')
    if video['review_status'] not in {'reviewed_known_artifacts','reviewed_with_failures','reviewed_pass'}:
        raise ValueError('Actual encoded video review is still required')
    if not video.get('review_notes'):raise ValueError('Missing actual encoded findings')
    if sha(output/'encoded_overview.png')!=video['encoded_overview_sha256']:raise ValueError('Changed encoded overview')
    review=read(output/'encoded_review/manifest.json')
    if review['video_sha256']!=video['video_sha256'] or review['status']!=video['review_status']:
        raise ValueError('Encoded crop review mismatch')
    if not review.get('notes') or len(review.get('evidence',[]))!=4:raise ValueError('Incomplete decoded crop review')
    for evidence in review['evidence']:
        if sha(evidence['path'])!=evidence['sha256']:raise ValueError('Changed decoded crop evidence')
    return video


def copy_tree_contents(source,destination):
    """Copy bytes only: the shared mount rejects directory metadata changes."""
    destination.mkdir(parents=True,exist_ok=False)
    for path in sorted(source.rglob('*')):
        target=destination/path.relative_to(source)
        if path.is_dir():target.mkdir(exist_ok=True)
        elif path.is_file():shutil.copyfile(path,target)
        else:raise ValueError(f'Unsupported publication entry: {path}')


def publish(output,destination):
    checked=audit(output,True);video=verify_video_review(output)
    if destination.exists():
        receipt=read(destination/'publication.json')
        if receipt['source_audit_sha256']!=sha(output/'integrity_audit.json'):raise ValueError('Immutable publication mismatch')
        verify_hashes(destination,receipt['hashes']);return
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=destination.name+'_publish_',dir=destination.parent))
    request=read(output/'request.json');frames=stage/'frames';frames.mkdir()
    for frame in request['ordered_frame_ids']:
        source=output/'frames'/frame/'frame.png';shutil.copyfile(source,frames/f'{frame}.png')
        if sha(frames/f'{frame}.png')!=checked['render_hashes'][frame]:raise ValueError('Published frame mismatch')
    for name in ['video.mp4','video_manifest.json','request.json','integrity_audit.json','camera_path.png','encoded_overview.png','checks.jsonl','progress.json']:
        shutil.copyfile(output/name,stage/name)
    for name in ['contact_sheets','visual_reviews','pixel_traces','encoded_review']:
        if (output/name).exists():copy_tree_contents(output/name,stage/name)
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_dynamic_grid_background_guard.md'
    shutil.copyfile(report,stage/'report.md')
    with zipfile.ZipFile(stage/'frames.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for frame in request['ordered_frame_ids']:archive.write(frames/f'{frame}.png',f'frames/{frame}.png')
    with zipfile.ZipFile(stage/'frames.zip') as archive:
        import hashlib
        if len(archive.namelist())!=150:raise ValueError('Frame archive count mismatch')
        for frame in request['ordered_frame_ids']:
            if hashlib.sha256(archive.read(f'frames/{frame}.png')).hexdigest()!=checked['render_hashes'][frame]:raise ValueError('Frame archive checksum mismatch')
    files={str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    atomic_json(stage/'publication.json',dict(source_root=str(output),source_audit_sha256=sha(output/'integrity_audit.json'),
        repository_commit=head,video_sha256=video['video_sha256'],hashes=files,unique_source_times=150,
        visual_status_counts=checked['visual_status_counts'],artifact_free_claimed=False,
        unresolved_lipstick_artifact=True,geometry_retained_in_source_root=True))
    verify_hashes(stage,files);os.replace(stage,destination)
    print(f'published={destination} actual_source_times=150 unresolved_object_defect=True',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','encode','diagram','encoded-crops','publish']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=Path('/mnt/data/dec5_dynamic_grid_150'))
    p.add_argument('--require-reviews',action='store_true');a=p.parse_args()
    if a.action=='encode':encode(a.output)
    elif a.action=='diagram':camera_diagram(a.output)
    elif a.action=='encoded-crops':encoded_crops(a.output)
    elif a.action=='publish':publish(a.output,a.destination)
    else:print(audit(a.output,a.require_reviews))
