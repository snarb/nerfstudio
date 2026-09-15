"""Audit, native-pixel review and packaging for true 3456x6144 output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
from render_cinematic_6k_output import BASE, OUT, SCALE, WIDTH, HEIGHT, read, write, sha, scale_camera

OLD=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_v2')


def audit():
    from joint_temporal_texture import cameras, HELD_CAMERAS
    q=read(BASE/'request.json'); config=read(OUT/'request.json')
    assert sha(BASE/'request.json')==config['parent_request_sha256']
    assert sha(Path(__file__).with_name('render_cinematic_6k_output.py'))==config['script_sha256']
    canonical=read(BASE/'frames/000899/result.json')['source_cameras']; records=[]
    assert len(canonical)==62 and not set(canonical)&HELD_CAMERAS
    for original,new in zip(q['inventory'],config['inventory']):
        frame=original['frame_id'];rows,_,_=cameras(frame)
        names=[r['physical_camera'] for r in rows];assert names==canonical
        baseline=BASE/'frames'/frame/'result.json'
        if original['index']<126:assert names==read(baseline)['source_cameras']
        expected=scale_camera(original['camera'],WIDTH,HEIGHT)
        assert new['camera']==expected and new['frame_id']==frame
        assert expected['transform_matrix']==original['camera']['transform_matrix']
        bounds=None
        if original['index']>=118:
            source=scale_camera(next(r for r in rows if r['physical_camera']==q['camera_path_report']['endpoint_train_camera']),5461,3072)
            u=(np.array([0,WIDTH-1])+.5-expected['cx'])/expected['fl_x']*source['fl_x']+source['cx']-.5
            v=(np.array([0,HEIGHT-1])+.5-expected['cy'])/expected['fl_y']*source['fl_y']+source['cy']-.5
            assert np.all((u>=0)&(u<=5460)) and np.all((v>=0)&(v<=3071))
            bounds=[u.tolist(),v.tolist()]
        records.append(dict(frame=frame,camera_order_matches=True,target_pose_exact=True,
            baseline_result_sha256=sha(baseline) if baseline.exists() else None,train_uv_bounds=bounds))
    # Strict original HD-equivalent visibility bounds imply interior native taps.
    native_lower=(np.array([2,2])+.5)*SCALE-.5
    native_upper=(np.array([1917,1077])+.5)*SCALE-.5
    assert np.all(native_lower>0) and np.all(native_upper<[5460,3071])
    write(OUT/'preflight.json',dict(status='pass',request_sha256=sha(OUT/'request.json'),
        audit_script_sha256=sha(__file__),source_camera_order=canonical,records=records,
        source_visibility_native_uv_limits=[native_lower.tolist(),native_upper.tolist()],
        all_train_sampling_in_bounds=True,target_intrinsics_scale=[3.2,3.2],
        intrinsics_convention='rays use pixel + 0.5; scale cx directly; source UV uses half-pixel transform'))
    print('preflight pass: 150 camera orders/poses, 32 ending bounds',flush=True)


def review(frames):
    dest=OUT/'review';dest.mkdir(exist_ok=True)
    records=[]
    for frame in frames:
        boxes=({'hair':(1580,800,2092,1312),'ear':(900,2160,1412,2672),'lips':(1910,3210,2422,3722)}
            if frame=='001083' else {'hair':(670,440,1182,952),'ear':(860,1660,1372,2172),'lips':(2440,1920,2952,2432)})
        if frame=='001123':
            boxes.update(ear=(900,1280,1412,1792),lips=(2420,2480,2932,2992),nose_seam=(2440,1920,2952,2432))
        native=Image.open(OUT/'frames'/frame/'frame.png').convert('RGB');assert native.size==(3456,6144)
        hd=Image.open(OLD/'frames'/frame/'frame.png').convert('RGB');assert hd.size==(1080,1920)
        control=hd.resize(native.size,Image.Resampling.BICUBIC)
        for name,box in boxes.items():
            canvas=Image.new('RGB',(1024,540),(25,25,25));draw=ImageDraw.Draw(canvas)
            for i,im in enumerate([control,native]):
                canvas.paste(im.crop(box),(512*i,28));draw.text((512*i+8,8),['HD output enlarged (control only)','True 6K output: native 1:1'][i],fill='white')
            path=dest/f'{frame}_{name}_AB.png';canvas.save(path)
            difference=np.asarray(native.crop(box),float)-np.asarray(control.crop(box),float)
            records.append(dict(frame=frame,box=list(box),region=name,path=str(path),sha256=sha(path),
                mean_abs_difference=float(np.abs(difference).mean()),diagnostic_control_only=True))
        canvas=Image.new('RGB',(690,640),(25,25,25));draw=ImageDraw.Draw(canvas)
        for i,im in enumerate([hd,native]):
            im=im.copy();im.thumbnail((345,614));canvas.paste(im,(345*i,26));draw.text((345*i+5,7),['Prior HD output','True 6K output'][i],fill='white')
        canvas.save(dest/f'{frame}_overview_AB.png')
    write(dest/'canary_crops.json',dict(records=records,output_pixels_upsampled=False,
        native_pixels_shown_one_to_one=True,control_only_bicubic_upscale=True,visual_status='pending'))


def temporal(frames,box_override=None,region='hair'):
    """Diagnostic fixed native pixel patches; no resampling of the new output."""
    assert all(int(b)-int(a)==2 for a,b in zip(frames,frames[1:]))
    dest=OUT/'review';dest.mkdir(exist_ok=True)
    with Image.open(OUT/'frames'/frames[0]/'frame.png') as im:first=np.asarray(im)
    yy,xx=np.where(first.max(2)>12)
    x=int(np.clip((xx.min()+xx.max())/2-192,0,3456-384))
    y=int(np.clip(yy.min()+128,0,6144-384));box=(x,y,x+384,y+384)
    if box_override is not None:box=tuple(box_override)
    assert box[2]-box[0]==384 and box[3]-box[1]==384
    canvas=Image.new('RGB',(384*len(frames),824),(25,25,25));draw=ImageDraw.Draw(canvas)
    records=[]
    for i,frame in enumerate(frames):
        for j,root in enumerate([OLD,OUT]):
            with Image.open(root/'frames'/frame/'frame.png') as im:
                if j==0:im=im.resize((3456,6144),Image.Resampling.BICUBIC)
                canvas.paste(im.crop(box),(384*i,28+412*j))
            draw.text((384*i+4,6+412*j),frame+' '+['HD enlarged control','Native 6K 1:1'][j],fill='white')
        records.append(dict(frame=frame,image_sha256=sha(OUT/'frames'/frame/'frame.png')))
    suffix='' if region=='hair' else '_'+region
    path=dest/f'temporal_{frames[0]}_{frames[-1]}{suffix}.png';canvas.save(path)
    write(path.with_suffix('.json'),dict(box=list(box),records=records,sha256=sha(path),visual_status='pending'))
    print(path,flush=True)


def status():
    progress=read(OUT/'progress.json');pid=progress.get('pid')
    def check(args):
        r=subprocess.run(args,text=True,capture_output=True);return dict(returncode=r.returncode,output=(r.stdout+r.stderr).strip())
    log=OUT/'render_full.log';done=list((OUT/'frames').glob('*/complete.json'))
    record=dict(utc=datetime.now(timezone.utc).isoformat(),progress=progress,complete=len(done),
        controller=check(['ps','-p',str(pid),'-o','pid,etime,rss,stat']) if pid else None,
        remote_worker=check(['ssh','ubuntu@dev3',"pgrep -af '[p]ython /fsx/tmp/lookcloser_cinematic_6k_output_v1/render_cinematic_6k_output.py remote' || true"]),
        disk=check(['df','-h',str(OUT)]),
        gpu=check(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader']),
        log_tail=check(['tail','-n','3',str(log)]) if log.exists() else None,
        oom_evidence=('out of memory' in log.read_text().lower()) if log.exists() else False)
    parallel=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_parallel_endings_v1')
    if (parallel/'progress.json').exists():
        pp=read(parallel/'progress.json');ppid=pp.get('pid')
        parallel_check=dict(utc=record['utc'],progress=pp,controller=check(['ps','-p',str(ppid),'-o','pid,etime,rss,stat']),
            gpu=record['gpu'],disk=record['disk'],remote_workers=record['remote_worker'])
        record['parallel_ending_worker']=parallel_check
        with (parallel/'checks.jsonl').open('a') as f:f.write(json.dumps(parallel_check)+'\n')
    history=read(OUT/'supervision.json') if (OUT/'supervision.json').exists() else []
    history.append(record);write(OUT/'supervision.json',history)
    with (OUT/'checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def package():
    provenance=OUT/'provenance';provenance.mkdir(exist_ok=True)
    frozen=provenance/'package_started.py';shutil.copyfile(__file__,frozen)
    package_hash=sha(frozen)
    write(OUT/'packaging_request.json',dict(packager_sha256=package_hash,request_sha256=sha(OUT/'request.json'),
        concurrent_encodes=2,encoder_threads_each=8,input_decoder_threads_each=4,
        native_dimensions=[3456,6144],resize=False))
    audit()
    from joint_temporal_texture import ROOT as COLOR
    config=read(OUT/'request.json');q=read(BASE/'request.json')
    assert sha(COLOR/'parameters.npz')==config['fixed_profiles_sha256']
    assert sha(COLOR/'exposure.json')==config['fixed_exposure_sha256']
    assert sha(Path(__file__).with_name('compose_cinematic_train_ending.py'))==read(BASE/'train_ending/request.json')['script_sha256']
    ids=[f'{899+2*i:06d}' for i in range(150)];records=[]
    for frame,row in zip(ids,config['inventory']):
        folder=OUT/'frames'/frame;receipt=read(folder/'complete.json')
        assert folder.resolve().is_relative_to(OUT.resolve())
        if folder.is_symlink():
            assert not os.path.isabs(os.readlink(folder))
            assert folder.resolve()==(OUT/'parallel_completed'/frame).resolve()
        assert receipt['request_sha256']==sha(OUT/'request.json')
        for name,digest in receipt['hashes'].items():assert sha(folder/name)==digest
        result=read(folder/'result.json');assert result['camera']==row['camera']
        assert result['output_dimensions']==[3456,6144] and not result['hd_pixels_upsampled']
        with Image.open(folder/'frame.png') as im:assert im.size==(3456,6144)
        provenance=read(folder/'source_provenance.json');sources=provenance['sources']
        assert all(r['dimensions']==[6144,3072] for r in sources)
        assert provenance['worker_sha256']==config['script_sha256'] and not provenance['resize']
        if row['index']<126:
            assert result['checks']['source_ids_recomputed_at_6k']
            with Image.open(folder/'source_ids.png') as im:assert im.size==(6144,3456)
        records.append(dict(frame=frame,image_sha256=sha(folder/'frame.png'),kind=result['kind'],
            receipt_sha256=sha(folder/'complete.json'),native_sources=len(sources),seconds=result['seconds'],
            relative_payload=str(folder.resolve().relative_to(OUT.resolve())),directory_link=folder.is_symlink()))
    assert sum(r['kind']=='3d_render' for r in records)==118
    assert sum(r['kind']=='explicit_3d_to_train_dissolve' for r in records)==8
    assert sum(r['kind']=='real_train_rgb' for r in records)==24
    assert len({r['image_sha256'] for r in records})==150
    out=OUT/'presentation';out.mkdir(exist_ok=True);seq=out/'sequence';seq.mkdir(exist_ok=True)
    for i,frame in enumerate(ids):
        link=seq/f'{i:06d}.png'
        if not link.exists():link.symlink_to(OUT/'frames'/frame/'frame.png')
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    specifications=[
        ('video.mp4','libx265',['-crf','16','-tag:v','hvc1','-x265-params','pools=8:frame-threads=2:log-level=warning']),
        ('video_h264.mp4','libx264',['-crf','16','-level:v','6.0'])]
    def encode_one(specification):
        filename,codec,extra=specification
        target=out/filename
        command=['ffmpeg','-y','-nostdin','-v','warning','-threads','4','-framerate','24','-i',str(seq/'%06d.png'),
            '-frames:v','150','-c:v',codec,'-preset','medium',*extra,'-pix_fmt','yuv420p',
            '-threads','8','-movflags','+faststart',str(target)]
        write(out/(filename+'.command.json'),dict(argv=command,no_resize_filter=True))
        print('encoding',filename,flush=True)
        with (out/(filename+'.log')).open('w') as log:
            subprocess.run(command,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','4','-count_frames','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration,color_space,color_transfer,color_primaries,color_range','-of','json',str(target)],env=env,text=True))['streams'][0]
        assert probe['width']==3456 and probe['height']==6144 and probe['r_frame_rate']=='24/1'
        assert int(probe['nb_read_frames'])==150 and abs(float(probe['duration'])-6.25)<1e-6
        write(out/(filename+'.probe.json'),probe)
        return filename,probe
    with ThreadPoolExecutor(max_workers=2) as pool:probes=dict(pool.map(encode_one,specifications))
    print('archiving native PNGs',flush=True)
    with zipfile.ZipFile(out/'frames.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for frame in ids:archive.write(OUT/'frames'/frame/'frame.png',frame+'.png')
    with zipfile.ZipFile(out/'frames.zip') as archive:
        assert len(archive.namelist())==150
        for row in records:
            data=archive.read(row['frame']+'.png');assert data.startswith(b'\x89PNG\r\n\x1a\n')
            assert hashlib.sha256(data).hexdigest()==row['image_sha256']
    reviewdir=OUT/'review';reviewdir.mkdir(exist_ok=True)
    for begin in range(0,150,10):
        canvas=Image.new('RGB',(1080,808),(25,25,25));draw=ImageDraw.Draw(canvas)
        for j,frame in enumerate(ids[begin:begin+10]):
            with Image.open(OUT/'frames'/frame/'frame.png') as im:
                im.thumbnail((216,384));x=j%5*216;y=j//5*404;canvas.paste(im,(x,y+20))
            draw.text((x+5,y+3),frame,fill='white')
        canvas.save(reviewdir/f'overview_{begin:03d}.jpg',quality=95)
    selected=[0,24,48,72,92,112,118,125,126,149];decoded=reviewdir/'decoded';decoded.mkdir(exist_ok=True)
    expression="select='"+'+'.join(f'eq(n,{i})' for i in selected)+"'"
    with (decoded/'decode.log').open('w') as log:
        subprocess.run(['ffmpeg','-y','-v','warning','-i',str(out/'video.mp4'),'-vf',expression,
            '-vsync','0','-threads','4',str(decoded/'%03d.png')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    canvas=Image.new('RGB',(1080,808),(25,25,25));draw=ImageDraw.Draw(canvas);decoded_records=[]
    for number,index in enumerate(selected,1):
        path=decoded/f'{number:03d}.png'
        with Image.open(path) as im:
            assert im.size==(3456,6144);im.thumbnail((216,384));x=(number-1)%5*216;y=(number-1)//5*404;canvas.paste(im,(x,y+20))
        draw.text((x+4,y+3),f'{index}: {ids[index]}',fill='white')
        decoded_records.append(dict(index=index,frame=ids[index],path=str(path),sha256=sha(path)))
    canvas.save(reviewdir/'decoded_overview.jpg',quality=95)
    write(decoded/'manifest.json',dict(video_sha256=sha(out/'video.mp4'),records=decoded_records))
    color_checks=[]
    for sample in decoded_records:
        if sample['index'] not in [92,149]:continue
        with Image.open(sample['path']) as im:actual=np.asarray(im.convert('RGB'))[::8,::8].astype(np.float32)
        with Image.open(OUT/'frames'/sample['frame']/'frame.png') as im:expected=np.asarray(im.convert('RGB'))[::8,::8].astype(np.float32)
        mask=expected.max(2)>16;difference=actual[mask]-expected[mask]
        bias=difference.mean(0);median=np.median(difference,axis=0)
        color_checks.append(dict(frame=sample['frame'],matched_native_pixel_stride=8,samples=int(mask.sum()),
            mean_rgb_delta_8bit=bias.tolist(),median_rgb_delta_8bit=median.tolist(),
            mean_absolute_rgb_delta_8bit=np.abs(difference).mean(0).tolist(),
            broad_channel_bias_detected=bool(np.any(np.abs(bias)>2)|np.any(np.abs(median)>2))))
    assert not any(r['broad_channel_bias_detected'] for r in color_checks),'Investigate decoded RGB color bias'
    write(reviewdir/'color_sanity.json',dict(checks=color_checks,metric_purpose='Matched-pixel broad color/range sanity, not quality ranking',
        probes={name:{key:probe.get(key,'unspecified') for key in ['color_space','color_transfer','color_primaries','color_range']}
            for name,probe in probes.items()},
        encoding_color_tags_explicit=False,limitation='Unspecified matrix/transfer/primaries metadata can be interpreted differently by other players'))
    parallel=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_parallel_endings_v1')
    provenance=OUT/'provenance';provenance.mkdir(exist_ok=True);parallel_bindings={}
    if (parallel/'parallel_request_symlink.json').exists():
        pr=read(parallel/'parallel_request_symlink.json')
        assert sha(Path(__file__).with_name('parallel_cinematic_6k_endings.py'))==pr['wrapper_sha256']
        assert pr['renderer_sha256']==config['script_sha256']
        for name in ['parallel_request.json','parallel_request_symlink.json','publication_checks.jsonl','scheduling_checks.jsonl',
                'failed_renameat2_wrapper.py','filesystem_publication_test/result.json']:
            target=provenance/('parallel_'+name.replace('/','_'));shutil.copyfile(parallel/name,target)
            parallel_bindings[str(target)]=sha(target)
    independent=Path('/mnt/data/dec5_cinematic_6k_independent_review')
    independent_bindings={}
    if (independent/'review.json').exists():
        review_record=read(independent/'review.json')
        assert review_record['request_sha256']==sha(OUT/'request.json')
        assert review_record['report_sha256']==sha(independent/'review.md')
        for name in ['review.json','review.md']:
            target=provenance/('independent_'+name);shutil.copyfile(independent/name,target)
            independent_bindings[str(target)]=sha(target)
    write(out/'manifest.json',dict(status='integrity_pass_visual_review_pending',records=records,
        request_sha256=sha(OUT/'request.json'),probes=probes,output_dimensions=[3456,6144],
        all150_output_images_unique=True,files={name:dict(sha256=sha(out/name),bytes=(out/name).stat().st_size)
            for name in ['video.mp4','video_h264.mp4','frames.zip']},
        source_resolution=[6144,3072],source_crop_resolution=[5461,3072],synthetic_detail=False,
        pure_3d_count=118,dissolve_count=8,pure_train_count=24,artifact_free_approval=False,
        internal_relative_directory_links_only=True,zip_contains_actual_png_bytes=True,
        parallel_provenance_bindings=parallel_bindings,packager_sha256=package_hash,
        concurrent_independent_encodes=2,independent_review_bindings=independent_bindings))
    print('packaged',out,flush=True)


def seal(notes_path):
    notes=read(notes_path);assert notes['status']=='approved_with_known_residuals'
    reviewed=notes['reviewed_paths'];assert len(reviewed)>=18
    for path in reviewed:assert Path(path).is_file()
    expected=[str(OUT/'review'/f'overview_{i:03d}.jpg') for i in range(0,150,10)]
    assert set(expected).issubset(reviewed)
    assert str(OUT/'review/decoded_overview.jpg') in reviewed
    config=read(OUT/'request.json');manifest=read(OUT/'presentation/manifest.json')
    assert manifest['status']=='integrity_pass_visual_review_pending'
    assert manifest['packager_sha256']==sha(__file__)
    assert manifest['output_dimensions']==[3456,6144] and len(manifest['records'])==150
    for name,record in manifest['files'].items():assert sha(OUT/'presentation'/name)==record['sha256']
    for path,digest in manifest.get('parallel_provenance_bindings',{}).items():assert sha(path)==digest
    for path,digest in manifest.get('independent_review_bindings',{}).items():assert sha(path)==digest
    for frame in manifest['records']:
        folder=OUT/'frames'/frame['frame'];assert sha(folder/'frame.png')==frame['image_sha256']
        assert sha(folder/'complete.json')==frame['receipt_sha256']
    assert sha(Path(__file__).with_name('render_cinematic_6k_output.py'))==config['script_sha256']
    provenance=OUT/'provenance';provenance.mkdir(exist_ok=True)
    names=['render_cinematic_6k_output.py','review_cinematic_6k_output.py','parallel_cinematic_6k_endings.py','nice_cinematic_6k_endings.py',
        'convert_dec5_5a3_pq16_to_exr.py','compose_cinematic_train_ending.py',*config['dependencies']]
    snapshots={}
    for name in names:
        source=Path(__file__).with_name(name);target=provenance/name;shutil.copyfile(source,target)
        snapshots[name]=dict(path=str(target),sha256=sha(target))
    notes['bindings']={path:sha(path) for path in reviewed}
    write(OUT/'visual_review.json',notes)
    bindings={str(OUT/path):sha(OUT/path) for path in [
        'request.json','preflight.json','packaging_request.json','visual_review.json','presentation/manifest.json',
        'presentation/video.mp4','presentation/video_h264.mp4','presentation/frames.zip',
        'review/decoded/manifest.json','review/color_sanity.json','checks.jsonl']}
    write(OUT/'delivery.json',dict(status='complete_reviewed_with_known_residuals',
        completed_utc=datetime.now(timezone.utc).isoformat(),output_dimensions=[3456,6144],
        fps=24,frames=150,duration_seconds=6.25,artifacts=manifest['files'],bindings=bindings,
        snapshots=snapshots,no_hd_output_upscale=True,synthetic_detail=False,
        source_optical_footprint_at_final_lens=[5461/1.9,3072/1.9],
        known_residuals=notes['known_residuals'],artifact_free_approval=False))
    print('sealed',OUT/'delivery.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','review','temporal','status','package','seal']);p.add_argument('--frames',nargs='+',default=['001083','001197']);p.add_argument('--notes',type=Path);p.add_argument('--box',nargs=4,type=int);p.add_argument('--region',default='hair');a=p.parse_args()
    if a.action=='audit':audit()
    elif a.action=='review':review(a.frames)
    elif a.action=='temporal':temporal(a.frames,a.box,a.region)
    elif a.action=='status':status()
    elif a.action=='package':package()
    else:seal(a.notes)
