"""Native A/B crops, chronological contact sheets and verified 6K-source delivery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import zipfile
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw
from render_cinematic_6k_texture import BASE, OUT, REMOTE, REMOTE_ROOT, RAW, read, write, sha


def status():
    progress=read(OUT/'progress.json');pid=progress.get('pid')
    def check(args):
        result=subprocess.run(args,text=True,capture_output=True)
        return dict(returncode=result.returncode,output=(result.stdout+result.stderr).strip())
    record=dict(utc=datetime.now(timezone.utc).isoformat(),progress=progress,
        complete=len(list((OUT/'frames').glob('*/complete.json'))),
        controller=check(['ps','-p',str(pid),'-o','pid,etime,rss,stat']) if pid else None,
        remote=check(['ssh',REMOTE,f"pgrep -af '[p]ython {REMOTE_ROOT}/render_cinematic_6k_texture.py remote' || true"]),
        gpu=check(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader']),
        log_tail=check(['tail','-n','4',str(OUT/'render_full.log')]))
    record['oom_evidence_in_render_log']='out of memory' in (OUT/'render_full.log').read_text().lower()
    path=OUT/'supervision.json';history=read(path) if path.exists() else []
    history.append(record);write(path,history)
    print(json.dumps(dict(utc=record['utc'],complete=record['complete'],progress=progress,
        controller=record['controller'],gpu=record['gpu'],oom_evidence=record['oom_evidence_in_render_log'])),flush=True)


def review(frames):
    dest=OUT/'review';dest.mkdir(exist_ok=True)
    for frame in frames:
        new=OUT/'frames'/frame/'frame.png'
        if not new.exists():continue
        old=BASE/'presentation'/'frames'/frame/'frame.png'
        images=[Image.open(old).convert('RGB'),Image.open(new).convert('RGB')]
        boxes=[('hair',(180,100,564,484)),('face',(500,480,884,864)),('cloth',(360,1440,744,1824))]
        if int(frame)<1060:
            # Position diagnostic crops using the baseline's black background;
            # these image coordinates never enter rendering or source selection.
            yy,xx=np.where(np.asarray(images[0]).max(2)>12)
            x=int(np.clip((xx.min()+xx.max())/2-192,0,696));y=int(np.clip(yy.min()+20,0,1536))
            face_y=int(np.clip(y+.22*(yy.max()-yy.min()),0,1536))
            boxes=[('hair',(x,y,x+384,y+384)),('face',(x,face_y,x+384,face_y+384)),('cloth',(360,1440,744,1824))]
        for name,box in boxes:
            canvas=Image.new('RGB',(768,414),(25,25,25));draw=ImageDraw.Draw(canvas)
            for i,im in enumerate(images):
                canvas.paste(im.crop(box),(384*i,30));draw.text((384*i+8,8),['HD source','Native 6K source'][i],fill='white')
            canvas.save(dest/f'{frame}_{name}_AB.png')
        canvas=Image.new('RGB',(540,500),(25,25,25));draw=ImageDraw.Draw(canvas)
        for i,im in enumerate(images):
            im.thumbnail((270,480));canvas.paste(im,(270*i,20));draw.text((270*i+5,3),['HD source','Native 6K source'][i],fill='white')
        canvas.save(dest/f'{frame}_overview_AB.png')


def temporal(frames):
    """Same fixed crop over consecutive real times; no temporal interpolation."""
    dest=OUT/'review';dest.mkdir(exist_ok=True)
    assert all(int(b)-int(a)==2 for a,b in zip(frames,frames[1:]))
    first=np.asarray(Image.open(BASE/'presentation'/'frames'/frames[0]/'frame.png'))
    yy,xx=np.where(first.max(2)>12)
    x=int(np.clip((xx.min()+xx.max())/2-80,0,920));y=int(np.clip(yy.min()+70,0,1760))
    box=(x,y,x+160,y+160)
    canvas=Image.new('RGB',(160*len(frames),370),(25,25,25));draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(frames):
        for j,base in enumerate([BASE/'presentation',OUT]):
            im=Image.open(base/'frames'/frame/'frame.png');canvas.paste(im.crop(box),(160*i,25+185*j))
            draw.text((160*i+3,4+185*j),f'{frame} '+['HD','6K'][j],fill='white')
    name=f'temporal_{frames[0]}_{frames[-1]}'
    canvas.save(dest/(name+'.png'))
    write(dest/(name+'.json'),dict(frames=frames,box=box,same_fixed_crop=True,
        image_sha256=sha(dest/(name+'.png')),visual_status='pending'))


def decoded_samples(out, ids, env):
    selected=[0,24,48,72,92,112,118,125,126,149]
    dest=OUT/'review'/'decoded';dest.mkdir(exist_ok=True)
    expression="select='"+'+'.join(f'eq(n,{i})' for i in selected)+"'"
    with (dest/'decode.log').open('w') as log:
        subprocess.run(['ffmpeg','-y','-v','warning','-i',str(out/'video.mp4'),
            '-vf',expression,'-vsync','0',str(dest/'%03d.png')],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    records=[];canvas=Image.new('RGB',(1080,808),(25,25,25));draw=ImageDraw.Draw(canvas)
    for number,index in enumerate(selected,1):
        path=dest/f'{number:03d}.png';im=Image.open(path);assert im.size==(1080,1920)
        records.append(dict(index=index,frame=ids[index],path=str(path),sha256=sha(path)))
        im.thumbnail((216,384));x=(number-1)%5*216;y=(number-1)//5*404
        canvas.paste(im,(x,y+20));draw.text((x+4,y+3),f'{index}: {ids[index]}',fill='white')
    canvas.save(OUT/'review'/'decoded_overview.jpg',quality=95)
    write(dest/'manifest.json',dict(video_sha256=sha(out/'video.mp4'),records=records))


def package():
    from joint_temporal_texture import ROOT as COLOR
    ids=[f'{899+2*i:06d}' for i in range(150)];q=read(BASE/'request.json')
    config=read(OUT/'request.json');assert sha(BASE/'request.json')==config['parent_request_sha256']
    dependencies=['joint_temporal_texture.py','render_patchmatch_camera_path.py',
        'bake_joint_temporal_mesh.py','diffusion_mesh_repair.py','native_texture_footprint.py']
    dependency_hashes={name:sha(Path(__file__).with_name(name)) for name in dependencies}
    for name,digest in dependency_hashes.items():assert digest==q['script_hashes'][name]
    ending='compose_cinematic_train_ending.py';dependency_hashes[ending]=sha(Path(__file__).with_name(ending))
    assert dependency_hashes[ending]==read(BASE/'train_ending'/'request.json')['script_sha256']
    assert sha(Path(__file__).with_name('render_cinematic_6k_texture.py'))==config['script_sha256']
    assert sha(Path(__file__).with_name('convert_dec5_5a3_pq16_to_exr.py'))==config['decoder_sha256']
    assert sha(COLOR/'parameters.npz')==config['fixed_profiles_sha256']
    assert sha(COLOR/'exposure.json')==config['fixed_exposure_sha256']
    assert q['ordered_frame_ids']==ids
    records=[]
    for frame,row in zip(ids,q['inventory']):
        folder=OUT/'frames'/frame;receipt=read(folder/'complete.json')
        assert receipt['request_sha256']==sha(OUT/'request.json')
        for name,digest in receipt['hashes'].items():assert sha(folder/name)==digest
        result=read(folder/'result.json');assert result['camera']==row['camera']
        assert Image.open(folder/'frame.png').size==(1080,1920)
        sources=read(folder/'source_provenance.json')['sources']
        assert all(r['dimensions']==[6144,3072] for r in sources)
        records.append(dict(frame=frame,image_sha256=sha(folder/'frame.png'),kind=result['kind'],
            receipt_sha256=sha(folder/'complete.json'),native_sources=len(sources)))
    assert sum(r['kind']=='3d_render' for r in records)==118
    assert sum(r['kind']=='explicit_3d_to_train_dissolve' for r in records)==8
    assert sum(r['kind']=='real_train_rgb' for r in records)==24
    assert len({r['image_sha256'] for r in records})==150
    out=OUT/'presentation';out.mkdir(exist_ok=True);seq=out/'sequence';seq.mkdir(exist_ok=True)
    for i,frame in enumerate(ids):
        link=seq/f'{i:06d}.png'
        if not link.exists():link.symlink_to(OUT/'frames'/frame/'frame.png')
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    for filename,crf,pixel in [('video.mp4','16','yuv420p'),('master_444.mp4','10','yuv444p')]:
        with (out/(filename+'.log')).open('w') as log:
            subprocess.run(['ffmpeg','-y','-v','warning','-framerate','24','-i',str(seq/'%06d.png'),
                '-frames:v','150','-c:v','libx264','-preset','slow','-crf',crf,'-pix_fmt',pixel,
                '-threads','8','-movflags','+faststart',str(out/filename)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    with zipfile.ZipFile(out/'frames.zip','w',compression=zipfile.ZIP_STORED) as archive:
        for frame in ids:archive.write(OUT/'frames'/frame/'frame.png',frame+'.png')
    with zipfile.ZipFile(out/'frames.zip') as archive:
        assert archive.testzip() is None and len(archive.namelist())==150
        for row in records:assert hashlib.sha256(archive.read(row['frame']+'.png')).hexdigest()==row['image_sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(out/'video.mp4')],env=env,text=True))['streams'][0]
    assert probe['width']==1080 and probe['height']==1920 and probe['r_frame_rate']=='24/1'
    assert int(probe['nb_read_frames'])==150 and abs(float(probe['duration'])-6.25)<1e-6
    decoded_samples(out,ids,env)
    for begin in range(0,150,10):
        canvas=Image.new('RGB',(1080,808),(25,25,25));draw=ImageDraw.Draw(canvas)
        for j,frame in enumerate(ids[begin:begin+10]):
            im=Image.open(OUT/'frames'/frame/'frame.png');im.thumbnail((216,384))
            x=j%5*216;y=j//5*404;canvas.paste(im,(x,y+20));draw.text((x+5,y+3),frame,fill='white')
        canvas.save(OUT/'review'/f'overview_{begin:03d}.jpg',quality=95)
    write(out/'manifest.json',dict(status='integrity_pass_visual_review_pending',records=records,
        request_sha256=sha(OUT/'request.json'),probe=probe,
        dependency_hashes=dependency_hashes,all150_output_images_unique=True,
        files={name:dict(sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in ['video.mp4','master_444.mp4','frames.zip']},
        source_resolution=[6144,3072],source_crop_resolution=[5461,3072],synthetic_detail=False,
        pure_3d_count=118,dissolve_count=8,pure_train_count=24,artifact_free_approval=False))
    print('packaged',out,flush=True)


def seal(images,note):
    """Bind an explicit visual verdict to the verified final deliverables."""
    manifest=read(OUT/'presentation'/'manifest.json')
    for name,row in manifest['files'].items():assert sha(OUT/'presentation'/name)==row['sha256']
    inspected={name:sha(OUT/name) for name in images}
    assert 'review/decoded_overview.jpg' in inspected
    review=dict(status='reviewed_native_sources_with_known_residuals',inspected_image_hashes=inspected,
        note=note,artifact_free_approval=False,fine_temporal_aliasing_ruled_out=False)
    write(OUT/'manual_visual_review.json',review)
    report=Path(__file__).parents[1]/'experiments'/'dec5_cinematic_6k_texture.md'
    shutil.copyfile(report,OUT/'report.md')
    snapshot=OUT/'script_snapshot';snapshot.mkdir(exist_ok=True)
    names=list(manifest['dependency_hashes'])+['render_cinematic_6k_texture.py',
        'review_cinematic_6k_texture.py','convert_dec5_5a3_pq16_to_exr.py']
    for name in names:shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    subprocess.run(['scp','-q',f'{REMOTE}:{RAW}/meta.json',str(OUT/'source_meta.json')],check=True)
    assert sha(OUT/'source_meta.json')=='676fe66283498eaac0bb59af03428970f348a2306cec064aa1113cddc274288b'
    manifest['status']='reviewed_native_6k_source_video_with_known_residuals'
    write(OUT/'presentation'/'manifest.json',manifest)
    names=['request.json','manual_visual_review.json','report.md','source_meta.json','supervision.json',
        'presentation/manifest.json','presentation/video.mp4','presentation/master_444.mp4','presentation/frames.zip']
    write(OUT/'delivery.json',dict(status=manifest['status'],bindings={name:sha(OUT/name) for name in names},
        script_snapshot={p.name:sha(p) for p in snapshot.iterdir()},frame_count=150,fps=24,duration_seconds=6.25,
        output_dimensions=[1080,1920],native_source_dimensions=[6144,3072],artifact_free_approval=False))
    print('sealed',OUT/'delivery.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['review','temporal','package','status','seal'])
    p.add_argument('--frames',nargs='+',default=['001083','001123','001197']);p.add_argument('--images',nargs='+');p.add_argument('--note');a=p.parse_args()
    if a.action in ['review','temporal']:globals()[a.action](a.frames)
    elif a.action=='seal':
        if not a.images or not a.note:p.error('seal requires explicitly inspected --images and a --note')
        seal(a.images,a.note)
    else:status() if a.action=='status' else package()
