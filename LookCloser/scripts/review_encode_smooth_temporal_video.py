"""Nonblocking review sheets and strict 150-time-frame video publication."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import ROOT,read,sha,atomic_json,cameras,exr,display
from render_smooth_temporal_mesh_video import OUTPUT,verify_request
from finalize_local_mesh_repair import verify_hashes

REGIONS={'lipstick_hand':(280,970,730,1420),'ear_hair':(690,780,930,1160),'face':(300,660,900,1240)}


def completed(output):
    request=verify_request(output);results=[];request_hash=sha(output/'request.json')
    for record in request['inventory']:
        frame=output/'frames'/record['frame_id'];complete=frame/'complete.json'
        if not complete.exists():continue
        receipt=read(complete)
        if receipt['request_sha256']!=request_hash:raise ValueError('Incompatible completed frame')
        verify_hashes(frame,receipt['hashes']);results.append((record,read(frame/'result.json')))
    return request,results


def sheets(output,first,last):
    request,ready=completed(output);ready={r['index']:(r,result) for r,result in ready}
    root=output/'contact_sheets';root.mkdir(exist_ok=True)
    for start in range(first,last,4):
        indices=list(range(start,min(start+4,150)))
        if not all(i in ready for i in indices):continue
        target=root/f'{start:03d}_{indices[-1]:03d}';target.mkdir(exist_ok=True)
        expected={ready[i][0]['frame_id']:ready[i][1]['render_sha256'] for i in indices}
        if (target/'manifest.json').exists() and read(target/'manifest.json')['render_hashes']==expected and read(target/'manifest.json').get('sheet_version')==2:continue
        images=[];references=[];ref_records=[]
        profile=read(ROOT/'camera_profiles.json');gains=dict(zip(profile['physical_cameras'],profile['rgb_gain']))
        gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
        for index in indices:
            record,result=ready[index];frame=record['frame_id'];im=Image.open(output/'frames'/frame/'frame.png').convert('RGB');images.append(im)
            rows,_,_=cameras(frame);cc=np.array([r['transform_matrix'] for r in rows])[:,:3,3]
            selected=int(np.argmin(np.linalg.norm(cc-np.array(record['camera']['transform_matrix'])[:3,3],axis=1)));row=rows[selected]
            rgb=np.rint(display(exr(row['file_path'])*np.array(gains[row['physical_camera']]),gain)*255).clip(0,255).astype(np.uint8)
            ref=Image.fromarray(np.rot90(rgb));references.append(ref)
            ref_records.append({'frame_id':frame,'physical_camera':row['physical_camera'],'source_sha256':sha(row['file_path']),
                'purpose':'posthoc nearest real train reference, NOT exact target GT'})
            for name,box in REGIONS.items():
                compare=Image.new('RGB',((box[2]-box[0])*2,box[3]-box[1]+24));draw=ImageDraw.Draw(compare)
                compare.paste(ref.crop(box),(0,24));compare.paste(im.crop(box),(box[2]-box[0],24))
                draw.text((3,4),'nearby real train reference',fill='white');draw.text((box[2]-box[0]+3,4),'novel-view prediction',fill='white')
                compare.save(target/f'{frame}_{name}.png')
        overview=Image.new('RGB',(540*len(images),504));draw=ImageDraw.Draw(overview)
        for k,(im,ref,index) in enumerate(zip(images,references,indices)):
            x=k*540;overview.paste(ref.resize((270,480)),(x,24));overview.paste(im.resize((270,480)),(x+270,24))
            draw.text((x+3,4),f'{ready[index][0]["frame_id"]}: real train | prediction',fill='white')
        overview.save(target/'overview.png')
        for name in ['lipstick_hand','ear_hair','face']:
            box=REGIONS[name];ww,hh=box[2]-box[0],box[3]-box[1]
            panel=Image.new('RGB',(ww*2,(hh+24)*2));draw=ImageDraw.Draw(panel)
            for k,(im,index) in enumerate(zip(images,indices)):
                x=k%2*ww;y=k//2*(hh+24);panel.paste(im.crop(box),(x,y+24));draw.text((x+4,y+4),ready[index][0]['frame_id'],fill='white')
            panel.save(target/f'{name}.png')
        atomic_json(target/'manifest.json',{'sheet_version':2,'indices':indices,'render_hashes':expected,'regions_portrait_xyxy':REGIONS,
            'references':ref_records,'reference_rgb_used_for_prediction':False,'visual_status':'pending'})
        print(f'review_sheet={start}..{indices[-1]}',flush=True)


def encode(output):
    request,ready=completed(output)
    if [r['frame_id'] for r,_ in ready]!=request['ordered_frame_ids'] or len(ready)!=150:
        raise ValueError('Encoding requires every one of the 150 chronological source instants')
    sequence=output/'video_frames';sequence.mkdir(exist_ok=True)
    for index,(record,result) in enumerate(ready):
        source=output/'frames'/record['frame_id']/'frame.png';destination=sequence/f'{index:05d}.png'
        if destination.exists():
            if sha(destination)!=sha(source):raise ValueError('Changed video input frame')
        else:os.link(source,destination)
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    temp=output/'smooth_temporal_150.partial.mp4';video=output/'smooth_temporal_150.mp4'
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-framerate','30','-i',str(sequence/'%05d.png'),
        '-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-threads','8','-movflags','+faststart',str(temp)],env=env,check=True)
    import json
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(temp)],env=env,text=True))['streams'][0]
    if int(probe['nb_read_frames'])!=150 or abs(float(probe['duration'])-5)>1e-3:raise ValueError('Encoded temporal inventory mismatch')
    os.replace(temp,video)
    encoded_review(output)
    atomic_json(output/'video_manifest.json',{'status':'encoded_requires_temporal_visual_review','video_sha256':sha(video),
        'video':str(video),'ffprobe':probe,'source_frames':request['ordered_frame_ids'],'source_frame_count':150,
        'camera_periodic':True,'actor_motion_periodic_not_claimed':True,'frame_interpolation':False,
        'request_sha256':sha(output/'request.json'),'render_hashes':{r['frame_id']:result['render_sha256'] for r,result in ready}})


def encoded_review(output):
    """Decode uniformly spaced actual MP4 frames for a temporal context panel."""
    request=verify_request(output);video=output/'smooth_temporal_150.mp4'
    env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
    decoded=output/'encoded_review';decoded.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(video),
        '-vf','select=not(mod(n\\,10)),scale=216:384','-vsync','vfr',str(decoded/'%03d.png')],env=env,check=True)
    samples=sorted(decoded.glob('*.png'))
    if len(samples)!=15:raise ValueError('Encoded temporal review sample inventory mismatch')
    panel=Image.new('RGB',(216*5,408*3));draw=ImageDraw.Draw(panel)
    for k,path in enumerate(samples):
        x=k%5*216;y=k//5*408;panel.paste(Image.open(path),(x,y+24))
        draw.text((x+3,y+4),f'{request["ordered_frame_ids"][k*10]} / {k/3:.2f}s',fill='white')
    panel.save(output/'encoded_temporal_overview.png')
    atomic_json(output/'encoded_review.json',{'video_sha256':sha(video),'sample_indices':list(range(0,150,10)),
        'encoded_temporal_overview_sha256':sha(output/'encoded_temporal_overview.png')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['sheets','encode','watch','decoded'])
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int,default=150)
    a=p.parse_args()
    if a.action=='sheets':sheets(a.output,a.first,a.last)
    elif a.action=='encode':encode(a.output)
    elif a.action=='decoded':encoded_review(a.output)
    else:
        while True:
            sheets(a.output,0,150)
            _,ready=completed(a.output)
            if len(ready)==150:
                encode(a.output);break
            progress=read(a.output/'progress.json')
            if progress.get('stage')=='workers_finished':raise RuntimeError('Incomplete worker inventory')
            time.sleep(30)


if __name__=='__main__':main()
