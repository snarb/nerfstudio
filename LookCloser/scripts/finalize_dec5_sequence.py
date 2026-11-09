#!/usr/bin/env python3
"""Durable CPU-only postflight for the full native50 campaign.

Waits for sealed worker completion, validates selected checkpoint receipts,
and makes a GT/render review video. It neither fits nor selects checkpoints.
Display FPS is explicit and is not a claim about capture timing calibration.
"""
from pathlib import Path
import argparse
import collections
import subprocess
import time
from PIL import Image,ImageDraw
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def selected(root,frame,mode='best'):
    directory=root/'frames'/frame
    if read(directory/'progress.json',{}).get('phase')!='complete':return None
    revision=read(directory/'progress.json')['revision'];training=directory/('training_'+revision)
    if not read(training/'complete.json'):return None
    best=read(training/'best.json')
    if mode=='final106316':
        final=next(r for r in read(training/'history.json') if r['step']==106316)
        latest=read(training/'latest.json')
        if latest['step']!=106316:raise ValueError('Completed full recipe lacks final checkpoint')
        best=dict(path=latest['path'],sha256=latest['sha256'],metrics=final,renders=final['renders'])
    checkpoint=Path(best['path']);renders=Path(best['renders'])
    if not checkpoint.is_file() or not (renders/'0.jpg').is_file():
        raise ValueError('Sealed best artifact missing for '+frame)
    return dict(frame=frame,revision=revision,best=best,checkpoint=checkpoint,renders=renders)


def finalize(root,manifest,fps,mode='best'):
    output=root/'review50'
    if mode!='best':output=output/mode
    output.mkdir(parents=True,exist_ok=True);frames_dir=output/'frames';frames_dir.mkdir(exist_ok=True)
    rows=[]
    for index,frame in enumerate(manifest['frames']):
        row=selected(root,frame,mode)
        if row is None:raise ValueError('Incomplete production frame '+frame)
        if sha(row['checkpoint'])!=row['best']['sha256']:raise ValueError('Selected checkpoint checksum mismatch '+frame)
        with Image.open(row['renders']/'0.jpg') as panel:
            w,h=panel.size
            if (w,h)!=(12288,3072):raise ValueError('Expected complete native6144x3072 GT/render pair')
            parts=[panel.crop(box).transpose(Image.Transpose.ROTATE_90).resize((540,1080),Image.Resampling.LANCZOS)
                   for box in [(0,0,w//2,h),(w//2,0,w,h)]]
            canvas=Image.new('RGB',(1080,1112));canvas.paste(parts[0],(0,32));canvas.paste(parts[1],(540,32))
            draw=ImageDraw.Draw(canvas);draw.text((10,8),'GT  '+frame,fill='white');draw.text((550,8),'LookCloser '+mode+' '+frame,fill='white')
            canvas.save(frames_dir/f'{index:04d}.jpg',quality=93)
        metrics=row['best']['metrics'];rows.append(dict(frame=frame,revision=row['revision'],
            checkpoint=str(row['checkpoint']),checkpoint_sha256=row['best']['sha256'],step=metrics['step'],
            renders=str(row['renders']),**{k:metrics[k] for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}))
        write(output/'progress.json',dict(completed=len(rows),total=len(manifest['frames']),time=time.time()))
    video=output/('gt_vs_'+mode+'.mp4');temporary=output/('gt_vs_'+mode+'.tmp.mp4')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),
        '-i',str(frames_dir/'%04d.jpg'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(temporary)],check=True)
    temporary.replace(video)
    # Ten evenly spaced complete-frame comparisons, without fitting a held-out ROI.
    indices=sorted({round(i*(len(rows)-1)/9) for i in range(10)})
    sheet=Image.new('RGB',(1080,556*((len(indices)+1)//2)))
    for position,index in enumerate(indices):
        with Image.open(frames_dir/f'{index:04d}.jpg') as image:
            sheet.paste(image.resize((540,556),Image.Resampling.LANCZOS),((position%2)*540,(position//2)*556))
    sheet.save(output/'contact_sheet.jpg',quality=90)
    result=dict(frames=len(rows),per_frame=rows,selection_mode=mode,
        mean_selected_metrics={k:sum(r[k] for r in rows)/len(rows) for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
        selected_step_counts=dict(collections.Counter(str(r['step']) for r in rows)),
        revision_counts=dict(collections.Counter(r['revision'] for r in rows)),
        video=str(video),video_sha256=sha(video),display_fps=fps,
        capture_timing_calibrated=False,checkpoint_selection_changed=False,
        checkpoint_checksums_verified=True,heldout_camera=manifest['heldout_camera'],
        hair_flicker_solution_established=False,finished=time.time())
    write(output/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--display-fps',type=int,default=24);p.add_argument('--wait',action='store_true')
    p.add_argument('--temporal-hair-reference',type=Path);a=p.parse_args()
    if not 1<=a.display_fps<=120:raise ValueError('Invalid display FPS')
    manifest=read(a.root/'manifest.json')
    if len(manifest['frames'])!=50:raise ValueError('Expected fifty production frames')
    previous=read(a.root/'campaign_complete.json',{})
    if previous.get('motion_compensated_review') and (not a.temporal_hair_reference or previous.get('native_head_review')):return
    while not all(selected(a.root,f) is not None for f in manifest['frames']):
        if not a.wait:raise ValueError('Production queue has not finished')
        time.sleep(30)
    results={}
    for mode in ['best','final106316']:
        path=a.root/'review50'
        if mode!='best':path=path/mode
        results[mode]=read(path/'complete.json') or finalize(a.root,manifest,a.display_fps,mode)
    from dec5_temporal_review import measure
    temporal_path=a.root/'review50/temporal'
    temporal=read(temporal_path/'complete.json') or measure(a.root,manifest,results,temporal_path,a.temporal_hair_reference)
    native_head=None
    if a.temporal_hair_reference:
        from dec5_native_head_review import render
        native_path=a.root/'review50/native_head'
        native_head=read(native_path/'complete.json') or render(a.root,manifest,results,native_path,a.temporal_hair_reference,a.display_fps)
    write(a.root/'campaign_complete.json',dict(results['best'],fixed_final_comparison=results['final106316'],
        motion_compensated_review=temporal,
        native_head_review=native_head,
        purpose='Official per-frame PSNR best versus one common final FR.3 gate; selection remains unchanged'))


if __name__=='__main__':main()
