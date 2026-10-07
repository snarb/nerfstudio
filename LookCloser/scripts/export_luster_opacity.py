#!/usr/bin/env python3
"""Export selected learned RGB, native comparisons, and fixed Luster review paths."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw

from run_luster_opacity import configuration,evaluate,sha,write


def load(selected):
    import torch
    checkpoint=Path(selected['checkpoint'])
    source=checkpoint.parents[4]
    request=json.loads((source/'request.json').read_text())
    request=deepcopy(request)
    request['resume']=str(checkpoint)
    request['end_step']=selected['step']
    cfg=configuration(request)
    trainer=cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    trainer.pipeline.model.current_train_step=selected['step']
    if selected['step']>=20000:
        trainer.pipeline.model.config.feature_reweighting_strength=.3
        trainer.pipeline.model.field.feature_reweighting_strength=.3
    trainer.pipeline.eval()
    return trainer,request


def rotation_z(theta):
    c,s=math.cos(theta),math.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)


def render_path(trainer,request,output,kind,frames,ffmpeg):
    import torch
    from nerfstudio.cameras.cameras import Cameras,CameraType
    pipe=trainer.pipeline
    pipe.eval()
    meta=json.loads((Path(request['data'])/'transforms.json').read_text())
    cid=97 if kind=='face' else 11
    row=next(r for r in meta['frames'] if r['camera_id']==cid)
    pose=np.array(row['transform_matrix'],dtype=np.float32)
    if kind=='face':
        x0,y0,x1,y1=request['roi_boxes'][Path(row['file_path']).name]['face']
        size=max(x1-x0,y1-y0)*1.4
        x0=(x0+x1)/2-size/2;y0=(y0+y1)/2-size/2
        width=height=512
        fx=row['fl_x']*width/size;fy=row['fl_y']*height/size
        cx=(row['cx']-x0)*width/size;cy=(row['cy']-y0)*height/size
        # Every arm follows the same frozen path; model-dependent depth would
        # change the comparison trajectory when geometry is precisely the axis.
        path_manifest=Path(request['data']).parent/'review_paths.json'
        target=np.array(json.loads(path_manifest.read_text())['face_pivot'],dtype=np.float32)
        angles=np.linspace(-math.radians(15),math.radians(15),frames)
    else:
        width,height=480,640
        fx=row['fl_x']*width/row['w'];fy=row['fl_y']*height/row['h']
        cx=row['cx']*width/row['w'];cy=row['cy']*height/row['h']
        target=np.zeros(3,dtype=np.float32)
        angles=np.linspace(0,2*math.pi,frames,endpoint=False)
    output.mkdir(parents=True,exist_ok=True)
    contact=Image.new('RGB',(8*160,math.ceil(frames/8)*190),(30,30,30))
    draw=ImageDraw.Draw(contact)
    poses=[]
    for index,angle in enumerate(angles):
        rotation=rotation_z(float(angle))
        current=pose.copy()
        current[:3,:3]=rotation@pose[:3,:3]
        current[:3,3]=target+rotation@(pose[:3,3]-target)
        camera=Cameras(camera_to_worlds=torch.tensor(current[:3])[None],fx=fx,fy=fy,cx=cx,cy=cy,
                       width=width,height=height,camera_type=CameraType.PERSPECTIVE).to(pipe.device)
        with torch.no_grad():
            rendered=pipe.model.get_outputs_for_camera(camera)
        rgb=rendered['rgb'].float().cpu().numpy()
        if not np.isfinite(rgb).all():
            raise FloatingPointError(f'Nonfinite path {kind}/{index}')
        im=Image.fromarray(np.rint(rgb.clip(0,1)*255).astype('uint8'))
        im.save(output/f'frame_{index:04d}.png')
        thumb=im.copy();thumb.thumbnail((158,166));contact.paste(thumb,((index%8)*160,(index//8)*190+20))
        draw.text(((index%8)*160+3,(index//8)*190+3),str(index),fill='white')
        poses.append(current.tolist())
    contact.save(output/'contact.jpg')
    write(output/'path.json',dict(kind=kind,pivot=target.tolist(),camera_id=cid,poses=poses,
                                 intrinsics=dict(fx=fx,fy=fy,cx=cx,cy=cy,w=width,h=height)))
    if ffmpeg:
        with (output/'ffmpeg.log').open('w') as log:
            subprocess.run([ffmpeg,'-y','-framerate','24','-i',str(output/'frame_%04d.png'),
                            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(output/'learned_rgb.mp4')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)


def export(selection_path,output,frames,ffmpeg,paths=True):
    selected=json.loads(selection_path.read_text())
    if 'selected' in selected:
        selected=selected['selected']
    if output.exists():
        raise FileExistsError('Export destination must be new')
    output.mkdir(parents=True)
    trainer,request=load(selected)
    request['output']=str(output)
    request['eval_stride']=1
    model=trainer.pipeline.model
    original_settings=dict(adaptive_coarse_step_size=model.config.adaptive_coarse_step_size,
                           max_steps_per_ray=model.config.max_steps_per_ray,
                           corrected_arm_allocator=model.config.corrected_arm_allocator,
                           surface_num_samples=model.config.surface_num_samples,
                           surface_refine_steps=model.config.surface_refine_steps)
    model.config.adaptive_coarse_step_size=.0005
    model.config.max_steps_per_ray=16384
    model.config.corrected_arm_allocator=True
    model.config.surface_num_samples=512
    model.config.surface_refine_steps=12
    request['evaluation_model']={}  # These explicit export settings are authoritative.
    # This export is separate from the evaluator used for checkpoint selection.
    metrics=evaluate(trainer,request,selected['step'])
    trainer.pipeline.eval()
    native=Path(metrics['renders'])
    for path in native.glob('eval_img_*.png'):
        shutil.copy2(path,output/path.name)
    if paths:
        render_path(trainer,request,output/'face', 'face',frames,ffmpeg)
        render_path(trainer,request,output/'orbit','orbit',frames,ffmpeg)
    artifact_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'manifest.json',dict(checkpoint=selected['checkpoint'],checkpoint_sha256=sha(Path(selected['checkpoint'])),
        training_selection=selected,export_metrics=metrics,original_settings=original_settings,
        export_settings=dict(coarse_step=.0005,volume_cap=16384,corrected_arm_allocator=True,
                             surface_samples=512,surface_bisection=12),
        artifact_sha256=artifact_hashes,learned_rgb_only=True,masks_used=False,visual_review='pending'))
    print(json.dumps(dict(output=str(output),psnr=metrics['eval_all_psnr'],ssim=metrics['eval_all_ssim'],lpips=metrics['eval_all_lpips'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=32)
    p.add_argument('--no-paths',action='store_true')
    p.add_argument('--ffmpeg',default=shutil.which('ffmpeg') or '/home/pavlo/.conda/envs/lookcloser-jpeg/bin/ffmpeg')
    args=p.parse_args()
    if args.frames<1:p.error('--frames must be positive')
    export(args.selection,args.output,args.frames,args.ffmpeg,not args.no_paths)
