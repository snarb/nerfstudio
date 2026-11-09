#!/usr/bin/env python3
"""Exact-checkpoint native-model eval previews; no GT compositing.

Default: all eval views at <=640px width. --camera NAME --bbox X0 Y0 X1 Y1
renders a native HD crop by changing only calibrated raster bounds/intrinsics.
Use --code-root to import the sealed model/custom parser implementation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def save_rgb(path,value):
 im=Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8));im.save(path);return im

def display_rgb(value, transform):
 if transform == 'identity':return value
 if transform != 'linear-reinhard-srgb':raise ValueError(transform)
 linear=np.maximum(value,0);linear=linear/(1+linear)
 return np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)

def depth_color(depth,opacity):
 valid=np.isfinite(depth)&(opacity>.05)&(depth>0);result=np.zeros((*depth.shape,3),np.float32)
 if valid.any():
  lo,hi=np.quantile(depth[valid],[.02,.98]);t=np.clip((depth-lo)/max(hi-lo,1e-8),0,1)
  result[...,0]=t;result[...,1]=1-np.abs(2*t-1);result[...,2]=1-t;result[~valid]=0
  return result,dict(quantile02=float(lo),quantile98=float(hi),valid_opacity_threshold=.05)
 return result,dict(quantile02=None,quantile98=None,valid_opacity_threshold=.05)

def select_cameras(dataset,requested):
 names=[Path(p).stem for p in dataset.image_filenames]
 if not requested:return list(range(len(names)))
 result=[]
 for name in requested:
  if name.isdigit():candidates=[int(name)] if int(name)<len(names) else []
  else:candidates=[i for i,n in enumerate(names) if n==name or n.endswith('_'+name)]
  if len(candidates)!=1:raise ValueError(f'Ambiguous/missing eval camera {name}; available {names}')
  result.append(candidates[0])
 return list(dict.fromkeys(result))

def run(args):
 if args.code_root:
  # Script directory normally precedes PYTHONPATH; explicitly prefer sealed modules.
  sys.path.insert(0,str(args.code_root.resolve()/'LookCloser/scripts'));sys.path.insert(0,str(args.code_root.resolve()))
 import torch
 import cv2
 import yaml
 from nerfstudio.utils.eval_utils import eval_setup
 from probe_marshawn_occupancy import assert_consistent_bounds
 torch.set_num_threads(2)
 def configure(cfg):
  cfg.eval_checkpoint=args.checkpoint.resolve()
  if args.data:
   cfg.pipeline.datamanager.data=args.data;cfg.pipeline.datamanager.dataparser.data=args.data
  return cfg
 cfg,pipe,checkpoint,step=eval_setup(args.config,eval_num_rays_per_chunk=args.chunk_size,test_mode='val',update_config_callback=configure)
 if checkpoint.resolve()!=args.checkpoint.resolve():raise ValueError('eval_setup did not load requested checkpoint')
 model=pipe.model;model.current_train_step=step
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);runtime=state.get('runtime',{})
 if 'feature_reweighting_strength' in runtime:
  strength=float(runtime['feature_reweighting_strength']);model.config.feature_reweighting_strength=strength;model.field.feature_reweighting_strength=strength
 del state
 bounds=assert_consistent_bounds(model) if not args.skip_strict_bounds else dict(model_scene_aabb=model.scene_aabb.detach().cpu().tolist())
 dataset=pipe.datamanager.eval_dataset;selected=select_cameras(dataset,args.camera)
 if args.bbox and len(selected)!=1:raise ValueError('Native crop requires exactly one --camera')
 args.output.mkdir(parents=True,exist_ok=False);rows=[]
 for index in selected:
  name=Path(dataset.image_filenames[index]).stem;camera=copy.deepcopy(dataset.cameras[index:index+1]).to(model.device)
  source=dataset.get_image_float32(index).cpu().numpy()[...,:3];native_h,native_w=source.shape[:2]
  if (int(camera.width.item()),int(camera.height.item()))!=(native_w,native_h):raise ValueError('GT raster differs from calibrated dataset camera')
  if args.bbox:
   x0,y0,x1,y1=args.bbox
   if not 0<=x0<x1<=native_w or not 0<=y0<y1<=native_h:raise ValueError('Crop outside native calibrated raster')
   camera.cx-=x0;camera.cy-=y0;camera.width.fill_(x1-x0);camera.height.fill_(y1-y0);gt=source[y0:y1,x0:x1]
  else:
   factor=min(1.,args.max_width/native_w);camera.rescale_output_resolution(factor)
   w,h=int(camera.width.item()),int(camera.height.item());gt=cv2.resize(source,(w,h),interpolation=cv2.INTER_AREA)
  started=time.monotonic()
  with torch.inference_mode():outputs=model.get_outputs_for_camera(camera)
  arrays={k:v.detach().float().cpu().numpy() for k,v in outputs.items() if torch.is_tensor(v)}
  pred=arrays['rgb'][...,:3]
  if pred.shape!=gt.shape or not np.isfinite(pred).all():raise ValueError('Output RGB shape/nonfinite mismatch')
  opacity=arrays['accumulation'].squeeze(-1);depth=arrays['depth'].squeeze(-1)
  delta=np.abs(pred-gt);depth_image,depth_scale=depth_color(depth,opacity)
  display_gt=display_rgb(gt,args.display_transform);display_pred=display_rgb(pred,args.display_transform)
  display_delta=np.abs(display_pred-display_gt)
  images=[save_rgb(args.output/(name+'_'+suffix+'.png'),value) for suffix,value in [('gt',display_gt),('rgb',display_pred),('absdiff',display_delta),('opacity',np.repeat(opacity[...,None],3,axis=-1)),('depth',depth_image)]]
  w,h=images[0].size;panel=Image.new('RGB',(w*5,h+24),'black');draw=ImageDraw.Draw(panel)
  for j,(im,label) in enumerate(zip(images,['GT','prediction','absolute RGB difference','opacity','depth visualization'])):panel.paste(im,(j*w,24));draw.text((j*w+4,4),label,fill='white')
  panel.save(args.output/(name+'_review.jpg'),quality=90)
  np.savez_compressed(args.output/(name+'_buffers.npz'),rgb=pred,gt=gt,absolute_difference=delta,depth=depth,opacity=opacity,**({'samples':arrays['num_samples_per_ray']} if 'num_samples_per_ray' in arrays else {}))
  # Preview differences are diagnostic; final metrics use native all-eval selection.
  mse=float(np.mean((display_pred-display_gt)**2));row=dict(camera=name,dataset_index=index,source=str(dataset.image_filenames[index]),source_sha256=sha(dataset.image_filenames[index]),native_size=[native_w,native_h],render_size=[w,h],bbox=args.bbox,seconds=time.monotonic()-started,mean_absolute_difference=float(delta.mean()),preview_psnr=float(-10*np.log10(max(mse,1e-12))),preview_metric_space=args.display_transform,mean_opacity=float(opacity.mean()),depth_visualization=depth_scale,review_image=name+'_review.jpg',buffers=name+'_buffers.npz')
  rows.append(row);(args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(selected)))+'\n');print(json.dumps(dict(camera=name,size=[w,h],seconds=row['seconds'])),flush=True)
 import inspect
 modules={}
 for obj in [type(model),type(model.field),type(pipe),type(cfg.pipeline.datamanager.dataparser),cfg.pipeline.datamanager.dataparser._target,assert_consistent_bounds]:
  path=inspect.getsourcefile(obj)
  if path:modules[path]=sha(path)
 data=Path(cfg.pipeline.datamanager.dataparser.data);transforms=data if data.is_file() else data/'transforms.json';normal=pipe.datamanager.train_dataparser_outputs
 parser_yaml=yaml.dump(cfg.pipeline.datamanager.dataparser,sort_keys=True);(args.output/'dataparser_config.yml').write_text(parser_yaml)
 receipt=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config=str(args.config),config_sha256=sha(args.config),step=step,script_sha256=sha(__file__),runtime_sources=modules,transforms_sha256=sha(transforms),normalization=dict(transform=normal.dataparser_transform.tolist(),scale=float(normal.dataparser_scale)),live_bounds=bounds,bounds_sha256=hashlib.sha256(json.dumps(bounds,sort_keys=True).encode()).hexdigest(),dataparser_config_sha256=hashlib.sha256(parser_yaml.encode()).hexdigest(),native_model_api='model.get_outputs_for_camera',gt_compositing=False,rows=rows,limitations=['Low-resolution preview metrics are diagnostic, not native all-eval checkpoint metrics.','Depth display uses per-view2–98% scaling on opacity>.05; raw buffers retained.','Camera frustum scaling/cropping only; learned field, occupancy and AABB unchanged.'])
 receipt['display_transform']=args.display_transform
 receipt['buffer_color_space']='model-native, no display transform applied'
 (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(output=str(args.output),views=len(rows),step=step)))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['config','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--data',type=Path);p.add_argument('--code-root',type=Path);p.add_argument('--max-width',type=int,default=640);p.add_argument('--chunk-size',type=int,default=2048);p.add_argument('--camera',action='append');p.add_argument('--bbox',type=int,nargs=4);p.add_argument('--skip-strict-bounds',action='store_true')
 p.add_argument('--display-transform',choices=['identity','linear-reinhard-srgb'],default='identity');a=p.parse_args()
 if a.max_width<=0 or a.chunk_size<=0:p.error('positive max-width/chunk-size required')
 run(a)
