#!/usr/bin/env python3
"""Train-only native JPEG person matting, qualified by separate hair semantics."""
from pathlib import Path
import argparse,json,os,time
import cv2,numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from dec5_hair import VALIDATION,TEST
from add_dec5_neural_fallback import inverse_display
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['data','semantics','cameras','model','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--offset',type=int,default=0);p.add_argument('--stride',type=int,default=1);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(24*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 from transformers import AutoModelForImageSegmentation
 from pymatting import estimate_foreground_ml
 model=AutoModelForImageSegmentation.from_pretrained(str(a.model),trust_remote_code=True,local_files_only=True).eval().cuda().half();prep=transforms.Compose([transforms.Resize((2048,2048)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);d=json.load(open(a.data/'transforms.json'));train={f['physical_camera']:f for f in d['frames'] if f['file_path'] in d['train_filenames']};frames=json.load(open(a.cameras))['frames'][a.offset::a.stride]
 if len(train)!=59 or not frames or any(f['physical_camera'] not in train or f['physical_camera'] in VALIDATION+TEST for f in frames):raise ValueError('Only source train cameras may be matted')
 request=dict(data_sha256=sha(a.data/'transforms.json'),camera_manifest_sha256=sha(a.cameras),model_weights_sha256=sha(a.model/'model.safetensors'),conversion_receipt_sha256=sha(a.model/'conversion_receipt.json'),selected_cameras=[f['physical_camera'] for f in frames],input_resolution=[2048,2048],person_alpha_not_hair_semantics=True,foreground='PyMatting on bounded linear-sRGB JPEG then inverse fixed tonemap; float32 approximate foreground, not recovered HDR',uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__));atomic_json(a.output/'request.json',request);rows=[]
 for f in frames:
  start=time.time();name=f['physical_camera'];sf=train[name];image_path=a.data/sf['file_path'];image=cv2.imread(str(image_path))[...,::-1];semantic_path=a.semantics/(name+'.npz');prob=np.load(semantic_path)['probability'][...,0].astype(np.float32)/65535;portrait=np.rot90(image);hair=np.rot90(prob);yy,xx=np.nonzero(hair>.25)
  if not len(xx):raise ValueError('Empty hair crop')
  sx,sy=portrait.shape[1]/hair.shape[1],portrait.shape[0]/hair.shape[0];x0=max(0,int((xx.min()-96)*sx));x1=min(portrait.shape[1],int((xx.max()+97)*sx));y0=max(0,int((yy.min()-96)*sy));y1=min(portrait.shape[0],int((yy.max()+257)*sy));crop=np.ascontiguousarray(portrait[y0:y1,x0:x1]);inputs=prep(Image.fromarray(crop))[None].cuda().half()
  with torch.inference_mode():alpha=F.interpolate(model(inputs)[-1].float(),size=crop.shape[:2],mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
  rgb=crop.astype(np.float32)/255;linear_srgb=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);estimate=estimate_foreground_ml(linear_srgb.astype(np.float64),alpha.astype(np.float64),n_small_iterations=5,n_big_iterations=2).astype(np.float32).clip(0,1);foreground=estimate/np.maximum(1-estimate,1e-6)/10.570320292648677;linear=inverse_display(rgb);scale=1.;foreground[alpha>=.995]=linear[alpha>=.995]
  if not np.isfinite(foreground).all():raise ValueError('Nonfinite foreground estimate')
  path=a.output/(name+'.npz');np.savez_compressed(path,alpha=alpha.astype(np.float16),foreground=foreground.astype(np.float32),portrait_crop=np.array([x0,y0,x1,y1]),original_shape=np.array(image.shape),linear_scale=scale)
  rows.append(dict(camera=name,sha256=sha(path),source_rgb_sha256=sha(image_path),semantic_sha256=sha(semantic_path),seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),clipped_fraction=float((linear>scale).mean())));atomic_json(a.output/'progress.json',dict(completed=len(rows),total=len(frames),camera=name))
 atomic_json(a.output/'complete.json',dict(records=rows,request_sha256=sha(a.output/'request.json'),**request));print('Native source matting completed',len(rows),flush=True)
if __name__=='__main__':main()
