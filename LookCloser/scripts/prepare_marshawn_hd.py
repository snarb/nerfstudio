#!/usr/bin/env python3
"""Prepare hash-bound, undistorted HD Marshawn directly from Log3G10 TIFFs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text, read_images_text
from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses

RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
ROOT=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local')
EVAL={'HC','DC','KA'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def npm(xy):
 a=np.array([[x/y,1,(1-x-y)/y] for x,y in xy]).T
 return a*np.linalg.solve(a,[.3127/.329,1,(1-.3127-.329)/.329])[None,:]
M=np.linalg.solve(npm([(.64,.33),(.30,.60),(.15,.06)]),npm([(.780308,.304253),(.121595,1.493994),(.095612,-.084589)]))
def display(raw):
 x=raw.astype(np.float32)/65535
 x=np.maximum(np.expm1(x*(np.log(10)/.224282))/155.975327-.01,0)
 x=np.maximum(x@M.T,0);x=x/(1+x)
 return np.rint(np.clip(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055),0,1)*255).astype(np.uint8)
def prepare(out,workers):
 cv2.setNumThreads(1);torch.set_num_threads(2)
 cameras=read_cameras_text(RIG/'cameras.txt');images=read_images_text(RIG/'images.txt')
 assert len(images)==66
 poses=[];names=[]
 for im in images.values():
  w=np.eye(4);w[:3,:3]=im.qvec2rotmat();w[:3,3]=im.tvec;p=np.linalg.inv(w);p[:3,1:3]*=-1;poses.append(p);names.append(im.name.split('/')[0])
 poses=torch.tensor(np.array(poses),dtype=torch.float32);train=[i for i,n in enumerate(names) if n not in EVAL]
 normalized,transform=auto_orient_and_center_poses(poses[train],method='up',center_method='focus')
 scale=1/float(normalized[:,:3,3].abs().max());T=torch.eye(4);T[:3]=transform
 normalized=T@poses;normalized[:,:3,3]*=scale
 (out/'images').mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True)
 def one(item):
  idx,im=item;name=names[idx];cam=cameras[im.camera_id];src=ROOT/name/f'{name}.000408.tif';receipt=out/'receipts'/f'{name}.json'
  if receipt.exists():
   old=json.loads(receipt.read_text())
   if old.get('script_sha256')==sha(__file__) and sha(out/old['row']['file_path'])==old['output_sha256']:
    print('cached',name,flush=True);return old['row'],old
  raw=cv2.imread(str(src),cv2.IMREAD_UNCHANGED)
  if raw is None or raw.shape!=(4320,8192,3) or raw.dtype!=np.uint16:raise ValueError(str(src))
  fx,fy,cx,cy,*d=cam.params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  w,h=1920,round(4320*1920/8192);sx=w/8192;sy=h/4320;newK=K.copy();newK[0]*=sx;newK[1]*=sy
  # COLMAP pixel centres use .5; OpenCV remap indices use zero.
  srcK=K.copy();srcK[0,2]-=.5;srcK[1,2]-=.5
  dstK=newK.copy();dstK[0,2]-=.5;dstK[1,2]-=.5
  mx,my=cv2.initUndistortRectifyMap(srcK,np.array(d),None,dstK,(w,h),cv2.CV_32FC1)
  # Area prefilter display RGB before the single geometric remap.
  # Decode strips to avoid multiple full-sensor float temporaries.
  rgb=np.empty(raw.shape,np.uint8)
  for y in range(0,4320,128):rgb[y:y+128]=display(raw[y:y+128,:,::-1])
  small=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
  # Maps into resized source raster (integer-centre convention).
  xsmall=(mx+.5)*sx-.5;ysmall=(my+.5)*sy-.5
  hd=cv2.remap(small,xsmall,ysmall,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
  valid=((mx>=0)&(mx<=8191)&(my>=0)&(my<=4319)).astype(np.uint8)*255
  prefix='eval' if name in EVAL else 'train';rel=f'images/{prefix}_{name}.png';mask=f'masks/{name}.png'
  Image.fromarray(hd).save(out/rel);Image.fromarray(valid).save(out/mask)
  row=dict(file_path=rel,mask_path=mask,physical_camera=name,w=w,h=h,fl_x=float(newK[0,0]),fl_y=float(newK[1,1]),cx=float(newK[0,2]),cy=float(newK[1,2]),k1=0.,k2=0.,p1=0.,p2=0.,transform_matrix=normalized[idx].tolist())
  audit=dict(row=row,source=str(src),source_sha256=sha(src),output_sha256=sha(out/rel),mask_sha256=sha(out/mask),script_sha256=sha(__file__))
  write(receipt,audit);print('prepared',name,flush=True);return row,audit
 with ThreadPoolExecutor(max_workers=workers) as pool:result=list(pool.map(one,enumerate(images.values())))
 rows=[x[0] for x in result]
 meta=dict(camera_model='OPENCV',orientation_override='none',frames=rows,train_filenames=[r['file_path'] for r in rows if r['physical_camera'] not in EVAL],val_filenames=[r['file_path'] for r in rows if r['physical_camera'] in EVAL],test_filenames=[r['file_path'] for r in rows if r['physical_camera'] in EVAL])
 write(out/'transforms.json',meta);write(out/'preparation.json',dict(frame=1660,train=63,eval=3,normalization=T.tolist(),scale=scale,color='Log3G10/RWG -> linear Rec709 gain1 -> Reinhard -> sRGB',rig_sha256={n:sha(RIG/n) for n in ['cameras.txt','images.txt']},script_sha256=sha(__file__)))
 sheet=Image.new('RGB',(1152,650));draw=ImageDraw.Draw(sheet)
 for i,name in enumerate(['HC','DC','AA','GA','KA','OA']):
  r=next(r for r in rows if r['physical_camera']==name);im=Image.open(out/r['file_path']);im.thumbnail((384,300));x=i%3*384;y=i//3*325;sheet.paste(im,(x,y+20));draw.text((x,y),name,fill='white')
 sheet.save(out/'initial_contact.jpg')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();prepare(a.output,a.workers)
