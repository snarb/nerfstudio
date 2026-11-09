#!/usr/bin/env python3
"""Geometry-only train background support and opt-in <=6000step ray MLP fit.

prepare never trains. fit is a separate explicit command. Support excludes the
projected loose subject box plus protective band; it does not label holes or
free space inside that projected volume.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from audit_marshawn_geometry import project_points

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def mask_for_row(row,box,valid,band=16):
 lo,hi=np.asarray(box);xyz=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]]);uv,front=project_points(row,xyz)
 if not front.all():raise ValueError('Loose subject box crosses camera plane; conservative projection requires explicit handling')
 hull=cv2.convexHull((uv-.5).astype(np.float32)).reshape(-1,2);excluded=np.zeros((row['h'],row['w']),np.uint8)
 # Round outwards through a1px protective raster dilation, then requested band.
 cv2.fillConvexPoly(excluded,np.rint(hull).astype(np.int32),1);excluded=cv2.dilate(excluded,np.ones((2*band+3,2*band+3),np.uint8))
 return valid&~excluded.astype(bool),hull

def prepare(a):
 cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=False);data=a.data;m=json.loads((data/'transforms.json').read_text());b=json.loads(a.boxes.read_text());assert b['normalization_sha256']==sha(data/'preparation.json')
 train=set(m['train_filenames']);frames=[r for r in m['frames'] if r['file_path'] in train]
 if len(frames)!=63 or train.intersection(m['val_filenames']):raise ValueError('Exact63train +disjoint eval required')
 box=np.array(b['scopes']['subject']['loose']);(a.output/'masks').mkdir();records=[];tiles=[]
 witnesses=np.load(a.boxes.parent/'subject_witnesses.npz')['xyz']
 for row in frames:
  valid=np.array(Image.open(data/row['mask_path']))>0;eligible,hull=mask_for_row(row,box,valid,a.band)
  if eligible.sum()<4352:raise ValueError('Insufficient geometry-only BG support '+row['physical_camera'])
  uv,front=project_points(row,witnesses);inside=front&(uv[:,0]>=0)&(uv[:,0]<row['w'])&(uv[:,1]>=0)&(uv[:,1]<row['h']);x=np.floor(uv[inside,0]).astype(int);y=np.floor(uv[inside,1]).astype(int)
  if eligible[y,x].any():raise AssertionError('Known subject witness admitted into BG fitting support')
  name=row['physical_camera'];maskfile='masks/'+name+'.png';Image.fromarray(eligible.astype(np.uint8)*255).save(a.output/maskfile)
  records.append(dict(camera=name,file_path=row['file_path'],source_sha256=sha(data/row['file_path']),source_mask_sha256=sha(data/row['mask_path']),support_mask=maskfile,support_mask_sha256=sha(a.output/maskfile),eligible_pixels=int(eligible.sum()),eligible_fraction=float(eligible.mean()),subject_witnesses_checked=int(inside.sum()),projected_hull_pixel_indices=hull.tolist()))
  source=Image.open(data/row['file_path']).convert('RGB');rgb=np.array(source);overlay=rgb.copy();overlay[eligible]=np.rint(.65*rgb[eligible]+.35*np.array([0,255,50])).astype(np.uint8);overlay[~eligible&valid]=np.rint(.65*rgb[~eligible&valid]+.35*np.array([255,0,150])).astype(np.uint8)
  tile=Image.new('RGB',(640,190));d=ImageDraw.Draw(tile)
  for dx,im in [(0,source),(320,Image.fromarray(overlay))]:im.thumbnail((320,169));tile.paste(im,(dx,21))
  d.text((4,3),f'{name}: source | greenBGfit / magenta excluded',fill='white');tiles.append(tile)
 pages=[]
 for start in range(0,len(tiles),21):
  canvas=Image.new('RGB',(1920,1330))
  for j,t in enumerate(tiles[start:start+21]):canvas.paste(t,(j%3*640,j//3*190))
  name=f'support_contact_{start//21+1:02d}.jpg';canvas.save(a.output/name,quality=90);pages.append(name)
 request=dict(kind='geometry_only_train_ray_background_support',data=str(data),data_sha256=sha(data/'transforms.json'),allowed_serving_dataset_sha256=sha(data/'transforms.json'),normalization_sha256=sha(data/'preparation.json'),boxes_sha256=sha(a.boxes),subject_box=box.tolist(),protective_band_hd_pixels=a.band+1,script_sha256=sha(__file__),train_cameras=63,uses_validation_rgb=False,uses_test_rgb=False,semantic_masks_used=False,records=records,pages=pages,limitations=['Support outside projected loose box is conservative fitting support, not a label of every background pixel','These rays miss the compact AABB: empty-space loss here alone cannot supervise holes inside it','Frozen ray RGB is not depth or geometry; heldout BG quality must pass separately before foreground attachment'])
 write(a.output/'request.json',request);print(json.dumps(dict(output=str(a.output),train=63,min_eligible=min(r['eligible_fraction'] for r in records),max_eligible=max(r['eligible_fraction'] for r in records),fit_started=False)))

def fit(a):
 if not 1<=a.steps<=6000:raise ValueError('Background fit hard cap6000optimizer steps')
 import torch
 from dec5_ray_background import RayBackgroundNetwork
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;cv2.setNumThreads(1)
 request=json.loads((a.support/'request.json').read_text());data=Path(request['data']);meta=json.loads((data/'transforms.json').read_text());assert sha(data/'transforms.json')==request['data_sha256'] and request['train_cameras']==63 and request['uses_validation_rgb'] is False and request['uses_test_rgb'] is False
 by={r['physical_camera']:r for r in meta['frames'] if r['file_path'] in meta['train_filenames']};assert set(by)=={r['camera'] for r in request['records']};center=np.mean([np.array(r['transform_matrix'])[:3,3] for r in by.values()],axis=0);radius=max(np.linalg.norm(np.array(r['transform_matrix'])[:3,3]-center) for r in by.values());rng=np.random.RandomState(42);pool=[]
 for rec in request['records']:
  row=by[rec['camera']];assert sha(data/row['file_path'])==rec['source_sha256'] and sha(a.support/rec['support_mask'])==rec['support_mask_sha256'];mask=np.array(Image.open(a.support/rec['support_mask']))>0;y,x=np.nonzero(mask);take=rng.choice(len(x),4352,replace=False);x,y=x[take],y[take];pose=np.array(row['transform_matrix']);direction=np.stack(((x+.5-row['cx'])/row['fl_x'],-(y+.5-row['cy'])/row['fl_y'],-np.ones(len(x))),-1)@pose[:3,:3].T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True);origin=np.broadcast_to((pose[:3,3]-center)/radius,direction.shape);rgb=np.array(Image.open(data/row['file_path']).convert('RGB'))[y,x]/255;pool.append(np.concatenate([origin,direction,rgb],axis=1))
 a.output.mkdir(parents=True,exist_ok=False);pool=np.stack(pool);device=torch.device(a.device);training=torch.tensor(pool[:,:4096].reshape(-1,9),device=device,dtype=torch.float32);monitor=torch.tensor(pool[:,4096:].reshape(-1,9),device=device,dtype=torch.float32);model=RayBackgroundNetwork(1,5,64).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6);start=time.monotonic();history=[]
 for step in range(a.steps):
  batch=training[torch.randint(len(training),(4096,),device=device)];pred=model(batch[:,:3],batch[:,3:6]);objective=(pred-batch[:,6:]).square().mean();optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step()
  if (step+1)%1000==0 or step+1==a.steps:
   with torch.no_grad():mse=(model(monitor[:,:3],monitor[:,3:6])-monitor[:,6:]).square().mean()
   progress=dict(step=step+1,train_camera_withheld_pixel_psnr=float(-10*torch.log10(mse)),seconds=time.monotonic()-start);history.append(progress);write(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
 model.eval();model.requires_grad_(False);torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},center=center.tolist(),radius=float(radius),origin_levels=1,direction_levels=5,width=64,step=a.steps),a.output/'background.pt')
 write(a.output/'request.json',dict(data_sha256=request['data_sha256'],allowed_serving_dataset_sha256=request['allowed_serving_dataset_sha256'],uses_validation_rgb=False,uses_test_rgb=False,train_cameras=63,support_request_sha256=sha(a.support/'request.json'),script_sha256=sha(__file__),network_source_sha256=sha(Path(__file__).with_name('dec5_ray_background.py')),steps=a.steps,batch=4096,seed=42,checkpoint_selection='fixed final step; monitor does not select weights',normalization='prepared OpenGL world origins centered/radius-scaled; normalized world ray directions',geometry_only_masks=True))
 write(a.output/'complete.json',dict(checkpoint_sha256=sha(a.output/'background.pt'),train_cameras=63,steps=a.steps,frozen=True,component_gate_passed=False,reason='Train fit only; independent heldout BG and boundary quality still required',history=history));print(json.dumps(dict(output=str(a.output),steps=a.steps,component_gate_passed=False)))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 q=sub.add_parser('prepare');q.add_argument('--data',type=Path,required=True);q.add_argument('--boxes',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--band',type=int,default=16)
 q=sub.add_parser('fit');q.add_argument('--support',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--steps',type=int,default=5000);q.add_argument('--device',default='cuda')
 a=p.parse_args()
 if a.command=='prepare':
  if a.band<0:p.error('Nonnegative protective band required')
  prepare(a)
 else:fit(a)
