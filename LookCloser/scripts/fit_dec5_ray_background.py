#!/usr/bin/env python3
"""One fixed train-only ray-space background fit; never a foreground rendering."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np,torch
from dec5_hair import cv_pose,VALIDATION,TEST
from dec5_ray_background import RayBackgroundNetwork
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['data','serving-data','semantics','cameras','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--steps',type=int,default=5000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(42);rng=np.random.RandomState(42)
 d=json.load(open(a.data/'transforms.json'));native=json.load(open(a.serving_data/'transforms.json'));by={f['physical_camera']:f for f in native['frames']};frames=[f for f in d['frames'] if f['file_path'] in d['train_filenames']]
 if len(frames)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):raise ValueError('Train59 required')
 center=np.mean([cv_pose(f)[:3,3] for f in frames],axis=0).astype(np.float32);radius=max(float(np.linalg.norm(cv_pose(f)[:3,3]-center)) for f in frames)
 samples=[];records=[]
 for f in frames:
  name=f['physical_camera'];other=by[name];np.testing.assert_allclose(f['transform_matrix'],other['transform_matrix'],atol=1e-10,rtol=0);sx,sy=other['w']/f['w'],other['h']/f['h'];np.testing.assert_allclose([f['fl_x']*sx,f['fl_y']*sy,f['cx']*sx,f['cy']*sy],[other[k] for k in ['fl_x','fl_y','cx','cy']],atol=1e-5,rtol=0)
  semantic=a.semantics/(name+'.npz');v=np.load(semantic)['probability'].astype(np.float32)/65535;mask=cv2.erode((v[...,2]>.98).astype(np.uint8),np.ones((33,33),np.uint8))>0;y,x=np.nonzero(mask)
  if len(x)<4352:raise ValueError('Insufficient confident BG samples '+name)
  pick=rng.choice(len(x),4352,replace=False);x,y=x[pick],y[pick];pose=cv_pose(f).astype(np.float32);direction=np.stack(((x+.5-f['cx'])/f['fl_x'],(y+.5-f['cy'])/f['fl_y'],np.ones(len(x))),-1).astype(np.float32)@pose[:3,:3].T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True);origin=np.broadcast_to((pose[:3,3]-center)/radius,direction.shape);rgb=cv2.imread(str(a.data/f['file_path']))[y,x,::-1].astype(np.float32)/255
  samples.append(np.concatenate((origin,direction,rgb),-1));records.append(dict(camera=name,rgb_sha256=sha(a.data/f['file_path']),semantic_sha256=sha(semantic),eligible_pixels=int(mask.sum())))
 pool=np.stack(samples);train=torch.as_tensor(pool[:,:4096].reshape(-1,9),device='cuda');hold=torch.as_tensor(pool[:,4096:].reshape(-1,9),device='cuda');model=RayBackgroundNetwork().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6);started=time.time();history=[]
 atomic_json(a.output/'request.json',dict(data_sha256=sha(a.data/'transforms.json'),allowed_serving_dataset_sha256=sha(a.serving_data/'transforms.json'),records=records,seed=42,steps=a.steps,batch=4096,source='confident train BG probability >.98, eroded 16 HD pixels',uses_validation_rgb=False,uses_test_rgb=False,checkpoint_selection='fixed final step; no target or holdout selection',artifact_kind='background RGB function only; not foreground reconstruction'))
 for step in range(a.steps):
  batch=train[torch.randint(len(train),(4096,),device='cuda')];prediction=model(batch[:,:3],batch[:,3:6]);objective=(prediction-batch[:,6:]).square().mean();optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step()
  if (step+1)%1000==0 or step+1==a.steps:
   with torch.no_grad():mse=(model(hold[:,:3],hold[:,3:6])-hold[:,6:]).square().mean();row=dict(step=step+1,train_camera_withheld_pixel_psnr=float(-10*torch.log10(mse)),elapsed_seconds=time.time()-started)
   history.append(row);atomic_json(a.output/'progress.json',row)
 model.eval();torch.save(dict(state_dict=model.state_dict(),center=center.tolist(),radius=radius,origin_levels=1,direction_levels=5,width=64,step=a.steps),a.output/'background.pt');atomic_json(a.output/'history.json',history)
 camera_records=[]
 for f in json.load(open(a.cameras))['frames']:
  h,w=f['h'],f['w'];image=np.empty((h*w,3),np.uint8);pose=torch.as_tensor(cv_pose(f),device='cuda',dtype=torch.float32);origin=(pose[:3,3]-torch.as_tensor(center,device='cuda'))/radius
  for begin in range(0,h*w,65536):
   flat=torch.arange(begin,min(begin+65536,h*w),device='cuda');x=flat%w;y=flat//w;direction=torch.stack(((x+.5-f['cx'])/f['fl_x'],(y+.5-f['cy'])/f['fl_y'],torch.ones_like(x)),1).float()@pose[:3,:3].T;direction=torch.nn.functional.normalize(direction,dim=-1)
   with torch.no_grad():rgb=model(origin.expand(len(flat),3),direction)
   image[begin:begin+len(flat)]=torch.round(rgb.clamp(0,1)*255).byte().cpu().numpy()
  path=a.output/(f['physical_camera']+'.png');cv2.imwrite(str(path),image.reshape(h,w,3)[...,::-1]);camera_records.append(dict(camera=f['physical_camera'],png_sha256=sha(path)))
 atomic_json(a.output/'complete.json',dict(records=camera_records,checkpoint_sha256=sha(a.output/'background.pt'),request_sha256=sha(a.output/'request.json'),camera_manifest_sha256=sha(a.cameras),uses_validation_rgb=False,uses_test_rgb=False,elapsed_seconds=time.time()-started,parameter_count=sum(p.numel() for p in model.parameters())))
 print('Background-only train59 fit and known-pose prediction complete',flush=True)
if __name__=='__main__':main()
