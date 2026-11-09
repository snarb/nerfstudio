#!/usr/bin/env python3
"""Train-only calibrated background votes in normalized 3D; no learned geometry."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,torch,yaml
import torch.nn.functional as F
from dec5_hair import cv_pose,VALIDATION,TEST
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['data','training','semantics','sparse','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--resolution',type=int,default=256);a=p.parse_args()
 if not 32<=a.resolution<=512:raise ValueError('Grid resolution outside bounded experiment range')
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 request=json.load(open(a.training/'request.json'))
 if request['data_sha256']!=sha(a.data/'transforms.json'):raise ValueError('Normalization dataset mismatch')
 cfg=yaml.load(next(a.training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.dataparser.data=a.data;parsed=cfg.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train');payload=json.load(open(a.data/'transforms.json'));by={Path(f['file_path']).name:f for f in payload['frames']};frames=[by[Path(p).name] for p in parsed.image_filenames]
 if len(frames)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):raise ValueError('Train59 only')
 raw=np.asarray([f['transform_matrix'] for f in frames]);normal=parsed.cameras.camera_to_worlds.numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);offset=normal[0,:3,3]-scale*R@raw[0,:3,3]
 if np.max(np.linalg.norm(scale*raw[:,:3,3]@R.T+offset-normal[:,:3,3],axis=-1))>1e-4:raise ValueError('Normalization alignment failed')
 bounds=parsed.scene_box.aabb.numpy();masks=[];rotations=[];centers=[];intrinsics=[];records=[]
 for f in frames:
  name=f['physical_camera'];path=a.semantics/(name+'.npz');v=np.load(path)['probability'];mask=cv2.erode((v[...,2]>round(.98*65535)).astype(np.uint8),np.ones((33,33),np.uint8))
  if any(abs(f.get(k,0))>1e-12 for k in ['k1','k2','p1','p2']):raise ValueError('Expected calibrated zero-distortion data')
  if mask.shape!=(1080,1260) or mask.shape!=(f['h'],f['w']):raise ValueError('Use the exact HD fit raster for conservative empty-space evidence')
  masks.append(mask);pose=cv_pose(f);rotations.append(R@pose[:3,:3]);centers.append(scale*R@pose[:3,3]+offset);intrinsics.append([f[k] for k in ['fl_x','fl_y','cx','cy']]);records.append(dict(camera=name,semantic_sha256=sha(path),known_background_pixels=int(mask.sum())))
 masks=torch.as_tensor(np.stack(masks),device='cuda',dtype=torch.float32)[:,None];rot=torch.as_tensor(np.stack(rotations),device='cuda',dtype=torch.float32);cent=torch.as_tensor(np.stack(centers),device='cuda',dtype=torch.float32);K=torch.as_tensor(intrinsics,device='cuda',dtype=torch.float32);camera_offset=torch.einsum('cd,cdk->ck',cent,rot)
 @torch.no_grad()
 def votes(points):
  cam=torch.einsum('nd,cdk->cnk',points,rot)-camera_offset[:,None,:];z=cam[...,2];uv=cam[...,:2]/z.clamp_min(1e-8)[...,None]*K[:,None,:2]+K[:,None,2:];valid=(z>0)&(uv[...,0]>=.5)&(uv[...,0]<=1259.5)&(uv[...,1]>=.5)&(uv[...,1]<=1079.5);grid=2*uv/torch.as_tensor([1260.,1080.],device='cuda')-1;sample=F.grid_sample(masks,grid[:,None],mode='nearest',padding_mode='zeros',align_corners=False)[:,0,0];return ((sample>.5)&valid).sum(0).byte(),valid.sum(0).byte()
 resolution=a.resolution;total=resolution**3;bg=np.empty(total,np.uint8);seen=np.empty(total,np.uint8);lo,hi=torch.as_tensor(bounds,device='cuda',dtype=torch.float32);span=hi-lo
 for begin in range(0,total,32768):
  ids=torch.arange(begin,min(begin+32768,total),device='cuda');ijk=torch.stack((ids%resolution,(ids//resolution)%resolution,ids//(resolution**2)),1);points=lo+(ijk+.5)/resolution*span;b,v=votes(points);bg[begin:begin+len(ids)]=b.cpu().numpy();seen[begin:begin+len(ids)]=v.cpu().numpy()
  if begin%(1<<20)==0:atomic_json(a.output/'progress.json',dict(completed=begin+len(ids),total=total))
 source=np.load(a.sparse);hair=np.zeros(len(source['xyz']),int)
 for i,name in enumerate(source['camera_names']):
  select=np.nonzero(source['observation_camera']==i)[0];prob=np.load(a.semantics/(str(name)+'.npz'))['probability'];uv=source['uv'][select];x=np.clip(uv[:,0].astype(int),0,1259);y=np.clip(uv[:,1].astype(int),0,1079);pick=select[prob[y,x,0]>round(.9*65535)];hair+=np.bincount(source['observation_point'][pick],minlength=len(hair))
 xyz=scale*source['xyz']@R.T+offset;reliable=(hair>=2)&(xyz>=bounds[0]).all(-1)&(xyz<=bounds[1]).all(-1);audit={}
 for threshold in [3,8]:
  points=torch.as_tensor(xyz[reliable],device='cuda',dtype=torch.float32);b,v=votes(points);carved=(b>=threshold)&(v>=8);audit[str(threshold)]=dict(reliable_train_hair_points=int(reliable.sum()),raw_carved_points=int(carved.sum()),raw_carved_fraction=float(carved.float().mean()) if len(carved) else None,voxel_carved_fraction=float(((bg>=threshold)&(seen>=8)).mean()))
 out=a.output/'votes.npz';np.savez_compressed(out,background_votes=bg.reshape((resolution,)*3),visible_votes=seen.reshape((resolution,)*3),bounds=bounds,rotation=R,translation=offset,scale=scale)
 atomic_json(a.output/'complete.json',dict(frame_id=payload['frame_id'],votes_sha256=sha(out),data_sha256=sha(a.data/'transforms.json'),sparse_sha256=sha(a.sparse),records=records,resolution=resolution,grid_axes='z,y,x; normalized voxel centers',normalization=dict(rotation=R.tolist(),translation=offset.tolist(),scale=float(scale)),training_hair_witness_audit=audit,known_background='train probability >.98, eroded 16 HD pixels; no target RGB/masks',uses_validation_rgb=False,uses_test_rgb=False,depth_or_alpha_ground_truth=False,implementation_sha256=sha(__file__)))
 print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
