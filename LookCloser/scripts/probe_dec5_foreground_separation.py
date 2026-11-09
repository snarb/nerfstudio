#!/usr/bin/env python3
"""Frozen head-compartment mass on excluded-camera semantic rays.

Train SfM + train semantics define a common box. Excluded-camera semantics only
sample the diagnostic rays after fitting. This is not alpha/depth ground truth.
Uniform midpoint integration bypasses occupancy identically for both fields.
"""
from pathlib import Path
import argparse,json
import numpy as np,torch,yaml
from dec5_hair import cv_pose
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['data','parent-training','parent-checkpoint','candidate-training','candidate-checkpoint','sparse-hd','semantics-hd','reference','cameras','roi','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 source=np.load(a.sparse_hd);fg=np.zeros(len(source['xyz']),int)
 for i,name in enumerate(source['camera_names']):
  select=np.nonzero(source['observation_camera']==i)[0];v=np.load(a.semantics_hd/(str(name)+'.npz'))['probability'];uv=source['uv'][select];x=np.clip(uv[:,0].astype(int),0,v.shape[1]-1);y=np.clip(uv[:,1].astype(int),0,v.shape[0]-1);good=np.maximum(v[y,x,0],v[y,x,1])>32767;fg+=np.bincount(source['observation_point'][select[good]],minlength=len(fg))
 xyz=source['xyz'][fg>=2]
 if len(xyz)<100:raise ValueError('Too few train head points')
 lo,hi=np.quantile(xyz,[.005,.995],axis=0);span=hi-lo;lo-=.15*span;hi+=.15*span
 rng=np.random.RandomState(42);origins=[];directions=[];nears=[];fars=[];labels=[];names=[];coverage=[]
 for f in json.load(open(a.cameras))['frames']:
  name=f['physical_camera'];prob=np.load(a.reference/(name+'.npz'))['probability'].astype(float)/65535;roi=np.load(a.roi/(name+'.npz'))['roi'];pose=cv_pose(f)
  for label,mask in [('hair',prob[...,0]>.9),('background',prob[...,2]>.95)]:
   y,x=np.nonzero(mask&roi);pick=rng.choice(len(x),min(16384,len(x)),replace=False);x,y=x[pick],y[pick];d=np.stack(((x+.5-f['cx'])/f['fl_x'],(y+.5-f['cy'])/f['fl_y'],np.ones(len(x))),-1)@pose[:3,:3].T;d/=np.linalg.norm(d,axis=-1,keepdims=True);o=pose[:3,3];safe=np.where(np.abs(d)>1e-12,d,1e-12);t0=(lo-o)/safe;t1=(hi-o)/safe;near=np.maximum(np.minimum(t0,t1).max(-1),0);far=np.maximum(t0,t1).min(-1);valid=np.nonzero(far>near)[0];coverage.append(dict(camera=name,label=label,sampled_pixels=len(x),intersect_head_box=len(valid)))
   take=rng.choice(valid,min(512,len(valid)),replace=False);origins.append(np.broadcast_to(o,(len(take),3)));directions.append(d[take]);nears.append(near[take]);fars.append(far[take]);labels.extend([label]*len(take));names.extend([name]*len(take))
 origins=np.concatenate(origins);directions=np.concatenate(directions);nears=np.concatenate(nears);fars=np.concatenate(fars);arrays=dict(origins=origins,directions=directions,nears=nears,fars=fars,labels=np.array(labels),cameras=np.array(names));results={};fractions=(torch.arange(2048,device='cuda',dtype=torch.float32)+.5)/2048
 for label,training,checkpoint in [('parent',a.parent_training,a.parent_checkpoint),('candidate',a.candidate_training,a.candidate_checkpoint)]:
  request=json.load(open(training/'request.json'))
  if request['data_sha256']!=sha(a.data/'transforms.json'):raise ValueError('Field dataset mismatch')
  cfg=yaml.load(next(training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data;cfg.pipeline.datamanager.cache_images_type='uint8';pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(checkpoint,map_location='cpu',weights_only=False);pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
  payload=json.load(open(a.data/'transforms.json'));by={Path(f['file_path']).name:f for f in payload['frames']};raw=np.array([by[Path(p).name]['transform_matrix'] for p in pipe.datamanager.train_dataset.image_filenames]);normal=pipe.datamanager.train_dataset.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);offset=normal[0,:3,3]-scale*R@raw[0,:3,3];rotation=torch.as_tensor(R,device='cuda',dtype=torch.float32);translation=torch.as_tensor(offset,device='cuda',dtype=torch.float32);mass=np.empty(len(origins))
  with torch.no_grad():
   for begin in range(0,len(origins),64):
    end=min(begin+64,len(origins));o=torch.as_tensor(origins[begin:end],device='cuda',dtype=torch.float32);d=torch.as_tensor(directions[begin:end],device='cuda',dtype=torch.float32);near=torch.as_tensor(nears[begin:end],device='cuda',dtype=torch.float32);far=torch.as_tensor(fars[begin:end],device='cuda',dtype=torch.float32);distance=near[:,None]+(far-near)[:,None]*fractions;points=o[:,None]+d[:,None]*distance[...,None];points=(scale*(points.reshape(-1,3)@rotation.T)+translation);density=torch.cat([model.field.density_fn(q).reshape(-1) for q in points.split(65536)]).reshape(end-begin,2048);tau=density.clamp_min(0).mean(-1)*(far-near)*scale;mass[begin:end]=(-torch.expm1(-tau)).cpu().numpy()
  if fingerprint(model)!=before:raise ValueError('Probe changed frozen field')
  arrays[label+'_head_mass']=mass;groups={}
  for group in ['hair','background']:
   select=arrays['labels']==group;v=mass[select];groups[group]=dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),over_point1=float((v>.1).mean()),over_point5=float((v>.5).mean()))
  results[label]=dict(groups=groups,checkpoint_sha256=sha(checkpoint),frozen_state_unchanged=True);del pipe,model;torch.cuda.empty_cache()
 np.savez_compressed(a.output/'rays.npz',**arrays);atomic_json(a.output/'metrics.json',dict(results=results,head_box_raw_calibration=[lo.tolist(),hi.tolist()],train_head_points=len(xyz),box_source='train SfM points supported as hair/face by at least two train semantics; 0.5-99.5 percentiles plus 15 percent margin',coverage=coverage,samples_per_ray=2048,occupancy_bypassed=True,domain='excluded semantic rays intersecting a train-derived head compartment; not true alpha or metric depth GT',uses_validation_rgb_for_fitting=False,uses_test_rgb=False))
 print(json.dumps(results),flush=True)
if __name__=='__main__':main()
