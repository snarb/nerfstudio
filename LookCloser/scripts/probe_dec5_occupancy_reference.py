#!/usr/bin/env python3
"""Compare native adaptive traversal with all occupancy cells enabled.

This probes train-SfM witnesses, NOT independent 3D GT or held-out RGB. Density,
frequency, RGB weights and the adaptive step policy stay fixed. The original
occupancy mask is restored and the complete model fingerprint is checked.
"""
from pathlib import Path
import argparse,json
import numpy as np,torch,yaml
from dec5_hair import packed_depth_quantile,ray_to_z
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['data','training','checkpoint','sparse','semantics','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--per-class',type=int,default=1024);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(20*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 req=json.load(open(a.training/'request.json'))
 if req['data_sha256']!=sha(a.data/'transforms.json'):raise ValueError('Model/data mismatch')
 cfg=yaml.load(next(a.training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data;cfg.pipeline.datamanager.cache_images_type='uint8'
 pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step']
 if model.config.ray_sampling_mode!='adaptive' or model.config.occupancy_eval_dilation_radius:raise ValueError('Requires native undilated adaptive marcher')
 original=fingerprint(model);mask=model.occupancy_grid.binaries.clone();occupied=float(mask.float().mean());d=json.load(open(a.data/'transforms.json'));by={Path(f['file_path']).name:f for f in d['frames']};frames=[by[Path(p).name] for p in pipe.datamanager.train_dataset.image_filenames];names={f['physical_camera']:i for i,f in enumerate(frames)}
 source=np.load(a.sparse);raw=np.array([f['transform_matrix'] for f in frames]);normal=pipe.datamanager.train_dataset.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);offset=normal[0,:3,3]-scale*R@raw[0,:3,3];xyz=scale*source['xyz']@R.T+offset
 camera=np.array([names[str(source['camera_names'][i])] for i in source['observation_camera']]);uv=source['uv'];labels=np.full(len(uv),3,np.uint8)
 for i,f in enumerate(frames):
  ids=np.nonzero(camera==i)[0];sem=np.load(a.semantics/(f['physical_camera']+'.npz'))['probability'];sx,sy=sem.shape[1]/f['w'],sem.shape[0]/f['h'];x=np.clip((uv[ids,0]*sx).astype(int),0,sem.shape[1]-1);y=np.clip((uv[ids,1]*sy).astype(int),0,sem.shape[0]-1);v=sem[y,x].astype(float)/65535;labels[ids]=np.where(v[:,0]>.5,0,np.where(v[:,1]>.5,1,np.where(v[:,2]>.9,2,3)))
 rng=np.random.RandomState(42);selected=np.concatenate([rng.choice(np.nonzero(labels==k)[0],min(a.per_class,int((labels==k).sum())),replace=False) for k in range(4)]);selected=np.sort(selected);out={'observation_indices':selected,'labels':labels[selected]};gt=np.empty(len(selected),float)
 for j,index in enumerate(selected):
  i=camera[index];f=normal[i];gt[j]=-np.dot(xyz[source['observation_point'][index]]-f[:3,3],f[:3,2])
 cameras=pipe.datamanager.train_dataset.cameras.to('cuda')
 try:
  for profile in ['native','all_cells']:
   model.occupancy_grid.binaries.copy_(mask if profile=='native' else torch.ones_like(mask));expected=np.zeros(len(selected));median=np.zeros(len(selected));opacity=np.zeros(len(selected));samples=np.zeros(len(selected));saturation=[]
   for i,f in enumerate(frames):
    loc=np.nonzero(camera[selected]==i)[0]
    for start in range(0,len(loc),256):
     positions=loc[start:start+256];coords=uv[selected[positions]];rays=cameras[i:i+1].generate_rays(0,coords=torch.as_tensor(coords[:,::-1].copy(),device='cuda',dtype=torch.float32),keep_shape=False)
     with torch.no_grad():v=model(rays)
     expected[positions]=ray_to_z(v['depth'].reshape(-1).float().cpu().numpy(),coords,f);opacity[positions]=v['accumulation'].reshape(-1).float().cpu().numpy();samples[positions]=v['num_samples_per_ray'].reshape(-1).cpu().numpy();saturation.append(float(v['adaptive_saturation_rate']))
     arrays=[v[k].detach().float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']];dist=packed_depth_quantile(*arrays,v['packed_ray_indices'].cpu().numpy(),rays.nears.cpu().numpy(),rays.fars.cpu().numpy());median[positions]=ray_to_z(dist,coords,f)
   out[profile+'_expected']=expected;out[profile+'_median']=median;out[profile+'_opacity']=opacity;out[profile+'_samples']=samples
   atomic_json(a.output/(profile+'.json'),dict(completed=len(selected),mean_samples=float(samples.mean()),mean_chunk_saturation=float(np.mean(saturation))))
 finally:model.occupancy_grid.binaries.copy_(mask)
 if fingerprint(model)!=original:raise ValueError('Frozen model changed after restoring occupancy')
 out['gt_normalized_z']=gt;np.savez_compressed(a.output/'witnesses.npz',**out);rows={}
 for label,k in [('hair',0),('face_neck',1),('background',2),('uncertain',3),('all',None)]:
  pick=np.ones(len(selected),bool) if k is None else labels[selected]==k;row={'count':int(pick.sum())}
  if not pick.any():continue
  for profile in ['native','all_cells']:
   valid=pick&(out[profile+'_opacity']>=.1)&np.isfinite(out[profile+'_median'])&(out[profile+'_median']>0)&(gt>0);err=np.abs(out[profile+'_median'][valid]-gt[valid])/gt[valid];row[profile]=dict(valid=int(valid.sum()),relative_median=float(np.median(err)) if len(err) else None,relative_p90=float(np.quantile(err,.9)) if len(err) else None,mean_samples=float(out[profile+'_samples'][pick].mean()))
  both=pick&(out['native_opacity']>=.1)&(out['all_cells_opacity']>=.1)&(gt>0);row['median_changed_over_1pct_gt']=float((np.abs(out['native_median'][both]-out['all_cells_median'][both])>.01*gt[both]).mean()) if both.any() else None;rows[label]=row
 atomic_json(a.output/'metrics.json',dict(groups=rows,occupied_cell_fraction=occupied,checkpoint_sha256=sha(a.checkpoint),sparse_sha256=sha(a.sparse),frozen_state_unchanged=True,source='train SfM fitting witnesses; not independent 3D GT',uses_validation_rgb=False,uses_test_rgb=False,comparison='same density/FR/adaptive policy; native occupancy vs all cells enabled; original mask restored'))
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
