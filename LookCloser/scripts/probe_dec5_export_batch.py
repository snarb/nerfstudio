#!/usr/bin/env python3
"""Check native adaptive-ray partition invariance before increasing export batches."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch,yaml
from dec5_hair import packed_depth_quantile
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['data','training','checkpoint','cameras','roi','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--batches',type=int,nargs='+',default=[256,2048]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(24*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 request=json.load(open(a.training/'request.json'))
 if request['data_sha256']!=sha(a.data/'transforms.json'):raise ValueError('Dataset mismatch')
 cfg=yaml.load(next(a.training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data;cfg.pipeline.datamanager.cache_images_type='uint8';pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
 payload=json.load(open(a.data/'transforms.json'));by={Path(f['file_path']).name:f for f in payload['frames']};raw=np.asarray([by[Path(p).name]['transform_matrix'] for p in pipe.datamanager.train_dataset.image_filenames]);normal=pipe.datamanager.train_dataset.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);offset=normal[0,:3,3]-scale*R@raw[0,:3,3]
 from nerfstudio.cameras.cameras import Cameras
 packets=[]
 for f in json.load(open(a.cameras))['frames']:
  roi=np.load(a.roi/(f['physical_camera']+'.npz'))['roi'];y,x=np.nonzero(roi);pick=np.linspace(0,len(x)-1,min(2048,len(x)),dtype=int);uv=np.stack([y[pick]+.5,x[pick]+.5],-1);pose=np.array(f['transform_matrix']);c2w=np.concatenate([R@pose[:3,:3],(scale*R@pose[:3,3]+offset)[:,None]],-1);cam=Cameras(camera_to_worlds=torch.as_tensor(c2w,dtype=torch.float32)[None],fx=f['fl_x'],fy=f['fl_y'],cx=f['cx'],cy=f['cy'],width=f['w'],height=f['h']).to('cuda');packets.append((cam,uv))
 with torch.no_grad():model(packets[0][0].generate_rays(0,coords=torch.as_tensor(packets[0][1][:256],device='cuda',dtype=torch.float32),keep_shape=False))
 arrays={};timing={}
 for batch in a.batches:
  values=[];torch.cuda.synchronize();start=time.perf_counter()
  with torch.no_grad():
   for cam,uv in packets:
    for begin in range(0,len(uv),batch):
     rays=cam.generate_rays(0,coords=torch.as_tensor(uv[begin:begin+batch],device='cuda',dtype=torch.float32),keep_shape=False);v=model(rays);packed=[v[k].float().cpu().numpy() for k in ['packed_spacing_starts','packed_spacing_ends','packed_weights']];args=[*packed,v['packed_ray_indices'].cpu().numpy(),rays.nears.cpu().numpy(),rays.fars.cpu().numpy()];median=packed_depth_quantile(*args);front=packed_depth_quantile(*args,quantile=.1);values.append(np.column_stack([v['depth'].reshape(-1).cpu().numpy(),v['accumulation'].reshape(-1).cpu().numpy(),median,front]))
  torch.cuda.synchronize();timing[str(batch)]=time.perf_counter()-start;arrays[str(batch)]=np.concatenate(values)
 raw_expected={k:v[:,0].copy() for k,v in arrays.items()}
 for value in arrays.values():value[:,0]=np.where((value[:,1]>=.1)&np.isfinite(value[:,0])&(value[:,0]>0),value[:,0],0)
 baseline=arrays[str(a.batches[0])];comparisons={};passed=True
 for key,value in arrays.items():
  delta=np.abs(value-baseline);relative=delta/np.maximum(np.abs(baseline),1e-6);maxima=relative.max(0);coverage_equal=bool(np.array_equal(value[:,1]>=.1,baseline[:,1]>=.1));depth_valid_equal=bool(np.array_equal(value[:,2:]>0,baseline[:,2:]>0));ok=coverage_equal and depth_valid_equal and max(maxima[0],maxima[2],maxima[3])<1e-4 and delta[:,1].max()<1e-4;passed&=ok;comparisons[key]=dict(max_absolute_delta=delta.max(0).tolist(),max_relative_delta=maxima.tolist(),opacity_threshold_equal=coverage_equal,conditional_depth_validity_equal=depth_valid_equal,passes=bool(ok))
 if fingerprint(model)!=before:raise ValueError('Probe changed field weights')
 np.savez_compressed(a.output/'predictions.npz',**arrays);atomic_json(a.output/'metrics.json',dict(passes=bool(passed),rays=len(baseline),expected_validity='same exported opacity >=.1 and finite positive depth; raw rejected-ray depth is not a surface',columns=['expected_ray_distance','opacity','median_ray_distance','front10_ray_distance'],seconds=timing,comparisons=comparisons,checkpoint_sha256=sha(a.checkpoint),frozen_state_unchanged=True,uses_validation_rgb=False,uses_test_rgb=False));print(json.dumps(dict(passes=bool(passed),rays=len(baseline),seconds=timing,comparisons=comparisons)),flush=True)
if __name__=='__main__':main()
