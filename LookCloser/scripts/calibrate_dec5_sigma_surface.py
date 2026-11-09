#!/usr/bin/env python3
"""Calibrate one density level from high-confidence train SfM hair witnesses."""
from pathlib import Path
import argparse,json
import numpy as np,torch,yaml
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['data','training','checkpoint','sparse','semantics','output']:p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
 request=json.load(open(a.training/'request.json'))
 if request['data_sha256']!=sha(a.data/'transforms.json'):raise ValueError('Field data mismatch')
 cfg=yaml.load(next(a.training.rglob('config.yml')).read_text(),Loader=yaml.Loader);cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data;cfg.pipeline.datamanager.cache_images_type='uint8';pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);pipe.load_state_dict(state['pipeline'],strict=True);pipe.eval();model=pipe.model;model.current_train_step=state['step'];before=fingerprint(model)
 data=json.load(open(a.data/'transforms.json'));by={Path(f['file_path']).name:f for f in data['frames']};raw=np.asarray([by[Path(p).name]['transform_matrix'] for p in pipe.datamanager.train_dataset.image_filenames]);normal=pipe.datamanager.train_dataset.cameras.camera_to_worlds.cpu().numpy();R=normal[0,:3,:3]@raw[0,:3,:3].T;scale=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);offset=normal[0,:3,3]-scale*R@raw[0,:3,3];source=np.load(a.sparse);support=np.zeros(len(source['xyz']),int)
 for i,name in enumerate(source['camera_names']):
  selected=np.nonzero(source['observation_camera']==i)[0];prob=np.load(a.semantics/(str(name)+'.npz'))['probability'];uv=source['uv'][selected];x=np.clip(uv[:,0].astype(int),0,prob.shape[1]-1);y=np.clip(uv[:,1].astype(int),0,prob.shape[0]-1);observed=selected[prob[y,x,0]>round(.9*65535)];support+=np.bincount(source['observation_point'][observed],minlength=len(support))
 points=scale*source['xyz']@R.T+offset;valid=(support>=2)&(points>=model.scene_box.aabb[0].cpu().numpy()).all(-1)&(points<=model.scene_box.aabb[1].cpu().numpy()).all(-1)
 if valid.sum()<100:raise ValueError('Too few reliable train hair witnesses')
 with torch.no_grad():density=model.field.density_fn(torch.as_tensor(points[valid],device='cuda',dtype=torch.float32)).reshape(-1).cpu().numpy()
 threshold=float(np.quantile(density,.1))
 if threshold<=0 or not np.isfinite(density).all():raise ValueError('Cannot calibrate a positive finite level from train hair')
 if fingerprint(model)!=before:raise ValueError('Calibration changed frozen field')
 np.savez_compressed(a.output/'witnesses.npz',normalized_xyz=points[valid],density_normalized=density)
 atomic_json(a.output/'complete.json',dict(frame_id=data['frame_id'],checkpoint_sha256=sha(a.checkpoint),dataset_sha256=sha(a.data/'transforms.json'),sparse_sha256=sha(a.sparse),threshold_density_normalized=threshold,threshold_density_per_calibration_unit=threshold*float(scale),threshold_policy='10th percentile of density at train SfM points with >=2 hair probability >.9 observations; fixed before receiver evaluation',count=int(valid.sum()),density_percentiles={str(q):float(np.quantile(density,q)) for q in [0,.1,.25,.5,.9,1]},rotation=R.tolist(),translation=offset.tolist(),scale=float(scale),refine_steps=16,uses_validation_rgb=False,uses_test_rgb=False,independent_depth_gt=False,frozen_state_unchanged=True,implementation_sha256=sha(__file__)))
 print('Calibrated train-only sigma surface threshold',threshold,flush=True)
if __name__=='__main__':main()
