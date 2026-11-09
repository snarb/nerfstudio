#!/usr/bin/env python3
"""Read-only sampler identity/ray and frequency bootstrap diagnosis of a checkpoint."""
import argparse,json,hashlib,sys,copy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
 v=np.asarray(v).reshape(-1)
 return dict(count=len(v),nonzero_fraction=float((v>0).mean()),maxlevel_fraction=float((v>=15).mean()),histogram=np.bincount(np.clip(np.rint(v),0,15).astype(int),minlength=16).tolist(),quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist()) if len(v) else dict(count=0)
def run(a):
 sys.path.insert(0,str(a.code_root/'LookCloser/scripts'));sys.path.insert(0,str(a.code_root))
 import torch
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.utils.lookcloser_rng import fork_seeded_rng
 from nerfstudio.utils.lookcloser_projection import normalized_frequency_resolution
 from probe_marshawn_occupancy import assert_consistent_bounds
 torch.set_num_threads(2)
 def configure(c):c.eval_checkpoint=a.checkpoint.resolve();return c
 cfg,pipe,checkpoint,step=eval_setup(a.config,test_mode='val',update_config_callback=configure);model=pipe.model;model.current_train_step=step;bounds=assert_consistent_bounds(model)
 a.output.mkdir(parents=True,exist_ok=False);ds=pipe.datamanager.train_dataset
 meta=json.loads((a.data/'transforms.json').read_text());by={Path(r['file_path']).stem:r for r in meta['frames']};rows=[by[Path(p).stem] for p in ds.image_filenames]
 sampler=pipe.datamanager.train_pixel_sampler;count_before=sampler.sample_count
 with fork_seeded_rng(cfg.pipeline.training_seed,'pixel',step,pipe.device):rays,batch=pipe.datamanager.next_train(step)
 indices=batch['indices'].detach().cpu().numpy()[:128];rgb=batch['image'].detach().float().cpu().numpy()[:128]
 if batch['image'].dtype==torch.uint8:rgb/=255
 actual_o=rays.origins.detach().cpu().numpy()[:128];actual_d=rays.directions.detach().cpu().numpy()[:128]
 expected_rgb=[];expected_o=[];expected_d=[];imgs={};sample_records=[];sheet=Image.new('RGB',(8*160,16*85));draw=ImageDraw.Draw(sheet)
 for n,(camera,y,x) in enumerate(indices):
  row=rows[camera];path=ds.image_filenames[camera]
  if camera not in imgs:imgs[camera]=Image.open(path).convert('RGB')
  im=imgs[camera];expected_rgb.append(np.array(im)[y,x]/255)
  pose=np.array(row['transform_matrix']);local=np.array([(x+.5-row['cx'])/row['fl_x'],-(y+.5-row['cy'])/row['fl_y'],-1.]);direction=pose[:3,:3]@local;direction/=np.linalg.norm(direction);expected_o.append(pose[:3,3]);expected_d.append(direction)
  crop=im.crop((x-16,y-16,x+17,y+17)).resize((66,66),Image.Resampling.NEAREST);sx=n%8*160;sy=n//8*85;sheet.paste(crop,(sx,sy+18));draw.rectangle((sx+73,sy+18,sx+137,sy+82),fill=tuple(np.rint(np.clip(rgb[n],0,1)*255).astype(int)));draw.text((sx,sy),f'{row["physical_camera"]} {x},{y}',fill='white')
  sample_records.append(dict(camera_id=int(camera),physical_camera=row['physical_camera'],file=str(path),x=int(x),y=int(y),batch_rgb=rgb[n].tolist(),source_rgb=expected_rgb[-1].tolist()))
 sheet.save(a.output/'sampled128_sourcepatch_batchRGB.jpg',quality=90)
 errors=dict(rgb_max_abs=float(np.max(abs(rgb-np.array(expected_rgb)))),ray_origin_max_abs=float(np.max(abs(actual_o-np.array(expected_o)))),ray_direction_max_abs=float(np.max(abs(actual_d-np.array(expected_d)))))
 camera_error=max(float(np.max(abs(ds.cameras.camera_to_worlds[i].cpu().numpy()-np.array(row['transform_matrix'])[:3]))) for i,row in enumerate(rows))
 z=np.load(a.witnesses);xyz=z['xyz'];names=z['cameras'].tolist();first=z['source_pairs'][:,0];name_to_camera={r['physical_camera']:i for i,r in enumerate(rows)};wcam=np.array([name_to_camera[names[i]] for i in first]);positions=torch.tensor(xyz,device=model.device,dtype=torch.float32)
 with torch.no_grad():levels=model.freq_grid.query(positions).squeeze(-1).cpu().numpy()
 actual_maps=pipe.cached_freq_maps;f2d=np.zeros(len(xyz));uv_all=np.zeros((len(xyz),2));front=np.zeros(len(xyz),bool);map_identity=[]
 for ci,row in enumerate(rows):
  path=Path(ds.image_filenames[ci]);metadata=json.loads((a.data/'lookcloser_frequencies'/(path.stem+'.json')).read_text());map_identity.append(dict(dataset_camera_index=ci,camera=row['physical_camera'],filename=path.name,map_image=metadata['image'],identity_matches=metadata['image']==path.name))
  ids=np.flatnonzero(wcam==ci)
  if not len(ids):continue
  pose=np.array(row['transform_matrix']);local=(xyz[ids]-pose[:3,3])@pose[:3,:3];zz=-local[:,2];u=row['fl_x']*local[:,0]/zz+row['cx'];v=-row['fl_y']*local[:,1]/zz+row['cy'];uv_all[ids]=np.column_stack([u,v]);front[ids]=(zz>0)&(u>=0)&(u<row['w'])&(v>=0)&(v<row['h'])
  fmap=actual_maps[ci].detach().cpu().numpy() if torch.is_tensor(actual_maps[ci]) else np.asarray(actual_maps[ci]);iy=np.clip((v//8).astype(int),0,fmap.shape[0]-1);ix=np.clip((u//8).astype(int),0,fmap.shape[1]-1);f2d[ids]=fmap[iy,ix]
 cameras=ds.cameras.to(model.device);ids_t=torch.tensor(wcam,device=model.device);encoding_box=getattr(model.field,'encoding_aabb',model.field.aabb)
 with torch.no_grad():
  resolution,valid=normalized_frequency_resolution(positions,cameras.camera_to_worlds[ids_t],torch.cat([cameras.fx[ids_t],cameras.fy[ids_t]],dim=-1),torch.cat([cameras.width[ids_t],cameras.height[ids_t]],dim=-1),encoding_box,torch.tensor(f2d,device=model.device,dtype=torch.float32));target=model.freq_grid.freq_to_level(resolution).cpu().numpy()
 valid=valid.cpu().numpy()&front;box=np.array(bounds['model.scene_aabb']);inside=((xyz>=box[0])&(xyz<box[1])).all(1)
 sb=np.array(json.loads(a.boxes.read_text())['scopes']['subject']['tight']);subject=((xyz>=sb[0])&(xyz<=sb[1])).all(1);sg=np.load(a.subject_witnesses)['xyz']
 with torch.no_grad():subject_levels=model.freq_grid.query(torch.tensor(sg,device=model.device,dtype=torch.float32)).squeeze(-1).cpu().numpy()
 def group(mask):
  ok=mask&valid&inside
  return dict(grid_levels=stats(levels[mask&inside]),projected2d_target_levels=stats(target[ok]),comparison_points=int(ok.sum()),grid_below_projected_target_fraction=float((levels[ok]<target[ok]).mean()) if ok.any() else None,grid_zero_and_target_at_least8_fraction=float(((levels[ok]==0)&(target[ok]>=8)).mean()) if ok.any() else None)
 dense=model.freq_grid.grid.detach().cpu().numpy();occ=model.occupancy_grid.binaries.detach().cpu().numpy()
 np.savez_compressed(a.output/'frequency_witness_arrays.npz',xyz=xyz,actual_grid_level=levels,projected2d_target_level=target,f2d=f2d,valid=valid,inside=inside,subject=subject,camera=wcam,uv=uv_all,subject_xyz=sg,subject_grid_level=subject_levels)
 report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),config_sha256=sha(a.config),step=step,live_bounds=bounds,eval_only_no_optimizer_steps=True,sampler=dict(audited_samples=128,actual_batch_size=len(batch['indices']),sample_count_before=count_before,sample_count_after=sampler.sample_count,errors=errors,prepared_vs_dataset_pose_max_abs=camera_error,passed=errors['rgb_max_abs']<1e-6 and errors['ray_origin_max_abs']<1e-5 and errors['ray_direction_max_abs']<1e-5 and camera_error<1e-5,samples=sample_records),frequency_map_identity=map_identity,all_map_ids_match=all(x['identity_matches'] for x in map_identity),frequency=dict(full_grid=stats(dense),all_train_witnesses=group(np.ones(len(xyz),bool)),subject_box_witnesses=group(subject),subject_roi_witnesses=stats(subject_levels),occupancy_fraction=float(occ.mean()),grid_updates_expected_through_step=[s for s in range(1,step+1) if s%cfg.pipeline.grid_update_interval==0],update_batch_size=cfg.pipeline.grid_update_batch_size,projection_mode=cfg.pipeline.frequency_projection_mode,zero_initialization=True,sfm_initialization_enabled=bool(cfg.pipeline.datamanager.dataparser.load_3D_points)),limitations=['Sample audit draws a new deterministic train batch from loaded state, not replay of historical optimizer updates.','SIFT witness2D target uses one actual supporting train camera, not all-view paper median.','Grid mismatches diagnose support/bootstrap; they do not prove a corrected recipe will improve quality.'])
 (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(sampler=report['sampler']['errors'],sampler_passed=report['sampler']['passed'],map_ids=report['all_map_ids_match'],frequency=report['frequency'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['config','checkpoint','code-root','data','witnesses','subject-witnesses','boxes','output']:p.add_argument('--'+k,type=Path,required=True)
 run(p.parse_args())
