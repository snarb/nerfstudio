"""One actual no-grad frequency update per restored scratch model state, no optimizer."""
import argparse,hashlib,json,os,pickle,sys,time
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rng_digest(state):
 return dict(python=hashlib.sha256(pickle.dumps(state['python'])).hexdigest(),numpy=hashlib.sha256(pickle.dumps(state['numpy'])).hexdigest(),torch_cpu=tensor_sha(state['torch_cpu']),torch_cuda=[tensor_sha(x)for x in state['torch_cuda']])
def state_digest(pipe,skip_grid=False):
 h=hashlib.sha256()
 for k,v in sorted(pipe.state_dict().items()):
  if skip_grid and k.endswith('.freq_grid.grid'):continue
  h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main(a):
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);sys.path[:0]=[str(a.code),str(a.code/'LookCloser/scripts')]
 import torch
 from nerfstudio.utils.eval_utils import eval_setup
 from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
 from marshawn_semantic_occupancy import load_snapshot
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(0).total_memory)
 payload=torch.load(a.source/'checkpoint.ckpt',map_location='cpu',weights_only=False);rng=payload['rng_state'];step=int(payload['step'])
 if step!=111316 or len(rng['torch_cuda'])!=1:raise ValueError('Exact selected source and one-device RNG required')
 def config(cfg):cfg.eval_checkpoint=a.source/'checkpoint.ckpt';cfg.pipeline.model.occupancy_eval_dilation_radius=0;return cfg
 cfg,pipe,checkpoint,loaded_step=eval_setup(a.source/'config.yml',test_mode='val',update_config_callback=config);model=pipe.model
 if loaded_step!=step or not model.field.correct_sh_directions or model.field.density_activation!='trunc_exp' or pipe.config.frequency_projection_mode!='legacy':raise ValueError('Original selected recipe differs')
 if pipe.config.grid_update_batch_size!=2048 or pipe.config.grid_update_interval!=1024 or pipe.config.independent_rng_streams or pipe.config.datamanager.cpu_fas_prefetch:raise ValueError('Original sampling/RNG settings differ')
 if model.config.occupancy_dilation_radius!=0 or model.config.adaptive_coarse_step_size!=.00625 or model.config.corrected_arm_allocator:raise ValueError('Unexpected sampler change')
 model.config.feature_reweighting_strength=.3;model.field.feature_reweighting_strength=.3;model.current_train_step=step;pipe.train()
 core,active,ema,boxes,receipt=load_snapshot(a.source/'grid.json',a.source/'checkpoint.ckpt')
 if not np.array_equal(model.occupancy_grid.binaries.cpu().numpy(),active)or not np.array_equal(model.occupancy_grid.occs.cpu().numpy().reshape(ema.shape),ema):raise ValueError('Actual occupancy/EMA mismatch')
 cameras=pipe.datamanager.train_dataset.cameras;names=[Path(p).stem.split('_')[-1]for p in pipe.datamanager.train_dataset.image_filenames];available=list(pipe.cached_freq_maps)
 if len(names)!=63 or set(names)&{'DC','HC','KA'}or sorted(available)!=list(range(63)):raise ValueError('Global TRAIN camera/cache binding differs')
 for index in available:
  image=pipe.datamanager.train_dataset.image_filenames[index]
  if Path(image).stem.split('_')[-1]!=names[index]:raise ValueError('Camera/cache ID mismatch')
 original_grid=model.freq_grid.grid.detach().clone();initial_hash=state_digest(pipe);non_grid_hash=state_digest(pipe,True);initial_frequency_sha=tensor_sha(original_grid)
 if initial_frequency_sha!='81a1c6ec7866c4d18609260801d5c357b1b1a74fc83248714a27f75ad3cbe4d5':raise ValueError('Selected frequency source differs')
 saved=[]
 for label in ['clothing103','DC12']:
  folder=a.source/(label+'_trace/rays');audit=json.loads((folder/'audit.json').read_text())
  if audit['checkpoint_sha256']!=sha(a.source/'checkpoint.ckpt'):raise ValueError('Saved mass source differs')
  for ray in audit['rows']:
   p=folder/ray['profiles']['native']['trace']
   with np.load(p)as z:xyz=z['xyz'].copy();w=z['actual_weights'].reshape(-1).astype(float);mid=(z['t_start'].reshape(-1)+z['t_end'].reshape(-1))*.5
   band=np.zeros(len(w),bool)
   if 'witness_xyz'in ray['pixel']:
    depth=float((np.asarray(ray['pixel']['witness_xyz'])-ray['origin'])@np.asarray(ray['direction']));band=abs(mid-depth)<=.02
   saved.append(dict(label=label,group=ray['pixel'].get('diagnostic_group'),xyz=xyz,weights=w,band=band,trace_sha256=sha(p)))
 def coverage():
  groups={}
  for item in saved:
   levels=model.freq_grid.query(torch.from_numpy(item['xyz']).to(model.device)).reshape(-1).detach().cpu().numpy();w=item['weights'];band=item['band']
   for key in [item['label'],item['label']+'/'+str(item['group'])]:
    d=groups.setdefault(key,dict(rays=0,termination=0.,mass_level0=0.,band_mass=0.,band_level0=0.));d['rays']+=1;d['termination']+=float(w.sum());d['mass_level0']+=float(w[levels==0].sum());d['band_mass']+=float(w[band].sum());d['band_level0']+=float(w[band&(levels==0)].sum())
  for d in groups.values():d['level0_fraction']=d['mass_level0']/d['termination']if d['termination']else None;d['band_level0_fraction']=d['band_level0']/d['band_mass']if d['band_mass']else None
  return groups
 baseline_coverage=coverage();results={};original_generate=cameras.generate_rays;original_update=model.freq_grid.update_step;capture={};original_batch=pipe.config.grid_update_batch_size
 def generate(*args,**kwargs):
  ray=original_generate(*args,**kwargs);capture['camera_indices']=kwargs['camera_indices'].detach().cpu().numpy().reshape(-1);capture['coords']=kwargs['coords'].detach().cpu().numpy();capture['origins']=ray.origins.detach().cpu().numpy();capture['directions']=ray.directions.detach().cpu().numpy();return ray
 def update(*args,**kwargs):
  if args:raise ValueError('Expected actual keyword update API')
  for key in ['positions','rendered_depth','focals','patch_f2d']:capture[key]=kwargs[key].detach().float().cpu().numpy()
  capture['new_levels']=model.freq_grid.freq_to_level(kwargs['patch_f2d']*(kwargs['focals']/(kwargs['rendered_depth']+1e-6))).detach().cpu().numpy();return original_update(*args,**kwargs)
 def forward(module,inputs,outputs):
  capture['depth']=outputs['depth'].detach().float().cpu().numpy();capture['accumulation']=outputs['accumulation'].detach().float().cpu().numpy();capture['num_samples_per_ray']=outputs['num_samples_per_ray'].detach().cpu().numpy()
 cameras.generate_rays=generate;model.freq_grid.update_step=update;hook=model.register_forward_hook(forward);a.output.mkdir(parents=True,exist_ok=True)
 try:
  for count in [2048,8192]:
   model.freq_grid.grid.copy_(original_grid);pipe.config.grid_update_batch_size=count;_restore_rng_state(rng);before_rng=rng_digest(_capture_rng_state());capture.clear();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
   with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=cfg.mixed_precision):pipe._update_frequency_grid(step)
   torch.cuda.synchronize();elapsed=time.monotonic()-started;after_rng=rng_digest(_capture_rng_state())
   if len(capture['camera_indices'])!=count or not all(np.isfinite(v).all()for v in capture.values()):raise ValueError('Invalid actual update arrays')
   indices=capture['camera_indices'];coords=capture['coords'];H=cameras.height.cpu().numpy().reshape(-1);W=cameras.width.cpu().numpy().reshape(-1)
   if(indices<0).any()or(indices>=63).any()or(coords[:,0]<0).any()or(coords[:,0]>=H[indices]).any()or(coords[:,1]<0).any()or(coords[:,1]>=W[indices]).any():raise ValueError('Actual global camera/pixel bounds failed')
   positions=torch.from_numpy(capture['positions']).to(model.device);inside=((positions>=model.freq_grid.aabb_min_buf)&(positions<=model.freq_grid.aabb_max_buf)).all(-1);_,admitted,_,_=model.field._normalize_positions(positions)
   current=model.freq_grid.grid.detach();changed=current!=original_grid
   if (current<original_grid).any()or state_digest(pipe,True)!=non_grid_hash:raise ValueError('Non-frequency state changed or monotonicity failed')
   folder=a.output/str(count);folder.mkdir();np.savez_compressed(folder/'actual_update.npz',**capture,inside_frequency_bounds=inside.cpu().numpy(),admitted_native_selector=admitted.cpu().numpy(),frequency_grid_after=current.cpu().numpy())
   results[str(count)]=dict(samples=count,seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),rng_before=before_rng,rng_after=after_rng,camera_histogram={names[i]:int((indices==i).sum())for i in range(63)},inside_count=int(inside.sum()),admitted_count=int(admitted.sum()),opacity_above_point1=int((capture['accumulation'].reshape(-1)>.1).sum()),new_level15_cells=int(changed.sum()),grid_sha256=tensor_sha(current),coverage_before=baseline_coverage,coverage_after=coverage(),non_frequency_state_unchanged=True,arrays_sha256=sha(folder/'actual_update.npz'))
   model.freq_grid.grid.copy_(original_grid);_restore_rng_state(rng)
   if state_digest(pipe)!=initial_hash:raise ValueError('Full scratch state failed restoration')
 finally:
  hook.remove();cameras.generate_rays=original_generate;model.freq_grid.update_step=original_update;model.freq_grid.grid.copy_(original_grid);pipe.config.grid_update_batch_size=original_batch;_restore_rng_state(rng)
 if results['2048']['rng_before']!=results['8192']['rng_before']:raise ValueError('Arms did not start from identical RNG')
 output=dict(passed=True,kind='SCRATCH_ACTUAL_FREQUENCY_UPDATE',step=step,checkpoint_sha256=sha(a.source/'checkpoint.ckpt'),original_frequency_sha256=initial_frequency_sha,results=results,full_state_restored=state_digest(pipe)==initial_hash,optimizer_updates=0,training_mode=True,actual_AMP=cfg.mixed_precision,global_camera_sampling_preserved=True,projection='legacy unchanged integercoords and originaldepth policy',side_effect='Larger batch consumes different global RNG; not matched future training rays',interpretation='Singleupdate coverage feasibility, not quality proof or substitute for fullcoldtrajectory',source_pipeline_sha256=sha(a.code/'nerfstudio/pipelines/lookcloser_pipeline.py'),helper_sha256=sha(__file__))
 (a.output/'complete.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({n:{k:r[k]for k in ['inside_count','admitted_count','new_level15_cells','peak_allocated_bytes']}for n,r in results.items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
