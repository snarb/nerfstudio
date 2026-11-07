#!/usr/bin/env python3
"""Read-only final focused geometry quadrature; exact state replay, no gradients or updates."""
import argparse,copy,hashlib,importlib.util,json,os,random,sys,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 path=Path(path).resolve();s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
 if Path(m.__file__).resolve()!=path:raise ValueError('Import shadow: '+name)
 return m

def midpoint_distances(near,far,n):
 import torch
 if n not in(16,64,128,256,512,1024)or not bool(torch.all(far>near)):raise ValueError('Reviewed finite positive quadrature intervals required')
 return near[:,None]+(far-near)[:,None]*(torch.arange(n,device=near.device,dtype=near.dtype)+.5)/n

def main(policy_path):
 policy=read(policy_path);R=Path(policy['root']);R.mkdir(exist_ok=True);code=Path(policy['code_root']);sys.path[:0]=[str(code/'LookCloser/scripts'),str(code)]
 for p,h in policy['inputs_sha256'].items():
  if sha(p)!=h:raise ValueError('Input binding differs '+p)
 isolation=load('marshawn_gpu_isolation',R/'marshawn_gpu_isolation.py');isolation.require_environment(policy['gpu_uuid'])
 bindings={}
 for name,path in [('train_marshawn_semantic_admission',code/'LookCloser/scripts/train_marshawn_semantic_admission.py'),('marshawn_tuning_sampling',code/'LookCloser/scripts/marshawn_tuning_sampling.py'),('marshawn_semantic_occupancy',code/'LookCloser/scripts/marshawn_semantic_occupancy.py'),('clear_tau_state_helpers',R/'state_helpers.py')]:
  load(name,path);bindings[name]=dict(path=str(path.resolve()),sha256=sha(path))
 base=sys.modules['train_marshawn_semantic_admission'];sampling=sys.modules['marshawn_tuning_sampling'];helpers=sys.modules['clear_tau_state_helpers']
 import torch,numpy as np
 isolation_proof=isolation.verify_cuda(policy['gpu_uuid'],torch);(R/'cuda_isolation_before_checkpoint.json').write_text(json.dumps(isolation_proof,indent=2)+'\n')
 from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
 from nerfstudio.cameras.rays import RaySamples,Frustums
 import nerfstudio.fields.lookcloser_field as field_module
 assert Path(field_module.__file__).resolve()==(code/'nerfstudio/fields/lookcloser_field.py').resolve()
 bindings['nerfstudio.fields.lookcloser_field']=dict(path=field_module.__file__,sha256=sha(field_module.__file__))
 (R/'imported_helpers.json').write_text(json.dumps(bindings,indent=2)+'\n')
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(policy['allocator_GiB']*2**30/torch.cuda.get_device_properties(0).total_memory)
 parent=base.validate_spec(read(policy['parent_spec']));checkpoint=Path(policy['checkpoint']);loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);assert loaded['step']==policy['source_step']==19188
 for name,digest in loaded['tuning_runtime']['input_sha256'].items():
  p=(Path(parent['semantic_admission']['manifest']).parent/name[9:]if name.startswith('semantic/')else Path(parent['semantic_admission']['manifest'])if name=='semantic_manifest'else Path(parent['data'])/name)
  if sha(p)!=digest:raise ValueError('Checkpoint source input differs '+name)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 from marshawn_focused_clothing_factory import build_config
 cfg=build_config(parent,read(policy['anchor_audit']),checkpoint,R/'setup',policy['arm']);trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();pipe=trainer.pipeline;model=pipe.model;model.current_train_step=19188
 assert model.field.feature_reweighting_strength==1. and not pipe._feature_reweighting_switch_applied
 assert model.field.density_activation=='trunc_exp' and model.field.correct_sh_directions
 h=torch.tensor([-4.,0.,2.],device=model.device);torch.testing.assert_close(model.field.activate_density(h),torch.exp(h+1.))
 sampler=sampling.install_sampling(pipe,Path(parent['data']),'actor','body80',64);sampler.load_state_dict(loaded['tuning_runtime']['sampling_state']);pipe.datamanager.train_count=int(sampler.state_dict()['sample_count']);model._tuning_occupancy_capture={k:v.to(model.device)if torch.is_tensor(v)else v for k,v in loaded['actual_occupancy_capture'].items()}
 focus=None
 if 'frequency_focus' in parent:
  focus_module=load('marshawn_frequency_focus_runtime',code/'LookCloser/scripts/marshawn_frequency_focus_runtime.py');focus=focus_module.FocusRuntime(pipe,sampler,parent['frequency_focus']);focus.load_state_dict(loaded['tuning_runtime']['frequency_focus_state']);assert focus.state_dict()==loaded['tuning_runtime']['frequency_focus_state']
  for name in ('marshawn_frequency_focus_runtime','marshawn_frequency_sampling_proposal'):
   path=Path(sys.modules[name].__file__).resolve();assert path==(code/'LookCloser/scripts'/(name+'.py')).resolve();bindings[name]=dict(path=str(path),sha256=sha(path))
  (R/'imported_helpers.json').write_text(json.dumps(bindings,indent=2)+'\n')
 capture=loaded['actual_occupancy_capture'];assert torch.equal(model.semantic_learned_core.cpu(),capture['core'])and torch.equal(model.occupancy_grid.binaries.cpu(),capture['active'])
 assert model.semantic_receipt==capture['semantic_receipt'] and model.__class__.__name__=='SemanticAdmissionModel'
 cache=pipe.datamanager.train_image_dataloader.cached_collated_batch;cache_ids=cache['image_idx'].cpu().tolist();helpers.verify_cache_ids(cache_ids,42);cache_before=helpers.tree_digest(cache)
 assert trainer._loaded_rng_state is not None;_restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None;runtime=copy.deepcopy(loaded['tuning_runtime'])
 def state():return dict(pipeline=pipe.state_dict(),optimizers={k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict()for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state(),tuning_runtime=copy.deepcopy(runtime),actual_occupancy_capture=model._tuning_occupancy_capture)
 initial=helpers.verify_state_equality(loaded,state());saved_rng=_capture_rng_state();saved_sampler=copy.deepcopy(sampler.state_dict());saved_count=pipe.datamanager.train_count;params=[p for p in model.field.parameters()if p.requires_grad];names=[name for name,p in model.field.named_parameters()if p.requires_grad];parent_scale=float(trainer.grad_scaler.get_scale());model_training=model.training;step_before=model.current_train_step
 with np.load(policy['prefix_archive'])as z:arrays={k:z[k].copy()for k in z.files}
 assert len(arrays['point_idx'])==84 and len(np.unique(arrays['point_idx']))==31 and not set(map(int,arrays['point_idx']))&{63,64,71,72,2219,5556,5568}
 o=torch.tensor(arrays['origins'],device=model.device,dtype=torch.float32);d=torch.tensor(arrays['directions'],device=model.device,dtype=torch.float32);near=torch.tensor(arrays['t_start'],device=model.device,dtype=torch.float32);old_far=torch.tensor(arrays['t_end'],device=model.device,dtype=torch.float32);depth=torch.tensor(arrays['witness_depth'],device=model.device,dtype=torch.float32)
 band_half=torch.maximum(torch.full_like(depth,.005),.01*depth);closer_far=depth-torch.maximum(torch.full_like(depth,.01),.02*depth)
 source_cpu=read(policy['CPU_feasibility']);band_near=torch.tensor([r['band_start']for r in source_cpu['source_rows']],device=model.device,dtype=torch.float32);band_far=torch.tensor([r['band_end']for r in source_cpu['source_rows']],device=model.device,dtype=torch.float32)
 assert bool(((near<old_far)&(old_far<closer_far)&(closer_far<band_near)&(band_near<depth)&(depth<band_far)).all())
 assert bool(model.semantic_lookup(torch.tensor(arrays['witness_xyz'],device=model.device,dtype=torch.float32)).all())
 camera_ids=arrays['camera_idx'];cameras,counts=np.unique(camera_ids,return_counts=True);weight=np.array([1/len(cameras)/counts[np.flatnonzero(cameras==ca)[0]]for ca in camera_ids]);assert abs(weight.sum()-1)<1e-12;balance=torch.tensor(weight,device=model.device,dtype=torch.float32)
 cams=pipe.datamanager.train_dataset.cameras.to(model.device);order=[Path(p).stem.split('_')[-1]for p in pipe.datamanager.train_dataset.image_filenames];ray_errors=[]
 for j,r in enumerate(source_cpu['source_rows']):
  ca=order.index(r['camera']);uv=r['projected_uv'];rb=cams[ca:ca+1].generate_rays(0,coords=torch.tensor([[uv[1],uv[0]]],device=model.device),keep_shape=False);ray_errors.append(max(float((rb.origins[0]-o[j]).abs().max()),float((rb.directions[0]-d[j]).abs().max())))
 assert max(ray_errors)<1e-6
 report={'quadrature':{}};started=time.time()
 for kind,(left,right,counts) in {'old_prefix':(near,old_far,[512,1024,2048]),'closer_prefix':(near,closer_far,[512,1024,2048]),'band':(band_near,band_far,[256,512,1024])}.items():
  records={};previous=None;pending=list(counts)
  while pending:
   n=pending.pop(0);values=[];admitted=0
   with torch.inference_mode(),torch.autocast('cuda',enabled=trainer.mixed_precision):
    for j in range(0,84,8):
     l,r=left[j:j+8],right[j:j+8];t=l[:,None]+(r-l)[:,None]*(torch.arange(n,device=model.device,dtype=torch.float32)+.5)/n
     positions=(o[j:j+8,None,:]+d[j:j+8,None,:]*t[...,None]).reshape(-1,3);zeros=torch.zeros((len(positions),1),device=model.device)
     rs=RaySamples(frustums=Frustums(origins=positions,directions=torch.zeros_like(positions),starts=zeros,ends=zeros,pixel_area=torch.ones_like(zeros)))
     sigma,_=model.field.get_density(rs);sigma=sigma.float().reshape(len(l),n);admission=model.semantic_lookup(positions).reshape(len(l),n)
     assert bool(torch.all(sigma[~admission]==0));tau=sigma.mean(1)*(r-l);assert bool(torch.isfinite(tau).all())and bool((tau>=0).all())
     values.extend(tau.cpu().tolist());admitted+=int(admission.sum())
   new=np.array(values);record=dict(per_ray_tau=values,camera_balanced_tau=float(new@weight),admitted_samples=admitted,samples=84*n)
   if previous is not None:
    old=np.array(records[str(previous)]['per_ray_tau']);err=abs(old-new);resolved=(err<=.05)|(err<=.05*abs(new))
    record['convergence']=dict(previous=previous,max_absolute=float(err.max()),max_relative=float((err/np.maximum(abs(new),1e-8)).max()),unresolved_indices=np.flatnonzero(~resolved).tolist(),criterion='absolute<=.05 OR relative<=5%')
    if not pending and not resolved.all() and n<8192:pending.append(n*2)
   records[str(n)]=record;previous=n
  report['quadrature'][kind]=records
 assert sampler.state_dict()==saved_sampler and focus.state_dict()==loaded['tuning_runtime']['frequency_focus_state']
 assert helpers.tree_digest(_capture_rng_state())==helpers.tree_digest(saved_rng)
 final=helpers.verify_state_equality(loaded,state());assert final==initial and helpers.tree_digest(cache)==cache_before and all(p.grad is None for p in params)
 report.update(passed=True,seconds=time.time()-started,initial_fullstate_hashes=initial,final_fullstate_hashes=final,actual_Adam_updates=0,FR=1.,source_step=19188,source_family=policy['name'],max_projected_ray_direction_error=max(ray_errors),checkpoint_sha256=sha(checkpoint),policy_sha256=sha(policy_path),all84_fit_only=True,reserved_used=False,no_gradient=True,auxiliary_private_state_serialized_sha256=helpers.tree_digest(loaded['fork_runtime']['auxiliary_state']),camera_balancing_weights=weight.tolist(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
 (R/'probe_complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');trainer.shutdown();print(json.dumps(dict(passed=True,seconds=report['seconds'])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);main(p.parse_args().policy)
