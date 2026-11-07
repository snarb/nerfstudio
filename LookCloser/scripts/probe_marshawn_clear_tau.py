#!/usr/bin/env python3
"""Read-only full-state TRAIN4096 versus qualified-clear64 optical-thickness gradients."""
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
 if n not in(256,512,1024)or not bool(torch.all(far>near)):raise ValueError('Reviewed finite positive quadrature intervals required')
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
 parent=base.validate_spec(read(policy['parent_spec']));checkpoint=Path(policy['checkpoint']);loaded=torch.load(checkpoint,map_location='cpu',weights_only=False);assert loaded['step']==111316
 for name,digest in loaded['tuning_runtime']['input_sha256'].items():
  p=(Path(parent['semantic_admission']['manifest']).parent/name[9:]if name.startswith('semantic/')else Path(parent['semantic_admission']['manifest'])if name=='semantic_manifest'else Path(parent['data'])/name)
  if sha(p)!=digest:raise ValueError('Checkpoint source input differs '+name)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 cfg=base.build_config(dict(parent,output=str(R/'setup')),checkpoint);trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();pipe=trainer.pipeline;model=pipe.model;model.current_train_step=111316
 assert model.field.feature_reweighting_strength==.3 and pipe._feature_reweighting_switch_applied
 sampler=sampling.install_sampling(pipe,Path(parent['data']),'actor','body80',64);sampler.load_state_dict(loaded['tuning_runtime']['sampling_state']);pipe.datamanager.train_count=int(sampler.state_dict()['sample_count']);model._tuning_occupancy_capture={k:v.to(model.device)if torch.is_tensor(v)else v for k,v in loaded['actual_occupancy_capture'].items()}
 capture=loaded['actual_occupancy_capture'];assert torch.equal(model.semantic_learned_core.cpu(),capture['core'])and torch.equal(model.occupancy_grid.binaries.cpu(),capture['active'])
 assert model.semantic_receipt==capture['semantic_receipt'] and model.__class__.__name__=='SemanticAdmissionModel'
 cache=pipe.datamanager.train_image_dataloader.cached_collated_batch;cache_ids=cache['image_idx'].cpu().tolist();helpers.verify_cache_ids(cache_ids,42);cache_before=helpers.tree_digest(cache)
 assert trainer._loaded_rng_state is not None;_restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None;runtime=copy.deepcopy(loaded['tuning_runtime'])
 def state():return dict(pipeline=pipe.state_dict(),optimizers={k:v.state_dict()for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict()for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state(),tuning_runtime=copy.deepcopy(runtime),actual_occupancy_capture=model._tuning_occupancy_capture)
 initial=helpers.verify_state_equality(loaded,state());saved_rng=_capture_rng_state();saved_sampler=copy.deepcopy(sampler.state_dict());saved_count=pipe.datamanager.train_count;params=[p for p in model.field.parameters()if p.requires_grad];names=[name for name,p in model.field.named_parameters()if p.requires_grad];parent_scale=float(trainer.grad_scaler.get_scale());model_training=model.training;step_before=model.current_train_step
 target=read(policy['targets']);indices=policy['selected_target_indices'];assert len(indices)==64 and len(set(indices))==64
 chosen=[target['targets'][i]for i in indices];o=torch.tensor([r['origin']for r in chosen],device=model.device,dtype=torch.float32);d=torch.tensor([r['direction']for r in chosen],device=model.device,dtype=torch.float32);near=torch.tensor([r['near']for r in chosen],device=model.device);far=torch.tensor([r['far']for r in chosen],device=model.device)
 cams=pipe.datamanager.train_dataset.cameras.to(model.device);order=[Path(p).stem.split('_')[-1]for p in pipe.datamanager.train_dataset.image_filenames];ray_errors=[]
 for j,r in enumerate(chosen):
  c=order.index(r['camera']);rb=cams[c:c+1].generate_rays(0,coords=torch.tensor([[r['y']+.5,r['x']+.5]],device=model.device),keep_shape=False);ray_errors.append(max(float((rb.origins[0]-o[j]).abs().max()),float((rb.directions[0]-d[j]).abs().max())))
 assert max(ray_errors)<1e-6
 main_rb=main_batch=None;report={};started=time.time();render_rng=saved_rng
 def grad_once(fn):
  used=parent_scale;attempts=[]
  for _ in range(24):
   _restore_rng_state(render_rng)
   with torch.autocast('cuda',enabled=trainer.mixed_precision):loss,extra=fn()
   grads=torch.autograd.grad(loss*used,params,allow_unused=True);finite=bool(torch.isfinite(loss))and all(g is None or bool(torch.isfinite(g).all())for g in grads);attempts.append(dict(backward_scale=used,finite=finite))
   if finite:
    result=[None if g is None else(g.detach().float()/used).cpu()for g in grads];norm=sum(float(g.double().square().sum())for g in result if g is not None)**.5
    return result,dict(value=float(loss.detach()),unscaled_gradient_L2=norm,backoff_attempts=attempts,backward_scale=used,GradScaler_mutated=False,**extra)
   del grads,loss;used/=2
  raise FloatingPointError('Read-only backward backoff did not become finite')
 try:
  model.train();model.current_train_step=111317;sampler.set_step(111317);main_rb,main_batch=pipe.datamanager.next_train(111317)
  assert len(main_rb.origins)==4096
  render_rng=_capture_rng_state()
  report['main_ray_indices_sha256']=helpers.tree_digest(main_batch['indices']);report['main_sampling']=sampler.audit_state();report['main_sampling']['actual_batch_size']=4096
  _restore_rng_state(saved_rng)
  def main_loss():
   out=model(main_rb);losses=model.get_loss_dict(out,main_batch);return sum(losses.values()),dict(component_values={k:float(v.detach())for k,v in losses.items()})
  main_grad,report['main_reference']=grad_once(main_loss)
  report['quadrature']={}
  for n in(256,512,1024):
   t=midpoint_distances(near,far,n);positions=(o[:,None,:]+d[:,None,:]*t[...,None]).reshape(-1,3);zeros=torch.zeros((len(positions),1),device=model.device);rs=RaySamples(frustums=Frustums(origins=positions,directions=torch.zeros_like(positions),starts=zeros,ends=zeros,pixel_area=torch.ones_like(zeros)))
   def tau_loss():
    sigma,_=model.field.get_density(rs);sigma=sigma.float().reshape(64,n);admission=model.semantic_lookup(positions).reshape(64,n);assert bool(torch.all(sigma[~admission]==0));tau=sigma.mean(1)*(far-near);return tau.mean(),dict(per_ray_tau=tau.detach().cpu().tolist(),fixed_support_admitted_samples=int(admission.sum()),sample_count=64*n)
   g,rec=grad_once(tau_loss);dot=sum(float((a.double()*b.double()).sum())for a,b in zip(g,main_grad)if a is not None and b is not None);den=rec['unscaled_gradient_L2']*report['main_reference']['unscaled_gradient_L2'];rec['cosine_to_main']=dot/den if den else None;rec['gradient_ratio_to_main']={str(c):c*rec['unscaled_gradient_L2']/report['main_reference']['unscaled_gradient_L2']for c in(.001,.01)};rec['finite_nonzero_gradient']=rec['unscaled_gradient_L2']>0;assert rec['finite_nonzero_gradient'];report['quadrature'][str(n)]=rec;del g
  for lo,hi in[(256,512),(512,1024)]:
   a=np.array(report['quadrature'][str(lo)]['per_ray_tau']);b=np.array(report['quadrature'][str(hi)]['per_ray_tau']);err=abs(a-b);report[f'convergence_{lo}_{hi}']=dict(max_absolute=float(err.max()),mean_absolute=float(err.mean()),max_relative=float((err/np.maximum(abs(b),1e-8)).max()),median_relative=float(np.median(err/np.maximum(abs(b),1e-8))),fraction_within5percent_or005=float(np.mean((err<=.05)|(err<=.05*abs(b)))))
 finally:
  sampler.load_state_dict(saved_sampler);pipe.datamanager.train_count=saved_count;model.current_train_step=step_before;model.train(model_training);_restore_rng_state(saved_rng)
 final=helpers.verify_state_equality(loaded,state());assert final==initial and sampler.state_dict()==saved_sampler and helpers.tree_digest(cache)==cache_before and all(p.grad is None for p in params)
 report.update(passed=True,seconds=time.time()-started,initial_fullstate_hashes=initial,final_fullstate_hashes=final,cache_hash_before_after=cache_before,historical_cache_order_observed=False,seed42_current_cache_reconstruction=True,sampler_restored=True,main_rng_restored=True,params_optimizer_scaler_core_unchanged=True,actual_Adam_updates=0,FR=.3,source_step=111316,selected_target_indices=indices,max_actual_ray_direction_error=max(ray_errors),autocast_enabled=trainer.mixed_precision,autocast_dtype=str(torch.get_autocast_dtype('cuda')),parent_GradScaler_scale=parent_scale,gradient_handling='autograd.grad(loss*fixedscale), then float32divide. Read-only backwardscale halves ifnonfinite; actualGradScaler/optimizer state neverupdated.',no_native_alpha_equivalence_assumed=True,no_field_or_grid_updates=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(checkpoint),policy_sha256=sha(policy_path))
 (R/'probe_complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');trainer.shutdown();print(json.dumps(dict(passed=True,seconds=report['seconds'],main_gradient_norm=report['main_reference']['unscaled_gradient_L2'],tau512=report['quadrature']['512']['value'])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);main(p.parse_args().policy)
