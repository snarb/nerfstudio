"""Actual4kfullstate restore +newnear train/eval/AMP proof, zero updates."""
from pathlib import Path
import argparse,json,time,hashlib,copy,sys
import numpy as np,torch
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
from nerfstudio.scripts.train import _set_random_seed
from run_canonical_hdr_near_resume import load
from probe_marshawn_occupancy import digest_model

def equal(a,b):
 if torch.is_tensor(b):assert torch.equal(a.detach().cpu(),b.detach().cpu())
 elif isinstance(b,np.ndarray):assert np.array_equal(a,b)
 elif isinstance(b,dict):
  assert a.keys()==b.keys()
  for k in b:equal(a[k],b[k])
 elif isinstance(b,(list,tuple)):
  assert type(a)==type(b)and len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 a.output.mkdir(exist_ok=False);torch.set_num_threads(2);cfg,spec=load(a.config,a.spec);saved=torch.load(cfg.load_checkpoint,map_location='cpu',weights_only=False);original=_capture_rng_state();cfg=copy.deepcopy(cfg);cfg.output_dir=a.output/'setup_outputs'
 try:
  _set_random_seed(42);t=cfg.setup(local_rank=0,world_size=1);t.setup();p=t.pipeline;m=p.model;assert t._start_step==4001 and cfg.max_num_iterations-t._start_step==17959 and spec['parent_attempt_charge']+17959==22000
  equal(p.state_dict(),saved['pipeline'])
  for k,o in t.optimizers.optimizers.items():equal(o.state_dict(),saved['optimizers'][k])
  for k,s in t.optimizers.schedulers.items():equal(s.state_dict(),saved['schedulers'][k])
  equal(t.grad_scaler.state_dict(),saved['scalers']);equal(t._loaded_rng_state,saved['rng_state']);assert m.config.near_plane==m.collider.near_plane==.052 and m.config.eval_use_training_near_plane
  assert m.config.rgb_output_parameterization=='linear_softplus'and m.field.appearance_embedding_dim==0 and not m.renderer_rgb.clamp_output
  _restore_rng_state(t._loaded_rng_state);t._loaded_rng_state=None;equal(_capture_rng_state(),saved['rng_state']);before=digest_model(m);scale=float(t.grad_scaler.get_scale());assert scale==16384.;capture={};next_train=p.datamanager.next_train
  def next_capture(step):
   rays,batch=next_train(step);capture['rays']=rays;return rays,batch
  p.datamanager.next_train=next_capture;rows=[]
  for step in[4001,4002]:
   p.train()
   with torch.autocast(device_type='cuda',enabled=True):out,losses,metrics=p.get_train_loss_dict(step)
   assert set(losses)=={'rgb_loss','distortion_loss','observed_sfm_free_space','observed_sfm_surface_band'}
   grads=torch.autograd.grad(sum(losses.values())*scale,[q for q in m.parameters()if q.requires_grad],allow_unused=True);finite=all(bool(torch.isfinite(g).all())for g in grads if g is not None)
   rays=capture['rays'];train_min=float(rays.nears.min());assert train_min>=.052-1e-7;m.collider.eval();test=m.collider(copy.deepcopy(rays));eval_min=float(test.nears.min());assert eval_min>=.052-1e-7
   rows.append(dict(step=step,rays=len(out['rgb']),samples=int(metrics['num_samples_per_batch']),AMP_scale=scale,combined_scaled_gradients_finite=finite,train_min_near=train_min,eval_min_near=eval_min,all_losses_finite=all(bool(torch.isfinite(v))for v in losses.values())))
   assert digest_model(m)==before;del out,losses,metrics,grads
  equal(p.state_dict(),saved['pipeline'])
  for k,o in t.optimizers.optimizers.items():equal(o.state_dict(),saved['optimizers'][k])
  for k,s in t.optimizers.schedulers.items():equal(s.state_dict(),saved['schedulers'][k])
  equal(t.grad_scaler.state_dict(),saved['scalers'])
  report=dict(passed=all(r['combined_scaled_gradients_finite']and r['all_losses_finite']for r in rows),config_sha256=sha(a.config),spec_sha256=sha(a.spec),source_sha256=sha(__file__),parent_checkpoint_sha256=sha(cfg.load_checkpoint),runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),python=sys.version),actual_Trainer_setup_full_pipeline_exact=True,Adam_scheduler_scaler_values_exact=True,parent_RNG_all_components_exact=True,native_last_RNG_restore_exercised=True,parent_Adam_updates=3998,parent_scale=scale,parent_LR=t.optimizers.optimizers['fields'].param_groups[0]['lr'],new_NN_parameters_or_buffers=False,field_and_grid_unchanged=True,optimizer_updates=0,training_iterations_called=0,callbacks_not_run=[c.func.__name__ for c in t.callbacks],packets=rows,phase_start4001=4001,end_exclusive=21960,prior_charged_attempts=4041,new_attempt_cap=17959,total_attempt_upper_bound=22000,initial_dynamic_request_reset=64,trajectory_note=spec['trajectory_note'])
 finally:_restore_rng_state(original)
 equal(_capture_rng_state(),original);report['probe_process_global_RNG_restored']=True;(a.output/'proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],packets=rows)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
