"""Bound read-only native/dense replay of a strict margin-continuation CHILD.

An ordinary semantic eval_setup is deliberately insufficient: restore the
checkpoint's explicit selector and nonpersistent FR/LUT before every query.
"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def absolute(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 if Path(m.__file__).resolve()!=Path(path).resolve():raise ValueError('Import source mismatch')
 return m

def run(plan_path,arm):
 plan=read(plan_path);release=read(plan['release'])
 if release.get('reviewer')!='root'or release.get('readonly_authorized')is not True or release.get('plan_sha256')!=sha(plan_path):raise ValueError('Exact read-only release required')
 row=plan['models'][arm];archive=Path(row['archive']);out=Path(row['output']);out.mkdir(parents=True,exist_ok=False);code=Path(plan['code']);helpers=Path(plan['helpers'])
 for p,d in plan['files_sha256'].items():
  if sha(p)!=d:raise ValueError('Bound read-only source drift: '+p)
 for p,d in row['files_sha256'].items():
  if sha(archive/p)!=d:raise ValueError('Bound child artifact drift: '+p)
 sys.path[:0]=[str(code/'LookCloser/scripts'),str(code)]
 isolation=absolute('_margin_readonly_isolation',helpers/'marshawn_gpu_isolation.py');lease=isolation.require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
 import torch,numpy as np
 write(out/'GPU_identity.json',isolation.verify_cuda(lease['expected_physical_uuid'],torch));torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
 probe=absolute('probe_marshawn_occupancy',helpers/'probe_marshawn_occupancy.py');late=absolute('_margin_readonly_state',helpers/'train_marshawn_semantic_late_distortion.py')
 import marshawn_margin_continuation_policy as margin
 import marshawn_semantic_occupancy as semantic
 import train_marshawn_semantic_admission as training
 import nerfstudio.utils.eval_utils as eu
 from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
 modules=[isolation,probe,late,margin,semantic,training]
 bindings={m.__name__:dict(path=str(Path(m.__file__).resolve()),sha256=sha(m.__file__))for m in modules}
 for b in bindings.values():
  if plan['files_sha256'].get(b['path'])!=b['sha256']:raise ValueError('Actual imported helper not bound')
 write(out/'import_binding.json',bindings)
 checkpoint=archive/'checkpoint.ckpt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False);runtime=payload.get('margin_fork_runtime')
 if not runtime or runtime['selector_state']['binding']['arm']!=arm:raise ValueError('Strict CHILD selector lineage required; composed zero parent uses separate loader')
 keys=['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'];before={k:late.tree_digest(payload[k])for k in keys}
 def config(cfg):cfg.eval_checkpoint=checkpoint;cfg.pipeline.model.occupancy_eval_dilation_radius=0;return cfg
 result=eu.eval_setup(archive/'config.yml',test_mode='val',update_config_callback=config);cfg,pipe,_,step=result;model=pipe.model;pipe.eval();model.current_train_step=step
 if step!=row['step']or model.config.adaptive_coarse_step_size!=.0015625 or model.config.max_steps_per_ray!=1024 or model.config.corrected_arm_allocator or not model.field.correct_sh_directions or model.field.density_activation!='trunc_exp':raise ValueError('Actual source field/sampler recipe differs')
 if late.tree_digest(pipe.state_dict())!=before['pipeline']:raise ValueError('Actual inference pipeline differs from saved child')
 cache=runtime['model_cache'];model.config.feature_reweighting_strength=cache['FR'];model.field.feature_reweighting_strength=cache['FR'];pipe._feature_reweighting_switch_applied=cache['switch_applied'];model.field._feature_weight_lut=cache['feature_weight_LUT'].to(model.device);model.field._feature_weight_lut_strength=cache['strength']
 if cache['FR']!=.3 or not cache['switch_applied']:raise ValueError('Bound actual FR.3 runtime required')
 gate=margin.install_checkpoint_margin(model,payload,archive/'spec.json');gate.verify_installed()
 core,active,ema,_,record=semantic.load_snapshot(archive/'grid.json',checkpoint)
 if not np.array_equal(active,model.occupancy_grid.binaries.cpu().numpy())or not np.array_equal(ema,model.occupancy_grid.occs.cpu().numpy().reshape(ema.shape)):raise ValueError('Actual saved occupancy/EMA mismatch')
 _restore_rng_state(payload['rng_state']);rng=_capture_rng_state();rng_hash=late.tree_digest(rng);lut_hash=late.tree_digest(model.field._feature_weight_lut)
 state=late.tree_digest(pipe.state_dict());data=Path(read(archive/'spec.json')['parent_spec']);data=Path(read(data)['data']);ready=read(data/'ready.json')
 metrics=training.evaluate(SimpleNamespace(pipeline=pipe),ready,data,out,step);pipe.eval()
 late.verify_numeric_render_parity(str(archive/'renders'),out/'renders'/f'step-{step:09d}',{p.name:sha(p)for p in(archive/'renders').glob('*.png')},read(archive/'renders/metrics.json'),metrics)
 original_setup=eu.eval_setup;original_profiles=probe.occupancy_profiles;original_argv=sys.argv
 eu.eval_setup=lambda *a,**kw:result
 def profiles(native,occs,radius,loadedstep,warmup,*unused):
  if radius!=0 or loadedstep!=step or not np.array_equal(native,active)or not np.array_equal(occs.reshape(ema.shape),ema):raise ValueError('Native grid changed during trace')
  return dict(native=active),dict(record,source_selector=gate.binding,all_profiles_keep_density_gate=True,cpu_reconstruction_used=False)
 probe.occupancy_profiles=profiles
 try:
  for label,path in plan['queries'].items():
   request=read(path);ds=getattr(pipe.datamanager,request.get('split','eval')+'_dataset');names=[Path(x).stem.split('_')[-1]for x in ds.image_filenames]
   if names!=request['camera_order']:raise ValueError('Query camera order differs')
   errors=[]
   for px,w in zip(request['pixels'],request['rays']):
    camera=ds.cameras[px['camera']:px['camera']+1].to(model.device);ray=camera.generate_rays(0,coords=torch.tensor([[px['y']+.5,px['x']+.5]],device=model.device),keep_shape=False);errors.append(max(float(np.max(abs(ray.origins[0].cpu().numpy()-w['origin']))),float(np.max(abs(ray.directions[0].cpu().numpy()-w['direction'])))))
   if len(errors)!=len(request['pixels'])or max(errors)>1e-6:raise ValueError('Physical query chain mismatch')
   trace=out/(label+'_trace');trace.mkdir();gate.verify_installed();sys.argv=[probe.__file__,'--config',str(archive/'config.yml'),'--checkpoint',str(checkpoint),'--queries',path,'--output',str(trace/'rays'),'--data',str(data),'--uniform-samples','8192'];probe.main();gate.verify_installed();write(trace/'coordinate_chain.json',dict(max_error=max(errors),query_sha256=sha(path),actual_FR=.3,source_selector=gate.binding))
 finally:eu.eval_setup=original_setup;probe.occupancy_profiles=original_profiles;sys.argv=original_argv
 if late.tree_digest(pipe.state_dict())!=state or late.tree_digest(_capture_rng_state())!=rng_hash or late.tree_digest(model.field._feature_weight_lut)!=lut_hash:raise ValueError('Read-only pipeline/RNG/LUT changed')
 if before!={k:late.tree_digest(payload[k])for k in keys}:raise ValueError('Serialized source seven states changed')
 gate.verify_installed();write(out/'complete.json',dict(passed=True,step=step,checkpoint_sha256=sha(checkpoint),source_selector=gate.binding,actual_FR=.3,coarse_step=.0015625,cap=1024,native_R0=True,seven_states_unchanged=True,RNG_LUT_unchanged=True,optimizer_updates=0,occupancy_updates=0,frequency_updates=0,quality_promotion=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();run(a.plan,a.arm)
