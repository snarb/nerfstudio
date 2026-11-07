"""Same frozen quarter+.1 field: native/source-margin/restored; no updates."""
import argparse,copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def absolute(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 if Path(module.__file__).resolve()!=path.resolve():raise ValueError('Imported module identity mismatch')
 return module

def compare_code_images(first,last):
 import numpy as np
 if first.shape!=last.shape or first.dtype!=last.dtype:raise ValueError('Restored image shape/dtype differs')
 if first.dtype not in (np.uint8,np.uint16):raise ValueError('Expected saved integer RGB/alpha')
 delta=np.abs(first.astype(np.int64)-last.astype(np.int64))
 maximum=int(delta.max(initial=0))
 if maximum>1:raise ValueError('Restored FHD render exceeds one saved-code rounding tolerance')
 fraction=float((delta>0).mean()) if delta.size else 0.
 limit=1e-5 if first.dtype==np.uint8 else .001
 if fraction>limit:raise ValueError('Restored FHD changed fraction exceeds campaign replay policy')
 return dict(restored_bitwise_equal=bool(maximum==0),restored_max_code_error=maximum,restored_changed_values=int((delta>0).sum()),restored_tolerance_codes=1,restored_changed_fraction=fraction,restored_changed_fraction_limit=limit)

def verify_reference(first,reference,step):
 import numpy as np
 from PIL import Image
 path=first/'renders'/f'step-{step:09d}'
 for camera in ['DC','HC','KA']:
  for suffix in ['.png','_alpha.png']:
   compare_code_images(np.asarray(Image.open(path/(camera+suffix))),np.asarray(Image.open(reference/(camera+suffix))))
 compare_metrics(json.loads((path/'metrics.json').read_text()),json.loads((reference/'metrics.json').read_text()))

def compare_metrics(first,other):
 pairs=[(first['metrics'],other['metrics'])]
 pairs += [(first[key][name],other[key][name])for key in ['by_view','by_actor']for name in first[key]]
 pairs += list(zip(first['records'],other['records']))
 if len(first['records'])!=len(other['records']):raise ValueError('Metric domains differ')
 for a,b in pairs:
  if any(abs(a[key]-b[key])>1e-5 for key in ['psnr','ssim','lpips']):raise ValueError('Identity/source metric parity failed before B')

def verify_replay(first,other,step,queries):
 import numpy as np
 from PIL import Image
 first_render=first/'renders'/f'step-{step:09d}'
 other_render=other/'renders'/f'step-{step:09d}'
 for camera in ['DC','HC','KA']:
  for suffix in ['.png','_alpha.png']:
   compare_code_images(np.asarray(Image.open(first_render/(camera+suffix))),np.asarray(Image.open(other_render/(camera+suffix))))
 compare_metrics(json.loads((first_render/'metrics.json').read_text()),json.loads((other_render/'metrics.json').read_text()))
 for label in queries:
  paths=list((first/(label+'_trace/rays')).glob('pixel*_native.npz'))
  if not paths:raise ValueError('Missing identity native arrays')
  for path in paths:
   with np.load(path)as a,np.load(other/(label+'_trace/rays')/path.name)as b:
    if set(a.files)!=set(b.files)or any(not np.array_equal(a[k],b[k])for k in a.files):raise ValueError('All-ones identity native arrays differ before B')


def verify_density_restriction(first,changed,queries,lookup,device):
 import numpy as np
 import torch
 records=[]
 for label in queries:
  paths=list((first/(label+'_trace/rays')).glob('pixel*_uniform2x.npz'))
  if not paths:raise ValueError('Missing dense16384 selector proof arrays')
  for path in paths:
   with np.load(path)as a,np.load(changed/(label+'_trace/rays')/path.name)as b:
    if not np.array_equal(a['xyz'],b['xyz']):raise ValueError('Dense comparison positions changed')
    keep=lookup(torch.as_tensor(a['xyz'],device=device)).cpu().numpy().reshape(-1)
    before=a['sigma'].reshape(-1);after=b['sigma'].reshape(-1)
    if not np.isfinite(before).all()or not np.isfinite(after).all():raise ValueError('Nonfinite actual density')
    if (after[~keep]!=0).any()or not np.array_equal(before[keep],after[keep]):raise ValueError('Actual field density selector mismatch')
    records.append(dict(trace=label+'/'+path.name,retained=int(keep.sum()),excluded=int((~keep).sum()),nonzero_removed=int((before[~keep]>0).sum()),outside_density_exact_zero=True,admitted_density_bitwise_equal=True))
 return dict(passed=True,records=records,meaning='Actual dense field query, not CPU density reconstruction',occupancy_changed=False)


def main(a):
 policy=json.loads((a.root/'policy.json').read_text())
 inherited=Path(policy['inherited_identity_receipt'])
 if sha(inherited)!=policy['inherited_identity_receipt_sha256']:raise ValueError('Inherited identity proof drift')
 prior=json.loads(inherited.read_text())
 if not prior.get('all_ones_identity_passed_before_B')or not prior.get('passed')or not prior.get('seven_states_unchanged'):raise ValueError('Unqualified inherited identity proof')
 arm=next(x for x in policy['models']if x['name']==a.arm)
 if prior['source_checkpoint_sha256']!=arm['files_sha256']['checkpoint.ckpt']:raise ValueError('Inherited proof source differs')
 archive=Path(arm['archive']);branch=Path(arm['branch']);helpers=a.root/'helpers';out=a.root/a.arm;out.mkdir(exist_ok=True)
 sys.path[:0]=[str(helpers),str(branch/'code'),str(branch/'code/LookCloser/scripts')]
 isolation=absolute('_qualified_gpu_isolation',helpers/'marshawn_gpu_isolation.py')
 expected_uuid=os.environ.get('EXPECTED_GPU_UUID','')
 isolation.require_environment(expected_uuid)
 import torch,numpy as np
 write(out/'gpu_identity.json',isolation.verify_cuda(expected_uuid,torch))
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(policy['allocator_cap_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
 probe=absolute('probe_marshawn_occupancy',helpers/'probe_marshawn_occupancy.py')
 late=absolute('_qualified_tree_digest',helpers/'train_marshawn_semantic_late_distortion.py')
 selector=absolute('_readonly_point_selector',helpers/'marshawn_readonly_point_selector.py')
 point_module=absolute('_proven_source_point_admission',helpers/'marshawn_source_point_admission.py')
 margin_module=absolute('_geometric_source_margin',helpers/'marshawn_source_margin_admission.py')
 import marshawn_semantic_occupancy as semantic
 import nerfstudio.utils.eval_utils as eu
 from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
 from marshawn_occupancy_audit import dilate_one
 import train_marshawn_tuning as training
 bindings={m.__name__:dict(path=str(Path(m.__file__).resolve()),sha256=sha(m.__file__))for m in [probe,late,semantic,training,selector,point_module,margin_module,isolation]}
 for name in ['probe_marshawn_occupancy.py','train_marshawn_semantic_late_distortion.py']:
  if sha(helpers/name)!=policy['helpers_sha256'][name]:raise ValueError('Helper hash differs')
 write(out/'import_binding.json',bindings)
 cp=archive/'checkpoint.ckpt';payload=torch.load(cp,map_location='cpu',weights_only=False)
 keys=['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'];serialized={k:late.tree_digest(payload[k])for k in keys}
 if payload['step']!=arm['step']:raise ValueError('Wrong boundary state')
 def configure(cfg):
  cfg.eval_checkpoint=cp;cfg.pipeline.model.occupancy_eval_dilation_radius=0
  cfg.pipeline.datamanager.data=Path(policy['data']);cfg.pipeline.datamanager.dataparser.data=Path(policy['data']);return cfg
 result=eu.eval_setup(archive/'config.yml',test_mode='val',update_config_callback=configure);cfg,pipe,checkpoint,step=result;model=pipe.model;pipe.eval();model.current_train_step=step
 if step!=arm['step'] or model.config.adaptive_coarse_step_size!=arm['coarse_step']or model.field.density_activation!='trunc_exp'or not model.field.correct_sh_directions:raise ValueError('Actual field recipe differs')
 if not getattr(model.field,'_semantic_admission_installed',False):raise ValueError('Missing semantic admission')
 core,active,ema,_,record=semantic.load_snapshot(archive/'grid.json',cp)
 if not np.array_equal(active,model.occupancy_grid.binaries.cpu().numpy())or not np.array_equal(ema,model.occupancy_grid.occs.cpu().numpy().reshape(ema.shape)):raise ValueError('Live grid differs')
 with np.load(archive/'grid.npz')as z:admission=z['admission'].copy()
 def profiles(native,occs,radius,loadedstep,warmup,*args):
  if loadedstep!=step or radius!=0 or not np.array_equal(native,active)or not np.array_equal(occs.reshape(ema.shape),ema):raise ValueError('Frozen occupancy changed')
  return dict(native=active),dict(record,whole_field_reference=True,semantic_density_lookup=current_phase[0],traversal_admission_unchanged=True,cpu_reconstruction_used=False)
 old_setup=eu.eval_setup;old_profiles=probe.occupancy_profiles;oldargv=sys.argv
 eu.eval_setup=lambda *args,**kwargs:result;probe.occupancy_profiles=profiles
 initial_strength=(model.config.feature_reweighting_strength,model.field.feature_reweighting_strength)
 initial_cap=model.config.max_steps_per_ray
 current_phase=['native_A']
 source_file=Path(policy['source_grid'])
 if sha(source_file)!=policy['source_grid_sha256']:raise ValueError('Source point grid changed')
 with np.load(source_file,allow_pickle=False)as z:
  lookups={m:margin_module.GeometricSourcePointAdmission(torch.as_tensor(z['semantic_gate_pad1'],device=model.device),z['aabb'],margin=m,epsilon=2e-7,base_class=point_module.SourcePointAdmission)for m in [0.,.001,.002]}
 native_normalize=model.field._normalize_positions
 source_mask_hash=sha(source_file)
 if initial_cap!=1024 or model.config.corrected_arm_allocator:raise ValueError('Original legacy1024 sampler required')
 lut=model.field._feature_weight_lut.clone();lut_strength=model.field._feature_weight_lut_strength
 model.config.feature_reweighting_strength=arm['FR'];model.field.feature_reweighting_strength=arm['FR']
 rng=_capture_rng_state();state_before=late.tree_digest(pipe.state_dict());rng_before=late.tree_digest(rng)
 ready=json.loads((Path(policy['data'])/'ready.json').read_text());phases=[]
 try:
  for name in ['native_A','margin0_queries_only','margin001','margin002','native_A_restored']:
   cap=initial_cap;current_phase[0]=name
   selector.restore_normalization(model.field,native_normalize)
   margin={'margin0_queries_only':0.,'margin001':.001,'margin002':.002}.get(name)
   if margin is not None:
    verify_reference(out/'native_A',Path(arm['reference_renders']),step)
    selector.compose_normalization(model.field,lookups[margin])
   strength=arm['FR'];model.config.max_steps_per_ray=cap
   dest=out/name;dest.mkdir(exist_ok=True);_restore_rng_state(rng)
   model.config.feature_reweighting_strength=strength;model.field.feature_reweighting_strength=strength
   model.field._feature_weight_lut=lut.clone();model.field._feature_weight_lut_strength=lut_strength
   pipe.eval()
   metrics=None if name=='margin0_queries_only' else training.evaluate(SimpleNamespace(pipeline=pipe),ready,Path(policy['data']),dest,step);pipe.eval()
   for label,query in policy['queries'].items():
    request=json.loads(Path(query).read_text());ds=getattr(pipe.datamanager,request.get('split','eval')+'_dataset')
    names=[Path(x).stem.split('_')[-1]for x in ds.image_filenames]
    if names!=request['camera_order']:raise ValueError('Query camera ordering differs')
    errors=[]
    for px,witness in zip(request['pixels'],request['rays']):
     cam=ds.cameras[px['camera']:px['camera']+1].to(model.device);r=cam.generate_rays(0,coords=torch.tensor([[px['y']+.5,px['x']+.5]],device=model.device),keep_shape=False)
     errors.append(max(float(np.max(abs(r.origins[0].cpu().numpy()-witness['origin']))),float(np.max(abs(r.directions[0].cpu().numpy()-witness['direction'])))))
    if max(errors)>1e-6:raise ValueError('Physical ray coordinate mismatch')
    trace=dest/(label+'_trace');trace.mkdir(exist_ok=True)
    sys.argv=[probe.__file__,'--config',str(archive/'config.yml'),'--checkpoint',str(cp),'--queries',query,'--output',str(trace/'rays'),'--data',policy['data'],'--uniform-samples','8192'];probe.main()
    write(trace/'coordinate_chain.json',dict(max_error=max(errors),FR=strength,max_steps_per_ray=cap,query_sha256=sha(query)))
   if margin is not None:
    proof=verify_density_restriction(out/'native_A',dest,policy['queries'],lookups[margin],model.device)
    write(dest/'actual_density_selector_proof.json',proof)
   state_after=late.tree_digest(pipe.state_dict())
   if state_after!=state_before:raise ValueError('FR inference changed persistent model/grid state')
   if late.tree_digest(_capture_rng_state())!=rng_before:raise ValueError('Inference changed global RNG')
   phases.append(dict(name=name,FR=strength,max_steps_per_ray=cap,geometric_Linf_margin=margin,numerical_epsilon=2e-7,metrics=None if metrics is None else {k:metrics['metrics'][k]for k in ['psnr','ssim','lpips']},persistent_pipeline_sha256=state_after,global_rng_sha256=rng_before,actual_LUT_strength=model.field._feature_weight_lut_strength))
   write(out/'progress.json',dict(phases=phases))
 finally:
  model.config.max_steps_per_ray=initial_cap
  selector.restore_normalization(model.field,native_normalize)
  model.config.feature_reweighting_strength,model.field.feature_reweighting_strength=initial_strength
  model.field._feature_weight_lut=lut;model.field._feature_weight_lut_strength=lut_strength
  _restore_rng_state(rng);eu.eval_setup=old_setup;probe.occupancy_profiles=old_profiles;sys.argv=oldargv
 restored={k:late.tree_digest(payload[k])for k in keys}
 if restored!=serialized or late.tree_digest(pipe.state_dict())!=state_before or late.tree_digest(_capture_rng_state())!=rng_before:raise ValueError('Source/fullstate restoration mismatch')
 if model.field._normalize_positions is not native_normalize:raise ValueError('Original field method not restored')
 if model.config.max_steps_per_ray!=initial_cap:raise ValueError('Sampler cap restore mismatch')
 if not torch.equal(model.field._feature_weight_lut,lut)or model.field._feature_weight_lut_strength!=lut_strength:raise ValueError('Nonpersistent LUT restore mismatch')
 png_checks={}
 for camera in ['DC','HC','KA']:
  for suffix in ['.png','_alpha.png']:
   file=camera+suffix;first=out/'native_A/renders'/f'step-{step:09d}'/file;last=out/'native_A_restored/renders'/f'step-{step:09d}'/file
   from PIL import Image
   restored_image=compare_code_images(np.asarray(Image.open(first)),np.asarray(Image.open(last)))
   source_check=compare_code_images(np.asarray(Image.open(first)),np.asarray(Image.open(Path(arm['reference_renders'])/file)))
   png_checks[file]=dict(**restored_image,source_replay=source_check)
 first_metrics=json.loads((out/'native_A/renders'/f'step-{step:09d}'/'metrics.json').read_text());restored_metrics=json.loads((out/'native_A_restored/renders'/f'step-{step:09d}'/'metrics.json').read_text())
 source_metrics=json.loads((Path(arm['reference_renders'])/'metrics.json').read_text())
 metric_deltas={}
 domains=[('aggregate',first_metrics['metrics'],restored_metrics['metrics'])]+[(key,first_metrics['by_actor'][key],restored_metrics['by_actor'][key])for key in first_metrics['by_actor']]+[(key,first_metrics['by_view'][key],restored_metrics['by_view'][key])for key in first_metrics['by_view']]+[(str(i),x,y)for i,(x,y)in enumerate(zip(first_metrics['records'],restored_metrics['records']))]
 domains += [('source/'+domain,x,z)for domain,x,z in [('aggregate',first_metrics['metrics'],source_metrics['metrics'])]+[(key,first_metrics['by_actor'][key],source_metrics['by_actor'][key])for key in first_metrics['by_actor']]+[(key,first_metrics['by_view'][key],source_metrics['by_view'][key])for key in first_metrics['by_view']]+[(str(i),x,z)for i,(x,z)in enumerate(zip(first_metrics['records'],source_metrics['records']))]]
 for domain,x,y in domains:
  for key in ['psnr','ssim','lpips']:
   delta=abs(x[key]-y[key]);metric_deltas[domain+'/'+key]=delta
   if delta>1e-5:raise ValueError('Restored metric exceeds campaign replay tolerance')
 for label in policy['queries']:
  for path in (out/'native_A'/(label+'_trace/rays')).glob('pixel*_native.npz'):
   with np.load(path)as before,np.load(out/'native_A_restored'/(label+'_trace/rays')/path.name)as after:
    if any(not np.array_equal(before[k],after[k])for k in before.files):raise ValueError('Restored native ray changed')
 write(out/'complete.json',dict(passed=True,step=step,phases=phases,source_checkpoint_sha256=sha(cp),serialized_seven_state_sha256=serialized,seven_states_unchanged=True,private_focus_state_sha256=late.tree_digest(payload['tuning_runtime'].get('frequency_focus_state')),nonpersistent_LUT_restored=True,global_rng_restored=True,source_render_parity=png_checks,restored_metric_deltas=metric_deltas,restored_metric_tolerance=1e-5,optimizer_updates=0,frequency_updates=0,occupancy_updates=0,quality_promotion=False,actual_FR=arm['FR'],sampler_cap_restored=True,source_point_grid_sha256=source_mask_hash,original_normalization_restored=True,all_ones_identity_inherited_from_hashbound_prior=True,inherited_proof_sha256=policy['inherited_identity_receipt_sha256'],geometric_Linf_margins=[0.,.001,.002],numerical_epsilon=2e-7,limitation='Inference-only source point density restriction with fixed coarse traversal/core/EMA/frequency. Not a trained asset or whole-scene empty-space certificate.'))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
