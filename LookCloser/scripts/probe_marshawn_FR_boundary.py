"""Same frozen state FR1 -> FR.3 -> restored FR1; no optimizer or grid updates."""
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

def main(a):
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 policy=json.loads((a.root/'policy.json').read_text());arm=next(x for x in policy['models']if x['name']==a.arm)
 archive=Path(arm['archive']);branch=Path(arm['branch']);helpers=a.root/'helpers';out=a.root/a.arm;out.mkdir(exist_ok=True)
 sys.path[:0]=[str(helpers),str(branch/'code'),str(branch/'code/LookCloser/scripts')]
 import torch,numpy as np
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(policy['allocator_cap_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
 probe=absolute('probe_marshawn_occupancy',helpers/'probe_marshawn_occupancy.py')
 late=absolute('_qualified_tree_digest',helpers/'train_marshawn_semantic_late_distortion.py')
 import marshawn_semantic_occupancy as semantic
 import nerfstudio.utils.eval_utils as eu
 from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
 from marshawn_occupancy_audit import dilate_one
 import train_marshawn_tuning as training
 bindings={m.__name__:dict(path=str(Path(m.__file__).resolve()),sha256=sha(m.__file__))for m in [probe,late,semantic,training]}
 for name in ['probe_marshawn_occupancy.py','train_marshawn_semantic_late_distortion.py']:
  if sha(helpers/name)!=policy['helpers_sha256'][name]:raise ValueError('Helper hash differs')
 write(out/'import_binding.json',bindings)
 cp=archive/'checkpoint.ckpt';payload=torch.load(cp,map_location='cpu',weights_only=False)
 keys=['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture'];serialized={k:late.tree_digest(payload[k])for k in keys}
 if payload['step']!=75940:raise ValueError('Wrong boundary state')
 def configure(cfg):
  cfg.eval_checkpoint=cp;cfg.pipeline.model.occupancy_eval_dilation_radius=0
  cfg.pipeline.datamanager.data=Path(policy['data']);cfg.pipeline.datamanager.dataparser.data=Path(policy['data']);return cfg
 result=eu.eval_setup(archive/'config.yml',test_mode='val',update_config_callback=configure);cfg,pipe,checkpoint,step=result;model=pipe.model;pipe.eval();model.current_train_step=step
 if step!=75940 or model.config.adaptive_coarse_step_size!=arm['coarse_step']or model.field.density_activation!='trunc_exp'or not model.field.correct_sh_directions:raise ValueError('Actual field recipe differs')
 if not getattr(model.field,'_semantic_admission_installed',False):raise ValueError('Missing semantic admission')
 core,active,ema,_,record=semantic.load_snapshot(archive/'grid.json',cp)
 if not np.array_equal(active,model.occupancy_grid.binaries.cpu().numpy())or not np.array_equal(ema,model.occupancy_grid.occs.cpu().numpy().reshape(ema.shape)):raise ValueError('Live grid differs')
 with np.load(archive/'grid.npz')as z:admission=z['admission'].copy()
 def profiles(native,occs,radius,loadedstep,warmup,*args):
  if loadedstep!=step or radius!=0 or not np.array_equal(native,active)or not np.array_equal(occs.reshape(ema.shape),ema):raise ValueError('Frozen occupancy changed')
  return dict(native=active,base=core&admission,halo1=dilate_one(core)&admission,all_cells=np.ones_like(active)),dict(record,whole_field_reference=True,semantic_density_support_unchanged=True,cpu_reconstruction_used=False)
 old_setup=eu.eval_setup;old_profiles=probe.occupancy_profiles;oldargv=sys.argv
 eu.eval_setup=lambda *args,**kwargs:result;probe.occupancy_profiles=profiles
 initial_strength=(model.config.feature_reweighting_strength,model.field.feature_reweighting_strength)
 lut=model.field._feature_weight_lut.clone();lut_strength=model.field._feature_weight_lut_strength
 model.config.feature_reweighting_strength=1.;model.field.feature_reweighting_strength=1.
 rng=_capture_rng_state();state_before=late.tree_digest(pipe.state_dict());rng_before=late.tree_digest(rng)
 ready=json.loads((Path(policy['data'])/'ready.json').read_text());phases=[]
 try:
  for name,strength in [('FR1',1.),('FR03',.3),('FR1_restored',1.)]:
   dest=out/name;dest.mkdir(exist_ok=True);_restore_rng_state(rng)
   model.config.feature_reweighting_strength=strength;model.field.feature_reweighting_strength=strength
   model.field._feature_weight_lut=lut.clone();model.field._feature_weight_lut_strength=lut_strength
   pipe.eval()
   metrics=training.evaluate(SimpleNamespace(pipeline=pipe),ready,Path(policy['data']),dest,step);pipe.eval()
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
    write(trace/'coordinate_chain.json',dict(max_error=max(errors),FR=strength,query_sha256=sha(query)))
   state_after=late.tree_digest(pipe.state_dict())
   if state_after!=state_before:raise ValueError('FR inference changed persistent model/grid state')
   if late.tree_digest(_capture_rng_state())!=rng_before:raise ValueError('Inference changed global RNG')
   phases.append(dict(name=name,FR=strength,metrics={k:metrics['metrics'][k]for k in ['psnr','ssim','lpips']},persistent_pipeline_sha256=state_after,global_rng_sha256=rng_before,actual_LUT_strength=model.field._feature_weight_lut_strength))
   write(out/'progress.json',dict(phases=phases))
 finally:
  model.config.feature_reweighting_strength,model.field.feature_reweighting_strength=initial_strength
  model.field._feature_weight_lut=lut;model.field._feature_weight_lut_strength=lut_strength
  _restore_rng_state(rng);eu.eval_setup=old_setup;probe.occupancy_profiles=old_profiles;sys.argv=oldargv
 restored={k:late.tree_digest(payload[k])for k in keys}
 if restored!=serialized or late.tree_digest(pipe.state_dict())!=state_before or late.tree_digest(_capture_rng_state())!=rng_before:raise ValueError('Source/fullstate restoration mismatch')
 if not torch.equal(model.field._feature_weight_lut,lut)or model.field._feature_weight_lut_strength!=lut_strength:raise ValueError('Nonpersistent LUT restore mismatch')
 png_checks={}
 for camera in ['DC','HC','KA']:
  for suffix in ['.png','_alpha.png']:
   file=camera+suffix;first=out/'FR1/renders/step-000075940'/file;last=out/'FR1_restored/renders/step-000075940'/file
   from PIL import Image
   restored_image=compare_code_images(np.asarray(Image.open(first)),np.asarray(Image.open(last)))
   source_check=compare_code_images(np.asarray(Image.open(first)),np.asarray(Image.open(archive/'renders'/file)))
   png_checks[file]=dict(**restored_image,source_replay=source_check)
 first_metrics=json.loads((out/'FR1/renders/step-000075940/metrics.json').read_text());restored_metrics=json.loads((out/'FR1_restored/renders/step-000075940/metrics.json').read_text())
 source_metrics=json.loads((archive/'renders/metrics.json').read_text())
 metric_deltas={}
 domains=[('aggregate',first_metrics['metrics'],restored_metrics['metrics'])]+[(key,first_metrics['by_actor'][key],restored_metrics['by_actor'][key])for key in first_metrics['by_actor']]+[(key,first_metrics['by_view'][key],restored_metrics['by_view'][key])for key in first_metrics['by_view']]+[(str(i),x,y)for i,(x,y)in enumerate(zip(first_metrics['records'],restored_metrics['records']))]
 domains += [('source/'+domain,x,z)for domain,x,z in [('aggregate',first_metrics['metrics'],source_metrics['metrics'])]+[(key,first_metrics['by_actor'][key],source_metrics['by_actor'][key])for key in first_metrics['by_actor']]+[(key,first_metrics['by_view'][key],source_metrics['by_view'][key])for key in first_metrics['by_view']]+[(str(i),x,z)for i,(x,z)in enumerate(zip(first_metrics['records'],source_metrics['records']))]]
 for domain,x,y in domains:
  for key in ['psnr','ssim','lpips']:
   delta=abs(x[key]-y[key]);metric_deltas[domain+'/'+key]=delta
   if delta>1e-5:raise ValueError('Restored metric exceeds campaign replay tolerance')
 for label in policy['queries']:
  for path in (out/'FR1'/(label+'_trace/rays')).glob('pixel*_native.npz'):
   with np.load(path)as before,np.load(out/'FR1_restored'/(label+'_trace/rays')/path.name)as after:
    if any(not np.array_equal(before[k],after[k])for k in before.files):raise ValueError('Restored native ray changed')
 write(out/'complete.json',dict(passed=True,step=step,phases=phases,source_checkpoint_sha256=sha(cp),serialized_seven_state_sha256=serialized,seven_states_unchanged=True,private_focus_state_sha256=late.tree_digest(payload['tuning_runtime'].get('frequency_focus_state')),nonpersistent_LUT_restored=True,global_rng_restored=True,source_render_parity=png_checks,restored_metric_deltas=metric_deltas,restored_metric_tolerance=1e-5,optimizer_updates=0,frequency_updates=0,occupancy_updates=0,quality_promotion=False,limitation='Same-state function jump with fixed learned occupancy; not adapted training quality. Dense references keep semantic support and are not geometric truth.'))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
