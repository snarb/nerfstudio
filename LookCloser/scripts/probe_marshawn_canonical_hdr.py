"""Actual cold canonical HDR zero-update gate; no optimizer/scheduler step."""
from pathlib import Path
import argparse,json,os,sys,time,hashlib,copy
import numpy as np,torch,cv2,yaml
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from probe_marshawn_occupancy import digest_model
from marshawn_canonical_hdr_preflight import canonical_pq_from_unbaked_ap1
from marshawn_canonical_physical_hdr import common_sdr

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gradient(term,named,scale):
 gs=torch.autograd.grad(term*scale,[p for _,p in named],retain_graph=True,allow_unused=True);finite=all(bool(torch.isfinite(g).all())for g in gs if g is not None);values={}
 if finite:
  for(n,_),g in zip(named,gs):
   if g is None:continue
   key='color'if'mlp_color'in n else'geometry';values[key]=values.get(key,0)+float((g.detach().double()/scale).square().sum())
 return dict(finite=finite,L2=sum(values.values())**.5 if finite else None,groups={k:v**.5 for k,v in values.items()})
def main(a):
 a.output.mkdir(exist_ok=False);torch.set_num_threads(2);cv2.setNumThreads(2);original=_capture_rng_state();cfg=yaml.load(a.config.read_text(),Loader=yaml.Loader);cfg.output_dir=a.output/'setup_outputs';cfg.logging.profiler='none';_set_random_seed(42)
 try:
  trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();pipe=trainer.pipeline;m=pipe.model;assert trainer._start_step==0 and not any(o.state for o in trainer.optimizers.optimizers.values());assert m.field.appearance_embedding_dim==0 and not m.renderer_rgb.clamp_output
  for cb in trainer.callbacks:cb.run_callback_at_location(step=0,location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
  before=digest_model(m);keys=set(pipe.state_dict());scale=float(trainer.grad_scaler.get_scale());named=[(n,p)for n,p in m.named_parameters()if p.requires_grad];captured={};oldnext=pipe.datamanager.next_train
  def next_train(step):
   rays,batch=oldnext(step);captured['batch']=batch;captured['rays']=rays;return rays,batch
  pipe.datamanager.next_train=next_train;packets=[]
  for step in[0,1]:
   pipe.train()
   with torch.autocast(device_type='cuda',enabled=True):outputs,losses,metrics=pipe.get_train_loss_dict(step)
   assert set(losses)=={'rgb_loss','distortion_loss','observed_sfm_free_space','observed_sfm_surface_band'}
   batch=captured['batch'];ids=batch['indices'].detach().cpu().numpy();gt=batch['image'].detach().cpu().numpy();assert gt.dtype==np.float32;maximum=0.
   for i in range(min(64,len(ids))):
    cam,y,x=map(int,ids[i]);source=pipe.datamanager.train_dataset.image_filenames[cam];ref=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)[y,x,::-1];maximum=max(maximum,float(abs(gt[i]-ref).max()))
   assert maximum==0.;g={k:gradient(v,named,scale)for k,v in losses.items()};g['combined']=gradient(sum(losses.values()),named,scale);pq=canonical_pq_from_unbaked_ap1(outputs['rgb']);clamped=float((outputs['rgb'].float()*52.13360318135631<.005).float().mean());assert clamped<1. and g['rgb_loss']['groups'].get('color',0)>0
   rows=dict(step=step,actual_rays=len(gt),samples=int(metrics['num_samples_per_batch']),actual_loader_RGB_max_error=maximum,batch_dtype=str(gt.dtype),batch_min=float(gt.min()),batch_max=float(gt.max()),prediction_linear_min=float(outputs['rgb'].min()),prediction_linear_max=float(outputs['rgb'].max()),PQ_prediction_floor_clamp_fraction=clamped,gradient=g,ratios={k:g[k]['L2']/g['rgb_loss']['L2']if g[k]['finite']and g['rgb_loss']['finite']and g['rgb_loss']['L2']else None for k in losses if k!='rgb_loss'})
   packets.append(rows)
   assert digest_model(m)==before and set(pipe.state_dict())==keys and not any(o.state for o in trainer.optimizers.optimizers.values())
   del outputs,losses,metrics
  # HDRrenderer mustretainphysicalradiance>1 in BOTH train/eval.
  m.eval();rgb=torch.tensor([[2.,3.,4.]],device='cuda');weight=torch.ones((1,1),device='cuda');ridx=torch.zeros(1,dtype=torch.long,device='cuda');render=m.renderer_rgb(rgb=rgb,weights=weight,ray_indices=ridx,num_rays=1);assert torch.equal(render,rgb)
  # Physicalblack0 encodes the signedfloor, never add thefloor tolinearRGB.
  v=torch.tensor([0.,.005/52.13360318135631,.02,1.,100.],device='cuda',requires_grad=True);q=canonical_pq_from_unbaked_ap1(v);dg=torch.autograd.grad(q.sum(),v)[0];assert q[0]==q[1]and dg[0]==0 and dg[2]>0 and dg[-1]==0
  # Unequal-sample mixture must encode AFTERlinearcompositing.
  mixture=canonical_pq_from_unbaked_ap1(torch.tensor([1.],device='cuda'));wrong=canonical_pq_from_unbaked_ap1(torch.tensor([0.,2.],device='cuda')).mean();assert abs(float(mixture-wrong))>.01
  # Explicitall3EXRloader+commonSDRpreviews, sourceGT only, NOcalibrationfit.
  previews=[]
  for i,path in enumerate(pipe.datamanager.eval_dataset.image_filenames):
   actual=pipe.datamanager.eval_dataset.get_image_float32(i).cpu().numpy();ref=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1];assert np.array_equal(actual,ref)
   name=Path(path).stem.split('_')[-1];x=torch.from_numpy(actual[::3,::3].copy()).cuda();from PIL import Image
   for label,ev in [('unscalar',0.),('common_oldscalar',cfg.pipeline.model.canonical_old_scalar_EV[name])]:
    display=common_sdr(x,cfg.pipeline.model.canonical_ap1_to_709,ev);fn=name+'_'+label+'_SOURCE_GT.png';Image.fromarray(np.rint(display.cpu().numpy()*255).astype('uint8')).save(a.output/fn)
   previews.append(dict(camera=name,source=str(path),source_sha256=sha(path),all_pixels_loader_exact=True,max_linear=float(actual.max())))
  passed=all(p['gradient']['combined']['finite']and all(g['finite']for g in p['gradient'].values())for p in packets)
  result=dict(passed=passed,config_sha256=sha(a.config),source_sha256=sha(__file__),runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),python=sys.version),packets=packets,loaded_scale=scale,source_previews=previews,renderer_eval_retains_linear_above_one=True,physical_black_linear=0.,signed_PQ_floor_nits=.005,PQ_sample_vs_linear_compositing_distinct=True,field_and_occupancy_state_unchanged_after_initial_native_callback=True,no_optimizer_updates=True,no_scheduler_steps=True,new_NN_params_or_buffers=False,frequency_disabled=True,appearance_dim=0,full_train_AP1_units=True,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_end=time.time())
 finally:_restore_rng_state(original)
 result['process_global_RNG_restored']=True;(a.output/'probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=passed,packets=packets)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
