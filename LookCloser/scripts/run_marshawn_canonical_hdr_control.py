"""Bounded cold canonical physicalHDR controller; no launch until reviewed gates."""
from pathlib import Path
import argparse,csv,hashlib,json,os,signal,sys,threading,time,math
R=Path('/fsx/tmp/marshawn_001660_hd');P=R/'canonical_linear_HDR_PQMSE_STAGED'
sys.path.insert(0,str(R/'code_native/LookCloser/scripts'));sys.path.insert(0,str(R/'native_png_linear_distortion_STAGED/code'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.%d.tmp'%os.getpid());tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def load(path,specpath):
 import yaml
 s=json.loads(specpath.read_text());assert sha(path)==s['config_sha256']
 for f,h in s['sources_sha256'].items():assert sha(f)==h,f
 c=yaml.load(path.read_text(),Loader=yaml.Loader)
 assert c.max_num_iterations==22000 and c.machine.seed==42 and c.pipeline._target.__name__=='CanonicalAP1HDRPipeline'
 assert c.load_checkpoint is None and c.load_dir is None and c.load_step is None and c.eval_checkpoint is None
 assert c.pipeline.target_num_samples==1048576 and c.pipeline.datamanager.cache_images_type=='float32'and c.optimizers['fields']['scheduler'].max_steps==22000
 assert c.pipeline.model.rgb_output_parameterization=='linear_softplus'and c.pipeline.model.reconstruction_loss_type=='pq_mse'and c.pipeline.model.appearance_embedding_dim==0
 assert not c.pipeline.model.enable_frequency_grid and not c.pipeline.model.enable_feature_reweighting and c.pipeline.datamanager.pixel_sampler._target.__name__=='PixelSampler'
 return c,s

def exact_eval_config(config,checkpoint,eval_num_rays_per_chunk=None):
 import yaml
 c=yaml.load(Path(config).read_text(),Loader=yaml.Loader);c.eval_checkpoint=Path(checkpoint).resolve();c.load_checkpoint=None;c.load_dir=None;c.load_step=None;c.load_config=None
 if eval_num_rays_per_chunk is not None:c.pipeline.model.eval_num_rays_per_chunk=eval_num_rays_per_chunk
 out=Path(config).with_name('eval_exact_'+checkpoint.stem+'.yml');out.write_text(yaml.dump(c));return out

def train(path,specpath):
 import numpy as np,torch
 import nerfstudio.scripts.train as native
 cfg,spec=load(path,specpath);assert spec['root_training_authorized']and spec['zeroopt_passed']and spec['peer_review_passed']
 def loop(local_rank,world_size,config,global_rank=0):
  native._set_random_seed(config.machine.seed+global_rank);t=config.setup(local_rank=local_rank,world_size=world_size);t.setup();m=t.pipeline.model
  assert t._start_step==0 and not any(o.state for o in t.optimizers.optimizers.values());assert m.field.appearance_embedding_dim==0 and not m.renderer_rgb.clamp_output
  expected=torch.tensor(config.pipeline.datamanager.dataparser.marshawn_aabb)
  for b in[m.field.aabb,m.occupancy_grid.aabbs[0].reshape(2,3),m.scene_aabb.reshape(2,3)]:torch.testing.assert_close(b.cpu(),expected,rtol=0,atol=1e-6)
  atomic(P/'worker.json',dict(pid=os.getpid(),time=time.time(),cold_seed=42,start_step=0,end_exclusive=22000,attempt_cap=22000,initial_scaler=t.grad_scaler.get_scale(),initial_Adam_state_empty=True,physical_black=0.,PQ_encoding_floor_nits=.005))
  save=t.save_checkpoint
  def bound_save(step):
   save(step);ck=t.checkpoint_dir/('step-%09d.ckpt'%step);folder=P/'grids';folder.mkdir(exist_ok=True);grid=m.occupancy_grid
   np.savez_compressed(folder/('step%09d_core0.npz'%step),occs=grid.occs.detach().cpu().numpy(),active=grid.binaries.detach().cpu().numpy(),aabb=grid.aabbs.detach().cpu().numpy())
   atomic(folder/('step%09d_receipt.json'%step),dict(step=step,checkpoint=str(ck),checkpoint_sha256=sha(ck),policy='actualnativeLCcore0_nohalo_noCPUthresholdreconstruction',grid_sha256=sha(folder/('step%09d_core0.npz'%step))))
  t.save_checkpoint=bound_save;evaluate=t.eval_iteration
  def protected(step):
   if step>0 and step%2000==0:t.save_checkpoint(step)
   return evaluate(step)
  t.eval_iteration=protected;iteration=t.train_iteration
  def observed(step):
   if step>=22000:raise ValueError('Hard22000attemptcap')
   if step==2001:
    ck=t.checkpoint_dir/'step-000002000.ckpt';state=torch.load(ck,map_location='cpu',weights_only=False);assert state['step']==2000 and all(bool(torch.isfinite(v).all())for v in state['pipeline'].values()if torch.is_tensor(v));del state
    atomic(P/'gate2000.json',dict(worker_pid=os.getpid(),next_step=2001,saved_step=2000,attempts=2001,checkpoint=str(ck),checkpoint_sha256=sha(ck),time=time.time(),pause='SIGSTOPsameworkerbefore2001_noextraoptimizerstep'));os.kill(os.getpid(),signal.SIGSTOP)
   result=iteration(step)
   if step<64 or step%500==0:
    counts=[int(v['step'])for o in t.optimizers.optimizers.values()for v in o.state.values()if'step'in v];updates=max(counts)if counts else 0
    row=dict(step=step,attempts=step+1,Adam_updates=updates,AMP_skips=step+1-updates,scale=t.grad_scaler.get_scale(),LR=t.optimizers.optimizers['fields'].param_groups[0]['lr'],weights_finite=all(bool(torch.isfinite(p).all())for p in m.parameters())if step in[31,63]or step%2000==0 else None)
    with(P/'amp_trace.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   return result
  t.train_iteration=observed;t.train()
 native.train_loop=loop;native.main(cfg)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controller','train']);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--spec',type=Path,required=True);a,rest=parser.parse_known_args()
 if a.mode=='train':train(a.config,a.spec)
 else:
  cfg,spec=load(a.config,a.spec);dry='--dry-run'in rest
  if not dry:assert spec['root_training_authorized']and spec['zeroopt_passed']and spec['peer_review_passed']
  import run_bounded_ngp_quiet as quiet
  from marshawn_all_view_early_stop import configure_quiet_all_view_stop
  from run_marshawn_png_distortion_control import health_loop
  quiet.run_dir=lambda options:cfg.get_base_dir() # quietotherwisehardcodesinstant-ngp-bounded
  quiet.eval_config_for_step=exact_eval_config;final_eval=quiet.run_final_eval;terminal_cache={}
  def checked_eval(runpath,checkpoint,label,chunk=None):
   if str(checkpoint)in terminal_cache:return terminal_cache[str(checkpoint)]
   r=final_eval(runpath,checkpoint,label,chunk);assert Path(r['checkpoint']).resolve()==Path(checkpoint).resolve();return r
  quiet.run_final_eval=checked_eval
  def choose(metrics,models):
   rows=[]
   if Path(metrics).exists():
    for r in csv.DictReader(Path(metrics).open()):
     try:step=int(r['step']);values=[float(r[k])for k in['eval_all_psnr','eval_all_ssim','eval_all_lpips']]
     except(ValueError,KeyError,TypeError):continue
     ck=Path(models)/('step-%09d.ckpt'%step)
     if ck.exists()and all(map(math.isfinite,values)):rows.append(dict(checkpoint=str(ck),step=step,psnr=values[0],ssim=values[1],lpips=values[2],source='actualall3PQCSV'))
   terminal=Path(models)/'step-000021999.ckpt'
   if terminal.exists():
    result=checked_eval(Path(models).parent,terminal,'terminal21999',512);terminal_cache[str(terminal)]=result;v=result['results'];rows.append(dict(checkpoint=str(terminal),step=21999,psnr=v['psnr'],ssim=v['ssim'],lpips=v['lpips'],source='freshterminalall3PQ'))
   if not rows:return None,'no_complete_all3metrics'
   maximum=max(r['psnr']for r in rows);win=min((r for r in rows if r['psnr']>=maximum-.07-1e-12),key=lambda r:(r['lpips'],-r['psnr']));atomic(P/'selection.json',dict(candidates=rows,winner=win,rule='Highestall3PQPSNR/lowestLPIPSwithin.07dB'));return Path(win['checkpoint']),'all3PQPSNR_LPIPS_0.07'
  configure_quiet_all_view_stop(quiet,choose);quiet.train_command=lambda options:[sys.executable,str(Path(__file__)),'train','--config',str(a.config),'--spec',str(a.spec)];sys.argv=[sys.argv[0]]+rest;os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
  stop=threading.Event();thread=threading.Thread(target=health_loop,args=(cfg,stop),daemon=True)
  if not dry:thread.start()
  try:status=quiet.main()
  finally:
   stop.set()
   if not dry:thread.join(timeout=2)
  raise SystemExit(status)
