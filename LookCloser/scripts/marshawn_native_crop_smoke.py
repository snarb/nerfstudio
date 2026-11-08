"""Finite native warmup/resume/ARM canary; no quality claim or promotion."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,hashlib
from marshawn_native_crop_queue import read,write,sha


def run_child(worker,spec,stop,log):
 fd=os.environ.get('LOOKCLOSER_GPU_LEASE_FD');fds=(int(fd),)if fd else ()
 if not fd:raise ValueError('Qualified inherited global GPU lease required')
 command=[sys.executable,str(worker),'--spec',str(spec),'--smoke-steps',str(stop)]
 with Path(log).open('ab')as stream:r=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=fds)
 return r.returncode


def main(root):
 root=Path(root);O=root/'ops/smokes';O.mkdir(parents=True,exist_ok=True);settings=read(O/'settings.json');worker=Path(settings['worker'])
 if sha(worker)!=settings['worker_sha256']:raise ValueError('Smoke worker changed')
 ready=root/'data_native/ready.json';frequency=read(root/'data_native/frequency_ready.json')
 if frequency.get('ready_sha256')!=sha(ready):raise ValueError('Native66 frequency seal not current')
 warm=Path(settings['warm_spec']);arm=Path(settings['arm_spec']);wd=Path(read(warm)['output']);ad=Path(read(arm)['output'])
 first=None
 if not(wd/'latest.json').exists():
  first=run_child(worker,warm,4,O/'warm_initial.log');write(O/'warm_initial_exit.json',dict(returncode=first,time=time.time()))
  if first!=143:raise ValueError('Expected declared warmup pause exit143, got'+str(first))
 if read(wd/'latest.json')['step']<3:
  code=run_child(worker,warm,4,O/'warm_resume.log')
  if code!=0:raise ValueError('Warmup own-output resume failed'+str(code))
 if not(ad/'complete.json').exists():
  code=run_child(worker,arm,4100,O/'ARM.log')
  if code!=0:raise ValueError('Cold ARM canary failed'+str(code))
 cold=read(wd/'cold_setup_audit.json');resume=read(wd/'resume_setup_audit.json');rng=read(wd/'resume_rng_verification.json');acold=read(ad/'cold_setup_audit.json')
 for record in [cold,acold]:
  if(record.get('cold')is not True or record.get('resume')is not None or record.get('trainer_start_step')!=0 or record.get('occupied_cells')or record.get('nonzero_ema_cells')or record.get('optimizer_state_sizes')!={'fields':0}):raise ValueError('Canary not initially cold')
 if resume['trainer_start_step']!=1 or rng.get('passed')is not True or rng['restored_checkpoint_step']!=0:raise ValueError('Own-checkpoint/RNG resume evidence failed')
 if read(wd/'latest.json')['step']!=3 or read(ad/'latest.json')['step']!=4099:raise ValueError('Canary actual terminal steps differ')
 for output in [wd,ad]:
  result=read(output/'result.json');runtime=read(output/'runtime.json');progress=read(output/'progress.json')
  if result['technical_smoke']is not True or runtime['actual_iteration_limit']!=140000:raise ValueError('Technical/cap evidence differs')
  if not list((output/'renders').rglob('HC.png')):raise ValueError('Actual native HC canary render missing')
  if output==wd:
   if progress.get('cumulative_point_samples',0)<=0 or progress.get('current_train_rays')!=4096:raise ValueError('Actual fixed warmup point/ray counters missing')
  else:
   samples=progress.get('sampling_metrics',{})
   if samples.get('num_samples_per_batch',0)<=0 or 'adaptive_saturation_rate'not in samples:raise ValueError('Actual positive ARM sample/saturation metrics missing')
 row=dict(kind='NATIVE_CROP_WARM_RESUME_AND_COLD_ARM_SMOKE_V1',passed=True,quality_qualified=False,frame=1660,worker_sha256=sha(worker),native_ready_sha256=sha(ready),warm_completed_updates=4,ARM_completed_updates=4,ARM_start_step=4096,actual_terminal_steps=[3,4099],cold_audits=[cold,acold],resume_setup=resume,resume_RNG_verification=rng,warm_output=str(wd),ARM_output=str(ad),actual_render_directories=[str(wd/'renders'),str(ad/'renders')],same_owned_checkpoint_only=True,ROOT_image_and_runtime_review_pending=True,finished=time.time())
 write(O/'qualification.json',row);print(json.dumps(row),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
