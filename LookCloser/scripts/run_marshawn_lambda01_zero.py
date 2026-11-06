"""Exact source-bound lambda0.1 zero-update gate; reuse v2 control only on equal state."""
import argparse,fcntl,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def rows(gpu):
 s=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(x.split(',')[0]),mib=int(x.split(',')[1]))for x in s.splitlines()]
def main(root):
 policy=read(root/'zero_policy.json');release=read(root/'zero_release.json')
 if release.get('approved_zero_only')is not True or release['policy_sha256']!=sha(root/'zero_policy.json'):raise ValueError('Exact zero-only release required')
 if policy['max_seconds']>300 or policy['max_process_mib']>12288:raise ValueError('Unexpected zero scope')
 for name,digest in policy['artifacts_sha256'].items():
  if sha(root/name)!=digest:raise ValueError('Bound artifact changed: '+name)
 for name,digest in read(root/'source_inventory.json')['files'].items():
  if sha(root/'code'/name)!=digest:raise ValueError('Frozen source changed: '+name)
 spec_path=root/'specs/zero_lambda0p1/spec.json';spec=read(spec_path);out=Path(spec['output']);child_release=read(spec['release'])
 if spec['mode']!='zero' or spec['distortion_coefficient']!=.1 or child_release.get('technical_authorized')is not True or child_release.get('diagnostic_authorized')is not False:raise ValueError('Zero-only exact coefficient required')
 if out.exists():raise ValueError('Fresh zero output required')
 gpu=policy['gpu'];lock=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 uuid=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 if uuid!=policy['gpu_uuid']or rows(gpu):raise ValueError('Reviewed GPU is not exclusively idle')
 child=None;started=time.monotonic();checks=[]
 def check():
  active=rows(gpu)
  if any(x['pid']!=(child.pid if child else None)for x in active):raise RuntimeError('Unplanned GPU process')
  memory=sum(x['mib']for x in active)
  if memory>policy['max_process_mib']or time.monotonic()-started>policy['max_seconds']:raise RuntimeError('Zero resource limit')
  row=dict(time=time.time(),gpu=gpu,processes=active,memory_mib=memory,exclusive_global_lock=True);checks.append(row)
  with(root/'zero_resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
 def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root/'code')+':'+str(root/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 try:
  check()
  with(root/'zero_worker.log').open('x')as log:
   child=subprocess.Popen([policy['python'],str(root/'code/LookCloser/scripts/train_marshawn_semantic_late_distortion.py'),'--spec',str(spec_path)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root/'code/LookCloser')
   while child.poll()is None:check();time.sleep(5)
   rc=child.returncode;child=None;check()
  if rc:raise RuntimeError('Zero worker failed '+str(rc))
  control=Path(policy['control_zero_complete']);assert sha(control)==policy['control_zero_sha256']
  original,new=read(control),read(out/'complete.json')
  if original['status']!='zero_update_state_equal'or new['status']!='zero_update_state_equal':raise ValueError('Exact zero-state receipts required')
  keys=('pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture','semantic_support','cache_sha256','cache_sources','sampler','next_step','Adam','FR')
  for key in keys:
   if original['proof'][key]!=new['proof'][key]:raise ValueError('Control reuse state mismatch: '+key)
  config=out/'worker/lookcloser/frozen/config.yml';values=re.findall(r'^\s*distortion_loss_mult:\s*([^\s]+)',config.read_text(),re.M)
  if len(values)!=1 or float(values[0])!=.1:raise ValueError('Actual configured coefficient differs')
  write(root/'zero_complete.json',dict(passed=True,policy_sha256=sha(root/'zero_policy.json'),source_checkpoint_sha256=spec['parent_checkpoint_sha256'],actual_distortion_coefficient=.1,control_state_cache_sampler_RNG_all_equal=True,equal_proof_keys=list(keys),zero_receipt_sha256=sha(out/'complete.json'),control_zero_sha256=sha(control),new_optimizer_updates=0,peak_process_mib=max(x['memory_mib']for x in checks),elapsed_seconds=time.monotonic()-started,existing_control_trajectory_reusable=True,diagnostic_started=False,quality_claim=False))
 except BaseException as e:
  signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  write(root/'zero_failed.json',dict(error=repr(e),other_producers_modified=False));raise
 finally:lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
