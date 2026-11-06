"""Root-released, idle-exclusive final34376 semantic fork replay; zero optimization."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def rows(gpu):
 text=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(s.split(',')[0]),mib=int(s.split(',')[1]))for s in text.splitlines()]
def main(root):
 policy=json.loads((root/'policy.json').read_text());release=json.loads((root/'release.json').read_text())
 if release.get('approved_readonly')is not True or release['policy_sha256']!=sha(root/'policy.json'):raise ValueError('Exact readonly resource release required')
 for name,digest in policy['helpers_sha256'].items():
  if sha(root/'helpers'/name)!=digest:raise ValueError('Audit helper changed')
 c=Path(policy['campaign']);source=Path(policy['code_root']);inventory=Path(policy['source_inventory'])
 if sha(inventory)!=policy['source_inventory_sha256']:raise ValueError('Fork source inventory changed')
 for name,digest in json.loads(inventory.read_text())['files'].items():
  if sha(source/name)!=digest:raise ValueError('Frozen fork code changed: '+name)
 for name,digest in policy['queries_sha256'].items():
  if sha(c/name)!=digest:raise ValueError('Sealed query changed')
 archives={}
 for label,coefficient in [('control001',.01),('treatment1',1.)]:
  a=c/'independent_occupancy/semantic_late_pair_v2'/label/'step-000034376';receipt=json.loads((a/'archive_receipt.json').read_text())
  if sha(a/'archive_receipt.json')!=policy['archive_receipts_sha256'][label]or receipt['effective_distortion_coefficient']!=coefficient or receipt['actual_grid_schema']!=2:raise ValueError('Wrong fork archive')
  for name,digest in receipt['artifacts_sha256'].items():
   if sha(a/name)!=digest:raise ValueError('Fork archive changed: '+label+'/'+name)
  renders=c/'semantic_late_distortion_candidate_v2/runs'/('diagnostic_'+label)/'renders/step-000034376'
  import shutil
  shutil.copytree(renders,a/'renders',dirs_exist_ok=True);archives[label]=a
 if (root/'complete.json').exists():return
 if (root/'failed.json').exists():raise ValueError('Preserve failed audit generation')
 lease=None;started_wait=time.monotonic()
 while lease is None:
  for gpu in policy['gpus']:
   handle=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock').open('a+')
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:handle.close();continue
   if rows(gpu):handle.close();continue
   lease=handle;break
  if lease is None:
   write(root/'state.json',dict(phase='waiting_idle_global_lock',pid=os.getpid(),time=time.time()))
   if time.monotonic()-started_wait>policy['max_wait_seconds']:raise TimeoutError('No idle GPU; no replay started')
   time.sleep(30)
 child=None;owned=set();started=time.monotonic();checks=[]
 def check(phase):
  current=rows(gpu)
  if any(r['pid']not in owned for r in current):raise RuntimeError('Unexpected exclusive GPU process')
  memory=sum(r['mib']for r in current)
  if memory>policy['max_process_mib']or time.monotonic()-started>policy['max_seconds']:raise RuntimeError('Readonly resource budget exceeded')
  row=dict(time=time.time(),phase=phase,gpu=gpu,processes=current,memory_mib=memory,exclusive_global_lock=True);checks.append(row)
  with(root/'resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
  return row
 def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 env=dict(os.environ,PYTHONPATH=str(root/'helpers')+':'+str(source)+':'+str(source/'LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 try:
  for label,coefficient in [('control001',.01),('treatment1',1.)]:
   a=archives[label]
   for name,query in [('clothing103',c/'independent_occupancy/queries/clothing_103_predeclared.json'),('DC12',c/'independent_occupancy/DC_error12_15188/queries.json')]:
    out=a/(name+'_trace');health=a/(name+'_producer_health.json');before=check(label+'_'+name+'_before')
    command=[policy['python'],str(root/'helpers/audit_marshawn_semantic_checkpoint.py'),'--data',str(c/'data'),'--output',str(out),'--queries',str(query),'--receipt',str(a/'step-000034376_grid.json'),'--checkpoint',str(a/'checkpoint.ckpt'),'--config',str(a/'config.yml'),'--code-root',str(source),'--gpu',str(gpu),'--memory-cap-gib',str(policy['allocator_cap_gib']),'--expected-distortion',str(coefficient)]
    write(root/'state.json',dict(phase=label+'_'+name,pid=os.getpid(),gpu=gpu,time=time.time()))
    with(a/(name+'_trace.log')).open('w')as log:
     child=subprocess.Popen(command,env=env,cwd=source/'LookCloser',stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned.add(child.pid)
     while child.poll()is None:check(label+'_'+name);time.sleep(5)
     rc=child.returncode;child=None
    after=check(label+'_'+name+'_after');write(health,dict(audit_environment_healthy=rc==0,exclusive_idle_gpu=True,producer_completed=True,optimizer_updates=0,before_gpu=before,after_gpu=after,checkpoint_sha256=sha(a/'checkpoint.ckpt'),returncode=rc))
    if rc:raise RuntimeError('Readonly trace failed: '+label+'/'+name)
  lease.close();lease=None
  for script,out in [('compare_semantic_late_traces.py','clothing103'),('compare_semantic_late_dc.py','DC12')]:
   subprocess.run([policy['python'],str(root/'helpers'/script),'--campaign',str(c),'--output',str(root/out)],env=env,check=True)
  subprocess.run([policy['python'],str(root/'helpers/summarize_marshawn_depth_residuals.py'),'--comparison',str(root/'clothing103/comparison.json'),'--output',str(root/'clothing103/depth_residuals.json')],env=env,check=True)
  write(root/'complete.json',dict(passed=True,policy_sha256=sha(root/'policy.json'),optimizer_updates=0,producer_modified=False,elapsed_seconds=time.monotonic()-started,peak_process_mib=max(r['memory_mib']for r in checks),model_promotion=False))
 except BaseException as e:
  signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  write(root/'failed.json',dict(error=repr(e),producer_modified=False));raise
 finally:
  if lease is not None:lease.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
