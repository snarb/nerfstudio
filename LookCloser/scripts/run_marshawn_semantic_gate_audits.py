"""Delegated fixed-gate semantic replay, serial idle-exclusive GPU leases."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def rows(g):
 s=subprocess.check_output(['nvidia-smi','--id='+str(g),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(x.split(',')[0]),mib=int(x.split(',')[1]))for x in s.splitlines()]
def main(root):
 policy=read(root/'policy.json');release=read(root/'release.json')
 if not release.get('approved_readonly')or release['policy_sha256']!=sha(root/'policy.json'):raise ValueError('Exact delegated readonly binding required')
 for name,digest in policy['helpers_sha256'].items():
  if sha(root/'helpers'/name)!=digest:raise ValueError('Audit helper changed')
 c=Path(policy['campaign']);records=[]
 for name,digest in policy['queries_sha256'].items():
  if sha(c/name)!=digest:raise ValueError('Sealed query changed')
 for arm in policy['models']:
  branch=c/arm['branch'];manifest=branch/'source_manifest.json';spec=branch/'spec.json'
  if sha(manifest)!=arm['source_manifest_sha256']or sha(spec)!=arm['spec_sha256']:raise ValueError('Reviewed branch identity changed')
  for path,digest in read(manifest)['files_sha256'].items():
   if sha(branch/'code'/path)!=digest:raise ValueError('Frozen branch source changed')
  step=arm['gate_step'];run=branch/'run';complete=run/'renders'/f'step-{step:09d}/metrics.json'
  checkpoint=Path(arm.get('checkpoint_override',str(run/'checkpoints'/f'step-{step:09d}.ckpt')))
  required=[complete,checkpoint,run/'occupancy'/f'step-{step:09d}_grid.json']
  while not all(p.exists()for p in required):write(root/'state.json',dict(phase='waiting_gate_'+arm['name'],time=time.time(),pid=os.getpid()));time.sleep(10)
  configuration=read(spec)
  if configuration['radius']!=arm['radius']or configuration['width']!=arm['width']:raise ValueError('Completed recipe differs')
  if arm.get('expected_step')is not None and step!=arm['expected_step']:raise ValueError('Declared selected step changed')
  run=branch/'run';grid=run/'occupancy'/f'step-{step:09d}_grid.json';capture=read(grid);checkpoint=Path(arm.get('checkpoint_override',str(run/'checkpoints'/f'step-{step:09d}.ckpt')))
  if not checkpoint.exists():checkpoint=c/'independent_occupancy'/arm['progress_archive']/f'step-{step:09d}/checkpoint.ckpt'
  if sha(checkpoint)!=capture['checkpoint_sha256']or capture['schema']!=2 or capture['radius']!=arm['radius']:raise ValueError('Actual selected snapshot differs')
  out=root/arm['name'];out.mkdir(exist_ok=True)
  files={'checkpoint.ckpt':checkpoint,'grid.json':grid,'grid.npz':grid.with_suffix('.npz'),'config.yml':run/'worker/lookcloser/frozen/config.yml','source_manifest.json':manifest,'spec.json':spec,'gate_metrics.json':complete}
  for target,source in files.items():
   if not(out/target).exists():os.link(source,out/target)
   if sha(out/target)!=sha(source):raise ValueError('Protected archive differs')
  shutil.copytree(run/'renders'/f'step-{step:09d}',out/'renders',dirs_exist_ok=True)
  record=dict(name=arm['name'],step=step,radius=arm['radius'],width=arm['width'],archive=str(out),branch=str(branch),files_sha256={k:sha(out/k)for k in files});write(out/'archive_receipt.json',record);records.append(record)
  write(root/'selected_models.json',records)
  if(out/'arm_complete.json').exists():continue
  lease=None
  while lease is None:
   for gpu in policy['gpus']:
    claim=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock').open('a+')
    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:claim.close();continue
    if rows(gpu):claim.close();continue
    lease=claim;break
   if lease is None:write(root/'state.json',dict(phase='waiting_idle_'+arm['name'],time=time.time(),pid=os.getpid()));time.sleep(30)
  started=time.monotonic();child=None;owned=set();checks=[]
  def check(phase):
   active=rows(gpu)
   if any(x['pid']not in owned for x in active):raise RuntimeError('Unexpected exclusive GPU process')
   memory=sum(x['mib']for x in active)
   if memory>policy['max_process_mib']or time.monotonic()-started>policy['max_seconds_per_arm']:raise RuntimeError('Readonly resource budget exceeded')
   row=dict(time=time.time(),phase=phase,gpu=gpu,memory_mib=memory,processes=active,exclusive_idle=True);checks.append(row)
   with(out/'resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
   return row
  def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
  signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
  env=dict(os.environ,PYTHONPATH=str(root/'helpers')+':'+str(branch/'code')+':'+str(branch/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
  try:
   for label,query in [('clothing103',c/'independent_occupancy/queries/clothing_103_predeclared.json'),('DC12',c/'independent_occupancy/DC_error12_15188/queries.json')]:
    trace=out/(label+'_trace');before=check(label+'_before');write(root/'state.json',dict(phase=arm['name']+'_'+label,gpu=gpu,time=time.time(),pid=os.getpid()))
    command=[policy['python'],str(root/'helpers/audit_marshawn_semantic_checkpoint.py'),'--data',str(c/'data'),'--output',str(trace),'--queries',str(query),'--receipt',str(out/'grid.json'),'--checkpoint',str(out/'checkpoint.ckpt'),'--config',str(out/'config.yml'),'--code-root',str(branch/'code'),'--gpu',str(gpu),'--memory-cap-gib',str(policy['allocator_cap_gib'])]
    with(out/(label+'.log')).open('w')as f:
     child=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=branch/'code/LookCloser');owned.add(child.pid)
     while child.poll()is None:check(label);time.sleep(5)
     rc=child.returncode;child=None
    after=check(label+'_after');write(out/(label+'_producer_health.json'),dict(audit_environment_healthy=rc==0,exclusive_idle_gpu=True,producer_modified=False,optimizer_updates=0,before_gpu=before,after_gpu=after))
    if rc:raise RuntimeError('Gate replay failed '+arm['name']+'/'+label)
   write(out/'arm_complete.json',dict(passed=True,step=step,optimizer_updates=0,elapsed_seconds=time.monotonic()-started,peak_mib=max(x['memory_mib']for x in checks)))
  except BaseException as e:
   signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
   if child is not None and child.poll()is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   write(out/'failed.json',dict(error=repr(e),producer_modified=False));raise
  finally:lease.close()
 env=dict(os.environ,PYTHONPATH=str(root/'helpers'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 for script,label in [('compare_selected_traces.py','clothing103'),('compare_selected_dc.py','DC12')]:subprocess.run([policy['python'],str(root/'helpers'/script),'--campaign',str(c),'--output',str(root/label)],env=env,check=True)
 subprocess.run([policy['python'],str(root/'helpers/summarize_marshawn_depth_residuals.py'),'--comparison',str(root/'clothing103/comparison.json'),'--output',str(root/'clothing103/depth_residuals.json')],env=env,check=True)
 write(root/'complete.json',dict(passed=True,selected_models=records,optimizer_updates=0,producer_modified=False,quality_promotion=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
