"""Bounded idle-exclusive scratch update proof; owns only its child process group."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def rows(g):
 s=subprocess.check_output(['nvidia-smi','--id='+str(g),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(x.split(',')[0]),mib=int(x.split(',')[1]))for x in s.splitlines()]
def main(root):
 p=read(root/'policy.json');release=read(root/'release.json')
 if not release.get('approved_scratch_readonly')or release['policy_sha256']!=sha(root/'policy.json'):raise ValueError('Bound delegated scope required')
 for path,digest in p['artifacts'].items():
  if sha(path)!=digest:raise ValueError('Released artifact changed '+path)
 for path,digest in read(p['source_manifest'])['files_sha256'].items():
  if sha(Path(p['code'])/path)!=digest:raise ValueError('Frozen production source changed')
 if p['max_seconds']>900 or p['max_process_mib']>16384:raise ValueError('Resource scope exceeded')
 g=p['gpu'];lease=None;started_wait=time.monotonic()
 while lease is None:
  f=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{g}.lock').open('a+')
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:f.close();time.sleep(30);continue
  if rows(g):f.close();time.sleep(30);continue
  lease=f
 if subprocess.check_output(['nvidia-smi','--id='+str(g),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=p['gpu_uuid']:raise ValueError('GPU identity differs')
 child=None;start=time.monotonic();samples=[]
 def stop(signum,frame):raise RuntimeError('Own supervisor interrupted '+str(signum))
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
  command=[p['python'],str(root/'probe_marshawn_actor_frequency_update.py'),'--source',p['source'],'--code',p['code'],'--output',str(root/'results'),'--gpu',str(g)]
  with(root/'worker.log').open('w')as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll()is None:
    gpu=rows(g)
    if any(x['pid']!=child.pid for x in gpu):raise RuntimeError('Foreign GPU process under exclusive lease')
    used=sum(x['mib']for x in gpu)
    if used>p['max_process_mib']or time.monotonic()-start>p['max_seconds']:raise RuntimeError('Resource envelope exceeded')
    row=dict(time=time.time(),child=child.pid,gpu=g,process_mib=used,exclusive_global_lock=True);samples.append(row)
    with(root/'resources.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
    time.sleep(5)
   if child.returncode:raise RuntimeError('Scratch proof failed; source producer unchanged')
  result=read(root/'results/complete.json')
  if not result['passed']or not result['full_state_restored']or result['optimizer_updates']!=0:raise ValueError('Scratch proof incomplete')
  write(root/'complete.json',dict(passed=True,policy_sha256=sha(root/'policy.json'),result_sha256=sha(root/'results/complete.json'),seconds=time.monotonic()-start,peak_process_mib=max(x['process_mib']for x in samples),no_production_changes=True,optimizer_updates=0))
 except BaseException as e:
  signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  write(root/'failed.json',dict(error=repr(e),producer_modified=False));raise
 finally:lease.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
