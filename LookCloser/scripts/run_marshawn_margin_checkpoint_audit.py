"""Bound serial idle-exclusive read-only replay; never changes producers."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)
def identity():
 x=Path('/proc/self/stat').read_text().split(') ',1)[1].split();return dict(pid=os.getpid(),start=x[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def rows(gpu):
 x=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(r.split(',')[0]),mib=int(r.split(',')[1]))for r in x.splitlines()]
def main(root):
 policy=json.loads((root/'policy.json').read_text());release=json.loads(Path(policy['release']).read_text())
 if release.get('reviewer')!='root'or release.get('readonly_authorized')is not True or release.get('plan_sha256')!=sha(root/'policy.json'):raise ValueError('Exact root read-only release required')
 for p,h in policy['files_sha256'].items():
  if sha(p)!=h:raise ValueError('Bound source/helper drift: '+p)
 manifest=json.loads(Path(policy['source_manifest']).read_text())
 for p,h in manifest['files_sha256'].items():
  if sha(Path(policy['code'])/p)!=h:raise ValueError('Frozen science drift: '+p)
 for arm,row in policy['models'].items():
  for p,h in row['files_sha256'].items():
   if sha(Path(row['archive'])/p)!=h:raise ValueError('Child archive drift: '+p)
  dest=Path(row['output'])
  if (dest/'complete.json').exists():continue
  if dest.exists():raise ValueError('Preserve previous partial output; use reviewed new generation')
  lease=None
  while lease is None:
   for gpu in policy['gpus']:
    claim=open(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock','a+')
    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:claim.close();continue
    if rows(gpu):claim.close();continue
    lease=claim;break
   if lease is None:write(root/'state.json',dict(phase='waiting_idle',arm=arm,pid=os.getpid(),time=time.time()));time.sleep(30)
  child=None;started=time.monotonic();peak=0;lease_path=root/(arm+'_lease.json')
  def interrupted(sig,frame):raise RuntimeError('Interrupted '+str(sig))
  signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
  try:
   uuid=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();write(lease_path,dict(supervisor=identity(),gpu=gpu,gpu_uuid=uuid))
   with(root/(arm+'_worker.log')).open('w')as log:
    child=subprocess.Popen([policy['python'],str(Path(policy['helpers'])/'audit_marshawn_margin_checkpoint.py'),'--plan',str(root/'policy.json'),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,EXPECTED_GPU_UUID=uuid,LOOKCLOSER_GPU_LEASE_RECEIPT=str(lease_path),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
    while child.poll()is None:
     current=rows(gpu);mib=sum(x['mib']for x in current);peak=max(peak,mib)
     if any(x['pid']!=child.pid for x in current)or mib>policy['max_process_mib']or time.monotonic()-started>policy['max_seconds_per_state']:raise RuntimeError('Exclusive resource guard failed')
     state=dict(time=time.time(),phase='running',arm=arm,gpu=gpu,gpu_uuid=uuid,pid=os.getpid(),worker=child.pid,processes=current,mib=mib)
     with(root/(arm+'_resources.jsonl')).open('a')as f:f.write(json.dumps(state)+'\n')
     write(root/'state.json',state);time.sleep(5)
    if child.returncode or not(dest/'complete.json').exists():raise RuntimeError('Read-only replay failed '+str(child.returncode))
   write(root/(arm+'_resource_complete.json'),dict(passed=True,elapsed_seconds=time.monotonic()-started,peak_mib=peak,gpu=gpu,gpu_uuid=uuid,exclusive_global_lease=True,optimizer_updates=0))
  except BaseException as error:
   signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
   if child is not None and child.poll()is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   write(root/(arm+'_failed.json'),dict(error=repr(error),producer_modified=False));raise
  finally:lease.close()
 write(root/'complete.json',dict(passed=True,models=list(policy['models']),optimizer_updates=0,quality_promotion=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
