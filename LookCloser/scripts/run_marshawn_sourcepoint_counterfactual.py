"""Idle-exclusive current-best source-point counterfactual; no producer changes."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)
def rows(gpu):
 text=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(x.split(',')[0]),mib=int(x.split(',')[1]))for x in text.splitlines()]
def main(root):
 policy=json.loads((root/'policy.json').read_text());release=json.loads((root/'release.json').read_text())
 if not release.get('approved_readonly')or release['policy_sha256']!=sha(root/'policy.json'):raise ValueError('Missing exact readonly authorization')
 for name,digest in policy['helpers_sha256'].items():
  if sha(root/'helpers'/name)!=digest:raise ValueError('Helper drift')
 for path,digest in policy['queries_sha256'].items():
  if sha(path)!=digest:raise ValueError('Query drift')
 if sha(policy['source_grid'])!=policy['source_grid_sha256']:raise ValueError('Source grid drift')
 for arm in policy['models']:
  archive=Path(arm['archive']);branch=Path(arm['branch']);receipt=json.loads((archive/'archive_receipt.json').read_text())
  for name,digest in arm['files_sha256'].items():
   if sha(archive/name)!=digest or receipt['files_sha256'][name]!=digest:raise ValueError('Source archive drift')
  for name,digest in json.loads((archive/'source_manifest.json').read_text())['files_sha256'].items():
   if sha(branch/'code'/name)!=digest:raise ValueError('Source code drift')
  for name,digest in arm['reference_renders_sha256'].items():
   if sha(Path(arm['reference_renders'])/name)!=digest:raise ValueError('Reference FR render changed')
  dest=root/arm['name'];dest.mkdir(exist_ok=True)
  if(dest/'complete.json').exists():continue
  lease=None
  while lease is None:
   for gpu in policy['gpus']:
    claim=open(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock','a+')
    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:claim.close();continue
    if rows(gpu):claim.close();continue
    lease=claim;break
   if lease is None:write(root/'state.json',dict(phase='waiting_idle',arm=arm['name'],time=time.time(),pid=os.getpid()));time.sleep(30)
  child=None;start=time.monotonic();checks=[]
  def interrupted(sig,frame):raise RuntimeError('Interrupted '+str(sig))
  signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
  try:
   uuid=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
   with(dest/'probe.log').open('w')as log:
    child=subprocess.Popen([policy['python'],str(root/'helpers/probe_marshawn_sourcepoint_counterfactual.py'),'--root',str(root),'--arm',arm['name'],'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,EXPECTED_GPU_UUID=uuid,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
    while child.poll()is None:
     active=rows(gpu);mib=sum(x['mib']for x in active)
     if any(x['pid']!=child.pid for x in active)or mib>policy['max_process_mib']or time.monotonic()-start>policy['max_seconds_per_state']:raise RuntimeError('Exclusive resource guard failed')
     row=dict(time=time.time(),gpu=gpu,gpu_uuid=uuid,child=child.pid,processes=active,mib=mib);checks.append(row)
     with(dest/'resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
     write(root/'state.json',dict(phase='running',arm=arm['name'],time=time.time(),pid=os.getpid(),worker=child.pid,gpu=gpu));time.sleep(5)
    if child.returncode or not(dest/'complete.json').exists():raise RuntimeError('Probe failed code '+str(child.returncode))
   write(dest/'resource_complete.json',dict(elapsed_seconds=time.monotonic()-start,peak_mib=max(x['mib']for x in checks),gpu=gpu,gpu_uuid=uuid,exclusive_global_lease=True,optimizer_updates=0))
  except BaseException as error:
   signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
   if child is not None and child.poll()is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   write(dest/'failed.json',dict(error=repr(error),producer_modified=False));raise
  finally:lease.close()
 write(root/'complete.json',dict(passed=True,models=[x['name']for x in policy['models']],optimizer_updates=0,quality_promotion=False))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
