"""Reviewed idle-exclusive zero/32+32 gates for the separate quarter FR pair."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)
def rows(g):
 s=subprocess.check_output(['nvidia-smi','--id='+str(g),'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return [dict(pid=int(x.split(',')[0]),mib=int(x.split(',')[1]))for x in s.splitlines()]
def main(root):
 p=read(root/'technical_policy.json');r=read(root/'technical_release.json')
 if r.get('reviewer')!='root'or r.get('technical_authorized')is not True or r.get('policy_sha256')!=sha(root/'technical_policy.json'):raise ValueError('Exact technical release required')
 for name,digest in p['files_sha256'].items():
  if sha(root/name)!=digest:raise ValueError('Reviewed technical artifact changed '+name)
 for name,digest in read(root/'source_inventory.json')['files_sha256'].items():
  if sha(root/'code'/name)!=digest:raise ValueError('Frozen code changed '+name)
 lease=None
 while lease is None:
  for gpu in p['gpus']:
   claim=open(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{gpu}.lock','a+')
   try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:claim.close();continue
   if rows(gpu):claim.close();continue
   lease=claim;break
  if lease is None:write(root/'technical_state.json',dict(phase='waiting_idle',pid=os.getpid(),time=time.time()));time.sleep(30)
 gpu_uuid=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 start=time.monotonic();child=None;checks=[]
 def interrupted(signum,frame):raise RuntimeError('Interrupted '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid,EXPECTED_GPU_UUID=gpu_uuid,PYTHONPATH=str(root/'code')+':'+str(root/'code/LookCloser/scripts'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
 def run(label,command):
  nonlocal child
  with(root/(label+'.log')).open('w')as log:
   child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=root/'code/LookCloser')
   while child.poll()is None:
    active=rows(gpu);memory=sum(x['mib']for x in active)
    if any(os.getpgid(x['pid'])!=child.pid for x in active)or memory>p['max_process_mib']or time.monotonic()-start>p['max_seconds']:raise RuntimeError('Technical resource/ownership guard failed')
    row=dict(time=time.time(),phase=label,gpu=gpu,gpu_uuid=gpu_uuid,processes=active,mib=memory);checks.append(row)
    with(root/'technical_resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
    write(root/'technical_state.json',dict(phase=label,pid=os.getpid(),child=child.pid,gpu=gpu,time=time.time()));time.sleep(5)
   if child.returncode:raise RuntimeError('Technical child failed '+label+' '+str(child.returncode))
   child=None
 try:
  py=p['python'];worker=str(root/'code/LookCloser/scripts/train_marshawn_quarter_FR.py')
  for name in ['zero_original_FR03','zero_hold_FR1']:run(name,[py,worker,'--spec',str(root/'specs'/name/'spec.json')])
  sequence=str(root/'run_marshawn_quarter_FR_technical.py')
  for name in ['technical_original_FR03','technical_hold_FR1']:
   run(name,[py,sequence,'--worker',worker,'--spec',str(root/'specs'/name/'spec.json'),'--zero-control',str(root/'runs/zero_original_FR03/complete.json'),'--zero-treatment',str(root/'runs/zero_hold_FR1/complete.json')])
  run('pair_comparison',[py,sequence,'--audit-pair',str(root/'runs/technical_original_FR03/technical_sequence.json'),str(root/'runs/technical_hold_FR1/technical_sequence.json')])
  result=read(root/'pair_comparison.log')
  if not all(result['later_segment_equality'].values()):raise ValueError('Actual paired main ray segments differ; scientific interpretation required')
  write(root/'technical_complete.json',dict(passed=True,pair=result,gpu=gpu,gpu_uuid=gpu_uuid,elapsed_seconds=time.monotonic()-start,peak_mib=max(x['mib']for x in checks),exclusive_global_lease=True,production_authorized=False,all_source1495_unchanged=True))
 except BaseException as error:
  signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  write(root/'technical_failed.json',dict(error=repr(error),source_producer_modified=False));raise
 finally:lease.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
