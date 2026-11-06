"""Exclusive dev3 GPU wrapper with durable minute-level process/GPU supervision."""
from pathlib import Path
import subprocess,os,sys,json,time,fcntl,hashlib
B=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106/native_semantic_grid_20000');O=B.parent/'native_reference_continuation_20000';release=B/'native_continuation_plan/root_release.json'
assert hashlib.sha256(release.read_bytes()).hexdigest()=='ac0622978ea5d7fa7eb1c88cfd46a7c65266df0b27bebbb4e7bda92d7348c136'
lockpath=Path('/tmp/lookcloser_tuning_gpu_%d_0.lock'%os.getuid());lock=lockpath.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def gpu():
 mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
 return dict(memory_mib=mem,compute_processes=processes)
initial=gpu();assert not initial['compute_processes'] and int(initial['memory_mib'].split(',')[0])<128,initial
command=[sys.executable,str(B/'code/run_marshawn_native_continuation.py'),'--release',str(release)]+sys.argv[1:];environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='0'
with (O/'production_stdout.log').open('a') as stream:
 process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment)
 (O/'launch.json').write_text(json.dumps(dict(controller_pid=os.getpid(),worker_pid=process.pid,command=command,time=time.time(),root_release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(),global_gpu_lock=str(lockpath),initial_gpu=initial,source_ancestry='immutable original20k; technical32 excluded'),indent=2)+'\n')
 while True:
  status=json.loads((O/'status.json').read_text()) if (O/'status.json').exists() else None
  with (O/'production_stdout.log').open('rb') as log:
   log.seek(max(0,(O/'production_stdout.log').stat().st_size-65536));tail=log.read().decode(errors='replace')
  row=dict(time=time.time(),controller_pid=os.getpid(),worker_pid=process.pid,worker_returncode=process.poll(),latest_progress=status,gpu=gpu(),oom_evidence_in_recent_log='out of memory' in tail.lower(),complete=(O/'complete.json').exists())
  with (O/'supervision.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
  if row['worker_returncode'] is not None:break
  time.sleep(60)
sys.exit(process.returncode)
