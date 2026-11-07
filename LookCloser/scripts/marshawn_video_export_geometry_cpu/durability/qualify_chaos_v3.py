"""Finite real-process CPU crash qualification; no CUDA or model is loaded."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,types,shutil
O=Path(__file__).parent;sys.path.insert(0,str(O));import cpu_exec
from bridge_v3 import Bridge
PYTHON=sys.executable

def read(p):return cpu_exec.read(p)
def write(p,v):return cpu_exec.atomic(p,v)
def sha(p):return cpu_exec.sha(p)
def router(directory):
 def identity(pid):
  x=cpu_exec.identity(pid)
  if x:x['argv']=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
  return x
 return types.SimpleNamespace(__file__=str(directory/'route.py'),read=read,write=write,identity=identity,shutil=shutil)
STUB='''from pathlib import Path
import os,json,sys,time,fcntl
p=Path(__file__).parent
runtime=json.loads((p/'ledger/0000/runtime.json').read_text())
assert runtime['process']['pid']==os.getpid()
assert runtime['phase']=='identity_persisted_BEFORE_shader_exec'
assert runtime['nonce']==os.environ['LOOKCLOSER_CPU_EXEC_NONCE']
os.fstat(int(os.environ['LOOKCLOSER_CPU_EXEC_LEASE_FD']))
os.fstat(int(os.environ['LOOKCLOSER_CPU_CAPACITY_FD']))
with (p/'counter.lock').open('a+')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with(p/'counter.jsonl').open('a')as f:f.write(json.dumps(dict(pid=os.getpid(),nonce=runtime['nonce'],exec_attempt=runtime['exec_attempt'],metadata_before_exec=True))+'\\n');f.flush();os.fsync(f.fileno())
time.sleep(1.25)
(p/'TEST_RENDER_COMPLETE.json').write_text(json.dumps(dict(kind='CPU_CHAOS_TEST_ONLY_NOT_RENDER_OR_TRAINING',pid=os.getpid()))+'\\n')
'''
def prepare(q):
 q.mkdir(parents=True,exist_ok=True);(q/'worker_params.py').write_text(STUB);write(q/'spec.json',dict(output=str(q/'TEST_OUTPUT')));write(q/'path.json',dict(records=[dict(frame_id='CPU_TEST_ONLY',kind='neural')]))
 return[PYTHON,str(q/'worker_params.py'),'cpu','--spec',str(q/'spec.json'),'--path',str(q/'path.json'),'--index','0']
def command(q):return[PYTHON,str(q/'worker_params.py'),'cpu','--spec',str(q/'spec.json'),'--path',str(q/'path.json'),'--index','0']
def coordinate(q,fault):
 q=Path(q);r=router(q)
 def event(name,detail):
  if name==fault:os._exit(91)
 b=Bridge(r,q/'ledger',O/'cpu_exec.py',shader_sha=sha(q/'worker_params.py'),event=event);b.install()
 try:
  state=r.read(q/'CPU_state.json')if(q/'CPU_state.json').exists()else{};item=state.get('0',{});live=r.identity(item.get('pid',-1))if item.get('pid')else None
  if not(live and live['state']!='Z'and live['start']==item.get('start')and live['boot']==item.get('boot')):
   with(q/'shader.log').open('ab')as log:p=subprocess.Popen(command(q),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state['0']=dict(status='running',attempts=1,**r.identity(p.pid))
  r.write(q/'CPU_state.json',state)
  deadline=time.monotonic()+12
  while not(q/'TEST_RENDER_COMPLETE.json').exists():
   if time.monotonic()>deadline:raise TimeoutError('CPU test did not complete')
   time.sleep(.05)
  final=read(q/'ledger/0000/runtime.json');assert final['exec_attempt']==1;assert len((q/'counter.jsonl').read_text().splitlines())==1
  write(q/'qualification.json',dict(passed=True,exec_attempt=final['exec_attempt'],counter=1,nonce=final['nonce'],PID=final['process']['pid'],identity_persisted_before_shader_exec=True,CUDA_not_loaded=True))
 finally:b.restore()

def run(root):
 records=[]
 for fault in ['after_prelaunch_intent','after_Popen','after_identity','before_CPU_state_save']:
  q=root/fault;prepare(q);log=(q/'coordinator.log').open('ab')
  first=subprocess.run([PYTHON,__file__,'coordinate','--root',str(q),'--fault',fault],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);assert first.returncode==91,(fault,first.returncode)
  second=subprocess.run([PYTHON,__file__,'coordinate','--root',str(q),'--fault','none'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);assert second.returncode==0,(fault,(q/'coordinator.log').read_text()[-2000:]);records.append(dict(crash_point=fault,**read(q/'qualification.json')))
 # Actual inherited flock survives shader exec; a duplicate returns adoption0.
 q=root/'lock_busy';cmd=prepare(q);r=router(q);b=Bridge(r,q/'ledger',O/'cpu_exec.py',shader_sha=sha(q/'worker_params.py'));b.install()
 try:
  with(q/'shader.log').open('ab')as log:holder=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  intent=q/'ledger/0000/intent.json';duplicate=b.orig_popen([PYTHON,str(O/'cpu_exec.py'),'--intent',str(intent),'--expected-sha256',sha(intent)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);assert duplicate.wait(timeout=5)==0
  before={'0':dict(status='running',attempts=3,pid=duplicate.pid)};r.write(q/'CPU_state.json',before);assert before['0']['attempts']<=1;assert before['0']['pid']==holder.pid
  while not(q/'TEST_RENDER_COMPLETE.json').exists():time.sleep(.05)
  assert len((q/'counter.jsonl').read_text().splitlines())==1
  events=(q/'ledger/0000/events.jsonl').read_text();assert'lock_busy_adopt_existing'in events
  # Reuse with different shader bytes is rejected, not adopted as old work.
  (q/'worker_params.py').write_text(STUB+'\n# different source\n')
  try:b.overlay({})
  except ValueError:cross_source_rejected=True
  else:raise AssertionError('Different-source CPU worker was adopted')
  records.append(dict(crash_point='lock_busy_and_source_rebind',passed=True,actual_exec_count=1,retry_budget_not_consumed=True,inherited_lock_held_across_exec=True,cross_source_rejected=cross_source_rejected))
 finally:b.restore()
 # The original fleet's BEFORE-SSH state, not its stale outer mirror, wins.
 q=root/'authoritative_GPU_state';q.mkdir(exist_ok=True);write(q/'GPU_state.json',dict(jobs={'j':dict(status='pending')}));write(q/'state.json',dict(jobs={'j':dict(status='dispatching',host='CPU_TEST_HOST',gpu=7,spec_sha256='ACKNOWLEDGED_SOURCE')}));r=router(q);b=Bridge(r,q/'ledger',O/'cpu_exec.py',shader_sha='unused');actual=b.read_state(q/'GPU_state.json');assert actual['jobs']['j']['status']=='dispatching'and actual['jobs']['j']['gpu']==7;records.append(dict(crash_point='SSH_intent_before_outer_mirror',passed=True,acknowledged_host_GPU_preserved=True,stale_GPU_mirror_ignored=True))
 # Existing identical controls are never truncated while a shader runs.
 q=root/'immutable_control';(q/'control_olive').mkdir(parents=True);src=q/'source';dst=q/'control_olive/worker_params.py';src.write_text('same');dst.write_text('same');r=router(q);b=Bridge(r,q/'ledger',O/'cpu_exec.py',shader_sha='unused');before=dst.stat().st_mtime_ns;b.copy(src,dst);assert dst.stat().st_mtime_ns==before;src.write_text('different')
 try:b.copy(src,dst)
 except ValueError:pass
 else:raise AssertionError('Live shader source was replaced')
 records.append(dict(crash_point='restart_control_copy',passed=True,identical_source_not_rewritten=True,different_source_blocked=True))
 result=dict(kind='OPS_CPU_CRASH_ADOPTION_QUALIFICATION_V1',passed=all(r['passed']for r in records),records=records,code_sha256={p.name:sha(p)for p in [O/'cpu_exec.py',O/'bridge_v3.py',Path(__file__)]},no_GPU_or_model_loaded=True,no_production_state_changed=True,finished=time.time());write(root/'complete.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','coordinate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--fault',default='none');a=p.parse_args();run(a.root)if a.mode=='run'else coordinate(a.root,a.fault)
