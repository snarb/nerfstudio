"""Real CPU exec leases bound concurrency even if controller metadata is lost."""
from pathlib import Path
import json,os,subprocess,sys,time
O=Path(__file__).parent;sys.path.insert(0,str(O));import cpu_exec
q=O/'qualification_capacity_v1';q.mkdir(exist_ok=True);shader=q/'worker_params.py';shader.write_text('''from pathlib import Path
import os,sys,json,time,fcntl
q=Path(__file__).parent;i=int(sys.argv[-1]);r=json.loads((q/'ledger'/f'{i:04d}'/'runtime.json').read_text());assert r['process']['pid']==os.getpid();assert r['exec_attempt']==1;os.fstat(int(os.environ['LOOKCLOSER_CPU_EXEC_LEASE_FD']));os.fstat(int(os.environ['LOOKCLOSER_CPU_CAPACITY_FD']))
def event(kind):
 with(q/'events.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with(q/'events.jsonl').open('a')as f:f.write(json.dumps(dict(event=kind,index=i,time=time.time(),pid=os.getpid()))+'\\n');f.flush()
event('start');time.sleep(.65);event('end')
''');children=[]
for i in range(3):
 d=q/'ledger'/f'{i:04d}';cpu_exec.prepare(d,[sys.executable,str(shader),'cpu','--index',str(i)],[shader],index=i,max_concurrent=2)
 p=subprocess.Popen([sys.executable,str(O/'cpu_exec.py'),'--intent',str(d/'intent.json'),'--expected-sha256',cpu_exec.sha(d/'intent.json')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True);children.append(p)
for p in children:
 out=p.communicate(timeout=10)[0];assert p.returncode==0,out.decode()
events=[json.loads(x)for x in(q/'events.jsonl').read_text().splitlines()];active=0;maximum=0
for e in sorted(events,key=lambda e:e['time']):active+=1 if e['event']=='start'else-1;maximum=max(active,maximum)
assert active==0 and maximum==2 and len(events)==6
r=dict(kind='CPU_EXEC_CAPACITY_INHERITED_LEASE_QUALIFICATION_V1',passed=True,actual_max_concurrent=maximum,configured_capacity=2,actual_exec_attempts=[cpu_exec.read(q/'ledger'/f'{i:04d}'/'runtime.json')['exec_attempt']for i in range(3)],CPU_coordinator_not_required_for_capacity=True,CUDA_or_model_not_loaded=True,source_code_sha256=cpu_exec.sha(__file__),exec_helper_sha256=cpu_exec.sha(O/'cpu_exec.py'),finished=time.time());cpu_exec.atomic(q/'complete.json',r);print(json.dumps(r))
