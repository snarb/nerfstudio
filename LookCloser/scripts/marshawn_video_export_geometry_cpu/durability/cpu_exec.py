"""Durable CPU launch ledger; same PID/lock survives exec of the frozen shader."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.%s.tmp'%os.getpid())
 with tmp.open('w')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def identity(pid=None):
 pid=os.getpid()if pid is None else int(pid);p=Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text().split(') ',1)[1].split();return dict(pid=pid,start=s[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=s[0])
 except FileNotFoundError:return None

def live(runtime):
 ident=identity(runtime['process']['pid'])
 return ident is not None and ident['state']!='Z'and all(ident[k]==runtime['process'][k]for k in ['pid','start','boot'])
def source(command,inputs):
 return dict(command=list(command),input_sha256={str(Path(p).resolve()):sha(p)for p in inputs},CUDA_VISIBLE_DEVICES='')
def prepare(directory,command,inputs,*,index,nonce=None,max_concurrent=3):
 import uuid
 d=Path(directory);d.mkdir(parents=True,exist_ok=True);src=source(command,inputs);key=digest(src);p=d/'intent.json'
 if p.exists():
  previous=read(p)
  if previous['source']!=src or previous['source_digest']!=key or previous['index']!=index or previous['max_concurrent']!=max_concurrent:raise ValueError('CPU intent belongs to another source/profile/index')
  return previous
 intent=dict(kind='CPU_RENDER_PRELAUNCH_INTENT_V1',index=index,nonce=nonce or uuid.uuid4().hex,source=src,source_digest=key,coordinator=identity(),max_concurrent=max_concurrent,capacity_directory=str(d.parent/'capacity'),created=time.time());atomic(p,intent);return intent

def runtime_for(directory,intent):
 p=Path(directory)/'runtime.json'
 if not p.exists():return None
 runtime=read(p)
 if runtime.get('source_digest')!=intent['source_digest']or runtime.get('nonce')!=intent['nonce']or runtime.get('intent_sha256')!=sha(Path(directory)/'intent.json'):raise ValueError('Different-source CPU process cannot be adopted')
 return runtime

def run(intent_path,expected_sha):
 p=Path(intent_path);d=p.parent
 if sha(p)!=expected_sha:raise ValueError('CPU intent changed before launch')
 intent=read(p)
 if intent['kind']!='CPU_RENDER_PRELAUNCH_INTENT_V1'or digest(intent['source'])!=intent['source_digest']:raise ValueError('Malformed source-bound CPU intent')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU exec must have no visible CUDA devices')
 for path,value in intent['source']['input_sha256'].items():
  if sha(path)!=value:raise ValueError('CPU render input/code changed '+path)
 lease=(d/'execution.lock').open('a+')
 try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  # Busy is adoption, never an execution attempt or a render failure.
  with(d/'events.jsonl').open('a')as f:f.write(json.dumps(dict(event='lock_busy_adopt_existing',process=identity(),nonce=intent['nonce'],time=time.time()))+'\n')
  return 0
 previous=runtime_for(d,intent);previous_attempt=previous['exec_attempt']if previous else 0
 if previous and live(previous):raise RuntimeError('Live CPU holder without its inherited lock')
 runtime=dict(kind='CPU_RENDER_EXEC_OWNER_V1',process=identity(),nonce=intent['nonce'],source_digest=intent['source_digest'],intent_sha256=sha(p),command=intent['source']['command'],exec_attempt=previous_attempt,phase='identity_persisted_waiting_CPU_capacity',lock_path=str(d/'execution.lock'),lease_fd=lease.fileno(),time=time.time());atomic(d/'runtime.json',runtime)
 capacity=Path(intent['capacity_directory']);capacity.mkdir(exist_ok=True);slot=None;slot_number=None
 while slot is None:
  for number in range(int(intent['max_concurrent'])):
   candidate=(capacity/(str(number)+'.lock')).open('a+')
   try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:candidate.close();continue
   slot=candidate;slot_number=number;break
  if slot is None:time.sleep(.1)
 runtime.update(exec_attempt=previous_attempt+1,phase='identity_persisted_BEFORE_shader_exec',capacity_slot=slot_number,capacity_fd=slot.fileno());atomic(d/'runtime.json',runtime)
 os.set_inheritable(lease.fileno(),True);os.set_inheritable(slot.fileno(),True)
 env=dict(os.environ,LOOKCLOSER_CPU_EXEC_NONCE=intent['nonce'],LOOKCLOSER_CPU_EXEC_SOURCE_DIGEST=intent['source_digest'],LOOKCLOSER_CPU_EXEC_LEASE_FD=str(lease.fileno()),LOOKCLOSER_CPU_CAPACITY_FD=str(slot.fileno()))
 command=intent['source']['command'];os.execvpe(command[0],command,env)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--intent',type=Path,required=True);parser.add_argument('--expected-sha256',required=True);a=parser.parse_args();raise SystemExit(run(a.intent,a.expected_sha256))
