"""Operational adapters only; the qualified route and both shaders stay unchanged."""
from pathlib import Path
import hashlib,json,os,subprocess,time,types
import cpu_exec

class AwaitCPUOwnership(RuntimeError):pass

def digest(v):return cpu_exec.digest(v)
def sha(p):return cpu_exec.sha(p)
def read(p):return cpu_exec.read(p)
class Bridge:
 def __init__(self,router,ledger,helper,*,shader_sha,source_guard=None,event=None,max_concurrent=3):
  self.router=router;self.root=Path(router.__file__).parent;self.ledger=Path(ledger);self.helper=Path(helper);self.shader_sha=shader_sha;self.source_guard=source_guard or(lambda:None);self.event=event or(lambda name,detail:None);self.orig_read=router.read;self.orig_write=router.write;self.orig_identity=router.identity;self.orig_copy=router.shutil.copy2;self.orig_popen=subprocess.Popen;self.launch_cache={};self.max_concurrent=max_concurrent
 def emit(self,name,detail):self.event(name,detail)
 def cpu_command(self,command):return isinstance(command,(list,tuple))and len(command)>2 and Path(str(command[1])).name=='worker_params.py'and command[2]=='cpu'
 def index(self,command):return int(command[command.index('--index')+1])
 def inputs(self,command):
  script=Path(command[1]);paths=[script,Path(command[command.index('--spec')+1]),Path(command[command.index('--path')+1])]
  for name in ['hybrid_params.json','root_release.json','focal_scale.py','one.py']:
   p=script.parent/name
   if p.exists():paths.append(p)
  spec=read(paths[1]);request=read(paths[2]);record=request['records'][self.index(command)]
  if record.get('kind')!='neural':raise ValueError('CPUexec contract accepts neural0..71 only; frozen GT one.py is separate')
  if spec.get('gathered_geometry'):
   paths.append(Path(spec['gathered_geometry'])/'frames'/f'{self.index(command):04d}'/'geometry_complete.json')
  if spec.get('data_template'):
   data=Path(spec['data_template'].format(frame_id=record['frame_id']));paths.extend([data/'ready.json',data/'transforms.json'])
  return paths
 def protected(self,path):
  p=Path(path)
  return p==self.root/'canonical_path.json'or(p.parent.parent==self.root and p.parent.name.startswith('control_'))
 def copy(self,src,dst,*a,**kw):
  if self.protected(dst)and Path(dst).exists():
   if sha(src)!=sha(dst):raise ValueError('Refuse replacing active control/code '+str(dst))
   return Path(dst)
  return self.orig_copy(src,dst,*a,**kw)
 def runtime_records(self):
  if not self.ledger.exists():return
  for p in self.ledger.glob('*/intent.json'):
   intent=read(p);runtime=cpu_exec.runtime_for(p.parent,intent)
   if runtime is not None:yield p.parent,intent,runtime
 def overlay(self,state):
  state=dict(state)
  for directory,intent,runtime in self.runtime_records()or[]:
   # Re-hash exact shader/spec/path/model-geometry inputs before adoption.
   src=intent['source'];current=cpu_exec.source(src['command'],src['input_sha256'])
   if current!=src:raise ValueError('CPU shader inputs changed since launch')
   if cpu_exec.live(runtime):
    key=str(intent['index']);previous=state.get(key,{})
    if previous.get('status')=='complete':continue
    state[key]=dict(previous,status='running',attempts=runtime['exec_attempt'],command=src['command'],**runtime['process'],CPU_exec_nonce=intent['nonce'],CPU_exec_source_digest=intent['source_digest'],adopted_from_durable_exec=True)
  return state
 def read_state(self,path):
  p=Path(path)
  if p==self.root/'GPU_state.json':
   authority=self.root/'state.json'
   if authority.exists():return self.orig_read(authority)
  result=self.orig_read(path)
  if p==self.root/'CPU_state.json':result=self.overlay(result)
  return result
 def write_state(self,path,value):
  p=Path(path)
  if p.name=='spec.json'and p.parent.parent==self.root and p.parent.name.startswith('control_'):
   value=dict(value,source_visibility_adapter_sha256=sha(self.root/'one.py'))
  if self.protected(p)and p.exists():
   blob=(json.dumps(value,indent=2)+'\n').encode()
   if p.read_bytes()!=blob:raise ValueError('Refuse replacing active control/request '+str(p))
   return
  if p==self.root/'CPU_state.json':
   normalized=self.overlay(value);value.clear();value.update(normalized)
   self.emit('before_CPU_state_save',dict(value=value))
  return self.orig_write(path,value)
 def identity(self,pid):
  result=self.orig_identity(pid)
  if result is None and pid in self.launch_cache:return self.launch_cache.pop(pid)
  return result
 def popen(self,command,*args,**kwargs):
  if not self.cpu_command(command):return self.orig_popen(command,*args,**kwargs)
  self.source_guard()
  if sha(command[1])!=self.shader_sha:raise ValueError('Unqualified CPU shader')
  if kwargs.get('env',{}).get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU shader must have no CUDA device')
  index=self.index(command);directory=self.ledger/f'{index:04d}';intent=cpu_exec.prepare(directory,command,self.inputs(command),index=index,max_concurrent=self.max_concurrent);self.emit('after_prelaunch_intent',dict(intent=intent))
  runtime=cpu_exec.runtime_for(directory,intent);launched=None
  if not(runtime and cpu_exec.live(runtime)):
   launched=self.orig_popen([command[0],str(self.helper),'--intent',str(directory/'intent.json'),'--expected-sha256',sha(directory/'intent.json')],*args,**kwargs);self.emit('after_Popen',dict(pid=launched.pid,index=index))
  deadline=time.monotonic()+15
  while time.monotonic()<deadline:
   runtime=cpu_exec.runtime_for(directory,intent)
   if runtime is not None and(cpu_exec.live(runtime)or launched is None or runtime['process']['pid']==launched.pid):
    process=dict(runtime['process'],argv=intent['source']['command']);self.launch_cache[process['pid']]=process
    self.emit('after_identity',dict(runtime=runtime));return types.SimpleNamespace(pid=process['pid'])
   if launched is not None and launched.poll()not in(None,0):raise RuntimeError('CPUexec exited before publishing an exact-source identity')
   time.sleep(.05)
  # No attempt is charged merely because publication/lock ownership is pending.
  raise AwaitCPUOwnership('Await exact-source CPU exec owner '+str(index))
 def install(self):
  if not(self.root/'CPU_state.json').exists()and self.ledger.exists():self.orig_write(self.root/'CPU_state.json',self.overlay({}))
  self.router.read=self.read_state;self.router.write=self.write_state;self.router.identity=self.identity;self.router.shutil.copy2=self.copy;subprocess.Popen=self.popen
 def restore(self):
  self.router.read=self.orig_read;self.router.write=self.orig_write;self.router.identity=self.orig_identity;self.router.shutil.copy2=self.orig_copy;subprocess.Popen=self.orig_popen
