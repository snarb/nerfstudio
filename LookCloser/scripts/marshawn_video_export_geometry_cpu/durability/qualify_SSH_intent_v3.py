"""Actual frozen tick with a CPU-only receiver; kill after the remote ACK."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,types,hashlib
O=Path(__file__).parent;sys.path.insert(0,str(O));import cpu_exec
from bridge_v3 import Bridge
C=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107');sys.path.insert(0,str(C/'code/LookCloser/scripts'));import marshawn_tuning_controller as fleet

def read(p):return cpu_exec.read(p)
def write(p,v):return cpu_exec.atomic(p,v)
def run_child(q,fault):
 q.mkdir(exist_ok=True);dummy=types.SimpleNamespace(__file__=str(q/'route.py'),read=read,write=write,identity=cpu_exec.identity,shutil=__import__('shutil'));bridge=Bridge(dummy,q/'ledger',O/'cpu_exec.py',shader_sha='unused')
 plan=dict(schema_version=1,poll_seconds=5,hosts={'CPU_TEST_HOST':dict(root=str(q/'mock_remote'),python=sys.executable,controller=str(fleet.__file__),gpus=[0,1])},jobs=[dict(name='TEST_ONLY_RENDER',host='CPU_TEST_HOST',command=[sys.executable,'-c','pass'],cwd=str(q))]);state=bridge.read_state(q/'GPU_state.json')
 def host_call(host,argv,**kw):
  if argv[0]=='_probe':
   receipt=read(q/'mock_remote_receipt.json')if(q/'mock_remote_receipt.json').exists()else None;alive=False
   if receipt:
    x=cpu_exec.identity(receipt['worker']['pid']);alive=x is not None and x['state']!='Z'and all(x[k]==receipt['worker'][k]for k in['pid','start','boot'])
   devices={str(i):dict(uuid='GPU-CPU-TEST-NO-CUDA-'+str(i),processes=[{'pid':receipt['worker']['pid']}]if receipt and alive and receipt['gpu']==i else[])for i in[0,1]};snap=dict(jobs={'TEST_ONLY_RENDER':dict(receipt,worker_alive=alive,supervisor_alive=alive)}if receipt else{},gpus=devices,gpu_error=None,free_bytes=10**12,controller_sha256=cpu_exec.sha(fleet.__file__))
   return subprocess.CompletedProcess([],0,json.dumps(snap),'')
  assert argv[0]=='_start';spec=json.loads(kw['input_text']);before=read(q/'state.json')['jobs']['TEST_ONLY_RENDER'];assert before['status']=='dispatching'and before['spec_sha256']==fleet.digest(spec)
  process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);identity=cpu_exec.identity(process.pid);receipt=dict(name='TEST_ONLY_RENDER',status='running',spec_sha256=fleet.digest(spec),gpu=spec['gpu'],host='CPU_TEST_HOST',worker=identity,supervisor=identity);write(q/'mock_remote_receipt.json',receipt)
  with(q/'mock_starts.jsonl').open('a')as f:f.write(json.dumps(receipt)+'\n');f.flush();os.fsync(f.fileno())
  if fault:os._exit(93) # ACK delivered and worker alive; no outer mirror save.
  return subprocess.CompletedProcess([],0,'ack','')
 fleet.host_call=host_call;fleet.tick(plan,state,q);write(q/'GPU_state.json',state);assert state['jobs']['TEST_ONLY_RENDER']['status']=='running';assert len((q/'mock_starts.jsonl').read_text().splitlines())==1;write(q/'qualified.json',dict(passed=True,host=state['jobs']['TEST_ONLY_RENDER']['host'],gpu=state['jobs']['TEST_ONLY_RENDER']['gpu'],starts=1,authoritative_state_before_SSH=True,CUDA_or_model_not_loaded=True))

def main(q):
 q.mkdir(parents=True,exist_ok=True);write(q/'GPU_state.json',dict(jobs={'TEST_ONLY_RENDER':dict(status='pending')}));log=(q/'test.log').open('ab')
 p=subprocess.run([sys.executable,__file__,'child','--root',str(q),'--fault'],stdout=log,stderr=subprocess.STDOUT);assert p.returncode==93,p.returncode
 p=subprocess.run([sys.executable,__file__,'child','--root',str(q)],stdout=log,stderr=subprocess.STDOUT);assert p.returncode==0,(q/'test.log').read_text()[-3000:]
 result=dict(kind='FROZEN_FLEET_TICK_REAL_CPU_ACK_CRASH_QUALIFICATION_V1',**read(q/'qualified.json'),frozen_controller_sha256=cpu_exec.sha(fleet.__file__),test_source_sha256=cpu_exec.sha(__file__),no_production_state_changed=True,finished=time.time());write(q/'complete.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','child']);p.add_argument('--root',type=Path,required=True);p.add_argument('--fault',action='store_true');a=p.parse_args();main(a.root)if a.mode=='run'else run_child(a.root,a.fault)
