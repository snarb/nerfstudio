#!/usr/bin/env python3
"""Read-only60s production health monitor; records process/progress/GPU/OOM proof."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except FileNotFoundError:return default

def process(pid):
 try:
  f=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  return dict(pid=int(pid),start=f[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=f[0])
 except (FileNotFoundError,ProcessLookupError):return None

def matches(expected,actual):
 return bool(expected and actual and actual['state']!='Z'and all(str(expected[k])==str(actual[k])for k in ['pid','start','boot']))

def atomic(p,a):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2)+'\n');t.replace(p)

def remote(root,job):
 script='''import pathlib,json,subprocess,re,time
r=pathlib.Path(ROOT);j=JOB
read=lambda p:json.loads(p.read_text())if p.exists()else None
receipt=read(r/'remote_queue/jobs'/j/'receipt.json');progress=read(r/'run/progress.json');runtime=read(r/'run/runtime.json')
expected=(receipt or {}).get('worker');actual=None
if expected:
 try:
  f=pathlib.Path('/proc/%s/stat'%expected['pid']).read_text().split(') ',1)[1].split();actual={'pid':expected['pid'],'start':f[19],'boot':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'state':f[0],'command':pathlib.Path('/proc/%s/cmdline'%expected['pid']).read_bytes().replace(b'\\0',b' ').decode()}
 except FileNotFoundError:pass
log=r/'remote_queue/jobs'/j/'worker.log';tail=''
if log.exists():
 with log.open('rb')as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
errors=[line for line in tail.splitlines()if re.search(r'out of memory|CUDA error|FloatingPointError|Insufficient disk|Traceback',line,re.I)]
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
apps=[]
for line in gpu.stdout.splitlines():
 cols=[x.strip()for x in line.split(',')]
 if len(cols)==3 and expected and cols[1]==str(expected['pid']):apps.append({'gpu_uuid':cols[0],'pid':int(cols[1]),'used_MiB':cols[2]})
print(json.dumps({'time':time.time(),'receipt':receipt,'worker_process':actual,'progress':progress,'runtime':runtime,'worker_GPU':apps,'GPU_query_ok':gpu.returncode==0,'error_evidence':errors[-5:],'complete':(r/'run/complete.json').exists(),'latest':read(r/'run/latest.json')}))
'''.replace('ROOT',repr(root)).replace('JOB',repr(job))
 result=subprocess.run(['ssh','-o','ConnectTimeout=10','ubuntu@silver-sable','python3','-'],input=script,text=True,capture_output=True,timeout=45,check=True)
 return json.loads(result.stdout)

def watch(root,out):
 out.mkdir(parents=True,exist_ok=True);lock=(out/'monitor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);last_log=0;last_status=None
 plan=read(root/'controller_plan.json');job=plan['jobs'][0]['name'];remote_root=str(Path(plan['hosts']['silver']['root']).parent)
 while True:
  try:
   state=read(root/'controller/state.json',{});controller=read(root/'controller/controller.json',{});result=remote(remote_root,job);receipt=result.get('receipt')or{};worker_ok=matches(receipt.get('worker'),result.get('worker_process'));controller_ok=matches(controller.get('process'),process((controller.get('process')or{}).get('pid')))
   progress=result.get('progress');status=state.get('jobs',{}).get(job,{}).get('status','unknown');age=time.time()-progress['time']if progress else None
   record=dict(time=time.time(),job=job,controller_alive=controller_ok,worker_identity_alive=worker_ok,status=status,progress_age_seconds=age,step=progress.get('step')if progress else None,complete=result['complete'],no_OOM_evidence=not any('out of memory'in s.lower()for s in result['error_evidence']),evidence=result)
   atomic(out/'status.json',record)
   # Compact evidence every poll; wall-clock log remains at most hourly unless
   # status changes, completion or a concrete process/error anomaly occurs.
   issue=bool(result['error_evidence'])or(status=='running'and not worker_ok)or(age is not None and age>180)
   if time.time()-last_log>=3600 or status!=last_status or issue or result['complete']:
    with(out/'checks.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
    last_log=time.time();last_status=status
   if result['complete']:return
  except Exception as error:atomic(out/'monitor_error.json',dict(time=time.time(),error=repr(error)))
  time.sleep(60)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--detach',action='store_true');a=p.parse_args()
 if a.detach:
  a.output.mkdir(parents=True,exist_ok=True)
  prior=read(a.output/'launch.json',{})
  if prior.get('process')and matches(prior['process'],process(prior['process']['pid'])):
   print(prior['process']['pid']);raise SystemExit(0)
  log=(a.output/'monitor.log').open('ab');child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(a.root),'--output',str(a.output)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);atomic(a.output/'launch.json',dict(time=time.time(),pid=child.pid,process=process(child.pid),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()));print(child.pid)
 else:watch(a.root,a.output)
