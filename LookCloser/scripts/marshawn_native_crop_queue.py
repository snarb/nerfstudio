"""Isolated native-crop frequency/arm queue using the qualified physical GPU DAG.

No old video state is loaded. `prepare` is CPU-only; launching requires the local
queue_config launch flag after actual native-image review. The fitter and neural
worker remain separate, source-hashed entry points.
"""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time


def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except FileNotFoundError:return default


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with t.open('w')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)


def immutable(path,value):
 p=Path(path)
 if p.exists()and read(p)!=value:raise ValueError('Immutable isolated queue input changed: '+str(p))
 if not p.exists():write(p,value)


def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def config(root):
 p=Path(root)/'ops/queue_config.json';c=read(p)
 if c['root']!=str(Path(root).resolve())or c['hard_max_iterations']!=140000:raise ValueError('Foreign queue root/cap')
 if c['monitor_seconds']>3600 or c['max_attempts']>3:raise ValueError('Invalid supervision/retry bound')
 for name,digest in c['code_sha256'].items():
  if sha(Path(root)/'ops/code'/name)!=digest:raise ValueError('Qualified queue dependency changed: '+name)
 return c


def call(host,argv,timeout=60):
 cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host['address'],shlex.join(list(map(str,argv)))]if host.get('address')else list(map(str,argv))
 return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=timeout)


def stage_code(root,c):
 """CPU transport only. Shared FSX host paths are not recopied."""
 root=Path(root)
 for host in c['hosts'].values():
  target=Path(host['root'])/'ops/code'
  call(host,['mkdir','-p',target])
  if Path(host['root'])!=root:
   subprocess.run(['rsync','-a',str(root/'ops/code')+'/',host['address']+':'+str(target)+'/'],check=True,capture_output=True,timeout=60)
  check="from pathlib import Path;import hashlib,json,sys;r=Path(sys.argv[1]);x=json.loads(sys.argv[2]);assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in x.items());print('source hashes verified')"
  call(host,[host['python'],'-c',check,target,json.dumps(c['code_sha256'])])


def camera_source(root,c,camera):
 data=Path(root)/'data_native';image=data/'images'/(camera+'.png');receipt=data/'receipts'/(camera+'.json')
 if not image.exists()or not receipt.exists():return None
 r=read(receipt);digest=r.get('image_sha256',r.get('rgb_sha256'))
 if r.get('frame')!=c['frame']or r.get('camera')!=camera or not digest or sha(image)!=digest:raise ValueError('Native crop receipt/image mismatch: '+camera)
 from PIL import Image
 width,height=Image.open(image).size
 if width%8 or height%8 or width<=0 or height<=0:raise ValueError('Native frequency lattice requires positive multiples of8')
 return dict(image=image,receipt=receipt,image_sha256=digest,width=width,height=height,receipt_sha256=sha(receipt))


def stage_camera(root,c,camera):
 src=camera_source(root,c,camera)
 if src is None:return False
 root=Path(root)
 for host in c['hosts'].values():
  hr=Path(host['root']);image=hr/'data_native/images'/(camera+'.png');dest=hr/'data_native/lookcloser_frequencies'/(camera+'.pt');sp=hr/'ops/frequency_specs'/(camera+'.json')
  source=Path(host['frequency_source_code']);preprocess=source/'nerfstudio/scripts/lookcloser_preprocess.py'
  if hr!=root:
   call(host,['mkdir','-p',image.parent,sp.parent,dest.parent])
   check="from pathlib import Path;import sys,hashlib;p=Path(sys.argv[1]);assert not p.exists()or hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2],'Foreign native image';print('safe')"
   call(host,[host['python'],'-c',check,image,src['image_sha256']])
   subprocess.run(['rsync','-a','--ignore-existing',str(src['image']),host['address']+':'+str(image)],check=True,capture_output=True,timeout=180)
   actual=call(host,[host['python'],'-c',"import hashlib,sys;from pathlib import Path;print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())",image]).stdout.strip()
   if actual!=src['image_sha256']:raise ValueError('Staged native image SHA mismatch')
   pre_sha=call(host,[host['python'],'-c',"import hashlib,sys;from pathlib import Path;print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())",preprocess]).stdout.strip()
  else:pre_sha=sha(preprocess)
  spec=dict(frame=c['frame'],camera=camera,image=str(image),image_sha256=src['image_sha256'],output=str(dest),source_code=str(source),preprocess_source_sha256=pre_sha,allocator_GiB=8,recipe=c['fixed_frequency_recipe'],native_crop_no_resize=True,source_crop_receipt_sha256=src['receipt_sha256'],fresh_native_map_no_HD_reuse=True)
  local=root/'ops/specs_by_host'/next(k for k,v in c['hosts'].items()if v==host)/(camera+'.json');immutable(local,spec)
  if hr==root:
   # Olive and DEV3 share the same canonical files and frozen preprocessing code.
   immutable(sp,spec)
  else:
   payload=json.dumps(spec,sort_keys=True)
   script="from pathlib import Path;import json,sys,os;p=Path(sys.argv[1]);r=json.loads(sys.argv[2]);assert not p.exists()or json.loads(p.read_text())==r,'Foreign frequency spec';t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(r,indent=2));os.replace(t,p)"
   call(host,[host['python'],'-c',script,sp,payload])
 write(root/'ops/staged_cameras'/(camera+'.json'),dict(camera=camera,image_sha256=src['image_sha256'],source_receipt_sha256=src['receipt_sha256'],native_width=src['width'],native_height=src['height'],all_hosts_verified=True))
 return True


def frequency_plan(root,c):
 jobs=[]
 for camera in c['cameras']:
  for attempt in range(c['max_attempts']):
   jobs.append(dict(name='nativefreq_%s_a%02d'%(camera,attempt),command=['{host_python}','{host_root}/ops/code/frequency_worker.py','--spec','{host_root}/ops/frequency_specs/'+camera+'.json'],env={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},cwd='{host_root}',camera=camera,attempt=attempt,progress_files=['{host_root}/data_native/lookcloser_frequencies/'+camera+'.progress.json'],metrics_file='{host_root}/data_native/lookcloser_frequencies/'+camera+'.complete.json'))
 return dict(schema_version=1,poll_seconds=c['poll_seconds'],hosts=c['hosts'],jobs=jobs)


def active_frequency_jobs(plan,state,staged):
 """One attempt per image; SSH-unknown/alive ownership never advances."""
 result=[]
 for camera in sorted(staged):
  candidates=[j for j in plan['jobs']if j['camera']==camera]
  for job in candidates:
   item=state['jobs'][job['name']]
   if item['status']=='orphaned'and item.get('worker_alive')is True:item['status']='unknown'
   dead=item.get('worker_alive')is False and item.get('supervisor_alive')is False
   if item['status']=='orphaned'and dead:item['status']='failed'
   if item['status']in ['failed','blocked']:
    if not dead:return result  # conservative until both identities are dead
    continue
   result.append(job);break
  else:raise RuntimeError('Native image exhausted bounded fit attempts: '+camera)
 return result


def gather(root,c,job,state):
 item=state['jobs'][job['name']]
 if item.get('status')!='complete':return False
 camera=job['camera'];host=c['hosts'][item['host']];base=Path(host['root'])/'data_native/lookcloser_frequencies';target=Path(root)/'data_native/lookcloser_frequencies';target.mkdir(parents=True,exist_ok=True)
 if Path(host['root'])!=Path(root):
  for suffix in ['.pt','.json','.complete.json']:
   subprocess.run(['rsync','-a',host['address']+':'+str(base/(camera+suffix)),str(target/(camera+suffix))],check=True,capture_output=True,timeout=120)
 sys.path[:0]=[str(Path(root)/'ops/code')];worker=module(Path(root)/'ops/code/frequency_worker.py','native_crop_frequency_check');spec=read(Path(root)/'ops/frequency_specs'/(camera+'.json'));worker.validate_spec(spec)
 if not worker.validate_existing(spec):raise ValueError('Incomplete native map after gather')
 receipt=read(target/(camera+'.complete.json'))
 if receipt.get('source_image_sha256')!=spec['image_sha256']:raise ValueError('Foreign frequency completion')
 write(Path(root)/'ops/frequency_verified'/(camera+'.json'),dict(camera=camera,source_image_sha256=spec['image_sha256'],map_sha256=sha(target/(camera+'.pt')),metadata_sha256=sha(target/(camera+'.json')),completion_sha256=sha(target/(camera+'.complete.json')),host=item['host'],source_crop_receipt_sha256=spec['source_crop_receipt_sha256'],native_fresh_fit=True))
 return True


def monitor(root,c,state,phase):
 root=Path(root);rows=state.get('host_snapshots',{});health={}
 for name,host in c['hosts'].items():
  snapshot=rows.get(name,{})
  health[name]=dict(gpus=snapshot.get('gpus'),error=snapshot.get('error'),gpu_error=snapshot.get('gpu_error'))
 record=dict(time=time.time(),phase=phase,controller_pid=os.getpid(),actual_controller_alive=True,physical_hosts=health,jobs={k:dict(status=v.get('status'),host=v.get('host'),gpu=v.get('gpu'),worker=v.get('worker'),worker_alive=v.get('worker_alive'),supervisor_alive=v.get('supervisor_alive'))for k,v in state['jobs'].items()},last_compact_state_and_GPU_OOM_evidence='Native supervisor jobs/*/checks.jsonl contain PID/progress/GPU/oom_evidence every poll',next_due=time.time()+c['monitor_seconds'])
 write(root/'ops/supervision/latest.json',record)
 with(root/'ops/supervision/checks.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')


def run_frequencies(root):
 root=Path(root).resolve();c=config(root)
 if c.get('launch_authorized')is not True:raise ValueError('Actual native crop review/launch is not enabled yet')
 held=(root/'ops/frequency_controller.lock').open('a+')
 try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 fleet=module(root/'ops/code/qualified_controller.py','native_crop_qualified_DAG');original_materialize=fleet.materialize
 def materialize(job,host,gpu):
  result=original_materialize(job,host,gpu);result['command']=[x.replace('{host_python}',host['python'])for x in result['command']];return result
 fleet.materialize=materialize;plan=frequency_plan(root,c);manifest=root/'ops/frequency_manifest.json';immutable(manifest,plan);plan,state=fleet.prepare(root/'ops/frequency_dispatch',manifest);staged=set();verified=set();last_monitor=0
 write(root/'ops/frequency_controller_identity.json',dict(process=fleet.identity(),config_sha256=sha(root/'ops/queue_config.json')))
 while True:
  if(root/'STOP.json').exists():raise SystemExit('New native tuning queue explicitly stopped')
  newly_staged=False
  for camera in c['cameras']:
   p=root/'ops/staged_cameras'/(camera+'.json')
   if p.exists():staged.add(camera)
   elif stage_camera(root,c,camera):staged.add(camera);newly_staged=True;break
  # No GPU supervisor is launched for an unavailable image. Other images fit
  # immediately; all66 image preparation need not finish before the first fit.
  ready=copy.deepcopy(plan);ready['jobs']=active_frequency_jobs(plan,state,staged);fleet.tick(ready,state,root/'ops/frequency_dispatch')
  for job in plan['jobs']:
   if job['camera']not in verified and gather(root,c,job,state):verified.add(job['camera'])
  write(root/'ops/frequency_status.json',dict(time=time.time(),staged=len(staged),verified=len(verified),total=len(c['cameras']),jobs={k:v['status']for k,v in state['jobs'].items()}))
  now=time.time()
  if now-last_monitor>=c['monitor_seconds']:monitor(root,c,state,'fresh_native_frequency');last_monitor=now
  if len(verified)==len(c['cameras']):write(root/'ops/frequency_complete.json',dict(cameras=c['cameras'],all_native_source_bound=True,no_HD_maps_reused=True,finished=now));return
  time.sleep(1 if newly_staged else c['poll_seconds'])


def watchdog(root):
 root=Path(root);c=config(root);held=(root/'ops/watchdog.lock').open('a+')
 try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 fleet=module(root/'ops/code/qualified_controller.py','native_crop_watchdog_ID');write(root/'ops/watchdog_identity.json',dict(process=fleet.identity(),source_sha256=sha(__file__)))
 while not(root/'ops/frequency_complete.json').exists()and not(root/'STOP.json').exists():
  with(root/'ops/queue_stdout.log').open('ab')as log:p=subprocess.Popen([sys.executable,__file__,'run-frequencies','--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  code=p.wait();write(root/'ops/last_exit.json',dict(time=time.time(),returncode=code))
  if code==0:return
  time.sleep(30)


def prepare(root):
 c=config(root);stage_code(root,c);fleet=module(Path(root)/'ops/code/qualified_controller.py','native_crop_prepare');p=frequency_plan(root,c);fleet.validate(p);immutable(Path(root)/'ops/frequency_manifest.json',p);fleet.prepare(Path(root)/'ops/frequency_dispatch',Path(root)/'ops/frequency_manifest.json');return dict(CPU_only=True,logical_images=len(c['cameras']),bounded_attempt_slots=len(p['jobs']),physical_GPUs=10,launch_authorized=c['launch_authorized'])


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run-frequencies','watchdog','launch','status']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 if a.action=='prepare':print(json.dumps(prepare(a.root)))
 elif a.action=='run-frequencies':run_frequencies(a.root)
 elif a.action=='watchdog':watchdog(a.root)
 elif a.action=='status':print(json.dumps(read(a.root/'ops/frequency_status.json',{}),indent=2))
 else:
  c=config(a.root)
  if c.get('launch_authorized')is not True:raise ValueError('Native crop launch not authorized')
  with(a.root/'ops/watchdog_stdout.log').open('ab')as log:r=subprocess.Popen([sys.executable,__file__,'watchdog','--root',str(a.root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps(dict(watchdog_pid=r.pid)))
if __name__=='__main__':main()
