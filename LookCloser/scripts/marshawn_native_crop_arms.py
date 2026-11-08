"""Quiet initial native-crop arm launcher; isolated stage plans and cold outputs.

This reuses the unchanged qualified physical-GPU controller. The matrix contains
explicit worker/spec hashes. Promotion is a separate ROOT-owned stage-plan edit,
never transfer of weights between arms.
"""
from pathlib import Path
import argparse,copy,fcntl,hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from marshawn_native_crop_queue import config,read,write,sha,immutable,call,module,monitor


def rebind(value,root,remote):
 """Translate only explicit host filesystem roots; no tensor/config rewriting."""
 if isinstance(value,dict):return {k:rebind(v,root,remote)for k,v in value.items()}
 if isinstance(value,list):return [rebind(v,root,remote)for v in value]
 if isinstance(value,str):
  pairs=[(str(root),str(remote)),('/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107','/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_140k_20261107')]
  for a,b in pairs:
   if value==a or value.startswith(a+'/'):return b+value[len(a):]
 return value


def verify_initial_spec(spec,root):
 if spec.get('hard_max_iterations')!=140000 or spec.get('train_count')!=66:raise ValueError('Native arm must retain hard140k and ALL66 train')
 if spec.get('seed')!=42:raise ValueError('Initial controlled matrix seed42 required')
 output=Path(spec['output']).resolve();output.relative_to((Path(root)/'arms').resolve())
 if output.exists()and any(output.iterdir()):raise ValueError('Initial arm output is not cold/empty')
 stage=read(spec['stage_plan'])
 if stage.get('schema')!=1 or stage.get('stop_after_iterations')!=32000:raise ValueError('Initial arm ends at32k before ROOT promotion')
 if any(k in spec for k in ['resume','load_checkpoint','load_dir','checkpoint','parent_checkpoint']):raise ValueError('Foreign checkpoint import into cold arm forbidden')
 return spec


def prepare(root,matrix_path):
 root=Path(root).resolve();c=config(root);matrix=read(matrix_path)
 if matrix.get('kind')!='NATIVE_CROP_INITIAL_ARM_MATRIX_V1' or len(matrix['arms'])>10:raise ValueError('Unknown/oversized arm matrix')
 if not(root/'data_native/frequency_ready.json').exists():raise ValueError('Fresh native66 maps not finalized')
 ready=read(root/'data_native/ready.json');fr=read(root/'data_native/frequency_ready.json')
 if not ready.get('training_eligible') or not ready.get('frequency_ready'):raise ValueError('Native input not sealed for training')
 if fr.get('ready_sha256')!=sha(root/'data_native/ready.json'):raise ValueError('Final frequency/data binding differs')
 worker=Path(matrix['worker']);worker.resolve().relative_to(root)
 if sha(worker)!=matrix['worker_sha256']:raise ValueError('Native worker code changed')
 for path,digest in matrix['worker_files_sha256'].items():
  if sha(path)!=digest:raise ValueError('Native worker overlay dependency changed')
 jobs=[];claims=set()
 for arm in matrix['arms']:
  name=arm['name'];host=c['hosts'][arm['host']];gpu=arm.get('gpu');slot=(arm['host'],gpu)
  if slot in claims:raise ValueError('Matched arms claim one physical slot twice')
  claims.add(slot)
  source=Path(arm['spec']);spec=verify_initial_spec(read(source),root);hr=Path(host['root']);native=rebind(spec,root,hr)if hr!=root else spec
  sp=root/'ops/arm_specs_by_host'/arm['host']/(name+'.json');immutable(sp,native)
  native_spec=hr/'ops/arm_specs'/(name+'.json');native_worker=Path(rebind(str(worker),root,hr))if hr!=root else worker
  if hr!=root:
   call(host,['mkdir','-p',hr/'code_overlay',native_spec.parent,hr/'arms',hr/'stage_plans'])
   subprocess.run(['rsync','-a',str(root/'code_overlay')+'/',host['address']+':'+str(hr/'code_overlay')+'/'],check=True,capture_output=True,timeout=60)
   subprocess.run(['rsync','-rLt','--exclude=frequency_specs/','--exclude=*.fit.lock','--exclude=*.partial*',str(root/'data_native')+'/',host['address']+':'+str(hr/'data_native')+'/'],check=True,capture_output=True,timeout=300)
   call(host,[host['python'],'-c',"from pathlib import Path;import sys,json; p=Path(sys.argv[1]);r=json.loads(sys.argv[2]);assert not p.exists()or json.loads(p.read_text())==r;p.write_text(json.dumps(r,indent=2))",native_spec,json.dumps(native)])
   stage=Path(spec['stage_plan']);remote_stage=Path(native['stage_plan']);call(host,[host['python'],'-c',"from pathlib import Path;import sys; p=Path(sys.argv[1]);s=sys.argv[2];assert not p.exists()or p.read_text()==s;p.write_text(s)",remote_stage,stage.read_text()])
  else:immutable(native_spec,native)
  command=[host['python'],str(native_worker),'--spec',str(native_spec)]
  jobs.append(dict(name='arm_%s_stage032000'%name,host=arm['host'],gpu=gpu,command=command,cwd=str(hr),env={'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},progress_files=[str(Path(native['output'])/'progress.json'),str(Path(native['output'])/'latest.json')],metrics_file=str(Path(native['output'])/'stage_paused.json')))
  write(root/'ops/arm_bindings'/(name+'.json'),dict(name=name,host=arm['host'],gpu=gpu,original_spec_path=str(source),original_spec_sha256=sha(source),native_spec_path=str(native_spec),native_spec_sha256=sha(sp),neural_parameter_values_unchanged=True,only_IO_roots_rebound=hr!=root,initialization='FULLY_COLD; no parent checkpoint; own-stage crash resume only',max_iterations=140000,initial_stage=32000))
 plan=dict(schema_version=1,poll_seconds=c['poll_seconds'],hosts=c['hosts'],jobs=jobs);fleet=module(root/'ops/code/qualified_controller.py','native_arms_prepare');fleet.validate(plan);manifest=root/'ops/arm_manifest.json';immutable(manifest,plan);fleet.prepare(root/'ops/arm_dispatch',manifest);return plan


def run(root,matrix_path):
 root=Path(root).resolve();c=config(root);matrix=read(matrix_path)
 if matrix.get('launch_authorized')is not True:raise ValueError('Native arm smoke/launch not released yet')
 proof=read(matrix['smoke_qualification'])
 if proof.get('passed',proof.get('status')=='pass')is not True:raise ValueError('Actual native setup+ARM smoke not passed')
 if sha(matrix['smoke_qualification'])!=matrix['smoke_qualification_sha256']:raise ValueError('Foreign smoke proof')
 held=(root/'ops/arm_controller.lock').open('a+')
 try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 # On restart, adopt the frozen existing plan and own worker receipts. Do not
 # demand that an already-running arm's output be empty again.
 fleet=module(root/'ops/code/qualified_controller.py','native_arms_DAG')
 if(root/'ops/arm_manifest.json').exists():plan,state=fleet.prepare(root/'ops/arm_dispatch',root/'ops/arm_manifest.json')
 else:prepare(root,matrix_path);plan,state=fleet.prepare(root/'ops/arm_dispatch',root/'ops/arm_manifest.json')
 write(root/'ops/arm_controller_identity.json',dict(process=fleet.identity(),matrix_sha256=sha(matrix_path)));last=0
 while True:
  if(root/'STOP.json').exists():return
  done=fleet.tick(plan,state,root/'ops/arm_dispatch');now=time.time()
  if now-last>=c['monitor_seconds']:monitor(root,c,state,'independent_cold_initial32k_arms');last=now
  write(root/'ops/arm_status.json',dict(time=now,jobs={n:j['status']for n,j in state['jobs'].items()},initial_stage=32000,hard_max_iterations=140000))
  if done:
   write(root/'ops/initial_arms_finished.json',dict(time=now,all_initial_jobs_complete=all(j['status']=='complete'for j in state['jobs'].values()),ROOT_promotion_review_next=True,weights_never_transferred_between_arms=True));return
  time.sleep(c['poll_seconds'])


def watchdog(root,matrix):
 root=Path(root);held=(root/'ops/arm_watchdog.lock').open('a+')
 try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 c=config(root);fleet=module(root/'ops/code/qualified_controller.py','native_arm_watch_ID');write(root/'ops/arm_watchdog_identity.json',dict(process=fleet.identity(),matrix_sha256=sha(matrix)))
 while not(root/'ops/initial_arms_finished.json').exists()and not(root/'STOP.json').exists():
  # Preparation/final native maps can finish independently while this waits.
  if not(root/'data_native/frequency_ready.json').exists():time.sleep(10);continue
  if read(matrix).get('launch_authorized')is not True:time.sleep(10);continue
  with(root/'ops/arm_stdout.log').open('ab')as log:p=subprocess.Popen([sys.executable,__file__,'run','--root',str(root),'--matrix',str(matrix)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  code=p.wait();write(root/'ops/arm_last_exit.json',dict(time=time.time(),returncode=code))
  if code==0:return
  time.sleep(30)


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','watchdog','launch']);p.add_argument('--root',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);a=p.parse_args()
 if a.action=='prepare':print(json.dumps(dict(jobs=len(prepare(a.root,a.matrix)['jobs']),GPU_started=False)))
 elif a.action=='run':run(a.root,a.matrix)
 elif a.action=='watchdog':watchdog(a.root,a.matrix)
 else:
  with(a.root/'ops/arm_watchdog_stdout.log').open('ab')as log:p=subprocess.Popen([sys.executable,__file__,'watchdog','--root',str(a.root),'--matrix',str(a.matrix)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print(json.dumps(dict(watchdog_pid=p.pid,training_GPU_started=False)))
if __name__=='__main__':main()
