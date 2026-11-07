"""Read-only paired-fork artifact preservation and durable local supervision."""
import argparse,fcntl,hashlib,json,os,shlex,shutil,subprocess,time
from pathlib import Path
ARMS=('original_mapped','source_margin002')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def identity_alive(rec):
 try:
  fields=Path('/proc/'+str(rec['pid'])+'/stat').read_text().split(') ',1)[1].split()
  return fields[0]!='Z'and fields[19]==str(rec['start'])and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==rec['boot']
 except (FileNotFoundError,KeyError,TypeError):return False

def remote_once(root):
 result=dict(time=time.time(),arms={})
 for arm in ARMS:
  run=root/'runs'/('production_'+arm);rows=[]
  for renders in sorted((run/'renders').glob('step-*')):
   dest=root/'review_archives'/arm/renders.name;receipt=dest/'receipt.json'
   if receipt.exists():rows.append(json.load(open(receipt)));continue
   cp=run/'checkpoints'/(renders.name+'.ckpt');grid=cp.with_name(renders.name+'_grid.json');npz=grid.with_suffix('.npz');region=run/'region_diagnostics'/(renders.name+'.json')
   if not all(p.exists()for p in [cp,grid,npz,region,renders/'metrics.json']):continue
   record=json.load(open(grid));assert sha(cp)==record['checkpoint_sha256']and sha(npz)==record['grid_archive_sha256']
   dest.mkdir(parents=True,exist_ok=True)
   if not(dest/'checkpoint.ckpt').exists():os.link(cp,dest/'checkpoint.ckpt')
   assert sha(dest/'checkpoint.ckpt')==record['checkpoint_sha256']
   for source,name in [(grid,'grid.json'),(npz,'grid.npz'),(region,'region_diagnostics.json'),(root/'specs'/('production_'+arm)/'spec.json','spec.json'),(root/'specs'/('production_'+arm)/'root_release.json','root_release.json'),(root/'source_manifest.json','source_manifest.json'),(root/'margin/manifest.json','margin_manifest.json')]:shutil.copy2(source,dest/name)
   configs=list((run/'worker').rglob('config.yml'));assert len(configs)==1;shutil.copy2(configs[0],dest/'config.yml');shutil.copytree(renders,dest/'renders',dirs_exist_ok=True)
   files={str(p.relative_to(dest)):sha(p)for p in dest.rglob('*')if p.is_file()and p.name!='checkpoint.ckpt'}
   row=dict(step=int(renders.name.split('-')[1]),checkpoint_sha256=record['checkpoint_sha256'],archive=str(dest),files_sha256=files,source_spec_sha256=sha(dest/'spec.json'),checkpoint_retained=True,explicit_gate_loader_required=True,time=time.time());write(receipt,row);rows.append(row)
  status={}
  for name in ['progress.json','latest.json','complete.json','retention.json']:
   p=run/name
   if p.exists():status[name]=json.load(open(p))
  job=root/'remote_queue/jobs'/('margin126316_production_'+arm)/'receipt.json';j=json.load(open(job))if job.exists()else{}
  result['arms'][arm]=dict(archives=rows,status=status,job_status=j.get('status','pending'),worker_alive=identity_alive(j.get('worker')),supervisor_alive=identity_alive(j.get('supervisor')),worker=j.get('worker'),gpu=j.get('gpu'),gpu_uuid=j.get('gpu_uuid'),heartbeat=j.get('heartbeat'),oom_evidence=j.get('latest_check',{}).get('oom_evidence'),complete=(run/'complete.json').exists())
 write(root/'review_archives/remote_state.json',result);return result

def watch(local,remote,host,remote_script):
 out=local/'review_archives';out.mkdir(exist_ok=True);lock=(out/'watcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(out/'watcher_launch.json',dict(pid=os.getpid(),time=time.time(),source_sha256=sha(__file__),GPU_queries=False,worker_mutations=False))
 while True:
  try:
   cmd=shlex.join(['python3',remote_script,'--remote-once','--root',remote]);p=subprocess.run(['ssh',host,cmd],check=True,capture_output=True,text=True,timeout=180);state=json.loads(p.stdout)
   for arm,a in state['arms'].items():
    for row in a['archives']:
     dest=out/arm/f"step-{row['step']:09d}"
     if(dest/'fetch_verified.json').exists():continue
     dest.mkdir(parents=True,exist_ok=True);subprocess.run(['rsync','-a','--exclude=checkpoint.ckpt',host+':'+row['archive']+'/',str(dest)+'/'],check=True,stdout=subprocess.DEVNULL,timeout=180)
     for rel,d in row['files_sha256'].items():assert sha(dest/rel)==d,rel
     write(dest/'fetch_verified.json',dict(time=time.time(),passed=True,checkpoint_sha256=row['checkpoint_sha256'],checkpoint_retained_remote=True))
   write(out/'watcher_state.json',dict(pid=os.getpid(),**state))
   with(out/'health.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),arms={arm:{k:a[k]for k in ['job_status','worker_alive','supervisor_alive','gpu','heartbeat','oom_evidence','complete']}for arm,a in state['arms'].items()}))+'\n')
   if all(a['complete']for a in state['arms'].values()):write(out/'fetch_complete.json',dict(time=time.time(),all_artifacts_verified=True,quality_approved=False));return
   if any(a['job_status']in ['failed','blocked']for a in state['arms'].values()):write(out/'producer_failure.json',dict(time=time.time(),state=state));return
  except Exception as e:
   with(out/'errors.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),error=repr(e)))+'\n')
  time.sleep(30)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--remote-once',action='store_true');p.add_argument('--remote-root');p.add_argument('--host',default='ubuntu@silver-sable');p.add_argument('--remote-script');a=p.parse_args()
 if a.remote_once:print(json.dumps(remote_once(a.root)))
 else:watch(a.root,a.remote_root,a.host,a.remote_script)
