#!/usr/bin/env python3
"""Release-gated detached controller and artifact watcher; idempotent per generation."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  return dict(pid=int(pid),start_ticks=int(fields[19]),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())if fields[0]!='Z'else None
 except (FileNotFoundError,ProcessLookupError):return None

def verify(root):
 p=read(root/'launch_plan.json');r=read(root/'root_release.json')
 if r.get('production_authorized')is not True or r.get('technical_only')is not False:raise ValueError('Explicit production root release required')
 for key,name in [('launch_plan_sha256','launch_plan.json'),('controller_plan_sha256','controller_plan.json'),('contract_sha256','contract.json'),('spec_sha256','spec.json')]:
  if r.get(key)!=sha(root/name):raise ValueError('Release mismatch: '+name)
 for path,digest in p['local_artifacts'].items():
  if sha(path)!=digest:raise ValueError('Launch artifact changed: '+path)
 return p

def run(root):
 root=Path(root).resolve();lock=(root/'launch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=verify(root)
 release=root/'root_release.json';subprocess.run(['scp',str(release),p['host']+':'+p['remote_release']],check=True)
 code='import pathlib,hashlib;assert hashlib.sha256(pathlib.Path('+repr(p['remote_release'])+').read_bytes()).hexdigest()=='+repr(sha(release))
 import shlex
 subprocess.run(['ssh',p['host'],shlex.join(['python3','-c',code])],check=True)
 prior=read(root/'watcher_process.json')if(root/'watcher_process.json').exists()else{}
 if not prior.get('process')or identity(prior['process']['pid'])!=prior['process']:
  log=(root/'artifact_watcher.log').open('ab');child=subprocess.Popen(p['watcher_command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  record=dict(time=time.time(),process=identity(child.pid),command=p['watcher_command']);(root/'watcher_process.json').write_text(json.dumps(record,indent=2)+'\n')
 result=subprocess.run(p['controller_launch_command'],capture_output=True,text=True,check=True)
 record=dict(time=time.time(),release_sha256=sha(release),controller_result=result.stdout.strip(),watcher=read(root/'watcher_process.json'),manual_visual_review_passed=False,geometry_approved=False)
 (root/'launch_receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
