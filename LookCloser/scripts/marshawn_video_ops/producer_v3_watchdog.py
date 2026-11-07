"""Detached native-preparation supervisor; no GPU ownership or field side effects."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time
from prepare_marshawn_tuning import write,sha

def main(a):
 lock=open(a.root/'operations/producer_v3_watchdog.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 release=json.loads(a.release.read_text())
 if not release.get('authorized'):raise ValueError('Operational producer release is not authorized')
 env=dict(os.environ,PYTHONPATH=str(a.root/'operations')+':'+str(a.root/'code/LookCloser/scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');attempt=0
 while True:
  attempt+=1
  with (a.root/'operations/producer_v3_stdout.log').open('a')as stream:child=subprocess.Popen([sys.executable,str(a.root/'operations/marshawn_video_producer_v3.py'),'--root',str(a.root),'--release',str(a.release)],env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll()is None:
   write(a.root/'operations/producer_v3_watchdog_state.json',dict(watchdog_pid=os.getpid(),producer_pid=child.pid,attempt=attempt,release_sha256=sha(a.release),checked_at=time.time()));time.sleep(20)
  write(a.root/'operations/producer_v3_watchdog_state.json',dict(watchdog_pid=os.getpid(),producer_pid=child.pid,exit_code=child.returncode,attempt=attempt,checked_at=time.time()))
  if child.returncode==0:return
  time.sleep(40)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--release',type=Path,required=True);main(p.parse_args())
