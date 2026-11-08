from pathlib import Path
import subprocess,json,os,time,fcntl,hashlib
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');P=G/'operations/finish_gcrole_generation_v121.py'
def write(p,q):t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+'\n');os.replace(t,p)
if __name__=='__main__':
 lock=open(G/'operations/generation_watchdog.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=hashlib.sha256(P.read_bytes()).hexdigest()
 while not(G/'operations/corrected_source_generation_REVIEW_READY.json').exists():
  if(G/'operations/HOLD.json').exists():time.sleep(5);continue
  assert hashlib.sha256(P.read_bytes()).hexdigest()==code
  with(G/'operations/generation_finisher_v121.log').open('ab')as f:p=subprocess.Popen(['/home/pavlo/.conda/envs/nerfstudio/bin/python',str(P)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  write(G/'operations/generation_watchdog_state.json',dict(watchdog=os.getpid(),worker=p.pid,code_sha256=code,created=time.time()))
  rc=p.wait()
  if rc:write(G/'operations/generation_watchdog_last_failure.json',dict(worker=p.pid,exitcode=rc,time=time.time()));time.sleep(20)
 write(G/'operations/generation_watchdog_complete.json',dict(time=time.time(),actual_review_ready=True,no_source_marker_published=True))
