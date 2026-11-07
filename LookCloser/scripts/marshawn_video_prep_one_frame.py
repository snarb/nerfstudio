"""Durable full-frame RAW/native-mask preparation; intentionally no data publication."""
from pathlib import Path
import os,sys,subprocess,json,fcntl,time
from prepare_marshawn_tuning import write,sha
r=Path(sys.argv[1]);frame=int(sys.argv[2]);code=Path(__file__).resolve().parent;overlay=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay');out=r/'preparation_logs';out.mkdir(exist_ok=True)
cameras=json.loads((r/'source_inventory.json').read_text())['cameras'];uid='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46';env=dict(os.environ,CUDA_VISIBLE_DEVICES=uid,PYTHONPATH=str(overlay/'src')+':'+str(overlay/'deps')+':'+str(code),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
write(out/f'{frame}_standalone_launch.json',dict(pid=os.getpid(),frame=frame,started=time.time(),source_inventory_sha256=sha(r/'source_inventory.json'),checkpoint_sha256=sha(overlay/'weights/sam3.pt'),data_publication=False,no_NeRF_or_learned_grid_state=True))
commands=[('sources',[sys.executable,str(code/'marshawn_video_prep_sources.py'),'--frames',str(frame),'--cameras',*cameras,'--output',str(r/'sources'),'--raw-native','--workers','2']),('masks',[sys.executable,str(code/'marshawn_video_prep_masks.py'),'--root',str(r),'--frame',str(frame),'--cameras',*cameras])]
for stage,command in commands:
 lock=None;kwargs={}
 if stage=='masks':
  lock=open('/tmp/lookcloser_tuning_gpu_'+str(os.getuid())+'_0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX);kwargs['pass_fds']=(lock.fileno(),)
 with (out/f'{frame}_standalone_{stage}.log').open('a') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,**kwargs)
 if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 write(out/f'{frame}_standalone_{stage}_done.json',dict(finished=time.time(),frame=frame,stage=stage))
write(out/f'{frame}_native_complete.json',dict(finished=time.time(),frame=frame,all66=True,data_publication=False))
