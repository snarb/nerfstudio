"""Durable three-view fresh-frame native preparation smoke; no field training."""
from pathlib import Path
import os,sys,subprocess,json,fcntl,time
from prepare_marshawn_tuning import write,sha
r=Path(sys.argv[1]);code=Path(__file__).resolve().parent;overlay=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay');out=r/'smoke1665';out.mkdir(exist_ok=True)
uid='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46';lock=open('/tmp/lookcloser_gpu_'+uid+'.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=uid,PYTHONPATH=str(overlay/'deps')+':'+str(code),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2');write(out/'launch.json',dict(pid=os.getpid(),GPU_UUID=uid,started=time.time(),no_field_training=True))
for stage,command in [('sources',[sys.executable,str(code/'marshawn_video_prep_sources.py'),'--frames','1665','--cameras','HC','DC','KA','--output',str(r/'sources'),'--raw-native','--workers','2']),('masks',[sys.executable,str(code/'marshawn_video_prep_masks.py'),'--root',str(r),'--frame','1665','--cameras','HC','DC','KA'])]:
 with (out/(stage+'.log')).open('a') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 write(out/(stage+'_done.json'),dict(command=command,finished=time.time()))
write(out/'complete.json',dict(no_training=True,finished=time.time()))
