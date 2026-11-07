from pathlib import Path
import os,sys,json,fcntl,subprocess,time
from prepare_marshawn_tuning import write
r=Path(sys.argv[1]);code=Path(__file__).resolve().parent;overlay=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay');uid='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46'
lock=open('/tmp/lookcloser_gpu_'+uid+'.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=uid,PYTHONPATH=str(overlay/'deps')+':'+str(code),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
write(r/'tail_native/extra_launch.json',dict(pid=os.getpid(),GPU_UUID=uid,started=time.time(),reason='Three-camera headtruncation and zero greyanchors require6viewqualification',no_field_training=True))
cmd=[sys.executable,str(code/'segment_marshawn_temporal_samples.py'),'--frames','1960','2020','--cameras','FC','MC','NA','--sources',str(r/'tail_native/sources'),'--output',str(r/'tail_native/candidates'),'--checkpoint',str(overlay/'weights/sam3.pt')]
with (r/'tail_native/sam3_extra.log').open('a') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
write(r/'tail_native/extra_done.json',dict(finished=time.time()))
