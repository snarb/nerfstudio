"""Durable, source-bound tail qualification; no field training or learned input."""
from pathlib import Path
import subprocess,os,json,sys,fcntl,time
from prepare_marshawn_tuning import sha,write

def main():
 root=Path(sys.argv[1]);code=Path(__file__).resolve().parent;python=sys.executable
 overlay=Path('/fsx/tmp/marshawn_001660_hd/two_actor_isolation/local_sam3_overlay');checkpoint=overlay/'weights/sam3.pt'
 assert sha(checkpoint)=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
 uid='GPU-7ed25c4d-10d1-cbe3-d0b2-76c21211bd46';lock=open('/tmp/lookcloser_gpu_'+uid+'.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=uid,PYTHONPATH=str(overlay/'src')+':'+str(overlay/'deps')+':'+str(code),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 write(root/'launch.json',dict(pid=os.getpid(),GPU_UUID=uid,checkpoint_sha256=sha(checkpoint),script_sha256=sha(__file__),no_field_training=True,policy='IndependentSAM3personcandidates6nativeframes; manualfighteridentity+geometricqualificationpending',start=time.time()))
 commands=[('decode',[python,str(code/'marshawn_video_prep_sources.py'),'--frames','1960','2020','--cameras','HC','DC','KA','--output',str(root/'sources'),'--native-rois',str(root/'full_rois.json'),'--workers','2']),('sam3',[python,str(code/'segment_marshawn_temporal_samples.py'),'--frames','1960','2020','--cameras','HC','DC','KA','--sources',str(root/'sources'),'--output',str(root/'candidates'),'--checkpoint',str(checkpoint)])]
 for name,command in commands:
  with (root/(name+'.log')).open('a') as log:
   subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  write(root/(name+'_done.json'),dict(stage=name,command=command,finished=time.time()))
 write(root/'complete.json',dict(stage='CANDIDATES_PENDING_IDENTITY_GEOMETRY',finished=time.time()))
if __name__=='__main__':main()
