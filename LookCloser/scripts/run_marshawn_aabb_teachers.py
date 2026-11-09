"""Short full-scene geometry teachers; no candidate AABB enters training.

Runs sequentially on one host. Writes process/GPU/progress evidence each minute,
keeps all evaluation checkpoints and selects eval-all PSNR/LPIPS after each job.
"""
import argparse,json,os,subprocess,time,datetime
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def write(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)

def main(a):
 a.output.mkdir(parents=True,exist_ok=True);floor=json.loads(a.floor.read_text());T=np.array(floor['world_to_box'])
 env=dict(os.environ,PYTHONPATH='/home/nvuser/lookcloser/bootstrap_tree/lc16_deps:/home/nvuser/lookcloser/code')
 for frame in a.frames:
  dest=a.output/str(frame);dest.mkdir(exist_ok=True)
  if (dest/'complete.json').exists():continue
  data=dest/'data';data.mkdir(exist_ok=True);rows=[]
  for p in sorted((a.sources/str(frame)).iterdir()):
   c=json.loads((p/'receipt.json').read_text())['camera_row'];pose=T@np.array(c['transform_matrix']);pose[:3,3]=(pose[:3,3]-[0,2,1])/5
   im=data/(p.name+'.png')
   if not im.exists():im.symlink_to((p/'source.png').resolve())
   row={k:c[k] for k in ['w','h','fl_x','fl_y','cx','cy']};row.update(file_path=im.name,transform_matrix=pose.tolist());rows.append(row)
  if len(rows)!=72:raise ValueError(f'{frame}: expected72 cameras, got{len(rows)}')
  held=['DC.png','HC.png','KA.png'];write(data/'transforms.json',dict(camera_model='OPENCV',frames=rows,train_filenames=[r['file_path']for r in rows if r['file_path']not in held],val_filenames=held,test_filenames=held))
  cmd=[a.python,str(a.launcher),'instant-ngp-bounded','--vis','tensorboard','--experiment-name',f'aabb_teacher_{frame}','--output-dir',str(dest/'runs'),'--max-num-iterations','6001','--steps-per-save','2000','--save-only-latest-checkpoint','False','--steps-per-eval-batch','500','--steps-per-eval-image','6000','--steps-per-eval-all-images','2000','--logging.local-writer.enable','False','--pipeline.model.log2-hashmap-size','21','--pipeline.model.render-step-size','0.001','nerfstudio-data','--data',str(data),'--eval-mode','filename','--orientation-method','none','--center-method','none','--auto-scale-poses','False','--downscale-factor','1']
  write(dest/'command.json',dict(command=cmd,technical_only=True,world_to_floor=T.tolist(),teacher_center=[0,2,1],teacher_scale=.2,candidate_box_used=False))
  with (dest/'train.log').open('w')as log:
   worker=subprocess.Popen(cmd,stdout=log,stderr=log,env=env);start=time.time();last=0
   while worker.poll() is None:
    if (a.output/'STOP').exists():worker.terminate();worker.wait();raise RuntimeError('User stop marker')
    if time.time()-last>=60:
     last=time.time();progress={};events=sorted((dest/'runs').rglob('events.out*'))
     if events:
      ev=EventAccumulator(str(events[-1]));ev.Reload()
      for tag in ev.Tags()['scalars']:
       if 'psnr'in tag.lower():z=ev.Scalars(tag)[-1];progress[tag]=dict(step=z.step,value=z.value)
     state=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frame=frame,controller_pid=os.getpid(),worker_pid=worker.pid,worker_alive=worker.poll()is None,elapsed_seconds=time.time()-start,progress=progress,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),oom='out of memory'in (dest/'train.log').read_text(errors='replace').lower())
     write(a.output/'status.json',state);(a.output/'supervision.jsonl').open('a').write(json.dumps(state)+'\n');print(frame,progress,flush=True)
    time.sleep(5)
  if worker.returncode:raise RuntimeError(f'{frame}: worker exit{worker.returncode}; see train.log')
  event=sorted((dest/'runs').rglob('events.out*'))[-1];ev=EventAccumulator(str(event));ev.Reload();scores={}
  for metric in ['psnr','ssim','lpips']:
   tag='Eval Images Metrics Dict (all images)/'+metric
   for v in ev.Scalars(tag):scores.setdefault(v.step,{})[metric]=v.value
  peak=max(v['psnr']for v in scores.values());step=min((s for s,v in scores.items()if v['psnr']>=peak-.07),key=lambda s:scores[s]['lpips']);ck=event.parent/'nerfstudio_models'/f'step-{step:09d}.ckpt'
  if not ck.exists():raise FileNotFoundError(ck)
  write(dest/'complete.json',dict(frame=frame,config=str(event.parent/'config.yml'),checkpoint=str(ck),step=step,scores=scores,selection='highest eval_all_psnr, LPIPS tie-break within0.07dB',elapsed_seconds=time.time()-start,technical_only=True));print(frame,'DONE',step,flush=True)
 write(a.output/'DONE.json',dict(frames=a.frames,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--floor',type=Path,required=True);p.add_argument('--launcher',type=Path,required=True);p.add_argument('--python',default='/home/nvuser/lookcloser/env/bin/python');p.add_argument('--frames',type=int,nargs='+',required=True);main(p.parse_args())
