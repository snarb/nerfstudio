#!/usr/bin/env python3
"""Shared train-only frequency queue, one persistent advisory lock per image."""
import argparse,fcntl,json,os,socket,time
from pathlib import Path
import torch
from nerfstudio.scripts.lookcloser_preprocess import LookCloserPreprocessConfig,load_image_as_tensor

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
 os.umask(0);torch.set_num_threads(2);m=json.loads((a.data/'transforms.json').read_text());out=a.data/'lookcloser_frequencies';out.mkdir(exist_ok=True)
 config=LookCloserPreprocessConfig(max_res=8192,train_steps_per_level=1000,ssim_threshold=.95,eval_patch_batch_size=1024)
 for rel in m['train_filenames']:
  source=a.data/rel;dst=out/(source.stem+'.pt');lock=out/(source.stem+'.lock')
  with os.fdopen(os.open(lock,os.O_RDONLY|os.O_CREAT,0o666),'rb') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if dst.exists() and dst.with_suffix('.json').exists():continue
   start=time.time();torch.manual_seed(42)
   print('start',source.stem,socket.gethostname(),os.getpid(),flush=True)
   image=load_image_as_tensor(source,device=torch.device('cuda'));tmp=out/(source.stem+f'.{os.getpid()}.tmp.pt')
   config._process_single_image(img_path=source,image_tensor=image,save_freq_path=tmp,debug_out_dir=None,scene_size=2.)
   metadata=json.loads(tmp.with_suffix('.json').read_text());metadata.update(ssim_threshold=config.ssim_threshold,train_steps_per_level=config.train_steps_per_level,eval_patch_batch_size=config.eval_patch_batch_size,fit_script_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),source_image_sha256=__import__('hashlib').sha256(source.read_bytes()).hexdigest(),host=socket.gethostname());tmp.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n');tmp.with_suffix('.json').replace(dst.with_suffix('.json'));tmp.replace(dst)
   del image;torch.cuda.empty_cache();print('done',source.stem,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
