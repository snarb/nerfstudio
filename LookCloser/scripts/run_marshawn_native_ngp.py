#!/usr/bin/env python3
"""Native Instant-NGP bounded sanity at scene_scale32; no LookCloser model/pipeline."""
from pathlib import Path
import csv,math,os,sys
import run_bounded_ngp_quiet as quiet

def select(metrics, directory):
 rows=list(csv.DictReader(Path(metrics).open()));eligible=[]
 for r in rows:
  try:
   step=int(r['step']);values=[float(r[k]) for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']]
  except (KeyError,ValueError,TypeError):continue
  checkpoint=Path(directory)/f'step-{step:09d}.ckpt'
  if 0<=step<22000 and checkpoint.exists() and all(map(math.isfinite,values)):eligible.append((checkpoint,*values))
 if not eligible:raise RuntimeError('No saved checkpoint with all-view PSNR/SSIM/LPIPS')
 best=max(x[1] for x in eligible);winner=min((x for x in eligible if x[1]>=best-.07-1e-12),key=lambda x:(x[3],-x[1]))
 return winner[0],'all_eval_psnr_lpips_tie_0.07'

def main():
 original=quiet.train_command
 def command(args):
  cmd=original(args);cmd[0]=str(Path(sys.executable).with_name('ns-train'))
  cmd[cmd.index('--auto-scale-poses')+1]='False'
  return cmd
 quiet.train_command=command;quiet.best_eval_checkpoint=select
 os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
 return quiet.main()
if __name__=='__main__':raise SystemExit(main())
