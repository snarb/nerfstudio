"""Retain selected, raw LPIPS minimum and terminal without changing training policy."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def choices(complete,rows):
 raw=min(rows,key=lambda r:(r['metrics']['lpips'],-r['metrics']['psnr'],r['step']))
 return {'established_selector':int(complete['selection']['step']),'raw_lpips_min':int(raw['step']),'terminal':int(complete['final_step'])}
def main(branch,out):
 out.mkdir(parents=True,exist_ok=True);run=branch/'run'
 while not(run/'complete.json').exists():
  (out/'state.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),phase='waiting_natural_completion'))+'\n');time.sleep(60)
 complete=json.loads((run/'complete.json').read_text());evaluations=json.loads((run/'metrics.json').read_text());selected=choices(complete,evaluations);records={}
 for label,step in selected.items():
  folder=out/label;folder.mkdir(exist_ok=True);cp=run/'checkpoints'/f'step-{step:09d}.ckpt';grid=run/'occupancy'/f'step-{step:09d}_grid.json';meta=json.loads(grid.read_text())
  if sha(cp)!=meta['checkpoint_sha256']:raise ValueError('Final retained checkpoint does not match actual grid')
  target=folder/'checkpoint.ckpt'
  if not target.exists():os.link(cp,target)
  if sha(target)!=meta['checkpoint_sha256']:raise ValueError('Protected final checkpoint changed')
  for p in [grid,grid.with_suffix('.npz'),run/'worker/lookcloser/frozen/config.yml',branch/'spec.json',branch/'source_manifest.json']:shutil.copy2(p,folder/p.name)
  shutil.copytree(run/'renders'/f'step-{step:09d}',folder/'renders',dirs_exist_ok=True)
  row=next(r for r in evaluations if r['step']==step);records[label]=dict(step=step,metrics={k:row['metrics'][k]for k in ['psnr','ssim','lpips']},checkpoint_sha256=meta['checkpoint_sha256'],grid_sha256=meta['grid_archive_sha256'],path=str(folder))
 result=dict(complete=True,choices=records,selector_and_plateau_unchanged=True,raw_min_ties='highestPSNR then earlieststep only for exact LPIPS ties',source_complete_sha256=sha(run/'complete.json'),source_evaluations_sha256=sha(run/'metrics.json'),helper_sha256=sha(__file__))
 (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.branch,a.output)
