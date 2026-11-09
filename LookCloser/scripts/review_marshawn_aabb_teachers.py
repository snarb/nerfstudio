"""Read-only selected-checkpoint AABB partitions and source-depth diagnostics."""
import argparse,json,hashlib,subprocess,os
from pathlib import Path

def main(a):
 env=dict(os.environ,PYTHONPATH='/home/nvuser/lookcloser/bootstrap_tree/lc16_deps:/home/nvuser/lookcloser/code')
 for f in a.frames:
  frame=a.campaign/str(f);record=json.loads((frame/'complete.json').read_text());ck=Path(record['checkpoint']);sha=hashlib.sha256(ck.read_bytes()).hexdigest()
  for cam in ['DC','HC','KA']:
   out=frame/(a.box.stem+'_'+cam)
   if not (out/'receipt.json').exists():
    with (frame/(a.box.stem+'_'+cam+'.log')).open('w')as log:
     subprocess.run([a.python,str(a.helpers/'with_cached_nerfacc.py'),str(a.helpers/'render_aabb_review.py'),'--config',record['config'],'--checkpoint',str(ck),'--checkpoint-sha256',sha,'--box',str(a.box),'--camera',cam,'--chunk','4096','--output',str(out)],stdout=log,stderr=log,env=env,check=True)
   source=a.sources/str(f)/cam
   with (frame/(a.box.stem+'_'+cam+'_source.log')).open('w')as log:
    subprocess.run([a.python,str(a.helpers/'render_source_depth_box_review.py'),'--partition',str(out),'--source',str(source/'source.png'),'--mask',str(source/'production_mask.png'),'--camera',cam,'--output',str(out/'source_depth')],stdout=log,stderr=log,check=True)
   print(f,cam,json.loads((out/'source_depth/diagnostics.json').read_text()),flush=True)
  (frame/(a.box.stem+'_review_complete.json')).write_text(json.dumps(dict(frame=f,checkpoint=str(ck),checkpoint_sha256=sha,box_sha256=hashlib.sha256(a.box.read_bytes()).hexdigest(),cameras=['DC','HC','KA']),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['campaign','helpers','box','sources']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--frames',type=int,nargs='+',required=True);p.add_argument('--python',default='/home/nvuser/lookcloser/env/bin/python');main(p.parse_args())
