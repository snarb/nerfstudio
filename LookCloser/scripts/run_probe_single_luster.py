"""Sequential frozen-depth statistic probe, CPU projection and CUDA evaluation."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write,log
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args();root=a.root;deadline=read(root/'preflight/manifest.json')['deadline']
if a.wait_pid:
 while Path(f'/proc/{a.wait_pid}').exists():
  if time.time()>deadline:raise TimeoutError('Campaign deadline')
  time.sleep(2)
sources=read(root/'preflight/sources.json')['sources'];cameras=sorted(set([7,12,95,150]+[s for v in sources.values() for s in v[:4]]))
env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_HOME']='/home/pavlo/.conda/envs/nerfstudio';env['PATH']='/home/pavlo/.conda/envs/nerfstudio/bin:'+env['PATH']
with (root/'logs/surface_probe_render.log').open('w') as f:
 subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/probe_depth_luster_hybrid.py'),'--root',str(root),'--cameras',*map(str,cameras)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
from probe_warp_luster_hybrid import variant_root,warp
from probe_single_luster_hybrid import DEFAULT,combine
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
results=[]
for kind in ['q25','q75','mode']:
 if time.time()>deadline:raise TimeoutError('Campaign deadline')
 variant=variant_root(root,kind);name='single_probe_'+kind
 for c in [7,12,95,150]:
  for source in sources[str(c)][:4]:warp(variant,c,source)
  combine(variant,c,name,{**DEFAULT,'count':4,'match_weight':.25,'match_limit':.06,'gain':1.})
 compute(root,name,[7,12,95,150]);result=summarize(root,name);results.append(result);write(root/'evaluation/single_probe_screen.json',results);log(root,'probe_single_evaluated',**result)
