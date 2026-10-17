"""Sequential frozen-depth statistic probe, CPU projection and CUDA evaluation."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write,log
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root;deadline=read(root/'preflight/manifest.json')['deadline']
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
 if time.time()>deadline:raise TimeoutError('Campaign deadline')
 time.sleep(2)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU already in use'
# Evaluation-only spatial diagnostics run in a short-lived child before model inference.
with (root/'logs/view_detail_spatial.log').open('w') as f:
 subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/diagnose_luster_hybrid.py'),'--root',str(root),'--name','single_cubic_view_detail','--device','cuda'],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/review_regression_luster_hybrid.py'),'--root',str(root),'--name','single_cubic_view_detail'],check=True)
sources=read(root/'preflight/sources.json')['sources'];cameras=sorted(set([7,12,95,150]+[s for v in sources.values() for s in v[:4]]))
env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env['CUDA_HOME']='/home/pavlo/.conda/envs/nerfstudio';env['PATH']='/home/pavlo/.conda/envs/nerfstudio/bin:'+env['PATH']
with (root/'logs/density_probe_render.log').open('w') as f:
 subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/density_probe_luster_hybrid.py'),'--root',str(root),'--cameras',*map(str,cameras)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
from density_warp_luster_hybrid import variant_root,warp
from density_single_luster_hybrid import DEFAULT,combine
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
results=[]
for kind in ['density_peak','density_center']:
 if time.time()>deadline:raise TimeoutError('Campaign deadline')
 variant=variant_root(root,kind);name='single_'+kind
 for c in [7,12,95,150]:
  for source in sources[str(c)][:4]:warp(variant,c,source)
  combine(variant,c,name,read(root/'candidates/single_near4_full_s8/config.json'))
 compute(root,name,[7,12,95,150]);result=summarize(root,name);results.append(result);write(root/'evaluation/single_density_screen.json',results);log(root,'density_single_evaluated',**result)
