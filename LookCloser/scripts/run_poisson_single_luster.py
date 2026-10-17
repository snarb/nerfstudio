"""CPU-only fixed-label merge ablations followed by serialized GPU evaluation."""
import argparse,sys,time,subprocess
from pathlib import Path
from luster_hybrid_common import read,write
from poisson_single_luster_hybrid import combine
from view_illumination_luster_hybrid import combine as illuminate
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root;scripts=root/'code/LookCloser/scripts';end=read(root/'preflight/manifest.json')['deadline'];names=[]
for anchor in [.02,.05]:
 if time.time()>end:raise TimeoutError('Campaign deadline')
 name=f'single_cubic_poisson{int(anchor*100)}';combine(root,'single_cubic_fixed',name,anchor);names.append(name)
name='single_cubic75_view_detail';illuminate(root,'single_cubic75_fixed',name,'detail',6.);names.append(name)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
 if time.time()>end:raise TimeoutError('Campaign deadline')
 time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
 if time.time()>end:raise TimeoutError('Campaign deadline')
 with (root/f'logs/{name}_evaluation.log').open('w') as f:
  subprocess.run([sys.executable,str(scripts/'evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=f,stderr=subprocess.STDOUT,check=True)
 result=summarize(root,name);results.append(result);write(root/'evaluation/poisson_view75_screen.json',results);print(result,flush=True)
with (root/'logs/cubic75_spatial.log').open('w') as f:
 subprocess.run([sys.executable,str(scripts/'diagnose_luster_hybrid.py'),'--root',str(root),'--name','single_cubic75_fixed','--device','cuda'],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(scripts/'review_regression_luster_hybrid.py'),'--root',str(root),'--name','single_cubic75_fixed'],check=True)
