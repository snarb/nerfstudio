"""CPU guarded-gradient and witness/cubic composition, then serial evaluation."""
import argparse,sys,time,subprocess
from pathlib import Path
from luster_hybrid_common import read,write
from gradient_guard_luster_hybrid import combine
from cubic75_fixed_luster_hybrid import combine as cubic
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root;scripts=root/'code/LookCloser/scripts';end=read(root/'preflight/manifest.json')['deadline']
combine(root,'single_cubic75_fixed','single_cubic75_gradient_guard',.02)
cubic(root,'single_witness75','single_witness75_cubic75')
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
 if time.time()>end:raise TimeoutError('Campaign deadline')
 time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in ['single_cubic75_gradient_guard','single_witness75_cubic75']:
 if time.time()>end:raise TimeoutError('Campaign deadline')
 with (root/f'logs/{name}_evaluation.log').open('w') as f:
  subprocess.run([sys.executable,str(scripts/'evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=f,stderr=subprocess.STDOUT,check=True)
 r=summarize(root,name);results.append(r);write(root/'evaluation/gradient_witness75_screen.json',results);print(r,flush=True)
