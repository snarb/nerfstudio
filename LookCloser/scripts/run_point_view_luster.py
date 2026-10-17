"""Evaluate frozen pointwise view correction after all prior GPU owners exit."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from point_view_merge_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline']
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
with (root/'logs/point_view_render.log').open('w') as stream:
    subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/point_view_luster_hybrid.py'),'--root',str(root)],stdout=stream,stderr=subprocess.STDOUT,check=True)
from screen_luster_hybrid import summarize
results=[]
for gain in [.5,1.]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_point_view'+str(int(gain*100))
    combine(root,'single_cubic75_fixed',name,gain)
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result);write(root/'evaluation/point_view_screen.json',results);print(result,flush=True)
