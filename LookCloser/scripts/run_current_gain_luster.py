"""Bracket conservative transfer strength with current contradiction-ranked sources."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from fixed_sampling_seam_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
for gain in [.85,.95]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name=f'single_current_gain{round(gain*100)}_seam'
    combine(root,'single_source_contradiction400_cubic75',name,dict(gain=gain,filter_support='geometry',seam_fade=3.,adaptive_seam=True,seam_tolerance=.02));names.append(name)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result);write(root/'evaluation/current_gain_screen.json',results);print(result,flush=True)
