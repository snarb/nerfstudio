"""Test positive TRAIN-witness admission of an extra G6-G16 detail band."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from broad_witness_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
for suffix,threshold in [('25',.25),('60',.60)]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_broad_witness'+suffix
    combine(root,'single_cubic_eligible_phase_rt2_seam',name,threshold);names.append(name)
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
    result=summarize(root,name);results.append(result);write(root/'evaluation/broad_witness_screen.json',results);print(result,flush=True)
