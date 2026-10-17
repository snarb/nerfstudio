"""Isolate the plate veto with the reference second median-label pass preserved."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from cubic75_fixed_luster_hybrid import combine as cubic
from exact_plate_merge_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
for suffix in ['10','20']:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    parent='single_exact_plate'+suffix
    while not all((root/f'candidates/{parent}/cam{c:03d}_receipt.json').exists() for c in [7,12,95,150]):
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        time.sleep(3)
    cubic(root,parent,parent+'_matched_cubic75')
    name=parent+'_matched_seam'
    combine(root,parent+'_matched_cubic75',name);names.append(name)
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
    result=summarize(root,name);results.append(result);write(root/'evaluation/exact_plate_matched_screen.json',results);print(result,flush=True)
