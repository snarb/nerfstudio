"""Test conservative measured-background veto with cubic source support."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from exact_plate_rank_luster_hybrid import combine
from exact_plate_merge_luster_hybrid import combine as seam

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
cases=[('10',{'plate_veto_level':'strict10'}),('20',{'plate_veto_level':'strict20'})]
for suffix,change in cases:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_exact_plate'+suffix;config={**read(root/'candidates/single_cubic_eligible_phase_rt2/config.json'),**change}
    for target in [7,12,95,150]:
        if not (root/f'candidates/{name}/cam{target:03d}_receipt.json').exists():combine(root,target,name,config)
    final=name+'_seam';seam(root,name,final);names.append(final)
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
    result=summarize(root,name);results.append(result);write(root/'evaluation/exact_plate_screen.json',results);print(result,flush=True)
