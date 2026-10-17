"""Isolate geometric source-ranking weights with the current fixed merge recipe."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from witness_benefit_select_luster_hybrid import combine
from cubic75_fixed_luster_hybrid import combine as cubic
from fixed_sampling_seam_luster_hybrid import combine as seam

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
cases=[('50',{'witness_phase_weight':.5}),('150',{'witness_phase_weight':1.5})]
for suffix,change in cases:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_witness_benefit'+suffix;config={**read(root/'candidates/single_risk_patch/config.json'),**change}
    for target in [7,12,95,150]:
        if not (root/f'candidates/{name}/cam{target:03d}_receipt.json').exists():combine(root,target,name,config)
    cubic(root,name,name+'_cubic75')
    final=name+'_seam';seam(root,name+'_cubic75',final,dict(filter_support='geometry',seam_fade=3.,adaptive_seam=True,seam_tolerance=.02));names.append(final)
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
    result=summarize(root,name);results.append(result);write(root/'evaluation/witness_benefit_screen.json',results);print(result,flush=True)
