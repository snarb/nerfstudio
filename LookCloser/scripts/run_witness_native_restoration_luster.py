"""Native TRAIN witness restoration evidence; sequential GPU evaluation."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from witness_native_restoration_rank_luster_hybrid import combine
from cubic75_once_luster_hybrid import combine as cubic
from fixed_sampling_seam_luster_hybrid import combine as seam

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--wait-pids',nargs='+',type=int,default=[])
p.add_argument('--cache-wait-pids',nargs='+',type=int,default=[])
a=p.parse_args();root=a.root;deadline=read(root/'preflight/manifest.json')['deadline']
def wait(pids):
    while any(Path(f'/proc/{pid}').exists() for pid in pids):
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        time.sleep(3)
# The preceding cache producer may touch the same TRAIN-to-TRAIN warp folders.
wait(a.cache_wait_pids)
names=[]
for suffix,weight,negative_only in [('negative100',1.,True),('signed50',.5,False)]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_witness_native_'+suffix
    config={**read(root/'candidates/single_source_contradiction400/config.json'),
            'witness_native_weight':weight,'witness_native_negative_only':negative_only}
    for target in [7,12,95,150]:
        if not (root/f'candidates/{name}/cam{target:03d}_receipt.json').exists():combine(root,target,name,config)
    cubic(root,name,name+'_cubic75')
    final=name+'_seam';seam(root,name+'_cubic75',final,dict(filter_support='geometry',seam_fade=3.,adaptive_seam=True,seam_tolerance=.02));names.append(final)
wait(a.wait_pids)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result);write(root/'evaluation/witness_native_restoration_screen.json',results);print(result,flush=True)
