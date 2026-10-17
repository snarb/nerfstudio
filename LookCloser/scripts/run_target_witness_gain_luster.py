"""Read completed TRAIN confidence warps, test bounded analytical gain."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from luster_hybrid_common import read, write
from target_witness_gain_luster_hybrid import projected_gain
from target_witness_gain_merge_luster_hybrid import combine

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--cache-pid',type=int,required=True)
p.add_argument('--wait-pids',nargs='+',type=int,default=[])
a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline']
parent='single_photometric_depth25_seam'
config=read(root/'candidates'/parent/'config.json')
cases=[('single_target_witness_gain50',.5),('single_target_witness_gain100',1.)]
for target in [7,12,95,150]:
    receipt=read(root/'candidates'/parent/f'cam{target:03d}_receipt.json')
    for source in receipt['sources']:
        while not (root/'witness_native_restoration'/f'cam{source:03d}'/'receipt.json').exists():
            if time.time()>deadline:raise TimeoutError('Campaign deadline')
            if not Path(f'/proc/{a.cache_pid}').exists():
                raise RuntimeError('Required TRAIN cache did not complete')
            time.sleep(3)
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        projected_gain(root,target,source,config)
    for name,strength in cases:
        combine(root,parent,name,target,strength)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name,_ in cases:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),
            '--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result)
    write(root/'evaluation/target_witness_gain_screen.json',results)
    print(result,flush=True)
