"""Serialize frozen TRAIN LPIPS confidence, hard source ranking and evaluation."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from luster_hybrid_common import read, write, sha
from patch_eligibility_rank_luster_hybrid import combine
from cubic75_once_luster_hybrid import combine as cubic
from fixed_sampling_seam_luster_hybrid import combine as seam

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--wait-pids',nargs='+',type=int,default=[])
a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline']
parent='single_photometric_depth25_seam'
base=read(root/'candidates/single_photometric_depth25/config.json')
cases=[('control',True,.06),('no_cleanup',False,.06),('match10',True,.10)];names=[]
for suffix,cleanup,limit in cases:
    name='single_patch_eligibility_'+suffix
    config={**base,'remove_source_islands':cleanup,'match_admission_limit':limit}
    for target in [7,12,95,150]:
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        if not (root/'candidates'/name/f'cam{target:03d}_receipt.json').exists():combine(root,target,name,config)
    cubic(root,name,name+'_cubic75')
    final=name+'_seam'
    seam(root,name+'_cubic75',final,dict(filter_support='geometry',seam_fade=3.,adaptive_seam=True,seam_tolerance=.02))
    names.append(final)
    if suffix=='control':
        evidence={}
        for target in [7,12,95,150]:
            evidence[str(target)]={}
            for tail in ['', '_sources', '_confidence']:
                left=root/'candidates'/parent/f'cam{target:03d}{tail}.npy'
                right=root/'candidates'/final/f'cam{target:03d}{tail}.npy'
                assert np.array_equal(np.load(left,mmap_mode='r'),np.load(right,mmap_mode='r'))
                evidence[str(target)][tail or 'rgb']={'reference_sha256':sha(left),'control_sha256':sha(right),'exact':True}
        write(root/'evaluation/patch_eligibility_zero_equivalence.json',evidence)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names[1:]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),
            '--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result)
    write(root/'evaluation/patch_eligibility_screen.json',results)
    print(result,flush=True)
