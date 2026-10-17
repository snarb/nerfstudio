"""Controlled high-frequency residual extrapolation with fixed photo ownership.

This is an experimental contrast control, not additional measured detail.
No geometry, coordinates, source labels or visibility threshold changes.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from luster_hybrid_common import read,write,sha
from fixed_sampling_seam_luster_hybrid import combine

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline']
parent='single_photometric_depth25_cubic75'
reference='single_photometric_depth25_seam';names=[]
for gain in [1.,1.1,1.3,1.6]:
    assert time.time()<deadline
    name=f'single_high_gain{round(gain*100)}_seam'
    # Compare actual scaled corrections at seams, keeping the final tolerance
    # unchanged. Otherwise a higher gain would silently weaken seam admission.
    combine(root,parent,name,dict(gain=gain,filter_support='geometry',seam_fade=3.,
        adaptive_seam=True,seam_tolerance=.02/gain))
    if gain==1:
        evidence={}
        for target in [7,12,95,150]:
            evidence[str(target)]={}
            for suffix in ['', '_sources', '_confidence']:
                left=root/'candidates'/reference/f'cam{target:03d}{suffix}.npy'
                right=root/'candidates'/name/f'cam{target:03d}{suffix}.npy'
                assert np.array_equal(np.load(left,mmap_mode='r'),np.load(right,mmap_mode='r'))
                evidence[str(target)][suffix or 'rgb']=dict(exact=True,reference_sha256=sha(left),control_sha256=sha(right))
        write(root/'evaluation/high_gain_zero_equivalence.json',evidence)
    else:
        names.append(name)
        output=root/'evaluation/high_gain_reference_tiles'/name;output.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name('compare_reference_tiles_luster_hybrid.py')),
            '--root',str(root),'--reference',reference,'--names',reference,name,
            '--output',str(output),'--count','3'],check=True)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    assert time.time()<deadline;time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
    assert time.time()<deadline
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_luster_hybrid.py')),
            '--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result)
    write(root/'evaluation/high_gain_screen.json',results);print(result,flush=True)
