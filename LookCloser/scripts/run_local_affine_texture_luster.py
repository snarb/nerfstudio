"""Matched broad additive/affine photo color fit; serialized canonical evaluation."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from luster_hybrid_common import read, write
from local_affine_texture_luster_hybrid import combine
import numpy as np
from luster_hybrid_common import sha

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--wait-pids', nargs='+', type=int, default=[])
a = p.parse_args()
root = a.root
deadline = read(root / 'preflight/manifest.json')['deadline']
names = []
for suffix, sigma, affine in [('control',0.,False),('additive32',32.,False),('affine32',32.,True)]:
    if time.time() > deadline:
        raise TimeoutError('Campaign deadline')
    name = 'single_local_color_' + suffix
    combine(root, 'single_photometric_depth25_seam', name, sigma, affine)
    if suffix == 'control':
        evidence = {}
        for c in [7, 12, 95, 150]:
            evidence[str(c)] = {}
            for tail in ['', '_sources', '_confidence']:
                left = root/'candidates/single_photometric_depth25_seam'/f'cam{c:03d}{tail}.npy'
                right = root/'candidates'/name/f'cam{c:03d}{tail}.npy'
                assert np.array_equal(np.load(left,mmap_mode='r'),np.load(right,mmap_mode='r'))
                evidence[str(c)][tail or 'rgb'] = dict(exact=True,reference_sha256=sha(left),control_sha256=sha(right))
        write(root/'evaluation/local_color_zero_equivalence.json', evidence)
    else:
        names.append(name)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time() > deadline:
        raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU in use'
from screen_luster_hybrid import summarize
results = []
for name in names:
    if time.time() > deadline:
        raise TimeoutError('Campaign deadline')
    with (root / f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable, str(root / 'code/LookCloser/scripts/evaluate_luster_hybrid.py'),
                        '--root', str(root), '--candidate', name, '--device', 'cuda'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = summarize(root, name)
    results.append(result)
    write(root / 'evaluation/local_color_screen.json', results)
    print(result, flush=True)
