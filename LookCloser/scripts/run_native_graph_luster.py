"""Native-pixel appearance seams, without the prior 4x4 label quantization.

The zero-transition control must exactly reproduce the existing source labels
and final image before the nonzero-transition experiment is constructed.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from luster_hybrid_common import read, sha, write
from appearance_graph_rank_luster_hybrid import combine
from cubic75_once_luster_hybrid import combine as cubic
from fixed_sampling_seam_luster_hybrid import combine as seam

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--wait-pids', nargs='+', type=int, default=[])
a = p.parse_args()
root = a.root
deadline = read(root / 'preflight/manifest.json')['deadline']
names = []
for suffix, weight in [('0', 0.), ('5', .05)]:
    name = 'single_native_graph' + suffix
    config = {**read(root / 'candidates/single_source_contradiction400/config.json'),
              'grid_step': 1, 'transition_cost': weight, 'fallback_cost': 1e4}
    for target in [7, 12, 95, 150]:
        if time.time() > deadline:
            raise TimeoutError('Campaign deadline')
        if not (root / f'candidates/{name}/cam{target:03d}_receipt.json').exists():
            combine(root, target, name, config)
        if weight == 0:
            actual = root / f'candidates/{name}/cam{target:03d}_sources.npy'
            expected = root / f'candidates/single_source_contradiction400/cam{target:03d}_sources.npy'
            assert np.array_equal(np.load(actual), np.load(expected)), 'Native graph zero control changed source labels'
        print('native_graph_camera_complete', name, target, flush=True)
    cubic(root, name, name + '_cubic75')
    final = name + '_seam'
    seam(root, name + '_cubic75', final, dict(filter_support='geometry', seam_fade=3., adaptive_seam=True, seam_tolerance=.02))
    names.append(final)
    if weight == 0:
        rows = []
        for target in [7, 12, 95, 150]:
            actual = root / f'candidates/{final}/cam{target:03d}.npy'
            expected = root / f'candidates/single_source_contradiction400_once_seam/cam{target:03d}.npy'
            assert np.array_equal(np.load(actual), np.load(expected)), 'Native graph zero control changed final RGB'
            rows.append(dict(camera=target, actual=str(actual), actual_sha256=sha(actual), reference=str(expected), reference_sha256=sha(expected)))
        write(root / 'evaluation/native_graph_zero_equivalence.json', dict(exact_match=True, rows=rows, code_sha256=sha(Path(__file__))))
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
        subprocess.run([sys.executable, str(root / 'code/LookCloser/scripts/evaluate_luster_hybrid.py'), '--root', str(root), '--candidate', name, '--device', 'cuda'], stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = summarize(root, name)
    results.append(result)
    write(root / 'evaluation/native_graph_screen.json', results)
    print(result, flush=True)
