"""Native-pixel appearance seams, without the prior 4x4 label quantization.

The existing zero-transition control is rechecked exactly before testing
stronger appearance-dependent spatial agreement at native pixel resolution.
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
# Recheck the already measured, exact native-pixel zero control against
# the unchanged implementation before trying stronger coherence weights.
base = read(root/'candidates/single_source_contradiction400/config.json')
module = root/'code/LookCloser/scripts/appearance_graph_rank_luster_hybrid.py'
for target in [7,12,95,150]:
    receipt = read(root/'candidates/single_native_graph0'/f'cam{target:03d}_receipt.json')
    assert receipt['code_sha256'] == sha(module)
    assert receipt['config'] == {**base,'grid_step':1,'transition_cost':0.,'fallback_cost':1e4}
    for tail in ['', '_sources', '_confidence']:
        actual=root/'candidates/single_native_graph0_seam'/f'cam{target:03d}{tail}.npy'
        expected=root/'candidates/single_source_contradiction400_once_seam'/f'cam{target:03d}{tail}.npy'
        assert np.array_equal(np.load(actual,mmap_mode='r'),np.load(expected,mmap_mode='r'))
print('existing_native_zero_control_rechecked',flush=True)
names = []
for suffix, weight in [('40', .4), ('160', 1.6)]:
    name = 'single_native_graph_strong' + suffix
    config = {**read(root / 'candidates/single_source_contradiction400/config.json'),
              'grid_step': 1, 'transition_cost': weight, 'fallback_cost': 1e4}
    for target in [7, 12, 95, 150]:
        if time.time() > deadline:
            raise TimeoutError('Campaign deadline')
        if not (root / f'candidates/{name}/cam{target:03d}_receipt.json').exists():
            combine(root, target, name, config)
        print('native_graph_camera_complete', name, target, flush=True)
    cubic(root, name, name + '_cubic75')
    final = name + '_seam'
    seam(root, name + '_cubic75', final, dict(filter_support='geometry', seam_fade=3., adaptive_seam=True, seam_tolerance=.02))
    names.append(final)
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
    write(root / 'evaluation/native_graph_strong_screen.json', results)
    print(result, flush=True)
