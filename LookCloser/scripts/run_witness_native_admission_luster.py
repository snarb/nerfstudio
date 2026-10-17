"""Wait for immutable TRAIN confidence, then test conservative fallback."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from luster_hybrid_common import read, write
from witness_native_admission_luster_hybrid import combine

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--cache-pid', type=int, required=True)
p.add_argument('--wait-pids', nargs='+', type=int, default=[])
a = p.parse_args()
root = a.root
deadline = read(root / 'preflight/manifest.json')['deadline']
parent = 'single_photometric_depth25_seam'
name = 'single_witness_native_admission4060'
for target in [7, 12, 95, 150]:
    receipt = read(root / 'candidates' / parent / f'cam{target:03d}_receipt.json')
    while not all((root / 'witness_native_restoration_projected' / f'cam{target:03d}' / f'src{s:03d}' / 'receipt.json').exists() for s in receipt['sources']):
        if time.time() > deadline:
            raise TimeoutError('Campaign deadline')
        if not Path(f'/proc/{a.cache_pid}').exists():
            raise RuntimeError('Confidence cache producer exited before completion')
        time.sleep(3)
    if time.time() > deadline:
        raise TimeoutError('Campaign deadline')
    combine(root, parent, name, target)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time() > deadline:
        raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU in use'
with (root / f'logs/{name}_evaluation.log').open('w') as stream:
    subprocess.run([sys.executable, str(root / 'code/LookCloser/scripts/evaluate_luster_hybrid.py'), '--root', str(root), '--candidate', name, '--device', 'cuda'], stdout=stream, stderr=subprocess.STDOUT, check=True)
from screen_luster_hybrid import summarize
result = summarize(root, name)
write(root / 'evaluation/witness_native_admission_screen.json', result)
print(result, flush=True)
