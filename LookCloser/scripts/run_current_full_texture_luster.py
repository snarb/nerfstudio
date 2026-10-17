"""Current source labels: partial broad appearance and literal full texture."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from luster_hybrid_common import read, write
from current_full_texture_luster_hybrid import combine

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--wait-pids', nargs='+', type=int, default=[])
a = p.parse_args()
root = a.root
deadline = read(root / 'preflight/manifest.json')['deadline']
names = []
for gain in [.25, 1.]:
    if time.time() > deadline:
        raise TimeoutError('Campaign deadline')
    name = f'single_current_full_texture{round(gain * 100):03d}'
    combine(root, 'single_photometric_depth25_seam', name, gain)
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
    write(root / 'evaluation/current_full_texture_screen.json', results)
    print(result, flush=True)
