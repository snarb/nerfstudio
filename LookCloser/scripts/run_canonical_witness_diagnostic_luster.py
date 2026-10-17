"""CPU-only matched appearance-normalization diagnostics; no production output."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from luster_hybrid_common import read,write

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
root=a.root;deadline=read(root/'preflight/manifest.json')['deadline'];results=[]
while Path(f'/proc/{a.wait_pid}').exists():
    assert time.time()<deadline
    time.sleep(3)
for target in [7,12,95,150]:
    assert time.time()<deadline
    sources=read(root/f'candidates/single_photometric_depth25_seam/cam{target:03d}_receipt.json')['sources'][:2]
    reference=root/'probes/actual_texture_witness_cpu' if target==7 else root/'probes/actual_texture_witness_expanded'/f'cam{target:03d}'
    assert read(reference/'summary.json')['metric_unchanged']
    out=root/'probes/canonical_actual_texture_witness'/f'cam{target:03d}'
    if not (out/'summary.json').exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('probe_canonical_actual_texture_witness_luster.py')),
            '--root',str(root),'--output',str(out),'--target',str(target),'--sources',*map(str,sources),'--device','cpu'],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('diagnose_matched_witness_pair_luster.py')),
        '--root',str(root),'--pilot',str(out),'--reference',str(reference)],check=True)
    results.append(read(out/'pair_ranking_diagnostic.json'))
    write(root/'evaluation/canonical_actual_texture_witness_diagnostic.json',dict(
        usage='evaluation_only_no_prediction_feedback',selection='first two existing automatic source-pool entries',rows=results))
