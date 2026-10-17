"""Sequential CPU diagnostics of the first two automatic sources per view."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from luster_hybrid_common import read,write

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--cameras',nargs='+',type=int,default=[12,95,150]);a=p.parse_args()
root=a.root;deadline=read(root/'preflight/manifest.json')['deadline'];results=[]
for target in a.cameras:
    assert time.time()<deadline
    sources=read(root/f'candidates/single_photometric_depth25_seam/cam{target:03d}_receipt.json')['sources'][:2]
    out=root/'probes/actual_texture_witness_expanded'/f'cam{target:03d}'
    if not (out/'summary.json').exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('probe_actual_texture_witness_luster.py')),
            '--root',str(root),'--output',str(out),'--target',str(target),'--sources',*map(str,sources),'--device','cpu'],check=True)
    while not all((root/'target_witness_perceptual_projected'/f'cam{target:03d}'/f'src{s:03d}'/'receipt.json').exists() for s in sources):
        assert time.time()<deadline
        time.sleep(3)
    subprocess.run([sys.executable,str(Path(__file__).with_name('diagnose_actual_texture_witness_luster.py')),
        '--root',str(root),'--pilot',str(out)],check=True)
    results.append(read(out/'pair_ranking_diagnostic.json'))
    write(root/'evaluation/actual_texture_witness_expanded_diagnostic.json',dict(
        usage='evaluation_only_no_prediction_feedback',selection='first two existing automatic source-pool entries',rows=results))
