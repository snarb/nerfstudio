"""Evaluate automatic source-selection guards after the current GPU owner exits."""
from pathlib import Path
import time
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
from luster_hybrid_common import read,write,log
root=Path('/fsx/tmp/luster_000540_hybrid_20261017T104440Z');deadline=read(root/'preflight/manifest.json')['deadline']
while Path('/proc/667299').exists():
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(5)
results=[]
for name in ['single_consensus_30','single_consensus_60','single_consensus_85','single_graph_s8']:
    while not all((root/f'candidates/{name}/cam{cid:03d}_receipt.json').exists() for cid in [7,12,95,150]):
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        time.sleep(5)
    if not (root/f'evaluation/{name}_metrics.json').exists():compute(root,name,[7,12,95,150])
    result=summarize(root,name);results.append(result);write(root/'evaluation/single_source_guards.json',results);log(root,'single_guard_measured',**result)
