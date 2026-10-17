"""Sequential CUDA evaluation for completed single-source candidates."""
import argparse,time
from pathlib import Path
from luster_hybrid_common import read,write,log
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int);p.add_argument('--names',nargs='+',required=True);p.add_argument('--tag',required=True);a=p.parse_args();deadline=read(a.root/'preflight/manifest.json')['deadline']
def wait_until(predicate):
    while not predicate():
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        time.sleep(2)
if a.wait_pid:wait_until(lambda:not Path(f'/proc/{a.wait_pid}').exists())
results=[]
for name in a.names:
    wait_until(lambda:all((a.root/f'candidates/{name}/cam{c:03d}_receipt.json').exists() for c in [7,12,95,150]))
    if not (a.root/f'evaluation/{name}_metrics.json').exists():compute(a.root,name,[7,12,95,150])
    result=summarize(a.root,name);results.append(result);write(a.root/f'evaluation/{a.tag}.json',results);log(a.root,'single_source_evaluated',**result)
