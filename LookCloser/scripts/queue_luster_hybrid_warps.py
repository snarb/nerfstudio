"""CPU-only durable warp worker; consumes completed native geometry receipts."""
import argparse
from pathlib import Path
import time
from luster_hybrid_common import read,log
from transfer_luster_hybrid import warp

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--count',type=int,default=8)
a=p.parse_args()
pending=[(int(t),s) for t,sources in read(a.root/'preflight/sources.json')['sources'].items() for s in sources[:a.count]]
started=time.monotonic()
while pending:
    ready=[pair for pair in pending if (a.root/'geometry'/f'cam{pair[1]:03d}'/'receipt.json').exists()]
    for target,source in ready:
        warp(a.root,target,source)
        pending.remove((target,source))
    if not ready:
        if time.monotonic()-started>23*3600:raise TimeoutError('Geometry queue deadline exceeded')
        time.sleep(10)
log(a.root,'warp_queue_complete',count=a.count)
