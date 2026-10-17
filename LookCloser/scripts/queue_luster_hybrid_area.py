"""CPU-only parallel area caches; waits for native geometry/point warps as needed."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing
from pathlib import Path
import time
from antialias_luster_hybrid import variant_root,warp_area
from luster_hybrid_common import read,write,log


def run(root,count,workers):
    variant_root(root)
    pending=[(int(c),s) for c,ids in read(root/'preflight/sources.json')['sources'].items() for s in ids[:count]]
    deadline=read(root/'preflight/manifest.json')['deadline']
    done=[];active={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while pending or active:
            if time.time()>deadline:raise TimeoutError('Campaign deadline reached')
            for pair in pending.copy():
                c,s=pair
                cached=root/'variants/area4/warps'/f'cam{c:03d}'/f'src{s:03d}'/'receipt.json'
                if cached.exists():
                    pending.remove(pair);done.append(pair);continue
                available=root/'warps'/f'cam{c:03d}'/f'src{s:03d}'/'receipt.json'
                if len(active)<workers and available.exists():
                    active[pool.submit(warp_area,root,c,s)]=pair;pending.remove(pair)
            if active:
                finished,_=wait(active,timeout=3,return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result();done.append(active.pop(future))
            else:time.sleep(3)
            write(root/'preflight/area_queue.json',dict(completed=done,active=list(active.values()),pending=pending,time=time.time()))
    log(root,'area_queue_complete',count=count,pairs=len(done))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.root,a.count,a.workers)
