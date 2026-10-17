"""Prepare disjoint CPU confidence caches while their sequential owner is stopped.

No GPU work. An existing incomplete source cache belongs to the stopped owner
and is excluded. The owner is always resumed, including on worker failure.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
import os
from pathlib import Path
import signal
import time
from luster_hybrid_common import read,log


def prepare_one(root,source,config):
    from source_surface_confidence_luster_hybrid import source_confidence
    return str(source_confidence(Path(root),source,config))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--owner-pid',type=int,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    process=Path(f'/proc/{a.owner_pid}');command=(process/'cmdline').read_bytes().split(b'\0')
    assert any(x.endswith(b'run_source_surface_luster.py') for x in command)
    assert str(a.root).encode() in command and process.stat().st_uid==os.getuid()
    os.kill(a.owner_pid,signal.SIGSTOP)
    try:
        for _ in range(100):
            status=(process/'status').read_text()
            if '\nState:\tT ' in status:break
            time.sleep(.01)
        else:raise RuntimeError('Owner did not stop')
        config=read(a.root/'candidates/single_cubic_eligible_phase_rt2/config.json')
        selected=read(a.root/'preflight/sources.json')['sources']
        ids=sorted({s for c in [7,12,95,150] for s in selected[str(c)][:config['count']]})
        pending=[s for s in ids if not (a.root/'source_surface_confidence'/f'cam{s:03d}').exists()]
        log(a.root,'source_surface_parallel_cpu_start',owner=a.owner_pid,sources=pending,workers=a.workers)
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(prepare_one,str(a.root),s,config):s for s in pending}
            for future in as_completed(futures):
                future.result();log(a.root,'source_surface_parallel_cpu_ready',source=futures[future])
    finally:
        if process.exists() and (process/'cmdline').read_bytes().split(b'\0')==command:
            os.kill(a.owner_pid,signal.SIGCONT)
            log(a.root,'source_surface_parallel_cpu_owner_resumed',owner=a.owner_pid)


if __name__=='__main__':main()
