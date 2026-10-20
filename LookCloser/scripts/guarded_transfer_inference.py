"""Run only our inference with bounded allocation and a self-only watchdog.

Never acquires the trainer's lease, sends it signals, edits its files, or launches
training. The monitor terminates this process on unsafe telemetry or query failure.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def telemetry():
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=8)
    total,used,free=map(int,raw.strip().split(','))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True,timeout=8)
    own=0
    for line in apps.strip().splitlines():
        pid,mem=line.split(',')
        if int(pid)==os.getpid():
            own=int(mem)
    return dict(total_mib=total,used_mib=used,free_mib=free,own_mib=own)


def safe(state, admission=False):
    return state['free_mib']>=65536 and state['used_mib']<24576 and state['own_mib']<6144


def run(a):
    assert 1<=a.chunk<=4096
    # This is exclusively our transfer lock, never the other campaign's lease.
    lease=(a.root.parent.parent/'gpu_inference.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=telemetry()
    if not safe(state,True):
        raise RuntimeError(f'GPU admission refused: {state}')
    os.environ['LUSTER_HYBRID_FRAME_MANIFEST']=str((a.root/'context.json').resolve())
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.nice(10)
    stop=threading.Event()
    path=a.root/'logs/gpu_guard.jsonl'
    def watch():
        while not stop.is_set():
            try:
                s=telemetry()
                with path.open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),pid=os.getpid(),**s))+'\n')
                if not safe(s):
                    raise RuntimeError(f'GPU guard threshold: {s}')
            except Exception as error:
                try:
                    with path.open('a') as f:
                        f.write(json.dumps(dict(time=time.time(),action='exit_own_process_only',error=str(error)))+'\n')
                finally:
                    os._exit(75)
            stop.wait(1)
    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4096/state['total_mib'],0)
    if a.occupancy_only:
        from occupancy_native_luster_hybrid import run as render
    elif a.aabb_only:
        from bounded_native_luster_hybrid import run as render
    else:
        from render_luster_hybrid import run as render
    try:
        render(a)
    finally:
        stop.set();monitor.join(timeout=10)
        lease.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cameras',nargs='+',type=int,required=True)
    p.add_argument('--chunk',type=int,default=1024)
    p.add_argument('--aabb-only',action='store_true')
    p.add_argument('--occupancy-only',action='store_true')
    run(p.parse_args())
