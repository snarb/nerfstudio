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


def cached_geometry_complete(root,cameras):
    """Verify previously completed frozen outputs without CUDA initialization."""
    receipts=[root/'geometry'/f'cam{cid:03d}'/'receipt.json' for cid in cameras]
    if not receipts or not all(p.exists() for p in receipts):
        return False
    import numpy as np
    from luster_hybrid_common import native_rows,read,sha,log
    context=read(root/'context.json')
    assert sha(context['checkpoint'])==context['checkpoint_sha256']
    assert sha(context['config'])==context['config_sha256']
    rows=native_rows();states=set()
    required={'rgb','alpha','ray_distance_mean','depth_z_q10','depth_z_q50','depth_z_q90'}
    for cid,path in zip(cameras,receipts):
        r=read(path)
        assert r['state']=='complete' and r['camera']==cid
        assert r['checkpoint_sha256']==context['checkpoint_sha256']
        assert r['row']==rows[cid], 'Cached geometry calibration differs'
        assert required<=set(r['arrays'])
        assert len(r['state_hash'])==64
        states.add(r['state_hash'])
        for key in r['arrays']:
            p=path.parent/f'{key}.npy'
            assert sha(p)==r['hashes'][key], 'Cached geometry array changed'
            array=np.load(p,mmap_mode='r')
            shape=(rows[cid]['h'],rows[cid]['w'])
            assert array.shape==((*shape,3) if key=='rgb' else shape)
    assert len(states)==1, 'Mixed frozen model states'
    log(root,'geometry_cache_verified_without_cuda',cameras=cameras,
        checkpoint_sha256=context['checkpoint_sha256'],state_hash=next(iter(states)))
    return True


def run(a):
    assert 1<=a.chunk<=4096
    os.environ['LUSTER_HYBRID_FRAME_MANIFEST']=str((a.root/'context.json').resolve())
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if cached_geometry_complete(a.root,a.cameras):
        return
    # This is exclusively our transfer lock, never the other campaign's lease.
    lease=(a.root.parent.parent/'gpu_inference.lock').open('a')
    # Queue our own renderers before importing Torch or allocating CUDA memory.
    # This permits CPU transfer for one frame alongside inference for the next,
    # while retaining exactly one of our GPU renderers at a time.
    fcntl.flock(lease,fcntl.LOCK_EX)
    state=telemetry()
    if not safe(state,True):
        raise RuntimeError(f'GPU admission refused: {state}')
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
