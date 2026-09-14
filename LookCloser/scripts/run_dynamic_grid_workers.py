"""Supervise disjoint dynamic-frame workers, optionally including foreground guards."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from joint_temporal_texture import read, sha, atomic_json
import render_smooth_temporal_mesh_video as renderer


def worker(output, index, ids):
    from temporal_texture_view_prior import install
    install(renderer)
    if renderer.verify_request(output)['recipe'].get('train_foreground_guard'):
        from train_foreground_guard import install as guard
        guard(renderer)
    original=renderer.atomic_json
    def write(path,payload):
        if Path(path)==output/'progress.json':path=output/'workers'/f'{index}.json'
        original(path,payload)
    renderer.atomic_json=write
    render_one=renderer.render_one
    def notified(root,record,manifest):
        write(root/'progress.json',dict(pid=os.getpid(),frame_id=record['frame_id'],index=record['index'],
            stage='preparing_foreground_or_resuming',utc=datetime.now(timezone.utc).isoformat()))
        return render_one(root,record,manifest)
    renderer.render_one=notified
    renderer.render(output,ids)


def supervise(output,count):
    if not 1<=count<=8:raise ValueError('Expected 1..8 render workers')
    if count>4:
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if memory<80000:raise ValueError('More than four render workers requires the 96GB host')
    request=renderer.verify_request(output);ids=request['ordered_frame_ids']
    if len(set(ids))!=150 or ids!=sorted(ids):raise ValueError('Expected 150 distinct chronological times')
    directory=output/'workers';directory.mkdir(exist_ok=True)
    lock=(output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    groups=[x.tolist() for x in np.array_split(ids,count)];jobs=[];start=time.monotonic()
    for index,group in enumerate(groups):
        log=(directory/f'{index}.log').open('a')
        command=[sys.executable,__file__,'worker','--output',str(output),'--worker-index',str(index),'--frames',*group]
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        jobs.append((index,group,process,log))
    atomic_json(directory/f'execution_{os.getpid()}.json',dict(supervisor_pid=os.getpid(),script_sha256=sha(__file__),
        request_sha256=sha(output/'request.json'),workers=[dict(index=i,pid=p.pid,frames=g) for i,g,p,_ in jobs]))
    while True:
        statuses=[]
        for index,group,process,_ in jobs:
            progress=directory/f'{index}.json';current=read(progress) if progress.exists() else {}
            statuses.append(dict(index=index,pid=process.pid,exit_code=process.poll(),
                progress=current if current.get('pid')==process.pid else {'stage':'preparing_foreground_or_starting'}))
        complete=len(list((output/'frames').glob('*/complete.json')))
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
        record=dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),workers=statuses,
            complete=complete,total=len(ids),gpu_process_memory=gpu,free_bytes=shutil.disk_usage(output).free,
            elapsed_seconds=time.monotonic()-start,stage='rendering' if any(p.poll() is None for _,_,p,_ in jobs) else 'workers_finished')
        atomic_json(output/'progress.json',record)
        import json
        with (output/'checks.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
        print(f'complete={complete}/150 active={sum(p.poll() is None for _,_,p,_ in jobs)} elapsed={record["elapsed_seconds"]:.1f}',flush=True)
        if record['stage']=='workers_finished':break
        time.sleep(30)
    for _,_,_,log in jobs:log.close()
    if any(p.returncode for _,_,p,_ in jobs) or complete!=150:raise RuntimeError('Incomplete campaign; retain failed workspaces')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['supervise','worker']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--worker-index',type=int);p.add_argument('--frames',nargs='+');a=p.parse_args()
    renderer.torch.set_num_threads(4)
    if a.action=='worker':worker(a.output,a.worker_index,a.frames)
    else:supervise(a.output,a.workers)
