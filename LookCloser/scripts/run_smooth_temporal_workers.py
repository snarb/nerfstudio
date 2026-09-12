"""Disjoint frame workers; parallelize rendering, never concurrent PatchMatch.

The frozen renderer is unmodified. Its progress writer is redirected per worker
for supervision only; frame artifact writes and numerical operations are identical.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import render_smooth_temporal_mesh_video as renderer
from joint_temporal_texture import read,sha,atomic_json


def worker(output,index,ids):
    if 'target_angle_sigma_degrees' in renderer.verify_request(output)['recipe']:
        from temporal_texture_view_prior import install
        install(renderer)
    original=renderer.atomic_json
    def write(path,payload):
        if Path(path)==output/'progress.json':path=output/'workers'/f'{index}.json'
        original(path,payload)
    renderer.atomic_json=write
    try:renderer.render(output,ids)
    except BaseException as error:
        write(output/'progress.json',{'pid':os.getpid(),'stage':'failed','error':repr(error)})
        raise


def supervise(output,count):
    if not 1<=count<=4:raise ValueError('Use one to four non-PatchMatch render workers')
    renderer.verify_request(output);directory=output/'workers';directory.mkdir(exist_ok=True)
    lock=(output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ids=read(output/'request.json')['ordered_frame_ids'];groups=[x.tolist() for x in np.array_split(ids,count)]
    if sorted(sum(groups,[]))!=ids or len(set(sum(groups,[])))!=150:raise ValueError('Worker partition mismatch')
    launched=[];logs=[];start=time.monotonic()
    for index,group in enumerate(groups):
        stream=(directory/f'{index}.log').open('a');logs.append(stream)
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--output',str(output),'--worker-index',str(index),'--frames',*group]
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        launched.append((index,group,process))
    atomic_json(directory/f'execution_{os.getpid()}.json',{'supervisor_pid':os.getpid(),'script_sha256':sha(__file__),
        'request_sha256':sha(output/'request.json'),'workers':[{'index':i,'pid':p.pid,'frames':g} for i,g,p in launched],
        'overlapping_frame_assignments':False,'concurrent_patchmatch_jobs':0})
    while True:
        statuses=[]
        for index,group,process in launched:
            path=directory/f'{index}.json';progress=read(path) if path.exists() else {}
            statuses.append({'worker':index,'pid':process.pid,'exit_code':process.poll(),
                'current_progress':progress if progress.get('pid')==process.pid else {'stage':'starting'},
                'frame_count':len(group)})
        complete=len(list((output/'frames').glob('*/complete.json')))
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
        record={'utc':datetime.now(timezone.utc).isoformat(),'supervisor_pid':os.getpid(),'workers':statuses,
            'complete':complete,'total':150,'gpu_process_memory':gpu,'free_bytes':shutil.disk_usage(output).free,
            'elapsed_seconds':time.monotonic()-start,'stage':'rendering' if any(s['exit_code'] is None for s in statuses) else 'workers_finished'}
        atomic_json(output/'progress.json',record)
        with (output/'checks.jsonl').open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
        print(f'complete={complete}/150 active={sum(s["exit_code"] is None for s in statuses)} elapsed={record["elapsed_seconds"]:.1f}',flush=True)
        if all(s['exit_code'] is not None for s in statuses):break
        time.sleep(30)
    for stream in logs:stream.close()
    if any(s['exit_code'] for s in statuses):raise RuntimeError('A worker failed; retain partial outputs and inspect its compact log')
    if complete!=150:raise RuntimeError('Workers exited without a complete 150-frame inventory')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['supervise','worker'])
    parser.add_argument('--output',type=Path,default=renderer.OUTPUT);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--worker-index',type=int);parser.add_argument('--frames',nargs='+');args=parser.parse_args()
    renderer.torch.set_num_threads(4)
    if args.action=='worker':worker(args.output,args.worker_index,args.frames)
    else:supervise(args.output,args.workers)


if __name__=='__main__':main()
