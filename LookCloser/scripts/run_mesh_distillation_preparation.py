"""Supervise independent RGB teacher render workers (not PatchMatch/training)."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from joint_temporal_texture import read, atomic_json
from prepare_mesh_distillation_dataset import OUTPUT, verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('1..8 workers only; measured host RAM/GPU limit')
    request=verify(a.output);logroot=a.output/'logs';logroot.mkdir(exist_ok=True)
    outstanding=[x['id'] for x in request['plan'] if not (a.output/'synthetic/views'/x['id']/'complete.json').exists()]
    jobs=[];handles=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OPENCV_IO_ENABLE_OPENEXR='1')
    for index in range(a.workers):
        subset=outstanding[index::a.workers]
        if not subset:continue
        handle=(logroot/f'worker_{index}.log').open('a');handles.append(handle)
        cmd=[sys.executable,str(Path(__file__).with_name('prepare_mesh_distillation_dataset.py')),
             'render','--output',str(a.output),'--views',*subset]
        job=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env)
        jobs.append((index,job,subset))
    start=time.time()
    try:
        while True:
            alive=[dict(worker=i,pid=j.pid,exit_code=j.poll(),assigned=len(ids)) for i,j,ids in jobs]
            done=len(list((a.output/'synthetic/views').glob('*/complete.json')))
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            errors=[]
            for log in logroot.glob('worker_*.log'):
                tail=log.read_text()[-6000:]
                if any(token in tail for token in ['Traceback','OutOfMemoryError','CUDA error','out of memory']):errors.append(str(log))
            row=dict(utc=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),workers=alive,
                     complete=done,total=324,gpu_process_memory=gpu.stdout.strip(),
                     free_bytes=shutil.disk_usage(a.output).free,error_logs=errors,elapsed_seconds=time.time()-start)
            with (a.output/'preparation_checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic_json(a.output/'supervisor_status.json',row)
            print(f'complete={done}/324 alive={sum(x["exit_code"] is None for x in alive)} errors={len(errors)}',flush=True)
            if errors or any(x['exit_code'] not in (None,0) for x in alive):raise RuntimeError('Render worker failed; preserve outputs')
            if all(x['exit_code']==0 for x in alive):break
            time.sleep(30)
    finally:
        for _,job,_ in jobs:
            if job.poll() is None:job.terminate()
        for _,job,_ in jobs:
            try:job.wait(timeout=30)
            except subprocess.TimeoutExpired:job.kill();job.wait()
        for h in handles:h.close()
    assert done==324
    print('All teacher views ready; finalize/audit/visual review still required.',flush=True)


if __name__=='__main__':main()
