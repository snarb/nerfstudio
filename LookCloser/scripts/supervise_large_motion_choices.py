"""Bounded fresh-process shards for the non-reentrant frozen RGB installer."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from joint_temporal_texture import read,sha,atomic_json
from render_smooth_temporal_mesh_video import verify_request
from right_arc_visibility_refinement import BASE,VARIANTS,CANARIES


def supervise(canary=False,mixed=False):
    root=BASE/('mixed_fresh_workers' if mixed else ('refined_canary_workers' if canary else 'full_fresh_workers'));root.mkdir(exist_ok=True)
    jobs=[]
    for variant in (VARIANTS[::-1] if mixed else VARIANTS):
        gate=canary or (mixed and variant=='right_high_arc_refined')
        ids=CANARIES if gate else verify_request(BASE/variant)['ordered_frame_ids']
        ids=[f for f in ids if not (BASE/variant/'frames'/f/'complete.json').exists()]
        chunk=6 if gate else 25
        for start in range(0,len(ids),chunk):jobs.append(dict(variant=variant,frames=ids[start:start+chunk]))
    pending=list(enumerate(jobs));active=[];finished=[]
    atomic_json(root/'execution.json',dict(supervisor_pid=os.getpid(),script_sha256=sha(__file__),maximum_workers=6,jobs=jobs))
    while pending or active:
        for item in list(active):
            index,p,log,job=item;code=p.poll()
            if code is not None:
                finished.append(dict(index=index,variant=job['variant'],pid=p.pid,exit_code=code));log.close();active.remove(item)
        if any(j['exit_code'] for j in finished):
            pending=[] # let active in-scope work finish; do not launch more failures
        while pending and len(active)<6:
            index,job=pending.pop(0);out=BASE/job['variant'];(out/'workers').mkdir(exist_ok=True)
            command=[sys.executable,str(Path(__file__).with_name('run_view_consistent_dynamic_video.py')),'worker','--output',str(out),'--worker-index',str(index),'--frames',*job['frames']]
            log=(root/f'{index}.log').open('a');p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            active.append((index,p,log,job));atomic_json(root/f'{index}_launch.json',dict(command=command,pid=p.pid,job=job))
        rec=dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),pending_jobs=len(pending),
            active=[dict(index=i,pid=p.pid,variant=j['variant']) for i,p,_,j in active],finished=finished,
            complete={v:len(list((BASE/v/'frames').glob('*/complete.json'))) for v in VARIANTS},
            free_bytes=shutil.disk_usage(BASE).free,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip())
        with (root/'checks.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        atomic_json(root/'progress.json',rec);print(json.dumps(rec),flush=True)
        if active:time.sleep(30)
    assert all(j['exit_code']==0 for j in finished),'Failed shard; retained receipts/logs'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--canary',action='store_true');p.add_argument('--mixed',action='store_true');a=p.parse_args();supervise(a.canary,a.mixed)
