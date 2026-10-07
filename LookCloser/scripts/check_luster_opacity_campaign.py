#!/usr/bin/env python3
"""Record an agent supervision check of active Luster workers and GPU evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def alive(pid):
    try:
        os.kill(int(pid),0)
        return True
    except (ProcessLookupError,PermissionError,ValueError):
        return False


def check(root):
    jobs=[]
    for path in root.rglob('supervision.jsonl'):
        lines=path.read_text().splitlines()
        if not lines:
            continue
        last=json.loads(lines[-1])
        workers=last.get('workers',[dict(pid=last.get('worker'),exit=last.get('exit'))])
        active=[w for w in workers if w.get('pid') and w.get('exit') is None]
        if active and alive(last['controller']):
            jobs.append(dict(log=str(path),controller=last['controller'],controller_alive=True,
                       workers=[dict(pid=w['pid'],alive=alive(w['pid'])) for w in active],
                       latest=last,check_age_seconds=time.time()-last['time']))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    row=dict(time=time.time(),checker=os.getpid(),active_jobs=jobs,gpu=gpu.stdout.strip(),
             gpu_processes=processes.stdout.strip(),gpu_error=gpu.stderr.strip(),
             all_expected_workers_alive=all(w['alive'] for job in jobs for w in job['workers']),
             oom_evidence=any(job['latest'].get('oom',False) for job in jobs))
    with (root/'logs/agent_checks.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(dict(active_jobs=len(jobs),all_expected_workers_alive=row['all_expected_workers_alive'],
                          gpu=row['gpu'],oom=row['oom_evidence'])),flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/pavlo/renders/luster_000470_opacity'))
    args=p.parse_args()
    check(args.root)
