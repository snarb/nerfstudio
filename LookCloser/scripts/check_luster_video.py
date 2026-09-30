"""Record a compact live campaign check, including worker liveness and GPU state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import psutil


def load(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def alive(pid):
    try:return psutil.Process(pid).is_running() and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();root=args.root
    manifest=load(root/'manifest.json');frames=manifest['frames']
    record=dict(time=time.time(),prepared=sum((root/'frames'/f/'data/audit_preprocessing.json').exists() for f in frames),
                frequencies=sum((root/'frames'/f/'data/frequency_complete.json').exists() for f in frames),
                snapshots=sum((root/'snapshots'/f'{f}.json').exists() for f in frames),total=len(frames),
                free_GiB=psutil.disk_usage(root).free/2**30,
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),queues={})
    prep=load(root/'progress.json')
    if prep:record['preparation']=dict(progress=prep,controller_alive=alive(prep['pid']),complete=(root/'preparation_complete.json').exists())
    for name in ['local','dev3']:
        status=load(root/'frequency_queue'/f'{name}.json')
        if status:
            status['status_age_seconds']=time.time()-status['time']
            if 'queue_pid' in status:status['queue_alive']=alive(status['queue_pid'])
            progress=status.get('progress') or {}
            if name=='local' and progress.get('pid'):status['worker_alive_checked']=alive(progress['pid'])
            record['queues'][name]=status
    stages=[]
    for path in (root/'frames').glob('*/logs/s*/status.json'):
        status=load(path)
        if status and status.get('exit') is None:
            stages.append(dict(path=str(path),controller_alive=alive(status['controller_pid']),worker_alive=alive(status['worker_pid']),**status))
    record['active_stage_records']=stages
    with (root/'agent_checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if prep:record['preparation_failure_count']=len(prep.get('failures',{}))
    print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','queues']},indent=2))
    for name,status in record['queues'].items():print(name,json.dumps({k:status.get(k) for k in ['frame','alive','queue_alive','worker_alive_checked','progress','oom','status_age_seconds']}))


if __name__=='__main__':main()
