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
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--compact',action='store_true');args=p.parse_args();root=args.root
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
            if name=='local' and status.get('frame'):
                log=root/'frames'/status['frame']/'logs/frequencies_queue.log'
                if log.exists():
                    with log.open('rb') as stream:
                        stream.seek(max(0,log.stat().st_size-6000));tail=stream.read().decode(errors='replace')
                    status['oom']='out of memory' in tail.lower()
            record['queues'][name]=status
    controllers=[]
    for process in psutil.process_iter(['pid','cmdline']):
        command=process.info['cmdline'] or []
        if any(Path(arg).name in ['run_luster_video_campaign.py','finish_luster_video_frame.py','export_luster_selection.py','render_luster_video_frame.py'] for arg in command) and any(arg.startswith(str(root)) for arg in command):
            controllers.append(dict(pid=process.pid,alive=alive(process.pid),command=command))
    record['campaign_processes']=controllers
    priority=load(root/'gpu_priority/status.json')
    if priority:record['gpu_priority']=dict(pid=priority['pid'],alive=alive(priority['pid']),age_seconds=time.time()-priority['time'],paused=len(priority.get('paused',[])))
    stages=[]
    for path in (root/'frames').glob('*/logs/s*/status.json'):
        status=load(path)
        if status and status.get('exit') is None:
            stages.append(dict(path=str(path),controller_alive=alive(status['controller_pid']),worker_alive=alive(status['worker_pid']),**status))
    record['active_stage_records']=stages
    record['campaign_status']=load(root/'campaign_status.json')
    with (root/'agent_checks.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if prep:record['preparation_failure_count']=len(prep.get('failures',{}))
    if args.compact:
        summary={k:record[k] for k in ['time','prepared','frequencies','snapshots','total','gpu']}
        summary['free_GiB']=round(record['free_GiB'],1)
        summary['campaign']=record['campaign_status']
        summary['controllers']=[dict(pid=row['pid'],alive=row['alive'],script=next((Path(arg).name for arg in row['command'] if arg.endswith('.py')),'')) for row in controllers]
        summary['stages']=[dict(stage='/'.join(Path(row['path']).parts[-4:-1]),controller=row['controller_alive'],worker=row['worker_alive'],
                                step=(row.get('progress') or {}).get('step'),phase=(row.get('progress') or {}).get('phase'),oom=row.get('oom')) for row in stages]
        summary['queues']={name:dict(frame=status.get('frame'),images=(status.get('progress') or {}).get('images'),
                                    queue_alive=status.get('queue_alive'),worker_alive=status.get('worker_alive_checked',status.get('alive')),
                                    oom=status.get('oom'),age=round(status['status_age_seconds'],1)) for name,status in record['queues'].items()}
        summary['gpu_priority']=record.get('gpu_priority')
        print(json.dumps(summary));return
    print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','queues']},indent=2))
    for name,status in record['queues'].items():print(name,json.dumps({k:status.get(k) for k in ['frame','alive','queue_alive','worker_alive_checked','progress','oom','status_age_seconds']}))


if __name__=='__main__':main()
