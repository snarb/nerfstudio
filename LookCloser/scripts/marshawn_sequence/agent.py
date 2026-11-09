"""One durable local GPU executor. Queue state survives controller/SSH termination."""
import argparse
import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path
from common import atomic,read


def identity(pid):
    try:
        text=Path('/proc/%s/stat'%pid).read_text().split(') ',1)[1].split()
        if text[0]=='Z':return None
        return dict(pid=pid,start_ticks=text[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'agent.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    process=None;active=None;last_check=0
    state=read(root/'agent_state.json') if (root/'agent_state.json').exists() else dict(jobs={})
    for key,row in state['jobs'].items():
        if row['status']=='running' and identity(row['process']['pid'])==row['process']:active=key
        elif row['status']=='running':row['status']='interrupted'
    while True:
        now=time.time()
        if (root/'STOP').exists():
            if active:
                row=state['jobs'][active]
                if identity(row['process']['pid'])==row['process']:os.kill(row['process']['pid'],signal.SIGTERM)
            atomic(root/'agent_state.json',dict(state,stopped=True,time=now));return
        queue=read(root/'queue.json') if (root/'queue.json').exists() else dict(jobs=[])
        if active:
            row=state['jobs'][active]
            if process is not None:code=process.poll()
            else:code=None if identity(row['process']['pid'])==row['process'] else -999
            if code is not None:
                done=Path(row['job']['completion'])
                valid=done.exists()
                if valid and row['job'].get('target'):
                    receipt=read(done)
                    exhausted=bool(row['job'].get('max_optimizer_attempts') and receipt.get('attempt_budget',{}).get('reserved_attempts',0)>=row['job']['max_optimizer_attempts'])
                    valid=receipt.get('completed_iterations',0)>=row['job']['target'] or receipt.get('plateau',False) or exhausted
                row.update(status='complete' if valid and code in (0,-999) else 'failed',exit_code=code,finished=now)
                if row['status']=='failed' and row['job'].get('retry_backoff_seconds'):
                    row['next_retry_at']=now+min(900,row['job']['retry_backoff_seconds']*2**(row['attempts']-1))
                atomic(root/'agent_state.json',state);process=None;active=None
        if not active:
            for job in queue['jobs']:
                key=job['id'];old=state['jobs'].get(key)
                if old and old['status']=='complete':continue
                if old and old.get('attempts',0)>=job.get('max_attempts',3):continue
                if old and now<old.get('next_retry_at',0):continue
                if any(state['jobs'].get(d,{}).get('status')!='complete' for d in job.get('dependencies',[])):continue
                log=root/'logs'/(key+'.log');log.parent.mkdir(exist_ok=True)
                with log.open('ab',buffering=0) as stream:
                    process=subprocess.Popen(job['command'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,**job.get('env',{})))
                row=dict(status='running',attempts=(old or {}).get('attempts',0)+1,process=identity(process.pid),job=job,started=now)
                state['jobs'][key]=row;active=key;atomic(root/'agent_state.json',state);break
        if now-last_check>=300:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
            check=dict(time=now,agent_pid=os.getpid(),active=active,worker_alive=bool(active and identity(state['jobs'][active]['process']['pid'])==state['jobs'][active]['process']),gpu=gpu.stdout.strip(),gpu_error=gpu.stderr.strip())
            if active:
                progress=Path(state['jobs'][active]['job'].get('progress','/nonexistent'))
                check['progress']=read(progress) if progress.exists() else None
            with (root/'supervision.jsonl').open('a') as stream:stream.write(__import__('json').dumps(check)+'\n')
            atomic(root/'heartbeat.json',check);last_check=now
        atomic(root/'agent_state.json',dict(state,time=now,active=active));time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
