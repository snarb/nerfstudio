#!/usr/bin/env python3
"""Durable, restartable frame queue. No SSH session or Codex process is required.

The controller uses only the standard library. Each GPU runs one frame pipeline;
live workers are adopted after a controller restart using PID AND /proc starttime.
Receipts, logs, code and checkpoints live on persistent storage; downloads may be
placed on a disposable cache. A failed frame does not prevent other frames running.
"""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())
    temp.replace(path)


def identity(pid):
    try:
        # The comm field can contain spaces and parentheses.
        text=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if text[0]=='Z' else text[19]
    except (FileNotFoundError,ProcessLookupError):return None


def alive(record):
    return bool(record and record.get('starttime') is not None and identity(record['pid'])==record['starttime'])


def gpu_status():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    return [dict(zip(('index','uuid','used_mib','total_mib','utilization'),[v.strip() for v in line.split(',')]))
            for line in output.splitlines()]


def snapshot(root, manifest):
    rows=[]
    for frame in manifest['frames']:
        directory=root/'frames'/frame
        row=read(directory/'progress.json',{'phase':'queued'})
        worker=read(directory/'worker.json')
        failures=read(directory/'failures.json',[])
        rows.append(dict(frame=frame,**row,worker_alive=alive(worker),worker=worker,
            failure_count=len(failures),oom_evidence=any('out of memory' in f['error'].lower() for f in failures)))
    controller=read(root/'controller.json')
    return dict(time=time.time(),controller=controller,controller_alive=alive(controller),frames=rows,gpus=gpu_status(),
                persistent_free_bytes=shutil.disk_usage(root).free,
                counts=dict(collections.Counter(row['phase'] for row in rows)))


def serve(root):
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/'controller.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    manifest=read(root/'manifest.json');last_health=0;children={}
    while True:
        for key,child in list(children.items()):
            if child.poll() is not None:children.pop(key)
        state=snapshot(root,manifest);write(root/'status.json',state)
        # At most hourly, including PIDs, progress, checkpoints, GPU memory and
        # OOM evidence. Each worker also updates its own progress during training.
        if time.time()-last_health>=1800:
            with (root/'health.jsonl').open('a') as log:log.write(json.dumps(state)+'\n')
            last_health=time.time()
        if all(row['phase']=='complete' for row in state['frames']):return
        leased={row['worker']['gpu'] for row in state['frames'] if row['worker_alive']}
        idle=[g for g in state['gpus'] if g['uuid'] not in leased and int(g['used_mib'])<1024]
        all_maps=all((root/'frames'/f/'frequency_ready.json').exists() for f in manifest['frames'])
        for row in state['frames']:
            if not idle:break
            if row['phase']=='complete' or row['worker_alive']:continue
            if row['phase']=='frequency_ready' and not all_maps:continue
            directory=root/'frames'/row['frame'];directory.mkdir(parents=True,exist_ok=True)
            failures=read(directory/'failures.json',[])
            if failures and time.time()-failures[-1]['time']<min(1800,60*2**min(len(failures),5)):continue
            if len(failures)>=manifest.get('max_attempts',12):continue
            gpu=idle.pop(0)['uuid'];revision=read(root/'recipe.json')
            env=os.environ.copy();env.update(manifest['environment'])
            code=Path(revision['scripts']).parents[1]
            env['PYTHONPATH']=str(code)+os.pathsep+revision['scripts']+os.pathsep+env.get('PYTHONPATH','')
            env.update(CUDA_VISIBLE_DEVICES=gpu,LOOKCLOSER_GPU_INDEX=gpu)
            command=[manifest['python'],str(Path(revision['scripts'])/'dec5_sequence_worker.py'),
                     '--root',str(root),'--frame',row['frame']]
            with (directory/'worker.log').open('a') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[row['frame']]=child
            write(directory/'worker.json',dict(pid=child.pid,starttime=identity(child.pid),gpu=gpu,
                command=command,launched=time.time(),revision=revision['revision']))
        time.sleep(20)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['serve','status']);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='serve':serve(args.root.resolve())
    else:
        manifest=read(args.root/'manifest.json');state=snapshot(args.root,manifest)
        print(json.dumps(dict(counts=state['counts'],gpus=state['gpus'],controller_alive=state['controller_alive'],persistent_free_gib=round(state['persistent_free_bytes']/(1<<30),1)),ensure_ascii=False))
        for row in state['frames']:
            if row['phase']!='queued' or row['worker_alive']:
                print(row['frame'],row['phase'],'alive='+str(row['worker_alive']),
                      row.get('completed',''),row.get('step',''),row.get('error','')[:160])


if __name__=='__main__':main()
