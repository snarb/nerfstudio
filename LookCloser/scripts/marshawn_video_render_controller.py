#!/usr/bin/env python3
"""Detached post-training dispatcher; waits for all72 qualified cold checkpoints.

Run under the campaign's existing durable service. Safe to restart: images,
geometry and completed units are provenance-checked by the per-frame renderer.
This executable does NOT launch training or synthesize unavailable GT frames.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from marshawn_video_path import write, raw_file


def read(p):return json.loads(Path(p).read_text())


def idle_gpu(gpu):
    p=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],
        capture_output=True,text=True)
    return p.returncode==0 and int(p.stdout.strip())<1024


def unit_locked(root,index):
    unit=root/'frames'/f'{index:04d}';unit.mkdir(parents=True,exist_ok=True)
    with (unit/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        return False


def supervise(a):
    spec=read(a.spec);request=read(a.path);root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={};failures={};logs={};leases={};last_hour=0
    inventory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    gpu_uuids={int(line.split(',')[0]):line.split(',')[1].strip() for line in inventory.stdout.splitlines()}
    if not set(a.gpus)<=set(gpu_uuids):raise ValueError('Unknown physical GPU index')
    if (root/'controller.json').exists():
        failures={int(k):v for k,v in read(root/'controller.json').get('failed_units',{}).items()}
    while True:
        now=time.time()
        for index,(process,gpu) in list(running.items()):
            status=process.poll()
            if status is None:continue
            logs.pop(index).close();leases.pop(index).close();del running[index]
            if status:failures[index]=failures.get(index,0)+1
        ready=[];missing=[]
        for record in request['records'][:72]:
            frame=record['frame_id'];completion=Path(spec['frame_root_template'].format(frame_id=frame))/'frame_completion.json'
            if not completion.exists():missing.append(frame);continue
            try:
                selected=read(completion)['selectors'][spec.get('checkpoint_kind','actor_lpips')]
                if not Path(selected['checkpoint']).exists():missing.append(frame);continue
            except (KeyError,ValueError):missing.append(frame);continue
            ready.append(frame)
        pending=[];waiting_gt=[]
        for record in request['records']:
            index=record['index'];done=root/'frames'/f'{index:04d}'/'complete.json'
            if done.exists() or index in running:continue
            if unit_locked(root,index):continue  # A pre-crash worker is still completing this unit.
            if record['kind']=='neural' and missing:continue
            if record['kind']=='actual_gt' and not raw_file(request['endpoint_train_camera'],int(record['frame_id']),spec['raw_roots']):
                waiting_gt.append(record['frame_id']);continue
            if failures.get(index,0)>=3:continue
            pending.append(index)
        for gpu in a.gpus:
            if not pending:break
            if any(g==gpu for _,g in running.values()):continue
            lease=Path('/tmp/lookcloser_tuning_gpu_%s_%s.lock'%(os.getuid(),gpu)).open('a+')
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease.close();continue
            if not idle_gpu(gpu):lease.close();continue
            index=pending.pop(0);log=(root/f'render_{index:04d}.log').open('a');logs[index]=log
            leases[index]=lease
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuids[gpu],MARSHAWN_RENDER_GPU_UUID=gpu_uuids[gpu],
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            code=Path(__file__).resolve().parents[2]
            env['PYTHONPATH']=str(code)+os.pathsep+str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH','')
            cmd=[sys.executable,str(Path(__file__).with_name('marshawn_video_render.py')),
                '--spec',str(a.spec),'--path',str(a.path),'--index',str(index)]
            running[index]=(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,
                start_new_session=True,pass_fds=(lease.fileno(),)),gpu)
        finished=sum((root/'frames'/f'{i:04d}'/'complete.json').exists() for i in range(120))
        state=dict(time=now,pid=os.getpid(),complete=finished,total=120,qualified_models=len(ready),
            waiting_models=missing,waiting_actual_GT=waiting_gt,failed_units=failures,
            workers={str(i):dict(pid=p.pid,gpu=g) for i,(p,g) in running.items()},
            status='rendering' if running else 'waiting_models' if missing else 'waiting_GT' if waiting_gt else 'ready')
        write(root/'controller.json',state)
        if all((root/'frames'/f'{i:04d}'/'complete.json').exists() for i in range(72)) and not (root/'neural_3s_complete.json').exists():
            subprocess.run([sys.executable,str(Path(__file__).with_name('marshawn_video_render.py')),
                '--spec',str(a.spec),'--path',str(a.path),'--encode-neural'],check=True)
        if now-last_hour>=3600:
            memory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
            state['GPU_memory_evidence']=memory.stdout.strip();state['worker_processes_alive']=all(p.poll() is None for p,_ in running.values())
            state['OOM_log_evidence']={str(i):'out of memory' in (root/f'render_{i:04d}.log').read_text(errors='replace').lower()
                for i in set(failures)|set(running)}
            with (root/'hourly_checks.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
            last_hour=now
        if finished==120 and not running:
            subprocess.run([sys.executable,str(Path(__file__).with_name('marshawn_video_render.py')),
                '--spec',str(a.spec),'--path',str(a.path),'--encode'],check=True)
            write(root/'controller.json',dict(state,status='encoded_visual_review_pending'));return
        if any(n>=3 for n in failures.values()) and not running and not pending:
            write(root/'controller.json',dict(state,status='failed_review_required'));raise RuntimeError('Render unit failed three times; see logs')
        time.sleep(min(a.poll_seconds,60))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--path',type=Path,required=True);p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--poll-seconds',type=int,default=60);supervise(p.parse_args())
