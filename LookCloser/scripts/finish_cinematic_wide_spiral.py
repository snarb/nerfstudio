"""Supervised postprocessing as soon as each complete126-frame shot is ready.

Does not approve visuals or publish: those require actual operator review.
Rendering and postprocessing have separate compact process/stage receipts.
"""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from joint_temporal_texture import read,sha,atomic_json
from cinematic_wide_spiral_centered import BASE,VARIANTS,RAW_IDS


def worker(variant):
    root=BASE/variant;steps=[('compose','compose_cinematic_train_ending.py',
        ['--root',str(root),'--compose'])]
    steps += [(action,'cinematic_wide_spiral_centered.py',[action,'--variant',variant])
              for action in ['raw_audit','audit','encode','sheets']]
    for stage,script,args in steps:
        atomic_json(root/'postprocess_progress.json',dict(pid=os.getpid(),stage=stage,status='running',
            utc=datetime.now(timezone.utc).isoformat()))
        command=[sys.executable,str(Path(__file__).with_name(script)),*args]
        with (root/f'postprocess_{stage}.log').open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    atomic_json(root/'postprocess_progress.json',dict(pid=os.getpid(),status='ready_for_actual_visual_review',
        stage='complete',utc=datetime.now(timezone.utc).isoformat()))


def supervise():
    pending=list(VARIANTS);active={};finished={};root=BASE/'postprocess';root.mkdir(exist_ok=True)
    while pending or active:
        for v,p in list(active.items()):
            if p.poll() is not None:finished[v]=p.returncode;del active[v]
        if any(finished.values()):raise RuntimeError('Postprocessing failed; retained stage logs')
        for v in pending.copy():
            if all((BASE/v/'frames'/f/'complete.json').exists() for f in RAW_IDS):
                active[v]=subprocess.Popen([sys.executable,__file__,'--variant',v],
                    stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
                pending.remove(v)
        if pending:
            render=read(BASE/'full_fresh_workers'/'progress.json')
            if any(j['exit_code'] for j in render['finished']):raise RuntimeError('Renderer failure; outputs retained')
        status=dict(utc=datetime.now(timezone.utc).isoformat(),supervisor_pid=os.getpid(),
            pending=pending,active={v:p.pid for v,p in active.items()},finished=finished,
            stages={v:read(BASE/v/'postprocess_progress.json') for v in VARIANTS if (BASE/v/'postprocess_progress.json').exists()},
            free_bytes=shutil.disk_usage(BASE).free,script_sha256=sha(__file__))
        with (root/'checks.jsonl').open('a') as f:f.write(__import__('json').dumps(status)+'\n')
        atomic_json(root/'progress.json',status)
        if pending or active:time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    worker(a.variant) if a.variant else supervise()
