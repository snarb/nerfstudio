"""Local exclusive UUID supervisor; default validates only, never launches CUDA."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_marshawn_semantic_grid_replay import validate, sha


def write(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def identity(pid):
    stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,start=stat[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def gpu_state(plan):
    def query(args):
        return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[x.strip().split(', ')for x in query(['--query-gpu=index,uuid']).splitlines()]
    matches=[r for r in rows if int(r[0])==plan['gpu']]
    if len(matches)!=1 or matches[0][1].lower()!=plan['gpu_uuid'].lower():raise ValueError('Physical GPU index/UUID changed')
    apps=[]
    for line in query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory']).splitlines():
        fields=[x.strip()for x in line.split(',')]
        if fields[0].lower()==plan['gpu_uuid'].lower():apps.append(dict(pid=int(fields[1]),MiB=int(fields[2])))
    return apps


def stop_owned(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def run(path,execute=False):
    plan,_=validate(path,execute)
    if not execute:
        print(json.dumps(dict(dry_run=True,plan_sha256=sha(path),GPU_launched=False,uuid=plan['gpu_uuid'])));return
    root=Path(plan['controller']);root.mkdir(exist_ok=True)
    with (root/'controller.lock').open('a')as controller_lock:
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'identity.json').exists()or (root/'complete.json').exists()or Path(plan['output']).exists():raise ValueError('Existing attempt/output; no duplicate automatic launch')
        child=None
        def interrupted(sig,_):raise RuntimeError('Supervisor interrupted '+str(sig))
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
        write(root/'identity.json',identity(os.getpid()))
        try:
            with open(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{plan["gpu"]}.lock','a')as lease:
                while True:
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        write(root/'state.json',dict(status='waiting_for_global_lease',time=time.time()));time.sleep(30);continue
                    if gpu_state(plan):
                        fcntl.flock(lease,fcntl.LOCK_UN);write(root/'state.json',dict(status='waiting_for_idle_GPU',time=time.time()));time.sleep(30);continue
                    break
                validate(path,True)
                receipt=root/'lease.json';write(receipt,dict(gpu=plan['gpu'],gpu_uuid=plan['gpu_uuid'],supervisor=identity(os.getpid()),global_flock_held=True,time=time.time()))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],EXPECTED_GPU_UUID=plan['gpu_uuid'],LOOKCLOSER_GPU_LEASE_RECEIPT=str(receipt))
                command=[plan['python'],str(Path(plan['helpers'])/'run_marshawn_semantic_grid_replay.py'),'--plan',str(path),'--execute']
                began=time.time()
                with (root/'worker.log').open('wb')as log:
                    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write(root/'worker_identity.json',identity(child.pid))
                    while child.poll()is None:
                        apps=gpu_state(plan)
                        if any(a['pid']!=child.pid or a['MiB']>plan['process_cap_MiB']for a in apps):raise RuntimeError('GPU exclusivity/process memory exceeded')
                        if time.time()-began>plan['max_seconds']+30:raise TimeoutError('Bounded GPU worker deadline')
                        row=dict(status='running',time=time.time(),worker=identity(child.pid),compute=apps,elapsed=time.time()-began)
                        write(root/'state.json',row)
                        with (root/'resource_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                        time.sleep(5)
                    if child.returncode:raise RuntimeError('GPU worker failed exit '+str(child.returncode))
                result=json.loads((Path(plan['output'])/'complete.json').read_text())
                if result.get('passed')is not True or result.get('optimizer_updates')!=0:raise ValueError('Missing complete immutable replay proof')
                write(root/'GPU_complete.json',dict(complete_sha256=sha(Path(plan['output'])/'complete.json'),elapsed=time.time()-began,lease_released_before_CPU_evaluation=True))
            write(root/'state.json',dict(status='CPU_evaluation',time=time.time()))
            with (root/'CPU_evaluation.log').open('wb')as log:
                child=subprocess.Popen([plan['python'],str(Path(plan['helpers'])/'evaluate_marshawn_semantic_grid_replay_cpu.py'),'--plan',str(path),'--execute'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                if child.wait():raise RuntimeError('CPU metrics/contact evaluation failed')
            write(root/'complete.json',dict(passed=True,plan_sha256=sha(path),time=time.time(),mandatory_manual_review=True,quality_promoted=False))
            write(root/'state.json',dict(status='complete',time=time.time()))
        except BaseException as exc:
            stop_owned(child);write(root/'failure.json',dict(error=repr(exc),time=time.time()));write(root/'state.json',dict(status='failed',error=repr(exc),time=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();run(a.plan.resolve(),a.execute)
