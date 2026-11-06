#!/usr/bin/env python3
"""Exclusive-idle bounded supervisor for the separately released4K technical proof."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def validate_policy(p):
    if p.get('scope')!='TECHNICAL_ONLY_4K_R0_W64' or p.get('production_authorized') is not False:raise ValueError('Technical-only scope required')
    if p['gpu'] not in range(8) or not 0<p['deadline_seconds']<=900 or not 0<p['candidate_mib_cap']<=12288:raise ValueError('Resource caps exceed reviewed range')
    if p['smoke_steps']!=1057 or p['pause_after']!=1024 or p['monitor_seconds']!=5:raise ValueError('Wrong actual-update/resume schedule')
    return p

def gpu_processes(uuid):
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    out=[]
    for line in rows.splitlines():
        cols=[x.strip()for x in line.split(',')]
        if len(cols)==3 and cols[0]==uuid:out.append(dict(pid=int(cols[1]),mib=int(cols[2])))
    return out

def run(policy_path):
    policy_path=Path(policy_path);p=validate_policy(read(policy_path));r=Path(p['root']);r.mkdir(parents=True,exist_ok=True)
    own=(r/'supervisor.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (r/'technical_complete.json').exists():
        if read(r/'technical_complete.json').get('policy_sha256')!=sha(policy_path):raise ValueError('Completed proof belongs to another policy')
        return
    release=read(p['release'])
    if release.get('technical_authorized')is not True or release.get('production_authorized')is not False or release.get('policy_sha256')!=sha(policy_path):raise ValueError('Exact root technical release required')
    for path,digest in p['artifacts'].items():
        if sha(path)!=digest:raise ValueError('Policy artifact changed: '+path)
    lock=Path('/tmp')/f"lookcloser_tuning_gpu_{os.getuid()}_{p['gpu']}.lock";gpu=lock.open('a+');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi','-i',str(p['gpu']),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if gpu_processes(uuid):raise RuntimeError('GPU has active compute; no interruption or overlap allowed')
    start=time.monotonic();children=[];peak=0;current=None
    atomic(r/'supervisor.json',dict(pid=os.getpid(),time=time.time(),policy_sha256=sha(policy_path),GPU_uuid=uuid,global_lock=str(lock)))
    def terminate(signum,frame):raise RuntimeError('Supervisor interrupted')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(p['gpu']),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=p['code_root']+'/LookCloser/scripts:'+p['code_root'])
        for phase,expected in [('cold',143),('resume',0)]:
            argv=[p['python'],p['worker'],'--spec',p['spec'],'--contract',p['contract'],'--smoke-steps','1057']
            log=(r/(phase+'.log')).open('ab');current=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
            children.append(dict(phase=phase,pid=current.pid,argv=argv))
            while current.poll() is None:
                elapsed=time.monotonic()-start;procs=gpu_processes(uuid);memory=sum(x['mib']for x in procs);peak=max(peak,memory)
                if any(x['pid']!=current.pid for x in procs):raise RuntimeError('Unexpected compute process on exclusive technical GPU')
                if memory>p['candidate_mib_cap'] or elapsed>p['deadline_seconds']:raise RuntimeError('Technical resource cap exceeded')
                with (r/'health.jsonl').open('a')as f:f.write(json.dumps(dict(time=time.time(),phase=phase,pid=current.pid,elapsed_seconds=elapsed,gpu_mib=memory,processes=procs))+'\n')
                time.sleep(p['monitor_seconds'])
            log.close()
            if current.returncode!=expected:raise RuntimeError(f'{phase} exit{current.returncode}, expected{expected}')
            if phase=='cold':
                update=read(Path(p['output'])/'frequency_step1024.json')
                if update['step']!=1024 or update['unique_touched_cells']<1 or not update['RNG_unchanged_by_grid_math_and_observer']:raise ValueError('Actual1024 update proof missing')
        output=Path(p['output']);proof=read(output/'resume_fullstate_proof.json');telemetry=read(output/'resume_Adam_telemetry.json');latest=read(output/'latest.json')
        if not proof['resume'] or latest['step']!=1056 or len(telemetry)!=32 or not any(x['Adam_updated']for x in telemetry):raise ValueError('Actual32-attempt full-state resume incomplete')
        if not all(x['combined_gradients_finite']for x in telemetry if x['Adam_updated']):raise ValueError('Nonfinite resumed gradients')
        for camera in ['DC','HC','KA']:
            if not (output/'renders/step-000001056'/f'{camera}.png').is_file():raise ValueError('Missing primaryV1 render')
        atomic(r/'technical_complete.json',dict(status='TECHNICAL_PASS_NO_PRODUCTION_RELEASE',time=time.time(),elapsed_seconds=time.monotonic()-start,peak_GPU_MiB=peak,children=children,policy_sha256=sha(policy_path),final_checkpoint_sha256=latest['checkpoint_sha256'],actual_resumed_Adam_updates=sum(x['Adam_updated']for x in telemetry),READY_created=False,production_authorized=False,quality_review_passed=False))
    except BaseException as error:
        if current is not None and current.poll() is None:
            os.killpg(current.pid,signal.SIGTERM)
            try:current.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(current.pid,signal.SIGKILL);current.wait()
        atomic(r/'technical_failure.json',dict(time=time.time(),error=repr(error),children=children,elapsed_seconds=time.monotonic()-start,peak_GPU_MiB=peak));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);a=p.parse_args();run(a.policy)
