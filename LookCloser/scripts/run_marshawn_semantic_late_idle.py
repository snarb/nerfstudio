#!/usr/bin/env python3
"""Released semantic30376 technical pair on exclusively idle GPU5; no queue handoff."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def verify_zero(a,b):
    if any(x['status']!='zero_update_state_equal'for x in (a,b)):raise ValueError('Both zero gates required')
    for key in ('pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture','semantic_support','cache_sha256','cache_sources','sampler','next_step','Adam','FR'):
        if a['proof'][key]!=b['proof'][key]:raise ValueError('Paired zero-state mismatch: '+key)



def validate(policy, release, digest):
    if release.get('approved_for_exclusive_technical') is not True or release.get('policy_sha256') != digest:
        raise ValueError('Exact exclusive technical release required')
    if policy['gpu'] != 5 or policy['max_seconds'] != 900 or policy['max_process_mib'] != 12288 or policy['allocator_cap_gib'] != 10:
        raise ValueError('Unreviewed resource limits')
    if policy['production_training'] is not False or policy['checks_interval_seconds'] != 5:
        raise ValueError('Technical-only scope required')
    if set(policy['specs']) != {'zero_control001','zero_treatment1','technical_control001','technical_treatment1'}:
        raise ValueError('Only zero plus32/32 paired technical gates')


def main(root):
    root=Path(root);policy=read(root/'technical_idle_policy.json')
    validate(policy,read(root/'technical_idle_release.json'),sha(root/'technical_idle_policy.json'))
    for path,digest in policy['artifacts_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen artifact changed: '+path)
    specs={k:read(p)for k,p in policy['specs'].items()}
    for name,spec in specs.items():
        if spec['mode']!=name.split('_')[0] or spec['distortion_coefficient']!=(.01 if name.endswith('control001') else 1.):raise ValueError('Technical spec mismatch')
        release=read(spec['release'])
        if release.get('reviewer')!='root' or release.get('technical_authorized')is not True or release.get('diagnostic_authorized')is not False or release.get('spec_sha256')!=sha(policy['specs'][name]):raise ValueError('Missing exact technical-only child release')
    lock=open(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_5.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if gpu[0].strip()!=policy['gpu_uuid'] or int(gpu[1])>512 or int(gpu[2])>5:raise ValueError('GPU5 is not freshly idle or identity changed')
    began=time.monotonic();proc=None;checks=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='5',PYTHONPATH=policy['code_root']+':'+policy['code_root']+'/LookCloser/scripts',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def check(phase):
        rows=[]
        for line in subprocess.check_output(['nvidia-smi','-i','5','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
            pid,mib=map(int,line.split(','));rows.append(dict(pid=pid,mib=mib))
        own={proc.pid}if proc is not None else set()
        if any(r['pid']not in own for r in rows):raise RuntimeError('Unexpected process on exclusively locked GPU5')
        memory=sum(r['mib']for r in rows)
        if memory>policy['max_process_mib']:raise RuntimeError('Candidate memory ceiling')
        if time.monotonic()-began>policy['max_seconds']:raise TimeoutError('Technical900s deadline')
        row=dict(time=time.time(),phase=phase,candidate_mib=memory,gpu_rows=rows)
        checks.append(row)
        with(root/'technical_idle_checks.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
    def run(name,argv):
        nonlocal proc
        check('before_'+name)
        with(root/(name+'.log')).open('x')as log:
            proc=subprocess.Popen([policy['python'],*map(str,argv)],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=policy['code_root']+'/LookCloser',start_new_session=True)
            while proc.poll()is None:check(name);time.sleep(5)
            result=proc.returncode;check('after_'+name);proc=None
            if result:raise RuntimeError(f'{name} exit {result}')
    def terminate(*_):raise RuntimeError('Technical supervisor interrupted')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        check('initial');worker=policy['worker'];runner=policy['runner']
        for key in specs:
            if Path(specs[key]['output']).exists():raise ValueError('Fresh technical outputs required; recovery needs review')
        for key in ('zero_control001','zero_treatment1'):run(key,[worker,'--spec',policy['specs'][key]])
        zeros=[Path(specs[k]['output'])/'complete.json'for k in ('zero_control001','zero_treatment1')];verify_zero(*map(read,zeros))
        for label in ('control001','treatment1'):
            key='technical_'+label;out=Path(specs[key]['output'])
            run(key+'_first32',[worker,'--spec',policy['specs'][key],'--stop-after-attempts','32'])
            if not(out/'checkpoints/step-000030408.ckpt').exists():raise ValueError('Exact first32 checkpoint absent')
            run(key+'_resumed32',[worker,'--spec',policy['specs'][key]])
            run(key+'_audit',[runner,'--worker',worker,'--spec',policy['specs'][key],'--zero-control',zeros[0],'--zero-treatment',zeros[1]])
        sequences=[Path(specs['technical_'+x]['output'])/'technical_sequence.json'for x in ('control001','treatment1')]
        run('pair_audit',[runner,'--audit-pair',*sequences]);check('complete')
        atomic(root/'technical_idle_complete.json',dict(passed=True,policy_sha256=sha(root/'technical_idle_policy.json'),elapsed_seconds=time.monotonic()-began,max_candidate_mib=max(r['candidate_mib']for r in checks),pair_audit=read(root/'pair_audit.log'),exclusive_global_lock_held=True,no_producer_interrupted=True,production_authorized=False,quality_claim=False))
    except BaseException as exc:
        if proc is not None and proc.poll()is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        atomic(root/'technical_idle_failed.json',dict(error=repr(exc),elapsed_seconds=time.monotonic()-began,other_processes_modified=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
