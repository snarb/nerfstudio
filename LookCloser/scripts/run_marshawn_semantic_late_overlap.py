#!/usr/bin/env python3
"""Bounded, separately released GPU3 technical gate for the frozen fork pair."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def atomic(p, value):
    p = Path(p); temp = p.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(temp,p)
def identity(pid):
    text = Path(f'/proc/{pid}/stat').read_text(); parts = text[text.rfind(')')+2:].split()
    return dict(pid=pid,start=parts[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=parts[0])
def validate(policy, release, digest):
    if release.get('approved_for_technical_overlap') is not True or release.get('policy_sha256') != digest:
        raise ValueError('Exact root release required')
    if policy['gpu'] != 3 or not 0 < policy['max_seconds'] <= 900 or not 0 < policy['max_process_mib'] <= 12288 or policy['allocator_cap_gib'] != 10 or policy['production_training'] is not False:
        raise ValueError('Unreviewed resource scope')
    if policy['checks_interval_seconds'] != 5 or policy['producer_stale_seconds'] != 180:
        raise ValueError('Unreviewed health check scope')
    if set(policy['specs']) != {'zero_control001','zero_treatment1','technical_control001','technical_treatment1'}:
        raise ValueError('Only two zero proofs and paired64 attempts')

def verify_zero(a,b):
    if any(v['status'] != 'zero_update_state_equal' for v in (a,b)): raise ValueError('Zero gate failed')
    for key in ('pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture','semantic_support','cache_sha256','cache_sources','sampler','next_step','Adam','FR'):
        if a['proof'][key] != b['proof'][key]: raise ValueError('Paired zero-state mismatch: '+key)

def main(root):
    root = Path(root); policy = read(root/'technical_policy.json')
    validate(policy,read(root/'technical_release.json'),sha(root/'technical_policy.json'))
    for path,digest in policy['artifacts_sha256'].items():
        if sha(path) != digest: raise ValueError('Artifact changed: '+path)
    specs = {k:read(v) for k,v in policy['specs'].items()}
    for key,spec in specs.items():
        if spec['mode'] != key.split('_')[0] or spec['distortion_coefficient'] != (.01 if key.endswith('control001') else 1.): raise ValueError('Wrong technical spec')
        release = read(spec['release'])
        if release.get('reviewer') != 'root' or release.get('technical_authorized') is not True or release.get('diagnostic_authorized') is not False or release.get('spec_sha256') != sha(policy['specs'][key]): raise ValueError('Missing technical-only child release')
    lock = open(f"/tmp/lookcloser_technical_overlap_{os.getuid()}_3.lock",'a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_uuid=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if gpu_uuid!=policy['gpu_uuid']:raise ValueError('GPU identity changed')
    expected = policy['producer_identity']; began = time.monotonic(); checks = []; proc = None
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=policy['code_root']+':'+policy['code_root']+'/LookCloser/scripts',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def check(phase):
        current = identity(expected['pid'])
        if any(current[k] != expected[k] for k in ('pid','start','boot')) or current['state'] == 'Z': raise RuntimeError('Producer identity changed or died')
        progress = read(policy['producer_progress'])
        if time.time()-progress['time'] > 180: raise RuntimeError('Producer progress stale')
        rows = []
        for line in subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).splitlines():
            pid,mib = map(int,line.split(',')); rows.append(dict(pid=pid,mib=mib))
        owned = {proc.pid} if proc is not None else set()
        if any(r['pid'] not in owned|{expected['pid']} for r in rows): raise RuntimeError('Unplanned GPU process')
        memory = sum(r['mib'] for r in rows if r['pid'] in owned)
        if memory > policy['max_process_mib']: raise RuntimeError('Candidate VRAM cap exceeded')
        if time.monotonic()-began > policy['max_seconds']: raise TimeoutError('Technical deadline')
        log = Path(policy['producer_log'])
        with log.open('rb') as f: f.seek(max(0,log.stat().st_size-65536)); tail=f.read()
        if b'CUDA out of memory' in tail or b'torch.OutOfMemoryError' in tail: raise RuntimeError('Producer OOM evidence')
        row = dict(time=time.time(),phase=phase,producer=current,producer_step=progress['step'],producer_progress_age=time.time()-progress['time'],gpu_rows=rows,candidate_mib=memory,producer_oom_evidence=False)
        checks.append(row)
        with (root/'technical_resource_checks.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        return row
    def run(name,argv,capped=False):
        nonlocal proc
        check('before_'+name)
        if capped:
            # Initialize only CUDA allocator; frozen worker subsequently restores its exact RNG.
            bootstrap = "import runpy,sys,torch;torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory);p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')"
            argv = ['-c',bootstrap,*argv]
        with (root/(name+'.log')).open('w') as log:
            proc = subprocess.Popen([policy['python'],*map(str,argv)],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=policy['code_root']+'/LookCloser',start_new_session=True)
            while proc.poll() is None: check(name); time.sleep(5)
            result = proc.returncode; check('after_'+name); proc=None
            if result: raise RuntimeError(f'{name} exit {result}')
    def terminate(*_): raise RuntimeError('Technical supervisor interrupted')
    signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
    try:
        first = check('initial'); worker=policy['worker']; runner=policy['runner']
        for key in ('zero_control001','zero_treatment1'): run(key,[worker,'--spec',policy['specs'][key]],True)
        zeros=[Path(specs[k]['output'])/'complete.json' for k in ('zero_control001','zero_treatment1')]
        verify_zero(*map(read,zeros))
        for label in ('control001','treatment1'):
            key='technical_'+label; spec=specs[key]; out=Path(spec['output'])
            # Fresh technical gate; no silent restarts or overwritten audit history.
            if (out/'latest.json').exists(): raise ValueError('Existing technical run requires explicit recovery review')
            run(key+'_first32',[worker,'--spec',policy['specs'][key],'--stop-after-attempts','32'],True)
            first_ckpt=out/'checkpoints/step-000030408.ckpt'
            if not first_ckpt.exists(): raise ValueError('Missing exact first32 checkpoint')
            run(key+'_resumed32',[worker,'--spec',policy['specs'][key]],True)
            if not (out/'complete.json').exists(): raise ValueError('Missing64-attempt completion')
            run(key+'_audit',[runner,'--worker',worker,'--spec',policy['specs'][key],'--zero-control',zeros[0],'--zero-treatment',zeros[1]])
        sequences=[Path(specs['technical_'+x]['output'])/'technical_sequence.json' for x in ('control001','treatment1')]
        run('pair_audit',[runner,'--audit-pair',*sequences])
        pair=json.loads((root/'pair_audit.log').read_text()); final=check('complete')
        if final['producer_step'] <= first['producer_step']: raise ValueError('No observed producer progress')
        atomic(root/'technical_complete.json',dict(passed=True,policy_sha256=sha(root/'technical_policy.json'),elapsed_seconds=time.monotonic()-began,max_candidate_mib=max(r['candidate_mib'] for r in checks),producer_before=first,producer_after=final,pair_audit=pair,production_authorized=False,quality_claim=False,historical_cache_order_observed=False,shared_producer_global_lock_untouched=True))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        atomic(root/'technical_failed.json',dict(error=repr(exc),elapsed_seconds=time.monotonic()-began,producer_modified=False)); raise

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
