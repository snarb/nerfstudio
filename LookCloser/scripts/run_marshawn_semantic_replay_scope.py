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
    if release.get('approved_for_readonly_overlap') is not True or release.get('policy_sha256') != digest:
        raise ValueError('Exact root release required')
    if policy['gpu'] != 3 or not 0 < policy['max_seconds'] <= 600 or not 0 < policy['max_process_mib'] <= 12288 or policy['allocator_cap_gib'] != 10 or policy['production_training'] is not False:
        raise ValueError('Unreviewed resource scope')
    if policy['checks_interval_seconds'] != 5 or policy['producer_stale_seconds'] != 180:
        raise ValueError('Unreviewed health check scope')
    if set(policy['specs']) != {'original','fork'}:
        raise ValueError('Only two read-only zero-update replays')

def verify_zero(a,b):
    if any(v['status'] != 'zero_update_state_equal' for v in (a,b)): raise ValueError('Zero gate failed')
    for key in ('pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture','semantic_support','cache_sha256','cache_sources','sampler','next_step','Adam','FR'):
        if a['proof'][key] != b['proof'][key]: raise ValueError('Paired zero-state mismatch: '+key)

def main(root):
    root = Path(root); policy = read(root/'readonly_policy.json')
    validate(policy,read(root/'readonly_release.json'),sha(root/'readonly_policy.json'))
    for path,digest in policy['artifacts_sha256'].items():
        if sha(path) != digest: raise ValueError('Artifact changed: '+path)
    specs = {k:read(v) for k,v in policy['specs'].items()}
    for key,spec in specs.items():
        if spec['mode']!='zero' or spec['construction']!=key or spec['distortion_coefficient']!=.01:raise ValueError('Only original/fork read-only replays')
        release = read(spec['release'])
        if release.get('reviewer') != 'root' or release.get('read_only_authorized') is not True or release.get('technical_authorized') is not False or release.get('diagnostic_authorized') is not False or release.get('spec_sha256') != sha(policy['specs'][key]): raise ValueError('Missing technical-only child release')
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
        with (root/'readonly_resource_checks.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
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
        first=check('initial');worker=policy['worker']
        for key in ('original','fork'):
            if Path(specs[key]['output']).exists():raise ValueError('Fresh read-only output required')
            run(key,[worker,'--spec',policy['specs'][key]],True)
        results={k:read(Path(v['output'])/'complete.json')for k,v in specs.items()}
        keys=('pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture','semantic_support','cache_sha256','cache_sources','sampler','next_step','Adam','FR')
        for key in keys:
            if results['original']['proof'][key]!=results['fork']['proof'][key]:raise ValueError('Restored paired states differ: '+key)
        from PIL import Image
        import numpy as np
        differences={}
        for camera in ('DC','HC','KA'):
            for suffix in ('.png','_alpha.png'):
                name=camera+suffix
                images=[np.asarray(Image.open(Path(specs[k]['output'])/'renders/step-000030376'/name)).astype(np.float64)for k in ('original','fork')]
                delta=np.abs(images[0]-images[1]);differences[name]=dict(max_code_delta=float(delta.max()),mean_code_delta=float(delta.mean()),changed_values=int(np.count_nonzero(delta)))
        metric_deltas={k:results['fork']['zero_metrics']['metrics'][k]-results['original']['zero_metrics']['metrics'][k]for k in ('psnr','ssim','lpips')}
        final=check('complete')
        if final['producer_step']<=first['producer_step']:raise ValueError('Producer progress not observed')
        atomic(root/'readonly_complete.json',dict(completed=True,optimizer_updates=0,policy_sha256=sha(root/'readonly_policy.json'),elapsed_seconds=time.monotonic()-began,max_candidate_mib=max(r['candidate_mib']for r in checks),producer_before=first,producer_after=final,fullstate_equal=True,paired_pixel_deltas=differences,paired_metric_deltas=metric_deltas,runtime_attributes={k:v['runtime_attributes']for k,v in results.items()},numeric_tolerance_approved=False,cause_established=False,production_authorized=False))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        atomic(root/'readonly_failed.json',dict(error=repr(exc),elapsed_seconds=time.monotonic()-began,producer_modified=False)); raise

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
