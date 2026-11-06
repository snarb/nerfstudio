#!/usr/bin/env python3
"""Durable CPU admission and completion audit for the reviewed experimental maps."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p); temporary = p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(temporary, p)


def qualified_job(job, now):
    if not job.get('worker', {}).get('pid') or not job.get('supervisor', {}).get('pid'):
        return False
    if not job.get('spec_sha256') or not job.get('started'):
        return False
    if job.get('status') == 'complete':
        return bool(job.get('finished'))
    return (job.get('status') == 'running' and job.get('worker_alive') is True
            and job.get('supervisor_alive') is True
            and 0 <= now-float(job.get('heartbeat', 0)) <= 120)


def ssh_json(address, code):
    r = subprocess.run(['ssh', address, 'python3 -'], input=code, text=True,
                       capture_output=True, timeout=240, check=True)
    return json.loads(r.stdout)


def admission_states(release):
    result = {}
    for source in release['admission_sources']:
        if source['host'] == 'local':
            state = read(source['state'])
        else:
            state = ssh_json(source['host'], 'import pathlib,json\nprint(pathlib.Path('+repr(source['state'])+').read_text())')
        for name in source['required_jobs']:
            result[name] = state.get('jobs', {}).get(name, {})
    return result


def validate_release(root):
    release = read(root/'root_release.json')
    if (release.get('reviewer') != 'root' or release.get('frequency_fit_authorized') is not True
            or release.get('admission_required') is not True):
        raise ValueError('Explicit conditional root release required')
    for file, key in [('production_plan.json', 'plan_sha256'), ('preparation.json', 'inventory_sha256'), ('release_templates.json', 'release_templates_sha256')]:
        if sha(root/file) != release[key]: raise ValueError('Released source changed: '+file)
    if sha(root/'source_transfer_inventory.json') != release['source_transfer_inventory_sha256']:
        raise ValueError('Source transfer inventory changed')
    preparation = read(root/'preparation.json')
    for file, digest in preparation['inventory'].items():
        if sha(root/file) != digest: raise ValueError('Frozen inventory changed: '+file)
    for camera, expected in read(root/'release_templates.json').items():
        actual = read(root/camera/'root_release.json')
        for key in ('spec_sha256', 'source_complete_sha256', 'reviewer'):
            if actual.get(key) != expected[key]: raise ValueError('Camera release mismatch')
        if actual.get('frequency_fit_authorized') is not True: raise ValueError('Camera fit not authorized')
    return release


def aggregate(root, plan):
    # A separate read-only CPU auditor runs beside the maps. It never writes READY.
    host = plan['hosts']['silver']
    payload = {'root': host['root'], 'generation': root.name, 'jobs': [j['name'] for j in plan['jobs']], 'source_inventory': read(root/'source_transfer_inventory.json')}
    script = (root/'aggregate_frequency_outputs.py').read_text()
    code = 'PAYLOAD='+repr(payload)+'\n'+script
    r = subprocess.run(['ssh', host['address'], host['python']+' -'], input=code,
                       text=True, capture_output=True, timeout=300, check=True)
    result = json.loads(r.stdout)
    write(root/'aggregate.json', result)


def watch(root):
    root = root.resolve()
    lock = (root/'admission_watcher.lock').open('a+')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: return
    release = validate_release(root)
    release_hash = sha(root/'root_release.json')
    plan = read(root/'production_plan.json')
    campaign = root.parent
    controller = campaign/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'
    state_root = campaign/(root.name+'_controller')
    write(root/'admission_watcher.json', {'pid': os.getpid(), 'started': time.time(), 'code_sha256': sha(__file__), 'release_sha256': release_hash})
    previous, last_log = None, 0.
    while True:
        try:
            if sha(root/'root_release.json') != release_hash: raise ValueError('Release changed during watch')
            validate_release(root)
            receipt = root/'admission_receipt.json'
            if not receipt.exists():
                jobs = admission_states(release)
                ready = len(jobs) == 7 and all(qualified_job(j, time.time()) for j in jobs.values())
                status = {'stage': 'admission_ready' if ready else 'waiting_training_initial_dispatch', 'jobs': {n: j.get('status', 'missing') for n, j in jobs.items()}}
                if ready:
                    write(receipt, {'admitted_at': time.time(), 'release_sha256': release_hash, 'watcher_sha256': sha(__file__), 'jobs': jobs})
            if receipt.exists():
                if read(receipt)['release_sha256'] != release_hash: raise ValueError('Admission bound to another release')
                state = read(state_root/'state.json') if (state_root/'state.json').exists() else {}
                jobs = state.get('jobs', {})
                status = {'stage': 'fitting_experimental_maps', 'jobs': {n:j['status'] for n,j in jobs.items()}}
                if len(jobs) == 63 and all(j['status'] == 'complete' for j in jobs.values()):
                    aggregate(root, plan)
                    write(root/'admission_status.json', {'stage': 'experimental_maps_complete_not_promoted', 'time': time.time(), 'aggregate_sha256': sha(root/'aggregate.json')})
                    return
                if any(j['status'] in ('failed', 'blocked', 'orphaned') for j in jobs.values()):
                    status['stage'] = 'requires_failure_review'
                else:
                    # Idempotent launch resumes a dead dispatcher, never healthy workers.
                    subprocess.run([str(campaign/'python'), str(controller), 'launch', '--root', str(state_root), '--manifest', str(root/'production_plan.json')], check=True, capture_output=True, text=True)
            key = json.dumps(status, sort_keys=True)
            if key != previous or time.time()-last_log >= 3600:
                with (root/'admission_checks.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(), **status))+'\n')
                previous, last_log = key, time.time()
            write(root/'admission_status.json', dict(time=time.time(), **status))
        except Exception as exc:
            write(root/'admission_status.json', {'stage': 'blocked_error', 'time': time.time(), 'error': repr(exc)})
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    a = parser.parse_args()
    if a.detach:
        with (a.root/'admission_stdout.log').open('ab') as log:
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(a.root.resolve())], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'watcher_pid': p.pid}))
    else: watch(a.root)
