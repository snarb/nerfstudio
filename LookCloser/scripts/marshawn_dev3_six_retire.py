"""Retire the exclusive six-map service at an actual completed-map boundary.

Stop only its CPU batch parent while the current immutable fitter finishes.
The fitter inherits the global lease and is never interrupted by this helper.
No unstarted map may be reassigned until the resulting retirement receipt.
"""
from pathlib import Path
import argparse, importlib, json, os, shlex, signal, subprocess, sys, time
import importlib.util


def read(path):
    return json.loads(Path(path).read_text())


def remote_retire(V):
    sys.path.insert(0, str(V / 'operations/future_sources/code'))
    old = importlib.import_module('marshawn_future_services')
    base = V / 'operations/future_frequency'; current = read(base / 'progress.json'); frame = current['frame']
    frame_root = base / ('%06d' % frame); identity_path = frame_root / 'worker_identity.json'
    identity = read(identity_path); parent = identity['controller']; child = identity.get('child')
    if old.alive(parent):
        args = Path('/proc/%d/cmdline' % parent['pid']).read_bytes()
        if b'marshawn_future_services.py' not in args or b'remote' not in args:
            raise ValueError('Not the exclusive DEV3 six-map CPU parent')
        os.kill(parent['pid'], signal.SIGSTOP)
    # Re-read after stopping the parent: its last atomic child identity is now
    # stable. A live child retains the inherited lease until its normal exit.
    identity = read(identity_path); child = identity.get('child')
    while old.alive(child):
        old.atomic(base / 'retiring_current_map.json', dict(frame=frame, controller=parent, child=child,
                   status='current_map_finishing_naturally', time=time.time()))
        time.sleep(2)
    # The last active camera's produced map must validate before parent death.
    camera = identity['camera']; spec = read(frame_root / (camera + '.json'))
    source = V / 'code/LookCloser/scripts/marshawn_video_frequency.py'
    module_spec = importlib.util.spec_from_file_location('_retire_map_validator', source)
    validator = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(validator)
    validator.validate_spec(spec)
    completed = validator.validate_existing(spec)
    if child is not None and not completed:
        if old.alive(parent):
            os.kill(parent['pid'], signal.SIGCONT)
        raise ValueError('Current map did not complete; parent resumed, no ownership reassignment permitted')
    if old.alive(parent):
        os.kill(parent['pid'], signal.SIGTERM); os.kill(parent['pid'], signal.SIGCONT)
    deadline = time.time() + 30
    while old.alive(parent) and time.time() < deadline:
        time.sleep(.25)
    if old.alive(parent) or old.alive(child):
        raise RuntimeError('Exclusive map process tree did not retire')
    lease = old.lock('/tmp/lookcloser_tuning_gpu_%d_0.lock' % os.getuid())
    if lease is None:
        raise RuntimeError('DEV3 GPU lease is still owned; do not reassign maps')
    lease.close()
    result = dict(kind='EXCLUSIVE_DEV3_SIX_RETIRED_AT_COMPLETED_MAP_V1', frame=frame, camera=camera,
                  controller=parent, child=child, parent_and_child_proved_dead=True, global_lease_free=True,
                  current_map_completed=completed, current_map_not_started=child is None,
                  current_map_sha256=old.sha(spec['output']) if completed else None,
                  current_metadata_sha256=old.sha(Path(spec['output']).with_suffix('.json')) if completed else None,
                  no_scientific_fitter_interrupted=True, time=time.time())
    old.atomic(base / 'exclusive_service_retired.json', result)
    print(json.dumps(result), flush=True)


def retire(V):
    sys.path.insert(0, str(V / 'operations/future_sources/code'))
    old = importlib.import_module('marshawn_future_services'); base = V / 'operations/future_frequency'
    controls = {}
    for name in ['watchdog_identity.json', 'controller_identity.json']:
        path = base / name
        if not path.exists():
            continue
        owner = read(path); controls[name] = owner
        if old.alive(owner):
            args = Path('/proc/%d/cmdline' % owner['pid']).read_bytes()
            if b'marshawn_future_services.py' not in args or b'decode' in args:
                raise ValueError('Refusing to signal a non-DEV3 CPU service')
            os.kill(owner['pid'], signal.SIGTERM)
    deadline = time.time() + 30
    while any(old.alive(row) for row in controls.values()) and time.time() < deadline:
        time.sleep(.25)
    if any(old.alive(row) for row in controls.values()):
        raise RuntimeError('DEV3 LOCAL control service still alive')
    remote = '/tmp/marshawn_fleet10_ops/marshawn_dev3_six_retire.py'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', 'ubuntu@dev3', 'mkdir -p /tmp/marshawn_fleet10_ops'], check=True)
    subprocess.run(['rsync', '-rt', __file__, 'ubuntu@dev3:' + remote], check=True)
    command = ['/home/ubuntu/anaconda3/envs/nerfstudio/bin/python', remote, '_remote', '--campaign', str(V)]
    subprocess.run(['ssh', '-o', 'BatchMode=yes', 'ubuntu@dev3', shlex.join(command)], check=True)
    result = read(base / 'exclusive_service_retired.json'); result['retired_LOCAL_controls'] = controls
    old.atomic(base / 'exclusive_service_retired.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['retire', '_remote']); p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); remote_retire(a.campaign) if a.action == '_remote' else retire(a.campaign)
