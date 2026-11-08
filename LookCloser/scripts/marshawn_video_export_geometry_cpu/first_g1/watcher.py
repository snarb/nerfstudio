"""Passive1695 completion gate, then one unchanged native generation1 export."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def current_gate(config):
    """No native checkpoint, GPU probe or lease before current job completion."""
    plan = read(Path(config['fleet']) / 'video_plan.json')
    state = read(Path(config['fleet']) / 'state.json')
    current = state['frames']['1695']
    job = state['jobs'][current['name']]
    if (current['name'] != 'frame001695_g01_a00'
            or current.get('model_generation') != 1 or current.get('source_generation') != 1
            or plan['frame_generations']['1695']['model_generation'] != 1
            or plan['frame_generations']['1695']['source_generation'] != 1):
        raise ValueError('First-g1 canary must follow the current corrected1695 claim')
    if job['status'] in ('failed', 'blocked', 'held_input_generation', 'orphaned'):
        raise ValueError('Current scientific generation requires review: ' + job['status'])
    return dict(ready=job['status'] == 'complete', current_job=current['name'],
                current_status=job['status'], frame=1695, index=7,
                no_early_checkpoint_allowed=True)


def main(args):
    O = args.root
    config_path = O / 'config.json'
    config = read(config_path)
    config_sha = sha(config_path)
    if config['poll_seconds'] < 30:
        raise ValueError('Passive polling must be at least30seconds')
    V6 = Path(config['generation_helpers'])
    sys.path[:0] = [str(V6)]
    import route as R
    import manifest as M
    def guard():
        if sha(config_path) != config_sha:
            raise ValueError('Live passive-canary config changed')
        for name, digest in config['files_sha256'].items():
            if sha(name) != digest:
                raise ValueError('Canary operational source changed: ' + name)
        release = read(V6 / 'root_release.json')
        if release.get('approved') is not True or sha(V6 / 'root_release.json') != config['generation_release_sha256']:
            raise ValueError('Released V6 binding changed')
        for name, digest in release['files_sha256'].items():
            if R.sha(name) != digest:
                raise ValueError('Frozen final renderer changed: ' + name)
        if R.sha(Path(config['fleet']) / 'video_plan.json') != release['fleet_plan_sha256']:
            raise ValueError('F2 plan differs from released generation routing')
    guard()
    if args.dry_run:
        return dict(passed=True, gate=current_gate(config), GPU_probe=False,
                    GPU_lease=False, GPU_or_shader_started=False,
                    released_helpers_unchanged=True)
    if config.get('authorized') is not True:
        raise ValueError('Passive canary not operationally released')
    lock = (O / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (O / 'complete.json').exists():
        return read(O / 'complete.json')
    while True:
        guard()
        gate = current_gate(config)
        write(O / 'status.json', dict(gate, phase='awaiting_genuine_current_g1_completion', time=time.time()))
        if gate['ready']:
            break
        time.sleep(config['poll_seconds'])
    plan = read(Path(config['fleet']) / 'video_plan.json')
    host = plan['hosts']['silver']
    # All following staging begins after scientific completion and uses no GPU.
    remote = Path(host['root']) / 'operations/video_first_g1_canary_v1'
    R.call(host, ['mkdir', '-p', str(remote)])
    subprocess.run(['rsync', '-a', str(V6 / 'collect.py'), str(V6 / 'policy.py'),
                    host['address'] + ':' + str(remote) + '/'], check=True)
    export = read(config['export_plan'])
    request = dict(read(export['path']), source_transforms='source_transforms.json')
    R.immutable(O / 'path.json', request)
    if sha(O / 'path.json') != config['canonical_path_sha256']:
        raise ValueError('Canary path is not byte-identical to the full final path')
    def collector(name):
        return remote / 'collect.py' if name == 'silver' else V6 / 'collect.py'
    prefix = M.build(Path(config['fleet']), request, collector, call=R.call, limit=8)
    row = prefix['records']['7']
    if row['assignment']['host'] != 'silver' or row['native_campaign'] != config['native_campaign']:
        raise ValueError('First g1 scientific model moved to an unqualified native host')
    R.immutable(O / 'prefix_manifest.json', prefix)
    local = O / 'control'
    local.mkdir(exist_ok=True)
    native_control = remote / 'control'
    for name in ['worker.py', 'worker_params.py', 'one.py', 'cap_loader.py', 'render_gate_loader.py',
                 'focal_scale.py', 'hybrid_params.json', 'root_release.json']:
        shutil.copy2(R.P / name, local / name)
    for name in ['entry.py', 'policy.py', 'verify.py']:
        shutil.copy2(V6 / name, local / name)
    shutil.copy2(O / 'path.json', local / 'path.json')
    shutil.copy2(O / 'prefix_manifest.json', local / 'manifest.json')
    source_tf = R.LOCAL / 'frames/001660/data/transforms.json'
    R.bound(source_tf, request['source_transforms_sha256'])
    shutil.copy2(source_tf, local / 'source_transforms.json')
    spec = dict(read(export['spec']), geometry_output=str(Path(host['root']) / 'video/geometry_cpu_float_v1'),
                frame_root_template=str(Path(row['native_campaign']) / 'frames/{frame_id}'),
                data_template='/__generation_entry_required__/{frame_id}/data',
                render_profile='frozen_R0_s8_g1', ray_chunk=2048,
                generation_manifest=str(native_control / 'manifest.json'),
                generation_manifest_sha256=sha(O / 'prefix_manifest.json'),
                source_visibility_adapter_sha256=R.sha(R.P / 'one.py'),
                legacy_hybrid_root=str(R.SOURCE_REMOTE / 'video/reference/legacy'),
                photometry_fit=str(R.SOURCE_REMOTE / 'video/reference/photometry_fit.json'))
    R.immutable(local / 'spec.json', spec)
    R.call(host, ['mkdir', '-p', str(native_control)])
    subprocess.run(['rsync', '-a', str(local) + '/', host['address'] + ':' + str(native_control) + '/'], check=True)
    sys.path[:0] = [str(R.LOCAL / 'code'), str(R.LOCAL / 'code/LookCloser/scripts')]
    import marshawn_tuning_controller as fleet
    command = [host['python'], str(native_control / 'entry.py'), '--manifest', str(native_control / 'manifest.json'),
               '--worker', str(native_control / 'worker.py'), '--', 'gpu',
               '--code', str(Path(row['native_campaign']) / 'code'), '--spec', str(native_control / 'spec.json'),
               '--path', str(native_control / 'path.json'), '--index', '7']
    name = 'geometry_first_g1_0007_v1'
    render_plan = dict(schema_version=1, poll_seconds=config['poll_seconds'],
                       hosts={'silver': dict(host, root=str(remote / 'dispatch'))},
                       jobs=[dict(name=name, host='silver', command=command, cwd=str(native_control),
                                  env={'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})])
    R.immutable(O / 'render_plan.json', render_plan)
    state = read(O / 'state.json') if (O / 'state.json').exists() else dict(manifest_sha256=fleet.digest(render_plan), jobs={name: dict(status='pending')})
    if state['manifest_sha256'] != fleet.digest(render_plan):
        raise ValueError('Existing canary GPU owner belongs to another request')
    while state['jobs'][name]['status'] != 'complete':
        guard()
        fleet.tick(render_plan, state, O)
        item = state['jobs'][name]
        if item.get('status') == 'orphaned' and item.get('worker_alive'):
            item.update(status='unknown', adoption_wait='Original worker alive; no replacement')
        elif item.get('status') in ('orphaned', 'failed') and item.get('worker_alive') is False and item.get('supervisor_alive') is False:
            reply = R.call(host, [host['python'], str(native_control / 'verify.py'), '--directory',
                           str(Path(host['root']) / 'video/geometry_cpu_float_v1/frames/0007'), '--manifest',
                           str(native_control / 'manifest.json'), '--path', str(native_control / 'path.json'), '--index', '7'], timeout=90)
            R.artifact_outcome(item, json.loads(reply.stdout), index=7, manifest_sha=sha(O / 'prefix_manifest.json'),
                               path_sha=sha(O / 'path.json'), verifier_sha=R.sha(V6 / 'verify.py'), policy_sha=R.sha(V6 / 'policy.py'))
        write(O / 'state.json', state)
        write(O / 'status.json', dict(phase='genuine_complete_native_geometry_gate', GPU_status=item['status'], time=time.time()))
        if item['status'] in ('blocked', 'failed'):
            raise ValueError('First g1 native export requires review: ' + str(item))
        if item['status'] != 'complete':
            time.sleep(config['poll_seconds'])
    destination = R.LOCAL / 'video/geometry_cpu_float_v1/frames/0007'
    destination.mkdir(exist_ok=True)
    if (destination / 'geometry_complete.json').exists():
        R.geometry_check(destination, row, request, sha(O / 'path.json'))
    source = Path(host['root']) / 'video/geometry_cpu_float_v1/frames/0007'
    subprocess.run(['rsync', '-a', '--exclude=GPU.lock', host['address'] + ':' + str(source) + '/', str(destination) + '/'], check=True)
    proof = R.geometry_check(destination, row, request, sha(O / 'path.json'))
    geometry = read(destination / 'geometry_complete.json')
    archive = geometry['native_HC_parity']['archive_path']
    subprocess.run(['rsync', '-a', host['address'] + ':' + archive, str(O / 'selected_HC.png')], check=True)
    R.bound(O / 'selected_HC.png', geometry['native_HC_parity']['archive_sha256'])
    done = dict(kind='GENUINE_FIRST_G1_NATIVE_EXPORT_RUNTIME_QUALIFICATION_V1', frame=1695, index=7,
                completed_iterations=140000, checkpoint_sha256=row['checkpoint_sha256'],
                selected_step=row['selected_step'], geometry=proof,
                canonical_geometry_directory=str(destination),
                full_canonical_path_sha256=sha(O / 'path.json'), prefix_manifest_sha256=sha(O / 'prefix_manifest.json'),
                native_HC_parity=geometry['native_HC_parity'],
                tensor_SHA_before=geometry['pipeline_tensor_SHA_before'], tensor_SHA_after=geometry['pipeline_tensor_SHA_after'],
                physical_GPU=geometry['physical_GPU'], scientific_worker_PID=geometry['PID'],
                selected_HC_copy_sha256=sha(O / 'selected_HC.png'),
                original_six_packages_unchanged=True, ROOT_visual_review_required=True, finished=time.time())
    write(O / 'complete.json', done)
    write(O / 'ROOT_review_ready.json', done)
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    arguments = parser.parse_args()
    try:
        print(json.dumps(main(arguments), indent=2))
    except (OSError, subprocess.SubprocessError) as error:
        write(arguments.root / 'transport_wait.json', dict(error=str(error), time=time.time(), no_GPU_preemption=True))
        raise SystemExit(75)
    except (ValueError, KeyError, TypeError) as error:
        write(arguments.root / 'review_required.json', dict(error=str(error), time=time.time()))
        raise SystemExit(2)
