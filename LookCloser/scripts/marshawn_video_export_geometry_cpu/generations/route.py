"""Generation-aware operational transport around the unchanged final shaders.

Runs only after all72 scientific jobs complete. Three physical host rows remain
three host rows; generation roots are inputs, never extra GPU capacity.
"""
from pathlib import Path
import argparse
import fcntl
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

from policy import (CPU_SHA, GPU_SHA, LOCAL, PROFILE_SHA, SEAL, SILVER,
                    bound, geometry_check, manifest_row, read, sha)
from manifest import build

P = LOCAL / 'operations/video_geometry_cpu_params_v2'
D = LOCAL / 'operations/video_geometry_cpu_durable_v3'
E = LOCAL / 'operations/video_geometry_cpu_encoding_v4'
UUID = LOCAL / 'operations/video_dev3_uuid_v5/dev3_entry.py'
UUID_SHA = '7da7311d19b4779483cec4423e430412754367ac2807dea9366d4226e71dd5cc'
SOURCE_REMOTE = Path('/cache/ubuntu/lookcloser/campaigns/marshawn_video_1660_2015_24fps_20261107')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def immutable(path, value):
    blob = (json.dumps(value, indent=2) + '\n').encode()
    if Path(path).exists():
        if Path(path).read_bytes() != blob:
            raise ValueError('Refuse changing active generation control: ' + str(path))
    else:
        write(path, value)


def call(host, command, **kwargs):
    argv = (['ssh', '-o', 'BatchMode=yes', host['address'], shlex.join(command)]
            if host.get('address') else command)
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value



def artifact_outcome(item, proof, *, index, manifest_sha, path_sha, verifier_sha, policy_sha):
    """Called only after both original GPU process identities are proven dead."""
    expected = dict(kind='SOURCE_BOUND_NATIVE_GEOMETRY_VERIFICATION_V2', index=index,
                    manifest_sha256=manifest_sha, path_sha256=path_sha,
                    verifier_sha256=verifier_sha, policy_sha256=policy_sha)
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError('Remote native artifact verification belongs to another request/source')
    if proof.get('verified') is True:
        item.pop('review_required', None)
        item.pop('review_reason', None)
        item.update(status='complete', completion_source='verified_own_artifact_after_supervisor_loss', geometry_proof=proof)
    elif proof.get('verified') is False and proof.get('review_required') is True:
        item.update(status='blocked', review_required=True,
                    review_reason='Original GPU worker and supervisor are dead; own native artifact is incomplete/invalid',
                    artifact_verification=proof)
    else:
        raise ValueError('Remote verifier returned no authoritative artifact outcome')


def main(args):
    O = Path(__file__).parent
    lock = (O / 'postprocess.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release_path = O / 'root_release.json'
    release = read(release_path)
    release_sha = sha(release_path)
    if release.get('approved') is not True:
        raise ValueError('Generation-aware export ROOT release required')
    def guard():
        bound(release_path, release_sha)
        for name, digest in release['files_sha256'].items():
            bound(name, digest)
        bound(args.fleet / 'video_plan.json', release['fleet_plan_sha256'])
    guard()
    if not 1 <= args.cpu_workers <= 3:
        raise ValueError('Qualified CPU capacity1..3')
    if (O / 'delivery_complete.json').exists():
        for name, digest in read(O / 'delivery_complete.json')['artifacts_sha256'].items():
            bound(name, digest)
        return
    plan = read(args.fleet / 'video_plan.json')
    hosts = plan['hosts']
    if set(hosts) != {'silver', 'olive', 'dev3'}:
        raise ValueError('Exactly three physical hosts required')
    export = read(args.export_plan)
    request = read(export['path'])
    if request['version'] != 4 or request.get('paused_GT_authorized') is not True:
        raise ValueError('Only released72neural+48paused2020 path accepted')
    request = dict(request, source_transforms='source_transforms.json')
    immutable(O / 'canonical_path.json', request)
    canonical = O / 'canonical_path.json'
    source_tf = LOCAL / 'frames/001660/data/transforms.json'
    bound(source_tf, request['source_transforms_sha256'])
    if (O / 'source_transforms.json').exists():
        bound(O / 'source_transforms.json', sha(source_tf))
    else:
        shutil.copy2(source_tf, O / 'source_transforms.json')
    out = LOCAL / 'video/geometry_cpu_delivery_params_v2'
    gather = LOCAL / 'video/geometry_cpu_float_v1'
    for directory in [out, out / 'frames', gather, gather / 'frames']:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o777)
    # Stage small operational metadata collectors before reserving any GPU.
    collection = {}
    for name, host in hosts.items():
        root = Path(host['root']) / 'operations/video_generation_v6'
        collection[name] = root / 'collect.py'
        call(host, ['mkdir', '-p', str(root)])
        if name == 'silver':
            subprocess.run(['rsync', '-a', str(O / 'collect.py'), str(O / 'policy.py'),
                            host['address'] + ':' + str(root) + '/'], check=True)
        elif root != O:
            raise ValueError('Shared operational root differs')
    manifest = build(args.fleet, request, lambda name: collection[name], call=call)
    if manifest['partial_qualification_only'] or len(manifest['records']) != 72:
        raise ValueError('Final export requires all72 genuine generation claims/completions')
    immutable(O / 'generation_manifest.json', manifest)
    manifest_path = O / 'generation_manifest.json'
    manifest_sha = sha(manifest_path)
    helper_names = ['worker.py', 'worker_params.py', 'one.py', 'cap_loader.py',
                    'render_gate_loader.py', 'focal_scale.py', 'hybrid_params.json', 'root_release.json']
    bound(P / 'worker.py', GPU_SHA)
    bound(P / 'worker_params.py', CPU_SHA)
    bound(P / 'hybrid_params.json', PROFILE_SHA)
    controls = {}
    base = read(export['spec'])
    for name, host in hosts.items():
        for generation in [0, 1]:
            # Same host entry and scheduler capacity, different immutable I/O spec.
            local = O / 'controls' / (name + '_g%d' % generation)
            native_control = Path(host['root']) / 'operations/video_generation_v6/controls' / (name + '_g%d' % generation)
            local.mkdir(parents=True, exist_ok=True)
            for helper in helper_names:
                destination = local / helper
                if destination.exists():
                    bound(destination, sha(P / helper))
                else:
                    shutil.copy2(P / helper, destination)
            for helper in ['entry.py', 'policy.py', 'verify.py']:
                shutil.copy2(O / helper, local / helper)
            shutil.copy2(UUID, local / 'dev3_entry.py')
            immutable(local / 'path.json', request)
            immutable(local / 'manifest.json', manifest)
            shutil.copy2(source_tf, local / 'source_transforms.json')
            root = Path(host['root'])
            if generation:
                from policy import GEN1, SGEN1
                root = SGEN1 if name == 'silver' else GEN1
            spec = dict(base, output=str(out),
                        geometry_output=str(Path(host['root']) / 'video/geometry_cpu_float_v1'),
                        gathered_geometry=str(gather),
                        frame_root_template=str(root / 'frames/{frame_id}'),
                        data_template='/__generation_entry_required__/{frame_id}/data',
                        render_profile='frozen_R0_s8_g1', ray_chunk=2048,
                        generation_manifest=str(native_control / 'manifest.json'),
                        generation_manifest_sha256=manifest_sha,
                        source_visibility_adapter_sha256=sha(P / 'one.py'))
            if name == 'silver':
                spec.update(legacy_hybrid_root=str(SOURCE_REMOTE / 'video/reference/legacy'),
                            photometry_fit=str(SOURCE_REMOTE / 'video/reference/photometry_fit.json'))
            immutable(local / 'spec.json', spec)
            call(host, ['mkdir', '-p', str(native_control)])
            if name == 'silver':
                subprocess.run(['rsync', '-a', str(local) + '/',
                                host['address'] + ':' + str(native_control) + '/'], check=True)
            controls[(name, generation)] = native_control
    # One global CPU specSHA across all120 final images. Per-record resolution
    # occurs only inside entry.py; original shader metadata truthfully binds it.
    cpu_control = controls[('olive', 0)]
    cpu_spec = read(cpu_control / 'spec.json')
    cpu_spec.update(generation_manifest=str(manifest_path))
    cpu_spec_path = O / 'CPU_spec.json'
    immutable(cpu_spec_path, cpu_spec)
    sys.path[:0] = [str(LOCAL / 'code'), str(LOCAL / 'code/LookCloser/scripts'), str(D), str(E)]
    import marshawn_tuning_controller as fleet
    import marshawn_video_render as renderer
    import cpu_exec
    import encoder
    encoder.install(renderer, E / 'root_release.json')
    generic = {name: dict(host, root=str(Path(host['root']) / 'operations/video_generation_v6/dispatch'))
               for name, host in hosts.items()}
    jobs = []
    for key, row in manifest['records'].items():
        index = int(key)
        host = hosts[row['assignment']['host']]
        control = controls[(row['assignment']['host'], int(row['generation']['model_generation']))]
        command = [host['python'], str(control / 'entry.py'), '--manifest', str(control / 'manifest.json'),
                   '--worker', str(control / 'worker.py')]
        if row['assignment']['host'] == 'dev3':
            command += ['--dev3-entry', str(control / 'dev3_entry.py'), '--dev3-sha', UUID_SHA]
        command += ['--', 'gpu', '--code', str(Path(row['native_campaign']) / 'code'),
                    '--spec', str(control / 'spec.json'), '--path', str(control / 'path.json'),
                    '--index', str(index)]
        jobs.append(dict(name='geometry_cpu_v1_%04d' % index, host=row['assignment']['host'],
                         command=command, cwd=str(control),
                         env={'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}))
    render_plan = dict(schema_version=1, poll_seconds=5, hosts=generic, jobs=jobs)
    immutable(O / 'render_plan.json', render_plan)
    authority = O / 'state.json'
    gpu = read(authority) if authority.exists() else dict(jobs={job['name']: dict(status='pending') for job in jobs})
    gathered = set(read(O / 'gathered.json')['indices']) if (O / 'gathered.json').exists() else set()
    cpuenv = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    # All existing canonical packages (notably first6) are verified before jobs
    # launch. Failed checks stop; no foreign cache is relabelled or overwritten.
    for key, row in manifest['records'].items():
        directory = gather / 'frames' / ('%04d' % int(key))
        if (directory / 'geometry_complete.json').exists():
            proof = geometry_check(directory, row, request, sha(canonical))
            gpu['jobs']['geometry_cpu_v1_%04d' % int(key)].update(status='complete', completion_source='exact_existing_canonical_geometry', geometry_proof=proof)
            gathered.add(int(key))
    write(authority, gpu)
    while True:
        guard()
        # fleet.tick writes authoritative state.json before remote side effects.
        # Always reload it after a coordinator crash, not a stale GPU_state copy.
        fleet.tick(render_plan, gpu, O)
        # Supervisor transport loss cannot turn an extant GPU worker into a
        # replacement attempt. A fully bound own artifact is authoritative only
        # once neither the old worker nor its supervisor is alive.
        for job in jobs:
            item = gpu['jobs'][job['name']]
            if item.get('status') == 'orphaned' and item.get('worker_alive'):
                item.update(status='unknown', adoption_wait='Original GPU worker still alive')
            elif (item.get('status') in ('orphaned', 'failed')
                  and item.get('worker_alive') is False
                  and item.get('supervisor_alive') is False):
                index = int(job['name'].rsplit('_', 1)[1])
                row = manifest['records'][str(index)]
                host = hosts[row['assignment']['host']]
                control = controls[(row['assignment']['host'], int(row['generation']['model_generation']))]
                directory = Path(host['root']) / 'video/geometry_cpu_float_v1/frames' / ('%04d' % index)
                try:
                    reply = call(host, [host['python'], str(control / 'verify.py'), '--directory', str(directory), '--manifest', str(control / 'manifest.json'), '--path', str(control / 'path.json'), '--index', str(index)], timeout=90)
                    proof = json.loads(reply.stdout)
                except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as error:
                    item.update(status='unknown', adoption_wait='Cannot prove original GPU artifact/transport', transport_error=str(error))
                    continue
                artifact_outcome(item, proof, index=index,
                                 manifest_sha=manifest_sha, path_sha=sha(canonical),
                                 verifier_sha=sha(O / 'verify.py'), policy_sha=sha(O / 'policy.py'))
        write(authority, gpu)
        for key, row in manifest['records'].items():
            index = int(key)
            state = gpu['jobs']['geometry_cpu_v1_%04d' % index]
            if index in gathered or state.get('status') != 'complete':
                continue
            host = hosts[row['assignment']['host']]
            source = Path(host['root']) / 'video/geometry_cpu_float_v1/frames' / ('%04d' % index)
            destination = gather / 'frames' / ('%04d' % index)
            if row['assignment']['host'] == 'silver':
                destination.mkdir(exist_ok=True)
                subprocess.run(['rsync', '-a', '--exclude=GPU.lock', host['address'] + ':' + str(source) + '/', str(destination) + '/'], check=True)
            geometry_check(destination, row, request, sha(canonical))
            gathered.add(index)
            write(O / 'gathered.json', dict(indices=sorted(gathered)))
        complete = set()
        running = []
        for index in sorted(gathered):
            directory = out / 'frames' / ('%04d' % index)
            marker = directory / 'complete.json'
            if marker.exists():
                receipt = read(marker)
                if receipt['binding']['path_sha256'] != sha(canonical) or receipt['binding']['spec_sha256'] != sha(cpu_spec_path):
                    raise ValueError('Existing delivery belongs to another generation spec; explicit migration required')
                for name, digest in receipt['images'].items():
                    bound(directory / name, digest)
                complete.add(index)
                continue
            # Retain the qualified canonical ledger/capacity locks. A foreign
            # pre-existing intent is rejected by exact-source prepare/adoption.
            ledger = D / 'CPU_exec' / ('%04d' % index)
            if (ledger / 'intent.json').exists():
                intent = read(ledger / 'intent.json')
                if cpu_exec.source(intent['source']['command'], intent['source']['input_sha256']) != intent['source']:
                    raise ValueError('CPU prelaunch source changed')
                runtime = cpu_exec.runtime_for(ledger, intent)
                if runtime and cpu_exec.live(runtime):
                    running.append(index)
        for index in sorted(gathered - complete - set(running)):
            if len(running) >= args.cpu_workers:
                break
            row = manifest['records'][str(index)]
            command = [hosts['olive']['python'], str(O / 'entry.py'), '--manifest', str(manifest_path),
                       '--worker', str(cpu_control / 'worker_params.py'), '--', 'cpu',
                       '--code', str(LOCAL / 'code'), '--spec', str(cpu_spec_path),
                       '--path', str(canonical), '--index', str(index)]
            inputs = [O / 'entry.py', O / 'policy.py', manifest_path, cpu_spec_path, canonical,
                      cpu_control / 'worker_params.py', cpu_control / 'hybrid_params.json',
                      cpu_control / 'root_release.json', cpu_control / 'focal_scale.py', cpu_control / 'one.py',
                      gather / 'frames' / ('%04d' % index) / 'geometry_complete.json',
                      *map(Path, row['local_source_inputs_sha256'])]
            ledger = D / 'CPU_exec' / ('%04d' % index)
            intent = cpu_exec.prepare(ledger, command, inputs, index=index, max_concurrent=args.cpu_workers)
            runtime = cpu_exec.runtime_for(ledger, intent)
            if runtime and runtime['exec_attempt'] >= 3:
                raise RuntimeError('Repeated actual CPU shader failures: ' + str(index))
            directory = out / 'frames' / ('%04d' % index)
            directory.mkdir(exist_ok=True)
            with (directory / 'CPU_generation_stdout.log').open('ab') as log:
                subprocess.Popen([hosts['olive']['python'], str(D / 'cpu_exec.py'), '--intent', str(ledger / 'intent.json'), '--expected-sha256', sha(ledger / 'intent.json')],
                                 cwd=cpu_control, env=cpuenv, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # The immutable PRElaunch intent is already durable. A crash before
            # runtime acknowledgment is safe: same execution lock, no duplicate
            # shader, and only actual exec attempts count against the limit.
            running.append(index)
        write(O / 'status.json', dict(time=time.time(), geometry_gathered=len(gathered), CPU_complete=len(complete), CPU_running=len(running), generation_manifest_sha256=manifest_sha))
        if len(complete) == 72:
            break
        failed = [name for name, item in gpu['jobs'].items() if item.get('status') in ('failed', 'blocked')]
        if failed:
            raise RuntimeError('Native generation geometry requires review: ' + ','.join(failed))
        time.sleep(5)
    for index in range(72, 120):
        with (O / 'GT_CPU.log').open('ab') as log:
            subprocess.run([hosts['olive']['python'], str(O / 'entry.py'), '--manifest', str(manifest_path), '--worker', str(cpu_control / 'one.py'), '--', 'gt', '--code', str(LOCAL / 'code'), '--spec', str(cpu_spec_path), '--path', str(canonical), '--index', str(index)], cwd=cpu_control, env=cpuenv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    renderer.encode(dict(cpu_spec, _spec_path=str(cpu_spec_path)), canonical)
    artifacts = {}
    for kind, label in [('lookcloser', 'LookCloser'), ('hybrid', 'Hybrid')]:
        source = out / ('Marshawn1660_3s_24fps_%s_native_GT2s.mp4' % kind)
        target = out / ('Marshawn1660_3sNeural_2sPaused2020_24fps_%s_4K.mp4' % label)
        if not target.exists():
            os.link(source, target)
        bound(target, sha(source))
        artifacts[str(target)] = sha(target)
    write(O / 'delivery_complete.json', dict(artifacts_sha256=artifacts,
          generation_manifest_sha256=manifest_sha, output_frames=120, fps=24,
          manual_final_visual_review_required=True, shader_and_profile_unchanged=True,
          CP_and_YAML_native_host_preserved=True, finished=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fleet', type=Path, required=True)
    p.add_argument('--export-plan', type=Path, required=True)
    p.add_argument('--cpu-workers', type=int, default=3)
    main(p.parse_args())
