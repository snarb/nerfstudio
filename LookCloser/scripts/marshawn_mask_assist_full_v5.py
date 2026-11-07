"""One-call durable isolated V5 native masks; live installation is a later gate."""
from pathlib import Path
import argparse, importlib, json, os, shlex, subprocess, sys, time
import marshawn_sam3_assist as io
import marshawn_sam3_assist_v5 as assist


def fleet_for(V):
    sys.path.insert(0, str(V / 'code/LookCloser/scripts'))
    return importlib.import_module('marshawn_tuning_controller')


def start(argv, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as stream:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def guard(V, frame):
    release = io.read(V / 'operations/sam3_assist/v5/full_release.json')
    for path, digest in release['files_sha256'].items():
        if io.sha(path) != digest:
            raise ValueError('V5 native released source changed: ' + path)
    if frame != 1675 and not release.get('future_frames_authorized'):
        raise ValueError('V5 future-frame recipe has not passed its all66 source-review release')
    return release


def factory(V, frame, root, release):
    source_path = V / 'sources' / str(frame) / 'source_ready.json'
    while not source_path.exists():
        time.sleep(10)
    source = io.read(source_path); cameras = source['cameras']
    if source['frame'] != frame or len(cameras) != 66 or not source['source_verified_all66']:
        raise ValueError('ALL66 current-frame source-ready required')
    remote_root = Path(release['remote_root']); code = V / 'operations/sam3_assist/code'
    R = remote_root / 'operations/sam3_assist/v5' / ('frame%06d' % frame)
    config_path = root / 'all66_config.json'
    if not config_path.exists():
        # The nonce is acknowledged and all broker-owned native-slot workers
        # prove dead before the fixed LOCAL/SG partition is committed.
        drain = subprocess.run([sys.executable, str(code / 'marshawn_native_frequency_drain.py'), '--campaign', str(V), '--frame', str(frame)],
                               check=True, capture_output=True, text=True)
        drain_path = Path(drain.stdout.strip().splitlines()[-1]); drained = io.read(drain_path)
        if not drained.get('drained'):
            raise ValueError('Native partition lacks an owned-frequency drain proof')
        local = set(cameras[::5]); records = []
        for camera in cameras:
            host = 'olive' if camera in local else 'silver'
            remote_task = (root if host == 'olive' else R) / 'tasks' / camera
            assist.pack(V, frame, camera, root / 'tasks' / camera, str(remote_task),
                        str(V / 'code' if host == 'olive' else remote_root / 'code'),
                        release['local_overlay'] if host == 'olive' else str(remote_root / 'tools/sam3_overlay'),
                        V / 'operations/marshawn_union_resolution_v5.py')
            original = V / 'frames' / ('%06d' % frame) / 'native_masks' / (camera + '.npz')
            records.append(dict(camera=camera, host=host, task=str(root / 'tasks' / camera / 'task.json'),
                                task_sha256=io.sha(root / 'tasks' / camera / 'task.json'),
                                original_native=str(original) if original.exists() else None,
                                original_native_sha256=io.sha(original) if original.exists() else None))
        hosts = dict(
            silver=dict(address=release['remote_host'], python=release['remote_python'], root=str(R / 'all66_dispatch'),
                        controller=str(remote_root / 'code/LookCloser/scripts/marshawn_tuning_controller.py'),
                        adapter=str(remote_root / 'operations/sam3_assist/code/marshawn_sam3_assist_v5.py'), gpus=[1, 5, 6, 7]),
            olive=dict(python=sys.executable, root=str(root / 'all66_LOCAL_dispatch'),
                       controller=str(V / 'code/LookCloser/scripts/marshawn_tuning_controller.py'),
                       adapter=str(code / 'marshawn_sam3_assist_v5.py'), gpus=[0]))
        config = dict(version=5, campaign=str(V), frame=frame, root=str(root / 'all66_controller'),
                      isolated_campaign=str(root / 'isolated_campaign'), records=records, hosts=hosts,
                      files_sha256=release['files_sha256'], precollected={}, drain_receipt=str(drain_path),
                      source_ready_sha256=io.sha(source_path), prepared_publication_allowed=False)
        io.write(config_path, config)
    config = io.read(config_path)
    if config.get('source_ready_sha256') and config['source_ready_sha256'] != io.sha(source_path):
        raise ValueError('Factory source-ready binding changed')
    marker = root / 'portable_inputs_staged.json'
    if not marker.exists():
        subprocess.run(['ssh', '-o', 'BatchMode=yes', release['remote_host'], shlex.join(['mkdir', '-p', str(R / 'tasks')])], check=True)
        # These are only same-frame source inputs and operational specs.
        subprocess.run(['rsync', '-rLt', '--exclude=returned/', '--exclude=campaign/', '--exclude=pre_clear/',
                        '--exclude=operation_receipt.json', '--exclude=canonical_operation_receipt.json',
                        str(root / 'tasks') + '/', release['remote_host'] + ':' + str(R / 'tasks') + '/'], check=True)
        for name in ['marshawn_sam3_assist_v5.py', 'marshawn_sam3_union_capture_v5.py']:
            subprocess.run(['rsync', '-rt', str(code / name), release['remote_host'] + ':' + str(remote_root / 'operations/sam3_assist/code') + '/'], check=True)
        io.write(marker, dict(configuration_sha256=io.sha(config_path), same_frame_only=True, time=time.time()))
    return config_path


def coordinate(V, frame):
    release = guard(V, frame); fleet = fleet_for(V)
    root = V / 'operations/sam3_assist/v5' / ('frame%06d' % frame); root.mkdir(parents=True, exist_ok=True)
    held = fleet.lock(root / 'factory.lock')
    if held is None:
        return
    io.write(root / 'factory_identity.json', fleet.identity())
    ready = root / 'all66_controller/all66_review_ready.json'
    if ready.exists():
        return
    # The reviewed1675 batch has its own already-running durable owner.
    config_path = root / 'all66_config.json'
    if frame != 1675 or not config_path.exists():
        config_path = factory(V, frame, root, release)
    config = io.read(config_path); batch_root = Path(config['root']); batch_root.mkdir(exist_ok=True)
    while not ready.exists():
        identity = batch_root / 'watchdog_identity.json'; old = io.read(identity) if identity.exists() else None
        if not fleet.alive(old):
            batch_source = release['batch_runtime'] if frame != 1675 else release['batch_1675']
            start([sys.executable, batch_source, 'watchdog', '--config', str(config_path)], batch_root / 'watchdog.log')
        time.sleep(5)
    receipt = io.read(ready)
    if not receipt['all_actor_unions_conserved'] or not receipt['all_canonical_post_clear_replays_exact'] or len(receipt['cameras']) != 66:
        raise ValueError('Incomplete V5 all66 result')
    # Retire only this exact partition's intent. A newer producer owner wins.
    if config.get('drain_receipt'):
        drained = io.read(config['drain_receipt']); state_path = V / 'operations/producer_v4_state.json'
        state = io.read(state_path)
        if state.get('native_frequency_drain_nonce') == drained['nonce']:
            state.pop('native_frequency_drain_nonce')
            state.update(stage='native_phase_complete', checked_at=time.time(), native_frequency_drained_nonce=drained['nonce'])
            io.write(state_path, state)


def watchdog(V, frame):
    fleet = fleet_for(V); root = V / 'operations/sam3_assist/v5' / ('frame%06d' % frame)
    held = fleet.lock(root / 'factory_watchdog.lock')
    if held is None:
        return
    io.write(root / 'factory_watchdog_identity.json', fleet.identity())
    while not (root / 'all66_controller/all66_review_ready.json').exists():
        guard(V, frame); p = root / 'factory_identity.json'; old = io.read(p) if p.exists() else None
        if not fleet.alive(old):
            start([sys.executable, __file__, 'coordinate', '--campaign', str(V), '--frame', str(frame)], root / 'factory.log')
        time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run', 'coordinate', 'watchdog'], nargs='?', default='run')
    p.add_argument('--campaign', type=Path, required=True); p.add_argument('--frame', type=int, required=True); a = p.parse_args()
    V = a.campaign.resolve(); root = V / 'operations/sam3_assist/v5' / ('frame%06d' % a.frame); root.mkdir(parents=True, exist_ok=True)
    if a.action == 'coordinate':
        coordinate(V, a.frame)
    elif a.action == 'watchdog':
        watchdog(V, a.frame)
    else:
        guard(V, a.frame); fleet = fleet_for(V); identity = root / 'factory_watchdog_identity.json'
        if not fleet.alive(io.read(identity) if identity.exists() else None):
            start([sys.executable, __file__, 'watchdog', '--campaign', str(V), '--frame', str(a.frame)], root / 'factory_watchdog.log')
        ready = root / 'all66_controller/all66_review_ready.json'
        while not ready.exists():
            time.sleep(5)
        print(str(ready), flush=True)
