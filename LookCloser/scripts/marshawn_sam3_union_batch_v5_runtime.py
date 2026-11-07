"""Durable isolated V5 runtime; optional same-frame historical replay binding."""
from pathlib import Path
import argparse, importlib, json, os, shlex, shutil, subprocess, sys, time
import marshawn_sam3_assist as io
import marshawn_sam3_assist_v5 as assist


def guard(config):
    for path, digest in config['files_sha256'].items():
        if io.sha(path) != digest:
            raise ValueError('V5 batch released input/code changed: ' + path)
    for row in config['records']:
        if io.sha(row['task']) != row['task_sha256'] or (row.get('original_native') and io.sha(row['original_native']) != row['original_native_sha256']):
            raise ValueError('V5 task or held-original archive changed: ' + row['camera'])


def gather(config, row):
    task = Path(row['task']).parent; returned = task / 'returned'; returned.mkdir(exist_ok=True)
    source = Path(io.read(row['task'])['remote_task'])
    frame = '%06d' % config['frame']
    for name in ['operation_receipt.json', 'canonical_operation_receipt.json', 'pre_clear',
                 'campaign/frames/' + frame + '/native_masks', 'campaign/frames/' + frame + '/person_candidates']:
        if row['host'] == 'silver':
            subprocess.run(['rsync', '-rt', config['hosts']['silver']['address'] + ':' + str(source / name), str(returned) + '/'], check=True)
        else:
            src = source / name; dst = returned / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    proof = io.read(returned / 'operation_receipt.json')
    replay = returned / 'pre_clear/canonical_post_clear' / (row['camera'] + '.npz')
    old_sha=io.sha(replay)
    if row.get('original_native_sha256') and old_sha != row['original_native_sha256']:
        raise ValueError('Fresh canonical post-clear replay differs from held original: ' + row['camera'])
    isolated = Path(config['isolated_campaign']); native = isolated / 'frames' / frame / 'native_masks'
    existing = native / (row['camera'] + '.json')
    if existing.exists():
        record = io.read(existing)
        if record['archive_sha256'] != proof['archive_sha256'] or io.sha(native / (row['camera'] + '.npz')) != proof['archive_sha256']:
            raise ValueError('Existing isolated V5 output differs; refuse overwrite')
    else:
        assist.collect(config['campaign'], task, isolated)
    # Portable local metadata for the separate geometry/identity source guard.
    # Transport equality never substitutes for the subsequent source review.
    frame_root = native.parent; record = io.read(existing)
    application = dict(proof['union_resolution'], version=5, frame=config['frame'], camera=row['camera'],
                       source_receipt_sha256=record['source_receipt_sha256'],
                       old_native_sha256=old_sha, new_native_sha256=proof['archive_sha256'],
                       pre_overlap_archive_sha256=proof['pre_clear_archive_sha256'],
                       pre_overlap_archive=str(frame_root / 'pre_clear' / row['camera'] / (row['camera'] + '.npz')),
                       pre_overlap_proof_sha256=proof['pre_clear_receipt_sha256'],
                       resolver_code_sha256=proof['v5_resolver_sha256'],
                       independent_UNKNOWN_and_veto_unchanged=True,
                       previous_pixels_used_as_final_foreground=False,
                       source_visual_review_pending=True,
                       operation_receipt_sha256=io.sha(returned / 'operation_receipt.json'))
    application_path = frame_root / 'union_resolution_inputs' / row['camera'] / 'application.json'
    if application_path.exists() and io.read(application_path) != application:
        raise ValueError('Existing isolated V5 application receipt differs')
    io.write(application_path, application)
    record['union_resolution'] = dict(version=5, receipt=str(application_path.relative_to(frame_root)),
                    receipt_sha256=io.sha(application_path),
                    actor_ownership_uncertain_pixels=proof['union_resolution']['eligible_restored_overlap_pixels'],
                    pre_overlap_archive_sha256=proof['pre_clear_archive_sha256'],
                    eligible_pre_union_sha256=proof['union_resolution']['eligible_pre_union_sha256'])
    io.write(existing, record)
    return dict(camera=row['camera'], task_sha256=row['task_sha256'],
                canonical_post_clear_replay_exact=True,
                original_native_sha256=old_sha, historical_same_frame_comparison=bool(row.get('original_native_sha256')), resolved_archive_sha256=proof['archive_sha256'],
                restored_pixels=proof['union_resolution']['eligible_restored_overlap_pixels'],
                source_receipt_sha256=io.read(row['task'])['original_source_receipt_sha256'],
                union_conserved=proof['union_resolution']['union_conserved'],
                operation_receipt_sha256=io.sha(returned / 'operation_receipt.json'))


def run(config_path):
    config = io.read(config_path); root = Path(config['root'])
    sys.path.insert(0, str(Path(config['campaign']) / 'code/LookCloser/scripts'))
    fleet = importlib.import_module('marshawn_tuning_controller'); held = fleet.lock(root / 'controller.lock')
    if held is None:
        return
    guard(config); io.write(root / 'controller_identity.json', fleet.identity())
    state = io.read(root / 'state.json') if (root / 'state.json').exists() else dict(jobs={}, cameras={})
    for row in config['records']:
        camera = row['camera']
        if camera not in state['cameras']:
            name = 'v5_%06d_%s_a00' % (config['frame'], camera)
            state['cameras'][camera] = dict(name=name, attempt=0)
            state['jobs'][name] = dict(status='pending')
            if camera in config.get('precollected', {}):
                state['cameras'][camera].update(collected=True, proof=config['precollected'][camera])
                state['jobs'][name] = dict(status='complete', prequalified=True)
    while True:
        guard(config); jobs = []
        for row in config['records']:
            item = state['cameras'][row['camera']]
            if item.get('collected'):
                continue
            host = config['hosts'][row['host']]; spec = io.read(row['task'])
            jobs.append(dict(name=item['name'], host=row['host'], command=[host['python'], host['adapter'], 'run', '--spec', str(Path(spec['remote_task']) / 'task.json')],
                             env={'SAM3_ASSIST_LEASE_RECEIPT': '{job_root}/receipt.json'}, metrics_file=str(Path(spec['remote_task']) / 'operation_receipt.json')))
        fleet.tick(dict(schema_version=1, poll_seconds=5, hosts=config['hosts'], jobs=jobs), state, root)
        for row in config['records']:
            item = state['cameras'][row['camera']]; job = state['jobs'][item['name']]
            if job['status'] == 'complete' and not item.get('collected'):
                item.update(collected=True, proof=gather(config, row))
                io.write(root / 'state.json', state)
            elif job['status'] == 'failed':
                snapshot = state.get('host_snapshots', {}).get(row['host'], {})
                remote = snapshot.get('jobs', {}).get(item['name'])
                if remote and not remote.get('supervisor_alive') and not remote.get('worker_alive') and item['attempt'] < 3:
                    item['attempt'] += 1; item['name'] = 'v5_%06d_%s_a%02d' % (config['frame'], row['camera'], item['attempt'])
                    state['jobs'][item['name']] = dict(status='pending')
                elif item['attempt'] >= 3:
                    raise RuntimeError('V5 inference exhausted retries: ' + row['camera'])
        io.write(root / 'state.json', state)
        collected = {c: item['proof'] for c, item in state['cameras'].items() if item.get('collected')}
        io.write(root / 'progress.json', dict(frame=config['frame'], complete=len(collected), total=len(config['records']),
                 time=time.time(), prepared_publication_allowed=False))
        if len(collected) == len(config['records']):
            io.write(root / 'all66_review_ready.json', dict(frame=config['frame'], cameras=collected,
                     all_canonical_post_clear_replays_exact=True, all_actor_unions_conserved=True,
                     prepared_publication_allowed=False, root_visual_review_required=True,
                     configuration_sha256=io.sha(config_path), time=time.time()))
            return
        time.sleep(5)


def watchdog(config_path):
    config = io.read(config_path); root = Path(config['root'])
    sys.path.insert(0, str(Path(config['campaign']) / 'code/LookCloser/scripts'))
    fleet = importlib.import_module('marshawn_tuning_controller'); held = fleet.lock(root / 'watchdog.lock')
    if held is None:
        return
    io.write(root / 'watchdog_identity.json', fleet.identity())
    while not (root / 'all66_review_ready.json').exists():
        guard(config)
        old = io.read(root / 'controller_identity.json') if (root / 'controller_identity.json').exists() else None
        if not fleet.alive(old):
            with (root / 'controller.log').open('ab') as stream:
                p = subprocess.Popen([sys.executable, __file__, 'run', '--config', str(config_path)], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            fleet.append(root / 'watchdog_checks.jsonl', dict(time=time.time(), started=fleet.identity(p.pid)))
        time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['run', 'watchdog']); p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.config) if a.action == 'run' else watchdog(a.config)
