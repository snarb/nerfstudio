"""Inactive V62 controller variant: unchanged NN scheduling with two CPU collectors.

The original run/watchdog logic is retained except source-bound asynchronous
collection. Native pool, source-ready checks, fleet jobs, retry and priority
policies are unchanged. The original sealed gather executes in owned workers.
"""
from pathlib import Path
import argparse, importlib, json, os, subprocess, sys, time
from marshawn_native_async_collect import Collectors, base_module, read

def run(config_path):
    base = base_module(read(config_path)); io=base.io; guard=base.guard; pool=base.pool; streaming=base.streaming
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
    collectors = Collectors(config_path)
    while True:
        guard(config); jobs = []
        allowed, pool_proof=pool.available(config['campaign'],config['hosts'],state)
        remote_ready=streaming.ready(config,root)
        io.write(root/'native_pool_snapshot.json',pool_proof)
        filtered_hosts={h:dict(spec,gpus=allowed[h])for h,spec in config['hosts'].items()}
        routed=[]; loads={}
        for original_row in config['records']:
            item=state['cameras'][original_row['camera']];job=state['jobs'][item['name']]
            route=job.get('host') or item.get('route')
            if job['status']=='pending' and not job.get('host'):
                camera_allowed=dict(allowed)
                if original_row['camera']not in remote_ready:camera_allowed['silver']=[]
                route=pool.choose(camera_allowed,loads)
                if route:item['route']=route
            if route:loads[route]=loads.get(route,0)+1
            row=dict(original_row,**original_row['variants'].get(route or 'silver',{}))
            row['host']=route or 'silver';routed.append(row)
            if not route and not item.get('collected'):continue
            item = state['cameras'][row['camera']]
            if item.get('collected'):
                continue
            host = config['hosts'][row['host']]; spec = io.read(row['task'])
            jobs.append(dict(name=item['name'], host=row['host'], command=[host['python'], host['adapter'], 'run', '--spec', str(Path(spec['remote_task']) / 'task.json')],
                             env={'SAM3_ASSIST_LEASE_RECEIPT': '{job_root}/receipt.json'}, metrics_file=str(Path(spec['remote_task']) / 'operation_receipt.json')))
        fleet.tick(dict(schema_version=1, poll_seconds=5, hosts=filtered_hosts, jobs=jobs), state, root)
        for row in routed:
            item = state['cameras'][row['camera']]; job = state['jobs'][item['name']]
            if job['status'] == 'complete' and not item.get('collected'):
                proof = collectors.result(row)
                if proof is not None:
                    item.update(collected=True, proof=proof)
                    io.write(root / 'state.json', state)
                else:
                    collectors.submit(row)
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
    base = base_module(read(config_path)); io=base.io; guard=base.guard
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','watchdog']);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    run(a.config) if a.action=='run' else watchdog(a.config)
