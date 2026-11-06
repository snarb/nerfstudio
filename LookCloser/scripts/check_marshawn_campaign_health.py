"""Record a compact, read-only fleet check without copying large training logs."""
import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time


def last_record(path):
    with path.open('rb') as stream:
        size = path.stat().st_size
        stream.seek(max(0, size - 4_000_000))
        return json.loads(stream.read().splitlines()[-1])


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=35)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()[-500:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    now = time.time()
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'type': 'root_fleet_health_check', 'controllers': {}}
    states = set(args.root.glob('*/state.json')) | set(args.root.glob('*/controller/state.json'))
    for state_path in sorted(states):
        controller = state_path.parent / 'controller.json'
        # Production controller roots only, not frozen code or worker state.
        if not controller.exists():
            continue
        try:
            state = json.loads(state_path.read_text())
            if 'jobs' not in state:
                continue
            meta = json.loads(controller.read_text())
            pid = meta['process']['pid']
            item = {'pid': pid, 'proc_present': Path(f'/proc/{pid}').exists(),
                    'jobs': {}}
            checks = state_path.parent / 'checks.jsonl'
            if checks.exists():
                snapshot = last_record(checks)
                item['controller_check_age_seconds'] = round(time.time()-snapshot['time'], 1)
                # A fresh host inventory also includes jobs owned by remote controllers.
                for host, inventory in snapshot.get('hosts', {}).items():
                    active = {}
                    for name, job in inventory.get('jobs', {}).items():
                        if job.get('status') != 'running':
                            continue
                        latest = job.get('latest_check', {})
                        active[name] = {'worker_alive': job.get('worker_alive'),
                            'oom': latest.get('oom_evidence'),
                            'step': max((p.get('step', -1) for p in latest.get('progress', {}).values()
                                         if isinstance(p, dict)), default=None)}
                    existing = result.setdefault('host_inventories', {}).get(host, {})
                    if inventory.get('time', 0) > existing.get('time', 0):
                        result['host_inventories'][host] = {
                            'time': inventory.get('time'), 'active_jobs': active}
            for name, job in state['jobs'].items():
                if job.get('status') not in ('running', 'failed', 'starting'):
                    continue
                latest = job.get('latest_check', {})
                progress = list(latest.get('progress', {}).values())
                item['jobs'][name] = {
                    'status': job.get('status'), 'host': job.get('host'),
                    'worker_alive': job.get('worker_alive', latest.get('worker_alive')),
                    'heartbeat_age_seconds': round(time.time()-job.get('heartbeat', now), 1),
                    'step': max((v.get('step', -1) for v in progress if isinstance(v, dict)), default=None),
                    'oom': latest.get('oom_evidence')}
            result['controllers'][str(state_path.parent.relative_to(args.root))] = item
        except (OSError, ValueError, KeyError) as error:
            result['controllers'][str(state_path.parent)] = {'error': str(error)}
    query = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader']
    native = args.root/'native_reference_continuation_20000'
    launch = json.loads((native/'launch.json').read_text()) if (native/'launch.json').exists() else {}
    native_pids = ','.join(str(launch[key]) for key in ('controller_pid', 'worker_pid') if key in launch) or '0'
    commands = {'local_gpu': query,
                'silver_gpu': ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'ubuntu@silver-sable']+query,
                'dev3_gpu': ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'ubuntu@dev3']+query,
                'dev3_native_processes': ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'ubuntu@dev3',
                    'ps', '-p', native_pids, '-o', 'pid=,stat=,etime=,comm=']}
    remote_root = '/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/sh_geometry_candidate_20261106/controller'
    remote_code = """import json,pathlib,time
p=pathlib.Path(REMOTE_ROOT)
s=json.loads((p/'state.json').read_text())
c=json.loads((p/'controller.json').read_text())
r={'controller_pid':c['process']['pid'],'controller_alive':pathlib.Path('/proc/'+str(c['process']['pid'])).exists(),'jobs':{}}
for n,j in s['jobs'].items():
 q=j.get('latest_check',{})
 r['jobs'][n]={'status':j.get('status'),'worker_alive':j.get('worker_alive',q.get('worker_alive')),'heartbeat_age_seconds':round(time.time()-j.get('heartbeat',time.time()),1),'oom':q.get('oom_evidence'),'step':max((v.get('step',-1) for v in q.get('progress',{}).values() if isinstance(v,dict)),default=None)}
print(json.dumps(r))
""".replace('REMOTE_ROOT', repr(remote_root))
    commands['remote_geometry_ABC'] = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'ubuntu@silver-sable', shlex.join(['python3', '-c', remote_code])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(command, cmd): key for key, cmd in commands.items()}
        for future, key in futures.items():
            try:
                result[key] = future.result()
            except Exception as error:
                result[key] = {'error': str(error)}
    native = args.root/'native_reference_continuation_20000'
    if (native/'supervision.jsonl').exists():
        result['native_supervision'] = last_record(native/'supervision.jsonl')
        result['native_status'] = json.loads((native/'status.json').read_text())
    watcher = args.root/'exp_followup/conditional_release_checks.jsonl'
    if watcher.exists():
        result['conditional_followup'] = last_record(watcher)
    with (args.root/'root_supervision.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
