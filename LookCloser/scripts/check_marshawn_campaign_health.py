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
    exp_root = '/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/exp_geometry_candidate_20261106/controller'
    commands['remote_geometry_exp'] = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'ubuntu@silver-sable', shlex.join(['python3', '-c', remote_code.replace(repr(remote_root), repr(exp_root))])]
    semantic_root = '/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/semantic_actor_candidate_v2_20261106/controller'
    commands['remote_semantic_actor'] = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'ubuntu@silver-sable', shlex.join(['python3', '-c', remote_code.replace(repr(remote_root), repr(semantic_root))])]
    for label, generation in [('r1', 'semantic_actor_r1_prepared_20261106'),
                              ('w32', 'semantic_actor_w32_prepared_20261106'),
                              ('sourcepoint', 'semantic_sourcepoint_candidate_20261106'),
                              ('cold_lambda01', 'semantic_cold_lambda01_candidate_20261106'),
                              ('cold_halfstep', 'semantic_cold_halfstep_candidate_20261107')]:
        if (args.root/generation/'release.json').exists():
            candidate_root = '/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/'+generation+'/controller'
            candidate_code = remote_code.replace(repr(remote_root), repr(candidate_root))
            candidate_code = candidate_code.replace("s=json.loads", "if not (p/'state.json').exists():\n print(json.dumps({'status':'released_controller_not_started'}));raise SystemExit(0)\ns=json.loads", 1)
            commands['remote_semantic_'+label] = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                'ubuntu@silver-sable', shlex.join(['python3', '-c', candidate_code])]
    technical = args.root/'native_reference_4k_technical_v2'
    semantic4k = args.root/'semantic_4k_technical_v1'
    if (semantic4k/'root_release.json').exists():
        technical_code = """import json,pathlib,time
p=pathlib.Path('/home/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106/semantic_4k_technical_v1/technical_supervision')
r={}
for name in ['supervisor.json','technical_complete.json','technical_failure.json']:
 if (p/name).exists():r[name]=json.loads((p/name).read_text())
if (p/'health.jsonl').exists():
 with (p/'health.jsonl').open('rb') as f:
  f.seek(max(0,(p/'health.jsonl').stat().st_size-65536));r['health']=json.loads(f.read().splitlines()[-1])
 r['health_age_seconds']=time.time()-r['health']['time']
for label,source in [('controller','supervisor.json'),('worker','health')]:
 pid=r.get(source,{}).get('pid');r[label+'_alive']=pathlib.Path('/proc/'+str(pid)).exists() if pid else None
print(json.dumps(r))
"""
        commands['remote_semantic4k_technical'] = [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'ubuntu@silver-sable',
            shlex.join(['python3', '-c', technical_code])]
    if (technical/'launch.json').exists():
        pid = json.loads((technical/'launch.json').read_text())['pid']
        commands['dev3_4k_executor'] = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            'ubuntu@dev3', 'ps', '-p', str(pid), '-o', 'pid=,stat=,etime=,comm=']
    pair = args.root/'native_resolution_pair_100000'
    if (pair/'durable_launch.json').exists():
        meta = json.loads((pair/'durable_launch.json').read_text())
        status = json.loads((pair/'status.json').read_text()) if (pair/'status.json').exists() else {}
        pids = {meta.get('pid'), meta.get('controller_pid'), status.get('worker_pid')}
        pids.discard(None)
        if pids:
            commands['dev3_resolution_pair_processes'] = [
                'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'ubuntu@dev3',
                'ps', '-p', ','.join(str(pid) for pid in sorted(pids)),
                '-o', 'pid=,stat=,etime=,comm=']
        result['native_resolution_pair'] = {
            'launch': meta, 'status': status,
            'status_age_seconds': round(time.time()-status.get('time', now), 1),
            'complete_receipt_present': (pair/'complete.json').exists(),
            'failure_receipts': [path.name for path in sorted(pair.glob('failure-*.json'))],
        }
        if (pair/'state.json').exists():
            state = json.loads((pair/'state.json').read_text())
            result['native_resolution_pair']['arms'] = {
                name: {'step': arm['cursor']['step'], 'stopped': arm['plateau']['stopped'],
                       'selected_step': arm['selection']['step']}
                for name, arm in state['arms'].items()}
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
    if (native/'complete.json').exists():
        complete = json.loads((native/'complete.json').read_text())
        result['native_completion'] = {
            'complete_receipt_present': True,
            'status_file_may_describe_last_precompletion_phase': True,
            'receipt': str(native/'complete.json'),
            'final_step': complete.get('final_step', complete.get('step')),
            'selected_step': complete.get('selected_step'),
        }
    watcher = args.root/'exp_followup/conditional_release_checks.jsonl'
    if watcher.exists():
        result['conditional_followup'] = last_record(watcher)
    maps = args.root/'resolution_4k_frequency_queue_4000_fleet'
    if (maps/'admission_status.json').exists():
        result['frequency_admission'] = json.loads((maps/'admission_status.json').read_text())
    for name in ['wait_health.jsonl', 'phase_health.jsonl']:
        if (technical/name).exists():
            result['native4k_'+name] = last_record(technical/name)
    if (technical/'complete.json').exists():
        complete = json.loads((technical/'complete.json').read_text())
        result['native4k_technical_completion'] = {
            'complete_receipt_present': True,
            'receipt': str(technical/'complete.json'),
            'passed': complete.get('passed'),
            'production_authorized': False,
        }
    with (args.root/'root_supervision.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
