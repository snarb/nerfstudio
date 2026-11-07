"""CPU-only retention of committed diagnostic gates; checkpoints stay remote."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ARMS = ('control', 'old7369', 'seven10481')
GATES = (91628, 92128, 93128, 95128, 99128, 103128)


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)


def policy_at(path):
    p = read(path)
    if p['version'] != 3 or sha(__file__) != p['retention_sha256']:
        raise ValueError('Exact v3 CPU retention helper required')
    release = read(Path(path).parent / 'guard_release.json')
    if release.get('reviewer') != 'root' or release.get('external_guard_authorized') is not True or release.get('policy_sha256') != sha(path):
        raise ValueError('Exact root guard release required')
    if 'diagnostic' not in release.get('authorized_phases', []):
        raise ValueError('Diagnostic phase not released')
    if release.get('technical_verification_sha256') != p['technical_verification_sha256']:
        raise ValueError('Root technical proof binding required')
    return p


def gate_record(root, arm, step):
    """Read only a gate published through atomic latest; no torch/model imports."""
    root = Path(root); run = root / 'runs' / ('diagnostic_' + arm)
    latest = run / 'latest.json'
    if not latest.exists(): return None
    committed = read(latest)
    if committed['step'] < step: return None
    spec_path = root / 'specs' / ('diagnostic_' + arm) / 'spec.json'
    spec = read(spec_path)
    if spec['mode'] != 'diagnostic' or Path(spec['output']) != run:
        raise ValueError('Unexpected diagnostic source/output')
    checkpoint = run / 'checkpoints' / f'step-{step:09d}.ckpt'
    grid = checkpoint.with_name(checkpoint.stem + '_grid.json')
    archive = grid.with_suffix('.npz'); receipt = read(grid)
    checkpoint_hash = sha(checkpoint)
    if receipt['checkpoint_step'] != step or receipt['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Checkpoint/grid mismatch')
    if Path(receipt['checkpoint']) != checkpoint or receipt['grid_archive_sha256'] != sha(archive):
        raise ValueError('Grid source/archive mismatch')
    if committed['step'] == step and (committed['checkpoint'] != str(checkpoint.relative_to(run)) or committed['sha256'] != checkpoint_hash):
        raise ValueError('Atomic latest/checkpoint mismatch')
    render = run / 'renders' / f'step-{step:09d}'
    if read(render / 'metrics.json')['step'] != step:
        raise ValueError('Render metric step mismatch')
    files = [render / name for name in ('DC.png', 'HC.png', 'KA.png', 'DC_alpha.png', 'HC_alpha.png', 'KA_alpha.png', 'metrics.json')]
    files += [grid, archive, run / 'clear_targets' / f'step-{step:09d}.json']
    return dict(arm=arm, step=step, source_checkpoint_sha256=spec['parent_checkpoint_sha256'],
                spec_sha256=sha(spec_path), worker_sha256=spec['worker_sha256'],
                source_inventory_sha256=sha(root / 'source_inventory.json'),
                anchor_audit_sha256=sha(root / 'anchor_audit.json'),
                checkpoint=dict(path=str(checkpoint), sha256=checkpoint_hash, bytes=checkpoint.stat().st_size,
                                retained_remote=True, copied_locally=False),
                observed_atomic_latest=committed, files_sha256={str(f.relative_to(run)): sha(f) for f in files},
                remote_run=str(run), remote_time=time.time(), no_pruning=True)


def inventory(policy, completed):
    root = Path(policy['remote_root']); records = []; retained = []
    for arm in ARMS:
        for step in GATES:
            if f'{arm}/{step}' in completed:
                cp = root / 'runs' / ('diagnostic_' + arm) / 'checkpoints' / f'step-{step:09d}.ckpt'
                grid = read(cp.with_name(cp.stem + '_grid.json'))
                retained.append(dict(arm=arm, step=step, path=str(cp), bytes=cp.stat().st_size,
                                     checkpoint_sha256=grid['checkpoint_sha256'], exists=True))
                continue
            record = gate_record(root, arm, step)
            if record is not None: records.append(record)
    return dict(records=records, retained_remote_checks=retained,
                completed_remote={arm: (root / 'runs' / ('diagnostic_' + arm) / 'complete.json').exists() for arm in ARMS})


def commit_gate(record, out, copy_file):
    dest = Path(out) / record['arm'] / f"step-{record['step']:09d}"
    if dest.exists():
        prior = read(dest / 'retention.json')
        if prior['checkpoint'] != record['checkpoint'] or prior['files_sha256'] != record['files_sha256']:
            raise ValueError('Never replace a retained gate')
        return
    staging = dest.with_name(dest.name + f'.partial-{os.getpid()}-{time.time_ns()}')
    staging.mkdir(parents=True, exist_ok=False)
    for rel, digest in record['files_sha256'].items():
        relative = Path(rel)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix == '.ckpt':
            raise ValueError('Only bounded review artifacts can be fetched')
        target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(str(Path(record['remote_run']) / relative), target)
        if sha(target) != digest: raise ValueError('Transferred gate artifact changed: ' + rel)
    atomic(staging / 'retention.json', dict(record, local_commit_time=time.time(), CPU_only=True, source_mutation=False))
    os.replace(staging, dest)


def run_once(policy, out):
    completed = {f"{read(p)['arm']}/{read(p)['step']}" for p in Path(out).glob('*/step-*/retention.json') if '.partial-' not in str(p)}
    remote_helper = Path(policy['remote_root']) / 'external_guard_v3/retention.py'
    remote_policy = remote_helper.with_name('policy.json')
    command = ['python3', str(remote_helper), 'inventory', '--policy', str(remote_policy), '--completed', json.dumps(sorted(completed))]
    reply = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', policy['address'], shlex.join(command)], capture_output=True, text=True, check=True, timeout=180)
    snapshot = json.loads(reply.stdout)
    for retained in snapshot['retained_remote_checks']:
        old = read(Path(out) / retained['arm'] / f"step-{retained['step']:09d}" / 'retention.json')['checkpoint']
        if old['bytes'] != retained['bytes'] or old['sha256'] != retained['checkpoint_sha256']:
            raise ValueError('Previously retained remote checkpoint metadata changed')
    def copy_file(source, target):
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', policy['address'] + ':' + source, str(target)], check=True, timeout=120)
    for record in snapshot['records']:
        commit_gate(record, out, copy_file)
        completed.add(f"{record['arm']}/{record['step']}")
    return dict(time=time.time(), retained_gates=len(completed), expected_gates=18,
                completed_remote=snapshot['completed_remote'], retained_remote_checks=snapshot['retained_remote_checks'],
                newly_hashed_checkpoints=[r['checkpoint'] for r in snapshot['records']],
                checkpoint_pruning=False, GPU_used=False)


def watch(policy_path, once=False):
    policy = policy_at(policy_path)
    expected = Path(policy['local_root']) / 'external_guard_v3/policy.json'
    if Path(policy_path).resolve() != expected.resolve(): raise ValueError('Local-origin retention only')
    out = expected.parent / 'retained_gates'; out.mkdir(exist_ok=True)
    lock = (out / 'fetch.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(out / 'process.json', dict(pid=os.getpid(), start=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19], policy_sha256=sha(policy_path)))
    try:
        errors = 0
        while True:
            try:
                result = run_once(policy, out); atomic(out / 'status.json', result)
                if result['retained_gates'] == 18:
                    atomic(out / 'complete.json', dict(result, passed=True)); return
                guard_done = expected.parent / 'diagnostic/local_done.json'
                if guard_done.exists():
                    raise RuntimeError('Guard terminal before all18 gates retained; preserve partial evidence')
                errors = 0
                if once: return
            except Exception as exc:
                errors += 1; atomic(out / 'last_error.json', dict(time=time.time(), error=repr(exc), consecutive=errors))
                if errors >= 3 or once:
                    atomic(out / 'failure.json', dict(time=time.time(), error=repr(exc), worker_interrupted=False)); raise
            time.sleep(30)
    finally:
        lock.close()


def launch(policy_path):
    p = policy_at(policy_path); out = Path(p['local_root']) / 'external_guard_v3/retained_gates'; out.mkdir(exist_ok=True)
    for terminal in ('complete.json', 'failure.json'):
        if (out / terminal).exists(): raise ValueError('Terminal retention evidence is immutable')
    log = (out / 'fetcher.log').open('ab')
    proc = subprocess.Popen([p['local_python'], str(Path(__file__).resolve()), 'watch', '--policy', str(policy_path.resolve())], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(retention_pid=proc.pid, CPU_only=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['launch', 'watch', 'once', 'inventory'])
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--completed', default='[]')
    args = parser.parse_args()
    if args.command == 'inventory': print(json.dumps(inventory(policy_at(args.policy), set(json.loads(args.completed)))))
    elif args.command == 'launch': launch(args.policy)
    else: watch(args.policy, once=args.command == 'once')
