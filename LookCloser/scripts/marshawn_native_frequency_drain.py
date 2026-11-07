"""External handshake: drain owned RGB fits before a native-mask partition.

The released frequency broker already observes producer_v4_state.json. This
helper uses that interface without changing the broker or frozen model code.
It never terminates a fit or claims a GPU occupied by another workload.
"""
from pathlib import Path
import argparse, hashlib, importlib, json, os, sys, time, uuid

NATIVE_GPUS = {1, 5, 6, 7}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def blockers(state, snapshot):
    """A missing remote receipt cannot disprove a delayed detached launcher."""
    blocked = {}
    jobs = snapshot['jobs']
    for name, row in state.get('jobs', {}).items():
        if row.get('gpu') not in NATIVE_GPUS or not row.get('host'):
            continue
        remote = jobs.get(name)
        if remote is None:
            blocked[name] = 'remote launch receipt absent; ownership unresolved'
        elif remote.get('supervisor_alive') or remote.get('worker_alive'):
            blocked[name] = 'owned supervisor or worker remains alive'
    for name, row in jobs.items():
        if row.get('gpu') in NATIVE_GPUS and (row.get('supervisor_alive') or row.get('worker_alive')):
            blocked[name] = 'remote owned process remains alive'
    return blocked


def drain(campaign, frame, timeout):
    V = Path(campaign).resolve(); base = V / 'operations/native_frequency_drain'
    base.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(V / 'code/LookCloser/scripts'))
    fleet = importlib.import_module('marshawn_tuning_controller')
    held = fleet.lock(base / 'request.lock')
    if held is None:
        raise RuntimeError('Another native drain handshake owns the request')
    config = read(V / 'operations/future_rgb_sg/config.json')
    host = dict(address=config['remote_host'], python=config['remote_python'],
                root=config['remote_root'] + '/operations/future_rgb_sg/dispatch',
                controller=config['remote_root'] + '/code/LookCloser/scripts/marshawn_tuning_controller.py')
    nonce = uuid.uuid4().hex; started = time.time()
    request_path = V / 'operations/producer_v4_state.json'
    prior = read(request_path) if request_path.exists() else None
    write(base / (nonce + '.previous_state.json'), prior)
    request = dict(stage='native_visual_receipts', frame=int(frame),
                   native_partition_intent=True, native_frequency_drain_nonce=nonce,
                   producer_pid=os.getpid(), checked_at=time.time(),
                   reason='Drain owned RGB frequency fits before native partition; no process preemption')
    write(request_path, request)
    deadline = started + timeout if timeout else None
    while True:
        # Refresh while waiting so an older manual-HOLD report cannot win.
        current = read(request_path)
        if current.get('native_frequency_drain_nonce') != nonce:
            raise RuntimeError('Native drain request superseded by another owner')
        request['checked_at'] = time.time(); write(request_path, request)
        priority_path = V / 'operations/future_rgb_sg/priority.json'
        observed = read(priority_path) if priority_path.exists() else {}
        acknowledged = (observed.get('producer', {}).get('native_frequency_drain_nonce') == nonce
                        and not NATIVE_GPUS.intersection(observed.get('eligible_map_gpus', []))
                        and observed.get('time', 0) >= started)
        if acknowledged:
            result = fleet.host_call(host, ['_probe', '--root', host['root']])
            result.check_returncode(); snapshot = json.loads(result.stdout)
            expected = hashlib.sha256(Path(fleet.__file__).read_bytes()).hexdigest()
            if snapshot.get('controller_sha256') != expected:
                raise ValueError('Remote generic controller source differs')
            state = read(V / 'operations/future_rgb_sg/state.json')
            pending = blockers(state, snapshot)
            record = dict(frame=frame, nonce=nonce, checked_at=time.time(),
                          broker_acknowledged=True, owned_map_blockers=pending,
                          priority_sha256=hashlib.sha256(priority_path.read_bytes()).hexdigest())
            write(base / (nonce + '.progress.json'), record)
            if not pending:
                record.update(kind='NATIVE_PARTITION_OWNED_FREQUENCY_DRAIN_V1',
                              drained=True, source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              gpu_slots=sorted(NATIVE_GPUS), priority_request=str(request_path),
                              priority_retained_until_native_assignment=True,
                              other_workloads_may_still_occupy_gpus=True)
                receipt = base / (nonce + '.ready.json'); write(receipt, record)
                print(str(receipt), flush=True)
                return receipt
        if deadline and time.time() >= deadline:
            raise TimeoutError('Owned frequency fits did not prove drained; priority intent retained')
        time.sleep(2)


def selftest():
    state = {'jobs': {'one': {'gpu': 5, 'host': 'silver'}, 'other': {'gpu': 3, 'host': 'silver'}}}
    assert blockers(state, {'jobs': {}}) == {'one': 'remote launch receipt absent; ownership unresolved'}
    remote = {'one': {'gpu': 5, 'worker_alive': True, 'supervisor_alive': False}}
    assert blockers(state, {'jobs': remote})
    remote['one']['worker_alive'] = False
    assert not blockers(state, {'jobs': remote})
    remote['orphan'] = {'gpu': 7, 'worker_alive': True}
    assert 'orphan' in blockers(state, {'jobs': remote})
    print('PASS: active/unknown/delayed/orphan owners block; dead owned fits permit native partition')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path); p.add_argument('--frame', type=int)
    p.add_argument('--timeout', type=float, default=0, help='0 waits indefinitely; failure never terminates fits')
    p.add_argument('--selftest', action='store_true'); a = p.parse_args()
    if a.selftest:
        selftest()
    else:
        if a.campaign is None or a.frame is None:
            p.error('--campaign and --frame are required')
        drain(a.campaign, a.frame, a.timeout)
