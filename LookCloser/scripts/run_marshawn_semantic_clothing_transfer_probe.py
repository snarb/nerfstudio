"""Bounded serial read-only clothing probes under one physical GPU lease."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(path):
    plan = json.loads(path.read_text()); root = path.parent
    assert plan['authorized_readonly'] is True and plan['max_seconds'] <= 900
    assert plan['allocator_GiB'] <= 12 and plan['process_MiB'] <= 16384
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == plan['gpu_uuid']
    assert not any((root / n).exists() for n in ['complete.json', 'failure.json', 'launch.json'])
    lock = open('/tmp/lookcloser_tuning_gpu_%d_%d.lock' % (os.getuid(), plan['gpu']), 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(plan['gpu']), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert uuid == plan['gpu_uuid']

    def apps():
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
        return [tuple(v.strip() for v in row.split(',')) for row in output.splitlines() if row.strip()]

    def verify():
        for name, digest in plan['inputs_sha256'].items():
            assert sha(name) == digest, name
        for source in plan['sources']:
            inventory = json.loads(Path(source['source_manifest']).read_text())
            for name, digest in inventory['files_sha256'].items():
                assert sha(Path(source['code_root']) / name) == digest, name

    assert not any(row[0] == uuid for row in apps()), 'GPU occupied; no interruption permitted'
    verify(); start = time.time(); child = None; peak = 0
    (root / 'launch.json').write_text(json.dumps(dict(time=start, pid=os.getpid(), gpu_uuid=uuid, plan_sha256=sha(path))) + '\n')
    try:
        for policy_path in plan['policies']:
            policy = json.loads(Path(policy_path).read_text())
            with (Path(policy['root']) / 'probe.log').open('w') as log:
                child = subprocess.Popen([plan['python'], plan['probe'], '--policy', policy_path], stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True,
                                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                while child.poll() is None:
                    rows = apps(); own = [r for r in rows if r[0] == uuid]
                    used = sum(int(r[2]) for r in own); peak = max(peak, used)
                    assert not any(int(r[1]) == child.pid and r[0] != uuid for r in rows), 'Wrong physical GPU'
                    if used > plan['process_MiB'] or time.time() - start > plan['max_seconds'] or any(int(r[1]) != child.pid for r in own):
                        raise RuntimeError('Resource, deadline or foreign-process guard')
                    row = dict(time=time.time(), name=policy['name'], child_pid=child.pid, GPU_MiB=used)
                    (root / 'status.json').write_text(json.dumps(row) + '\n')
                    with (root / 'health.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                    time.sleep(2)
                assert child.returncode == 0, policy['name'] + ' failed; preserve evidence'
                result = json.loads((Path(policy['root']) / 'probe_complete.json').read_text())
                assert result['passed'] and result['actual_Adam_updates'] == 0 and result['initial_fullstate_hashes'] == result['final_fullstate_hashes']
                child = None
        verify()
        (root / 'complete.json').write_text(json.dumps(dict(passed=True, elapsed_seconds=time.time() - start, peak_GPU_MiB=peak,
            plan_sha256=sha(path), source_reverified=True, no_optimizer_updates=True,
            receipts={p: sha(Path(json.loads(Path(p).read_text())['root']) / 'probe_complete.json') for p in plan['policies']}), indent=2) + '\n')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        (root / 'failure.json').write_text(json.dumps(dict(error=repr(error), elapsed_seconds=time.time() - start, peak_GPU_MiB=peak)) + '\n')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    main(parser.parse_args().plan)
