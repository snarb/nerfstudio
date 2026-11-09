"""Pause one existing worker at a fully exported checkpoint, then seal a copy.

This external monitor never creates a trainer, resumes a worker or changes its
configuration. CSV progress is distinct from sparse AMP progress; an overrun is
reported rather than mislabelled as an exact optimizer boundary.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def step_stem(step):
    if not isinstance(step, int) or not 0 <= step < 22000:
        raise ValueError('Expected a checkpoint step in [0,22000)')
    return 'step-%09d' % step


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def proc_state(pid):
    text = (Path('/proc')/str(pid)/'stat').read_text()
    # The process name in parentheses may contain spaces.
    fields = text[text.rfind(')')+2:].split()
    return fields[0], fields[19]


def latest_csv_step(path):
    if path is None or not path.exists():
        return None
    steps = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row['step']))
            except (KeyError, TypeError, ValueError):
                pass
    return max(steps) if steps else None


def run(args):
    stem = step_stem(args.step)
    marker = args.stage/'grids'/(stem+'_grid.json')
    state, birth = proc_state(args.worker_pid)
    started = time.time()
    reservation = args.stage/('external_gate_%d_reservation.json' % args.step)
    with reservation.open('x') as f:
        json.dump(dict(pid=args.worker_pid, birth=birth, marker=str(marker),
                       started=started, monitor_pid=os.getpid()), f, indent=2)
    while time.time()-started < args.timeout_seconds:
        try:
            state, current_birth = proc_state(args.worker_pid)
        except FileNotFoundError:
            print(json.dumps(dict(status='worker_terminal', signal_sent=False)), flush=True)
            return
        if current_birth != birth or state == 'Z':
            print(json.dumps(dict(status='worker_terminal_or_reused', signal_sent=False)), flush=True)
            return
        if not marker.exists():
            time.sleep(.1)
            continue
        try:
            record = json.loads(marker.read_text())
        except json.JSONDecodeError:
            time.sleep(.1)
            continue
        checkpoint = Path(record['checkpoint'])
        if record['checkpoint_step'] != args.step or checkpoint.name != stem+'.ckpt':
            raise ValueError('Unexpected checkpoint marker; no signal sent')
        if sha(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed; no signal sent')
        if sha(record['grid_archive']) != record['grid_archive_sha256']:
            raise ValueError('Grid archive changed; no signal sent')
        if proc_state(args.worker_pid)[1] != birth:
            raise ValueError('Worker PID reused; no signal sent')
        os.kill(args.worker_pid, signal.SIGSTOP)
        for _ in range(100):
            if proc_state(args.worker_pid)[0] in ('T', 't'):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('SIGSTOP not observed; no automatic CONT')
        csv_step = latest_csv_step(args.metrics)
        snapshot = args.stage/('immutable_gate_%d' % args.step)
        snapshot.mkdir(exist_ok=False)
        sealed = snapshot/checkpoint.name
        shutil.copyfile(checkpoint, sealed)
        if sha(sealed) != record['checkpoint_sha256']:
            raise ValueError('Snapshot changed; worker remains paused for review')
        sealed.chmod(0o444)
        result = dict(worker_pid=args.worker_pid, worker_birth=birth, state='T',
                      checkpoint=str(sealed), checkpoint_sha256=sha(sealed),
                      checkpoint_step=args.step, latest_csv_step=csv_step,
                      overrun_observed=csv_step is not None and csv_step > args.step,
                      CSV_interval=10, no_automatic_CONT=True,
                      note='Saved checkpoint boundary is not proof of the live optimizer boundary.',
                      original_checkpoint=str(checkpoint), grid_record=record)
        (snapshot/'gate.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        return
    raise TimeoutError('Monitor deadline reached; no automatic restart or CONT')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--worker-pid', type=int, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--metrics', type=Path)
    p.add_argument('--timeout-seconds', type=float, default=1200)
    run(p.parse_args())
