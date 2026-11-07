"""Versioned192GiB execution policy over the unchanged source decoder.

The original scientific configuration/source_ready hashes remain unchanged;
only the free-space scheduling threshold is supplied by this policy.
"""
from pathlib import Path
import argparse, importlib, json, os, signal, subprocess, sys, time


def load(V):
    sys.path.insert(0, str(V / 'operations/future_sources/code'))
    old = importlib.import_module('marshawn_future_services')
    policy = old.read(V / 'operations/future_sources/policy192.json')
    if old.sha(__file__) != policy['policy_source_sha256'] or old.sha(policy['original_config']) != policy['original_config_sha256']:
        raise ValueError('Source execution policy/config changed')
    c = old.read(policy['original_config']); old.guard(c)
    c = dict(c, reserve_bytes=192 * 2**30)
    if c['frame_budget_bytes'] != 8 * 2**30:
        raise ValueError('Original8GiB per-frame budget changed')
    return old, c, policy


def watchdog(V):
    old, c, policy = load(V); ops = V / 'operations/future_sources'; held = old.lock(ops / 'watchdog.lock')
    if held is None:
        return
    old.atomic(ops / 'watchdog_identity.json', old.identity())
    while not (ops / 'decode_complete.json').exists():
        load(V); owner = old.read(ops / 'decode_identity.json') if (ops / 'decode_identity.json').exists() else None
        child = old.read(ops / 'decode_child.json').get('child') if (ops / 'decode_child.json').exists() else None
        if not old.alive(owner) and not old.alive(child):
            with (ops / 'decode192_controller.log').open('ab') as stream:
                p = subprocess.Popen([sys.executable, __file__, 'decode', '--campaign', str(V)], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            old.append(ops / 'decode192_watchdog_checks.jsonl', dict(time=time.time(), process=old.identity(p.pid)))
        time.sleep(15)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['decode', 'watchdog']); p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); old, c, policy = load(a.campaign)
    if a.action == 'decode':
        old.decode(c, Path(policy['original_config']))
    else:
        watchdog(a.campaign)
