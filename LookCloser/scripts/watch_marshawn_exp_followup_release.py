"""Execute a hash-bound, root-reviewed conditional experiment release once ready.

This authorizes more experiments, never promotion of a model or a grid.
The existing launcher independently verifies local/remote inventories and the
completed parent. This watcher only supplies its exact completion hash after
the already-reviewed numerical conditions become true.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shlex
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible(parent, policy):
    if parent.get('technical_smoke') or parent.get('final_step', 0) < 106316:
        return False
    if not parent.get('plateau', {}).get('stopped'):
        return False
    selected = parent['selection']
    values = [selected['metrics'][k] for k in ('psnr', 'ssim', 'lpips')]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        raise ValueError('Nonfinite/invalid parent scores')
    if not 0 <= selected['metrics']['lpips'] <= policy['maximum_actor_lpips']:
        return False
    for actor, reference in policy['reference_actor_lpips'].items():
        value = selected['by_actor'][actor]['lpips']
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid per-actor score')
        if value > reference * policy['maximum_actor_ratio']:
            return False
    return True


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    policy_hash = sha(args.policy)
    if policy.get('conditional_experiment_launch_authorized') is not True:
        raise ValueError('No conditional experiment authorization')
    root = args.policy.parent
    lock = (root/'conditional_release.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign = root.parent
    log = root / 'conditional_release_checks.jsonl'
    while True:
        if sha(args.policy) != policy_hash:
            raise ValueError('Reviewed policy changed while watcher was running')
        for path, digest in policy['reviewed_artifacts'].items():
            if sha(path) != digest:
                raise ValueError('Reviewed artifact changed: ' + path)
        check = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     policy_sha256=policy_hash, parent_available=False)
        try:
            process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                      policy['parent_host'],
                                      shlex.join(['cat', policy['parent_complete_path']])],
                                     capture_output=True, timeout=25)
            check['parent_available'] = process.returncode == 0
            if process.returncode:
                check['read_error'] = process.stderr.decode(errors='replace')[-300:]
        except (OSError, subprocess.SubprocessError) as error:
            check['read_error'] = str(error)
        if check['parent_available']:
            parent = json.loads(process.stdout)
            if parent['config'] != policy['expected_parent_config']:
                raise ValueError('Completed parent configuration differs from reviewed parent')
            check['eligible'] = eligible(parent, policy)
            if not check['eligible']:
                # Completion exists but fails a scientific condition: do not
                # silently weaken it or keep pretending more waiting will help.
                atomic(root/'conditional_release_blocked.json', check)
                with log.open('a') as stream:
                    stream.write(json.dumps(check)+'\n')
                return
            release = dict(policy['release_fields'],
                           parent_complete_sha256=hashlib.sha256(process.stdout).hexdigest(),
                           conditional_policy_sha256=policy_hash)
            (root/'observed_parent_complete.json').write_bytes(process.stdout)
            release_path = root/'root_release.json'
            atomic(release_path, release)
            with (root/'conditional_launch.log').open('ab') as stream:
                subprocess.run([str(campaign/'python'), str(root/'launch_reviewed.py'),
                                '--release', str(release_path)], stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
                reporter = campaign/'cross_campaign_report_exp_followup'
                subprocess.run([str(campaign/'python'), str(reporter/'report_marshawn_cross_campaign.py'),
                                'launch', '--root', str(reporter)], stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            check['experiment_queue_launched'] = True
            check['model_promoted'] = False
            atomic(root/'conditional_release_complete.json', check)
        with log.open('a') as stream:
            stream.write(json.dumps(check)+'\n')
        if check.get('experiment_queue_launched'):
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
