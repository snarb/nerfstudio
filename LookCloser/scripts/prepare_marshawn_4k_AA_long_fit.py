#!/usr/bin/env python3
"""Prepare, never launch, source-bound AA longer-update frequency controls."""
import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extend_worker(source, steps=4000, parent_steps=1000):
    if (parent_steps, steps) not in ((1000, 4000), (4000, 16000)):
        raise ValueError('Only reviewed diagnostic budget transitions may be prepared')
    replacements = [
        ("'train_steps_per_level':" + str(parent_steps), "'train_steps_per_level':" + str(steps)),
        ("train_steps_per_level=" + str(parent_steps), "train_steps_per_level=" + str(steps)),
    ]
    if source.count(replacements[0][0]) != 1 or source.count(replacements[1][0]) != 2:
        raise ValueError('Unexpected frozen worker: recipe/config/metadata must agree')
    for before, after in replacements:
        source = source.replace(before, after)
    return source


def prepare(campaign, steps=4000, local=False):
    parent_steps = 1000 if steps == 4000 else 4000
    parent = campaign / ('resolution_4k_AA_frequency' + ('_4000' if parent_steps == 4000 else ''))
    dest = campaign / ('resolution_4k_AA_frequency_' + str(steps))
    if dest.exists():
        raise ValueError('Preserve existing generation; output already exists')
    oldspec = json.loads((parent / 'spec_silver.json').read_text())
    worker = parent / 'fit_marshawn_4k_AA_frequency.py'
    if sha(worker) != oldspec['worker_sha256'] or oldspec['recipe']['train_steps_per_level'] != parent_steps:
        raise ValueError('Frozen parent worker/recipe changed')
    original = worker.read_text()
    modified = extend_worker(original, steps, parent_steps)
    dest.mkdir()
    (dest / worker.name).write_text(modified)
    (dest / 'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile=str(parent_steps)+'/worker.py', tofile=str(steps)+'/worker.py')))
    spec = copy.deepcopy(oldspec)
    spec['recipe']['train_steps_per_level'] = steps
    spec['worker_sha256'] = sha(dest / worker.name)
    spec['release'] = spec['release'].replace('/' + parent.name + '/', '/' + dest.name + '/')
    oldjob = 'resolution_4k_AA_freq095' + ('_4000' if parent_steps == 4000 else '')
    newjob = 'resolution_4k_AA_freq095_' + str(steps)
    spec['output'] = spec['output'].replace('/' + oldjob + '/', '/' + newjob + '/')
    plan = json.loads((parent / 'production_plan.json').read_text())
    job = plan['jobs'][0]
    job['name'] = newjob
    job['command'] = [s.replace('/' + parent.name + '/', '/' + dest.name + '/') for s in job['command']]
    if local:
        remote_root = plan['hosts']['silver']['root']
        def localize(value):
            if isinstance(value, str): return value.replace(remote_root, str(campaign))
            if isinstance(value, list): return [localize(v) for v in value]
            if isinstance(value, dict): return {localize(k): localize(v) for k, v in value.items()}
            return value
        spec = localize(spec)
        host = localize(plan['hosts']['silver'])
        host.pop('address')
        plan['hosts'] = {'local': host}
        job['host'] = 'local'
    def write(name, value):
        (dest / name).write_text(json.dumps(value, indent=2) + '\n')
    write('spec_silver.json', spec)
    write('production_plan.json', plan)
    write('root_release.template.json', {'reviewer': 'root', 'frequency_fit_authorized': False, 'spec_sha256': sha(dest / 'spec_silver.json'), 'source_complete_sha256': spec['inputs'][spec['source_complete']]})
    write('preparation.json', {'status': 'prepared_not_launched_pending_root_release', 'parent_spec_sha256': sha(parent / 'spec_silver.json'), 'parent_worker_sha256': sha(worker), 'parent_result_sha256': sha(parent / 'result/complete.json'), 'source_worker_sha256': spec['worker_sha256'], 'spec_sha256': sha(dest / 'spec_silver.json'), 'plan_sha256': sha(dest / 'production_plan.json'), 'recipe_diff': {'train_steps_per_level': [parent_steps, steps]}, 'input_content_hashes_unchanged': sorted(spec['inputs'].values()) == sorted(oldspec['inputs'].values()), 'preprocess_source_unchanged': spec['preprocess_sha256'] == oldspec['preprocess_sha256'], 'full63_fit_authorized': False, 'training_authorized': False})
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=(4000, 16000), default=4000)
    parser.add_argument('--local', action='store_true')
    a = parser.parse_args()
    print(prepare(a.campaign, a.steps, a.local))
