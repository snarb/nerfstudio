#!/usr/bin/env python3
"""Prepare, never launch, the source-bound AA 4000-update frequency control."""
import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extend_worker(source):
    replacements = [
        ("'train_steps_per_level':1000", "'train_steps_per_level':4000"),
        ("train_steps_per_level=1000", "train_steps_per_level=4000"),
    ]
    if source.count(replacements[0][0]) != 1 or source.count(replacements[1][0]) != 2:
        raise ValueError('Unexpected frozen worker: recipe/config/metadata must agree')
    for before, after in replacements:
        source = source.replace(before, after)
    return source


def prepare(campaign):
    parent = campaign / 'resolution_4k_AA_frequency'
    dest = campaign / 'resolution_4k_AA_frequency_4000'
    if dest.exists():
        raise ValueError('Preserve existing generation; output already exists')
    oldspec = json.loads((parent / 'spec_silver.json').read_text())
    worker = parent / 'fit_marshawn_4k_AA_frequency.py'
    if sha(worker) != oldspec['worker_sha256'] or oldspec['recipe']['train_steps_per_level'] != 1000:
        raise ValueError('Frozen parent worker/recipe changed')
    original = worker.read_text()
    modified = extend_worker(original)
    dest.mkdir()
    (dest / worker.name).write_text(modified)
    (dest / 'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True), fromfile='1000/worker.py', tofile='4000/worker.py')))
    spec = copy.deepcopy(oldspec)
    spec['recipe']['train_steps_per_level'] = 4000
    spec['worker_sha256'] = sha(dest / worker.name)
    spec['release'] = spec['release'].replace('/resolution_4k_AA_frequency/', '/resolution_4k_AA_frequency_4000/')
    spec['output'] = spec['output'].replace('/resolution_4k_AA_freq095/', '/resolution_4k_AA_freq095_4000/')
    plan = json.loads((parent / 'production_plan.json').read_text())
    job = plan['jobs'][0]
    job['name'] = 'resolution_4k_AA_freq095_4000'
    job['command'] = [s.replace('/resolution_4k_AA_frequency/', '/resolution_4k_AA_frequency_4000/') for s in job['command']]
    def write(name, value):
        (dest / name).write_text(json.dumps(value, indent=2) + '\n')
    write('spec_silver.json', spec)
    write('production_plan.json', plan)
    write('root_release.template.json', {'reviewer': 'root', 'frequency_fit_authorized': False, 'spec_sha256': sha(dest / 'spec_silver.json'), 'source_complete_sha256': spec['inputs'][spec['source_complete']]})
    write('preparation.json', {'status': 'prepared_not_launched_pending_root_release', 'parent_spec_sha256': sha(parent / 'spec_silver.json'), 'parent_worker_sha256': sha(worker), 'parent_result_sha256': sha(parent / 'result/complete.json'), 'source_worker_sha256': spec['worker_sha256'], 'spec_sha256': sha(dest / 'spec_silver.json'), 'plan_sha256': sha(dest / 'production_plan.json'), 'recipe_diff': {'train_steps_per_level': [1000, 4000]}, 'inputs_unchanged': spec['inputs'] == oldspec['inputs'], 'preprocess_source_unchanged': spec['preprocess_sha256'] == oldspec['preprocess_sha256'], 'full63_fit_authorized': False, 'training_authorized': False})
    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    print(prepare(parser.parse_args().campaign))
