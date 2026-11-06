#!/usr/bin/env python3
"""Prepare 63 independent source-bound fits; never authorize or launch them."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def camera_worker(source, camera):
    if camera in ('DC', 'HC', 'KA') or len(camera) != 2 or not camera.isalpha() or not camera.isupper():
        raise ValueError('Only TRAIN camera identifiers are eligible')
    # Camera changes affect labels and map filenames only, never fit mathematics.
    for before, after, count in [("'camera':'AA'", "'camera':" + repr(camera), 2),
                                 ("maps/'train_AA.pt'", "maps/'train_" + camera + ".pt'", 1),
                                 ("'fitting_frequency_AA'", "'fitting_frequency_" + camera + "'", 1)]:
        if source.count(before) != count:
            raise ValueError('Frozen camera-label contract changed')
        source = source.replace(before, after)
    return source


def prepare(campaign, steps, fleet=False):
    name = 'resolution_4k_frequency_queue_' + str(steps) + ('_fleet' if fleet else '')
    dest = campaign / name
    if dest.exists():
        raise ValueError('Existing generation must be preserved')
    selected = campaign / ('resolution_4k_AA_frequency' + ('_4000' if steps == 4000 else ''))
    base = json.loads((selected / 'spec_silver.json').read_text())
    source = selected / 'fit_marshawn_4k_AA_frequency.py'
    if sha(source) != base['worker_sha256'] or base['recipe']['train_steps_per_level'] != steps:
        raise ValueError('Selected pilot source changed')
    data = campaign / 'resolution_4k_train63'
    records = json.loads((data / 'source_metadata.json').read_text())['records']
    records = [r for r in records if r['camera'] not in ('DC', 'HC', 'KA')]
    if len(records) != 63 or len({r['camera'] for r in records}) != 63:
        raise ValueError('Exactly 63 unique TRAIN records required')
    plan = json.loads((selected / 'production_plan.json').read_text())
    template = plan['jobs'][0]
    plan['jobs'] = []
    if fleet:
        plan['hosts']['silver']['gpus'] = list(range(8))
    remote = Path(plan['hosts']['silver']['root'])
    dest.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')
    releases = {}
    for record in records:
        camera = record['camera']
        folder = dest / camera
        folder.mkdir()
        worker = folder / source.name
        worker.write_text(camera_worker(source.read_text(), camera))
        spec = copy.deepcopy(base)
        spec['recipe']['camera'] = camera
        spec['actor_crops'] = record['crops']
        spec['worker_sha256'] = sha(worker)
        spec['image'] = str(remote / 'resolution_4k_train63' / record['image'])
        spec['source_complete'] = str(remote / 'resolution_4k_train63/complete.json')
        spec['inputs'] = {str(remote / 'resolution_4k_train63' / rel): sha(data / rel) for rel in [record['image'], 'complete.json', 'source_metadata.json', 'transforms.json', 'box.json', 'generation.json', 'receipts/' + camera + '.json']}
        job = copy.deepcopy(template)
        if fleet:
            job.pop('gpu', None)
        job['name'] = 'resolution4k_freq' + str(steps) + '_' + camera
        spec['output'] = str(remote / 'jobs' / job['name'] / 'run')
        spec['release'] = str(remote / name / camera / 'root_release.json')
        write(folder / 'spec_silver.json', spec)
        job['command'] = ['{host_root}/python', '{host_root}/' + name + '/' + camera + '/' + source.name, '--spec', '{host_root}/' + name + '/' + camera + '/spec_silver.json']
        plan['jobs'].append(job)
        releases[camera] = {'reviewer': 'root', 'frequency_fit_authorized': False, 'spec_sha256': sha(folder / 'spec_silver.json'), 'source_complete_sha256': sha(data / 'complete.json')}
    write(dest / 'production_plan.json', plan)
    write(dest / 'release_templates.json', releases)
    write(dest / 'preparation.json', {'status': 'prepared_not_launched', 'steps_per_level': steps, 'selected_pilot_worker_sha256': sha(source), 'source_complete_sha256': sha(data / 'complete.json'), 'plan_sha256': sha(dest / 'production_plan.json'), 'inventory': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}, 'release_policy': 'Root must first review selected AA result, then authorize each exact spec. Launch only one budget generation. Shared GPU locks, idle GPUs only. Training queue priority requires admission orchestration before dispatch; no preemption. No READY or NeRF training.', 'silver_gpus': plan['hosts']['silver']['gpus'], 'coordinate_frame': 'unchanged accepted floor frame'})
    return dest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--steps', type=int, choices=(1000, 4000), required=True)
    p.add_argument('--fleet', action='store_true')
    a = p.parse_args()
    print(prepare(a.campaign, a.steps, a.fleet))
