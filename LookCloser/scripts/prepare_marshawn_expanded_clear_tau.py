"""Freeze a new243-versus7369 qualified-clear target contrast; never launch GPU work."""
import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def prepare(campaign, remote_campaign):
    from marshawn_expanded_clear_tau import PROPOSAL_SHA, REVIEW_SHA, SMALL_SHA, COUNTS
    import numpy as np
    parent = campaign/'negative_tau_candidate_v1'
    root = campaign/'negative_tau_expanded_candidate_v1'
    remote = remote_campaign/root.name
    if root.exists(): raise ValueError('Preserve existing generation')
    proposal = campaign/'native_resolution_pair_100000/documentation/clear_background_polygon_proposal_v1'
    review = campaign/'clear_background_polygon_root_review_v1/review.json'
    assert sha(proposal/'proposal.json') == PROPOSAL_SHA and sha(review) == REVIEW_SHA
    assert sha(parent/'targets.json') == SMALL_SHA
    root.mkdir()
    shutil.copytree(parent/'code', root/'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['marshawn_expanded_clear_tau.py', 'train_marshawn_expanded_clear_tau.py']:
        shutil.copy2(Path(__file__).parent/name, root/'code/LookCloser/scripts'/name)
    for name in ['anchor_audit.json', 'targets.json', 'normalized_config.yml', 'normalized_config_parity.json']:
        shutil.copy2(parent/name, root/name)
    shutil.copytree(parent/'source_archives', root/'source_archives')
    shutil.copytree(proposal, root/'polygon_review', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(review, root/'polygon_review/root_approval.json')
    old = json.loads((parent/'source_inventory.json').read_text())['files']
    for rel, digest in old.items(): assert sha(root/'code'/rel) == digest, rel
    files = {str(p.relative_to(root/'code')): sha(p) for p in (root/'code').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert len(files) == len(old)+2
    write(root/'source_inventory.json', dict(parent_inventory_sha256=sha(parent/'source_inventory.json'),
          parent1503_byte_identical=True, files=files))
    p = json.loads((proposal/'proposal.json').read_text())
    targets = dict(AABB=p['AABB'], targets=[], polygon_review=dict(proposal='polygon_review/proposal.json',
                   review='polygon_review/root_approval.json', original_targets='targets.json'),
                   scope='Only reviewed7369 filtered TRAIN pixels, conditional source floor qualification')
    for record in p['records']:
        ca = record['camera']
        with np.load(proposal/f'{ca}_proposed_pixels.npz') as z:
            assert len(z['x']) == COUNTS[ca]
            for i in range(len(z['x'])):
                targets['targets'].append(dict(camera=ca, x=int(z['x'][i]), y=int(z['y'][i]), origin=z['origin'].tolist(),
                   direction=z['direction'][i].tolist(), near=float(z['near'][i]), far=float(z['far'][i]), floor_t=float(z['floor_t'][i])))
    write(root/'expanded_targets.json', targets)
    original = json.loads((parent/'specs/diagnostic_control/spec.json').read_text())
    for k in ['mode', 'negative_tau_coefficient', 'output', 'release']: original.pop(k)
    original.update(anchor_audit=str(remote/'anchor_audit.json'), worker_sha256=sha(root/'code/LookCloser/scripts/train_marshawn_expanded_clear_tau.py'),
                    frozen_source_hashes={str(remote/'code'/name): digest for name, digest in files.items()})
    hosts = dict(silver=dict(address='ubuntu@silver-sable', root=str(remote/'remote_queue'), python=str(remote_campaign/'python'),
                 controller=str(remote_campaign/'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'), gpus=list(range(8))))
    specs, plans = {}, {}
    for phase in ['zero', 'technical', 'diagnostic']:
        jobs = []
        for arm, coefficient, population in [('control', 0., 'small243'), ('small243', .001, 'small243'), ('expanded7369', .001, 'expanded7369')]:
            name = f'{phase}_{arm}'; folder = root/'specs'/name; folder.mkdir(parents=True)
            target_name = 'expanded_targets.json' if population == 'expanded7369' else 'targets.json'
            spec = dict(original, mode=phase, negative_tau_coefficient=coefficient, target_population=population,
                        targets=str(remote/target_name), targets_sha256=sha(root/target_name), output=str(remote/'runs'/name),
                        release=str(remote/'specs'/name/'root_release.json'))
            write(folder/'spec.json', spec); specs[name] = sha(folder/'spec.json')
            write(folder/'root_release.template.json', dict(reviewer='root', technical_authorized=False, diagnostic_authorized=False,
                      spec_sha256=specs[name], status='PREPARED_NOT_RELEASED'))
            phases = [('_first32', ['--stop-after-attempts', '32'], 'latest.json'), ('_resume32', [], 'complete.json')] if phase == 'technical' else [('', [], 'complete.json')]
            first = None
            for suffix, args, metrics in phases:
                jname = 'expanded_tau111316_'+name+suffix
                jobs.append(dict(name=jname, host='silver', command=[str(remote_campaign/'python'), str(remote/'code/LookCloser/scripts/train_marshawn_expanded_clear_tau.py'), '--spec', str(remote/'specs'/name/'spec.json')]+args,
                   cwd=str(remote/'code/LookCloser'), env=dict(PYTHONPATH=str(remote/'code')+':'+str(remote/'code/LookCloser/scripts'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                   LOOKCLOSER_GPU_LEASE_RECEIPT=str(remote/'remote_queue/jobs'/jname/'receipt.json')),
                   dependencies=[first] if first else [], progress_files=[str(remote/'runs'/name/'progress.json')],
                   metrics_file=str(remote/'runs'/name/metrics)))
                first = jname
        plans[phase] = dict(schema_version=1, poll_seconds=30, hosts=hosts, jobs=jobs)
        write(root/f'{phase}_plan.json', plans[phase])
    diff = ''.join(difflib.unified_diff((parent/'code/LookCloser/scripts/train_marshawn_clear_tau.py').read_text().splitlines(True),
                (root/'code/LookCloser/scripts/train_marshawn_expanded_clear_tau.py').read_text().splitlines(True),
                fromfile='original243_worker', tofile='expanded_population_worker'))
    (root/'worker.diff').write_text(diff)
    write(root/'preparation.json', dict(status='PREPARED_NOT_RELEASED', source_inventory_sha256=sha(root/'source_inventory.json'),
           source_files=len(files), parent1503_unchanged=True, proposal_sha256=PROPOSAL_SHA, root_review_sha256=REVIEW_SHA,
           old_targets_sha256=SMALL_SHA, expanded_targets_sha256=sha(root/'expanded_targets.json'),
           source_checkpoint_sha256=original['parent_checkpoint_sha256'], spec_sha256=specs,
           plan_sha256={phase:sha(root/f'{phase}_plan.json') for phase in plans},
           scientific_factor='Target population243 vs7369 atlambda.001, matched fresh111316control0; no actor/geometry/main sampler change',
           gates=[111816,112316,113316,115316], zero_and32plus32_proof_required=True, GPU_work_launched=False,
           no_global_free_space_or_recipe_promotion=True))
    print(json.dumps(dict(root=str(root), files=len(files), specs=specs), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--remote-campaign', type=Path, required=True)
    a = p.parse_args(); prepare(a.campaign, a.remote_campaign)
