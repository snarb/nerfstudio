"""Package two cold map-only trials with unchanged1495-file scientific source."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from assemble_marshawn_HD_frequency_variants import sha, write


def main(a):
    if a.output.exists():
        raise ValueError('Preserve trial generation')
    assembly = json.loads((a.assembly / 'assembly.json').read_text())
    assert assembly['CPU_only'] and not assembly['training_authorized']
    source_manifest = json.loads((a.recipe / 'source_manifest.json').read_text())
    inherited_release = json.loads((a.recipe / 'release.json').read_text())
    assert inherited_release['approved_for_training'] is True
    assert inherited_release['spec_sha256'] == sha(a.recipe / 'spec.json')
    assert all(inherited_release[k] is True for k in ('native_gpu_paths_passed', 'all_ones_identity_passed', 'adam_resume_passed', 'source_positive_support_passed'))
    assert len(source_manifest['files_sha256']) == 1495
    for name, digest in source_manifest['files_sha256'].items():
        assert sha(a.recipe / 'code' / name) == digest
    a.output.mkdir()
    shutil.copytree(a.recipe / 'code', a.output / 'code', copy_function=os.link,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(a.recipe / 'source_manifest.json', a.output / 'source_manifest.json')
    factory = Path(__file__).with_name('train_marshawn_HD_frequency_variant.py')
    isolation = Path(__file__).with_name('marshawn_gpu_isolation.py')
    for p in (factory, isolation):
        shutil.copy2(p, a.output / p.name)
    remote = a.remote_campaign / a.output.name
    remote_assembly = a.remote_campaign / a.assembly.name
    remote_recipe = a.remote_campaign / a.recipe.name
    parent = json.loads((a.recipe / 'spec.json').read_text())
    policy = dict(schema=1, science_root=str(remote / 'code'), science_manifest=str(remote / 'source_manifest.json'),
        science_manifest_sha256=sha(a.output / 'source_manifest.json'),
        factory_sha256=sha(factory), isolation_sha256=sha(isolation),
        parent_spec=str(remote_recipe / 'spec.json'), parent_spec_sha256=sha(a.recipe / 'spec.json'),
        parent_release=str(remote_recipe / 'release.json'), parent_release_sha256=sha(a.recipe / 'release.json'),
        assembly_sha256=sha(a.assembly / 'assembly.json'), arms={},
        cold_seed=42, no_iteration_ceiling=True, mandatory_end=106316, training_authorized=False)
    jobs = []
    for variant, gpu in [('lr_decay', 4), ('threshold093', 7)]:
        root = a.output / variant;root.mkdir()
        (root / 'code').symlink_to('../code', target_is_directory=True)
        shutil.copy2(a.recipe / 'source_manifest.json', root / 'source_manifest.json')
        spec = copy.deepcopy(parent)
        spec['data'] = str(remote_assembly / variant / 'data')
        spec['output'] = str(remote / variant / 'run')
        spec['frequency_threshold'] = .95 if variant == 'lr_decay' else .93
        spec['semantic_admission']['release_receipt'] = str(remote / variant / 'root_release.json')
        write(root / 'spec.json', spec)
        policy['arms'][variant] = dict(spec=str(remote / variant / 'spec.json'), spec_sha256=sha(root / 'spec.json'),
            ready_sha256=assembly['assemblies'][variant]['ready_sha256'], frequency_threshold=spec['frequency_threshold'])
        write(root / 'root_release.template.json', dict(approved_for_training=False,
            spec_sha256=sha(root / 'spec.json'), semantic_manifest_sha256=spec['semantic_admission']['manifest_sha256'],
            native_gpu_paths_passed=True, all_ones_identity_passed=True, adam_resume_passed=True, source_positive_support_passed=True,
            inherited_proofs_note='These four mechanism proofs belong to the unchanged reviewed quarter.1 parent; no new GPU proof is claimed.',
            inherited_parent_release=str(remote_recipe / 'release.json'),
            external_factory_policy_sha256='SET_FROM_POLICY_AFTER_REVIEW', artifacts_sha256={}))
        jobs.append(dict(name='quarter01_FM_' + variant, host='silver', gpu=gpu, dependencies=[],
            command=[a.python, str(remote / factory.name), '--policy', str(remote / 'policy.json'), '--variant', variant,
                     '--lease-receipt', '{job_root}/receipt.json'],
            cwd=str(remote / 'code/LookCloser'),
            env=dict(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
            progress_files=[str(remote / variant / 'run/progress.json'), str(remote / variant / 'run/latest.json')],
            metrics_file=str(remote / variant / 'run/complete.json')))
    write(a.output / 'policy.json', policy)
    for variant in policy['arms']:
        p = a.output / variant / 'root_release.template.json';r=json.loads(p.read_text())
        r['external_factory_policy_sha256'] = sha(a.output / 'policy.json')
        r['artifacts_sha256'] = {'../policy.json':sha(a.output / 'policy.json'),
                                '../' + factory.name:sha(factory), '../' + isolation.name:sha(isolation)}
        write(p, r)
    controller = a.remote_campaign / 'transport_v2_code/LookCloser/scripts/marshawn_tuning_controller.py'
    plan = dict(schema_version=1, poll_seconds=15,
        hosts=dict(silver=dict(address='ubuntu@silver-sable', python=a.python, controller=str(controller),
                               root=str(remote / 'host'), gpus=[4, 7])), jobs=jobs)
    write(a.output / 'controller_plan.json', plan)
    source_hashes = {str(remote / 'code' / n):h for n,h in source_manifest['files_sha256'].items()}
    for p in [a.output / 'policy.json', a.output / 'controller_plan.json', a.output / factory.name, a.output / isolation.name,
              a.output / 'source_manifest.json'] + list(a.output.glob('*/spec.json')):
        source_hashes[str(remote / p.relative_to(a.output))] = sha(p)
    for variant in policy['arms']:
        data = a.assembly / variant / 'data'
        ready = json.loads((data / 'ready.json').read_text())
        source_hashes[str(remote_assembly / variant / 'data/ready.json')] = sha(data / 'ready.json')
        for n,h in ready['inventory']['files'].items():
            source_hashes[str(remote_assembly / variant / 'data' / n)] = h
        for p in (data / 'lookcloser_frequencies').glob('*_censored.npy'):
            source_hashes[str(remote_assembly / variant / 'data/lookcloser_frequencies' / p.name)] = sha(p)
    source_hashes[str(remote_assembly / 'assembly.json')] = sha(a.assembly / 'assembly.json')
    source_hashes[str(remote_recipe / 'release.json')] = sha(a.recipe / 'release.json')
    source_hashes[str(remote_recipe / 'spec.json')] = sha(a.recipe / 'spec.json')
    write(a.output / 'review_manifest.json', dict(files_sha256=source_hashes, science_files_unchanged=1495,
        policy_sha256=sha(a.output / 'policy.json'), assembly_sha256=sha(a.assembly / 'assembly.json'),
        controller_plan_sha256=sha(a.output / 'controller_plan.json'),
        controller='Existing transport_v2, local-origin SSH; one global lease, physical UUID before Torch',
        source_changes=[], external_factory_changes=['Exact map provenance validator accepts sealed .93; original scientific train/build_config unchanged'],
        launch_authorized=False, quality_promotion=False))
    print(json.dumps(dict(root=str(a.output), policy_sha256=sha(a.output / 'policy.json'), bindings=len(source_hashes))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('assembly','recipe','output','remote-campaign'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--python',default='/home/ubuntu/lookcloser/envs/marshawn_leader/bin/python')
    main(p.parse_args())
