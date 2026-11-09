"""Path-only silver source execution alias; prepares a false-release parity gate."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

C = Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SC = '/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
NAME = 'frozen_LC_semantic_source_silver_parity_plan_v1'
P = C / NAME
REMOTE = SC + '/' + NAME
UUID = 'GPU-279718e7-9e60-e5f2-53aa-65ff7dd03f37'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, indent=2) + '\n')


def relocate(v):
    if isinstance(v, str):
        return v.replace(str(C), SC)
    if isinstance(v, list):
        return [relocate(x) for x in v]
    if isinstance(v, dict):
        return {relocate(k): relocate(x) for k, x in v.items()}
    return v


def main():
    P.mkdir(exist_ok=True)
    (P / 'code').mkdir(exist_ok=True)
    source = C / 'frozen_LC_semantic_capture_plan_v4'
    original = json.loads((source / 'manifest.json').read_text())
    for f in (source / 'code').glob('*.py'):
        shutil.copyfile(f, P / 'code' / f.name)
    base = C / 'frozen_LC_semantic_physical_plan_v3/code'
    for name in ['marshawn_LC_frozen_source.py', 'marshawn_gpu_isolation.py',
                 'marshawn_tuning_controller.py', 'prepare_marshawn_frozen_semantic_cpu.py']:
        shutil.copyfile(base / name, P / 'code' / name)
    worker = Path(__file__).with_name('run_marshawn_LC_silver_source_parity.py')
    shutil.copyfile(worker, P / 'code' / worker.name)
    config = Path(original['local_config'])
    config_text = config.read_text().replace(str(C), SC)
    (P / 'config_execution_alias.yml').write_text(config_text)
    m = relocate(original)
    m.update(local_config=REMOTE + '/config_execution_alias.yml',
             python=SC + '/python', expected_physical_UUID=UUID,
             output=REMOTE + '/parity', max_seconds=1800)
    m['bound_inputs'][REMOTE + '/config_execution_alias.yml'] = sha(P / 'config_execution_alias.yml')
    m['code_sha256'] = {f.name: sha(f) for f in sorted((P / 'code').glob('*.py'))}
    # Existing unchanged capture runner authorizes loader restoration only. Its
    # run() and TRAIN loop are never called by this separate parity-only worker.
    write(P / 'source_execution_manifest.json', m)
    source_release = dict(reviewer='root', capture_authorized=False,
                          purpose='SOURCE_LOADER_ONLY_NO_CAPTURE',
                          manifest_sha256=sha(P / 'source_execution_manifest.json'),
                          code_sha256=m['code_sha256'])
    write(P / 'source_loader_release_TEMPLATE.json', source_release)
    spec = dict(phase='silver_source_parity_only', output=REMOTE + '/parity',
                root_release=REMOTE + '/root_release.json',
                LC_source_plan=REMOTE + '/source_execution_manifest.json',
                LC_source_release=REMOTE + '/source_loader_release.json',
                LC_source_worker_name='run_marshawn_lc_semantic_capture_v3.py',
                max_seconds=1800, source_updates=0, optimizer_authorized=False,
                semantic_head_loaded=False, TRAIN_capture_authorized=False,
                selectors_rendered=False)
    write(P / 'spec.json', spec)
    bound = dict(m['bound_inputs'])
    for name, h in m['scientific_files_sha256'].items():
        bound[m['branch'] + '/code/' + name] = h
    for name, h in m['source_archive_sha256'].items():
        bound[m['local_checkpoint'] if name == 'checkpoint.ckpt' else m['source_archive'] + '/' + name] = h
    for name, h in m['code_sha256'].items():
        bound[REMOTE + '/code/' + name] = h
    bound[REMOTE + '/source_execution_manifest.json'] = sha(P / 'source_execution_manifest.json')
    command = [SC + '/python', REMOTE + '/code/' + worker.name, '--spec', REMOTE + '/spec.json', '--execute']
    controller = dict(schema_version=1, poll_seconds=15,
        hosts=dict(silver=dict(address='ubuntu@silver-sable', python=SC+'/python',
            controller=REMOTE+'/code/marshawn_tuning_controller.py', root=REMOTE+'/queue', gpus=[2])),
        jobs=[dict(name='source_RGB_alpha_parity_only', host='silver', gpu=2,
            command=command, cwd=REMOTE+'/code', dependencies=[],
            env=dict(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                PYTHONUNBUFFERED='1', EXPECTED_GPU_UUID=UUID,
                LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json'),
            metrics_file=REMOTE+'/parity/complete.json')])
    write(P / 'controller_plan.json', controller)
    write(P / 'root_review_TEMPLATE.json', dict(reviewer='root', LC_source_parity_authorized=False,
        optimizer_authorized=False, semantic_fit_authorized=False, physical_render_authorized=False,
        source_loader_only=True, spec_sha256={REMOTE+'/spec.json':sha(P/'spec.json')},
        controller_plan_sha256=sha(P/'controller_plan.json'), bound_files=bound))
    write(P / 'canonical_equivalence.json', dict(canonical_manifest=str(source/'manifest.json'),
        canonical_manifest_sha256=sha(source/'manifest.json'),
        alias_manifest_sha256=sha(P/'source_execution_manifest.json'),
        path_replacement=[str(C), SC], additional_execution_fields=['python','expected_physical_UUID','output','max_seconds','local_config','code_sha256','bound_inputs'],
        config_original_sha256=sha(config), config_execution_alias_sha256=sha(P/'config_execution_alias.yml'),
        config_only_path_replacement=config_text.replace(SC,str(C)) == config.read_text(),
        scientific_changes=[], source_seven_states_unchanged=original['expected_serialized_states_sha256']==m['expected_serialized_states_sha256'],
        source_RGB_alpha_only=True, no_CUDA_started=True))
    print(json.dumps(dict(package=str(P), bindings=len(bound), CUDA_started=False)))


if __name__ == '__main__':
    main()
