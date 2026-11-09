"""Seal four read-only native-density cases after all three arms terminate."""
import hashlib
import json
from pathlib import Path
import shutil

C = Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SC = '/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
P = C / 'below_floor_density_terminal_readonly_plan_v1'
R = SC + '/' + P.name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    import numpy as np
    if P.exists():
        raise FileExistsError('New density namespace must not overwrite prior attempts')
    inventory_path = C / 'below_floor_density_warm_production_plan_v1/terminal_CPU_grid/selected_checkpoint_inventory.json'
    inventory = json.loads(inventory_path.read_text())
    assert set(inventory) == {'source126316', 'production_control0', 'production_prior1e4', 'production_prior1e3'}
    queries = C / 'independent_occupancy/quarter_distortion_sourcepoint_counterfactual_v1/queries'
    query_files = [queries / 'clothing103.json', queries / 'shoe6.json']
    expected = ['e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c',
                'b8dc52856872bd3f9f26bc1eee3e94110e992bf69ebc60009b68718589f57dde']
    assert [sha(p) for p in query_files] == expected
    transforms = C / 'data/transforms.json'
    assert all(json.loads(p.read_text())['transforms_sha256'] == sha(transforms) for p in query_files)
    observations = [row for p in query_files for row in json.loads(p.read_text())['pixels']]
    actor_xyz = np.array([q['witness_xyz'] for q in observations], dtype=np.float64)
    assert actor_xyz.shape == (109, 3) and len(np.unique(actor_xyz, axis=0)) == 41
    cells_path = C / 'semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/active_r0_low_cells.json'
    cells = json.loads(cells_path.read_text())
    assert len(cells) == 7
    xyz = np.concatenate([np.array([x['center'] for x in cells], np.float64), actor_xyz])
    P.mkdir()
    (P / 'code').mkdir()
    np.savez_compressed(P / 'points.npz', xyz_float64=xyz, xyz_float32=xyz.astype(np.float32),
        world_to_box=np.eye(4), cohort=np.array(['fixed_low_cell'] * 7 + ['actor_observation'] * 109))
    write(P / 'point_provenance.json', dict(point_count=116, actor_observations=109,
        unique_actor_XYZ=41, floor_transform='IDENTITY', native_density_input_dtype='float32',
        no_fitting_or_new_labels=True, fixed_low_cells=cells, actor_observations_original=observations,
        sources={str(p): sha(p) for p in query_files + [cells_path, transforms]},
        coordinate_evidence='clothing103 witness_coordinates explicitly states same floor frame as prepared cameras; shoe6 inherits the identical transforms hash. No second transform.'))
    write(P / 'selected_checkpoint_inventory.json', inventory)
    for name in ['probe_marshawn_below_floor_terminal_density_readonly.py',
                 'compare_marshawn_below_floor_terminal_density_cpu.py']:
        shutil.copyfile(Path(__file__).with_name(name), P / 'code' / name)
    tech = C / 'below_floor_density_warm_technical_plan_v1'
    shutil.copyfile(tech / 'helpers/marshawn_tuning_controller.py', P / 'code/marshawn_tuning_controller.py')
    bound = dict(json.loads((tech / 'specs/control0_full64/spec.json').read_text())['frozen_source_hashes'])
    jobs = []
    hashes = {}
    source = SC + '/distortion03_warm_continuation_candidate_v1'
    for case, gpu in [('production_control0', 5), ('production_prior1e4', 6),
                      ('production_prior1e3', 7), ('source126316', 5)]:
        item = inventory[case]
        assert item['checkpoint'].startswith(SC + '/') and item['config'].startswith(SC + '/')
        spec = dict(root_release=R + '/root_release.json', checkpoint=item['checkpoint'],
            checkpoint_sha256=item['checkpoint_sha256'], step=item['step'], pipeline_digest=item['pipeline_digest'],
            config=item['config'], source_code=source + '/code', source_scripts=source + '/code/LookCloser/scripts',
            points=R + '/points.npz', output=R + '/outputs/' + case)
        path = P / 'specs' / (case + '.json')
        write(path, spec)
        remote = R + '/specs/' + path.name
        hashes[remote] = sha(path)
        bound[remote] = sha(path)
        bound[item['config']] = item['config_sha256']
        if 'completion' in item:
            bound[item['completion']] = item['completion_sha256']
        jobs.append(dict(name=case, host='silver', gpu=gpu,
            command=[SC + '/python', R + '/code/probe_marshawn_below_floor_terminal_density_readonly.py', '--spec', remote, '--execute'],
            cwd=R + '/code', dependencies=['production_control0'] if case == 'source126316' else [],
            env=dict(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json'),
            metrics_file=R + '/outputs/' + case + '/complete.json'))
    write(P / 'controller_plan.json', dict(schema_version=1, poll_seconds=10,
        hosts=dict(silver=dict(address='ubuntu@silver-sable', python=SC + '/python',
            controller=R + '/code/marshawn_tuning_controller.py', root=R + '/queue', gpus=[5, 6, 7])), jobs=jobs))
    for f in [*(P / 'code').glob('*.py'), P / 'points.npz', P / 'point_provenance.json',
              P / 'selected_checkpoint_inventory.json', P / 'controller_plan.json']:
        bound[str(f).replace(str(C), SC)] = sha(f)
    for p in query_files + [cells_path, transforms]:
        bound[str(p).replace(str(C), SC)] = sha(p)
    write(P / 'root_review_TEMPLATE.json', dict(reviewer='root', readonly_density_authorized=False,
        optimizer_authorized=False, head_authorized=False, rendering_authorized=False,
        spec_sha256=hashes, bound_files=bound, controller_plan_sha256=sha(P / 'controller_plan.json'),
        checkpoint_sha256={x['checkpoint']: x['checkpoint_sha256'] for x in inventory.values()},
        scope='Four zero-update actual native-density cases; seven fixed low-cell centers and109 sealed actor observations. No optical/free-space claims. Trained fields use their own actual pipeline digests.'))
    print(json.dumps(dict(package=str(P), jobs=4, point_count=116, bounds=len(bound), GPU_started=False)))


if __name__ == '__main__':
    main()
