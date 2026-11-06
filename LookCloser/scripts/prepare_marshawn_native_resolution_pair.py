"""Stage a hash-bound paired proposal; this helper never imports a model or launches training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def plain(node):
    import yaml
    if isinstance(node, yaml.MappingNode):
        return {key.value: plain(value) for key, value in node.value}
    if isinstance(node, yaml.SequenceNode):
        return [plain(value) for value in node.value]
    return node.value


def difference(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((difference(a.get(k), b.get(k), prefix + '.' + k) for k in sorted(set(a) | set(b))), [])
    return [] if a == b else [dict(key=prefix, HD=a, TRAIN4K=b)]


def main(args):
    import yaml
    from yaml.nodes import ScalarNode, SequenceNode
    C = args.campaign
    B = C / 'native_semantic_grid_20000'
    F = B / 'native_4k_continuation_plan'
    V = B / 'native_4k_technical_executor_v2'
    Q = C / 'native_reference_4k_technical_v2'
    P = C / 'native_reference_continuation_20000'
    out = args.plan
    out.mkdir(exist_ok=True)
    assert not (out / 'release_candidate.json').exists(), 'Never replace a reviewed generation'
    assert json.loads((Q / 'complete.json').read_text())
    parent = json.loads((Q / 'selected_parent.json').read_text())
    baseline = json.loads((P / 'state.json').read_text())['selection']
    assert baseline['step'] == parent['step'] == 100000
    assert baseline['checkpoint_sha256'] == parent['checkpoint_sha256']
    production = C / 'native_resolution_pair_100000'
    old = json.loads((B / 'native_continuation_plan/root_release.json').read_text())
    technical = json.loads((V / 'root_release.json').read_text())
    spec = json.loads((F / 'adapter/technical_probe_spec.json').read_text())
    source = {key: parent[key] for key in ['step', 'checkpoint', 'checkpoint_sha256', 'runtime', 'runtime_sha256', 'grid', 'grid_sha256']}
    code = out / 'code'
    code.mkdir(exist_ok=True)
    tests = out / 'tests'
    tests.mkdir(exist_ok=True)
    for name in ['run_marshawn_native_resolution_pair.py', 'run_marshawn_native_resolution_chunk.py',
                 'prepare_marshawn_native_resolution_pair.py']:
        shutil.copyfile(args.sources / name, code / name)
    shutil.copyfile(Path(old['evaluator']).parent / 'marshawn_tuning_metrics.py', code / 'marshawn_tuning_metrics.py')
    shutil.copyfile(args.test, tests / args.test.name)
    # The test imports ../scripts; use the same copied modules through a local alias.
    (out / 'scripts').symlink_to('code', target_is_directory=True)
    result = subprocess.run([sys.executable, str(tests / args.test.name)], capture_output=True, text=True)
    (out / 'cpu_tests.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    write(out / 'cpu_test_receipt.json', dict(returncode=result.returncode, python=sys.executable,
          no_GPU_or_model_import=True, tests_sha256=sha(tests / args.test.name), log_sha256=sha(out / 'cpu_tests.log')))
    arms = {}
    trees = {}
    for item in spec['configs']:
        name = 'HD' if item['train_scale'] == 1 else 'TRAIN4K'
        root = yaml.compose(Path(item['config']).read_text())
        def get(node, key):
            return next(v for k, v in node.value if k.value == key)
        def put(node, key, value):
            for index, (k, v) in enumerate(node.value):
                if k.value == key:
                    node.value[index] = k, value
                    return
            raise KeyError(key)
        def scalar(value):
            return ScalarNode('tag:yaml.org,2002:str', str(value))
        def path(value):
            return SequenceNode('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', [scalar(value)])
        gate = out / (name + '_RGB_GATE_NOT_APPROVED.json')
        shutil.copyfile(item['RGB_gate'], gate)
        assert json.loads(gate.read_text())['approved'] is False
        put(root, 'load_checkpoint', path(source['checkpoint']))
        put(root, 'output_dir', path(production / name / 'run'))
        put(root, 'experiment_name', scalar('native_resolution_pair_' + name))
        put(root, 'timestamp', scalar('from100000'))
        pipeline = get(root, 'pipeline')
        put(pipeline, 'final_grid_artifact_dir', path(production / name / 'grids'))
        put(pipeline, 'final_color_gate', scalar(gate))
        put(pipeline, 'final_color_gate_sha256', scalar(sha(gate)))
        config = out / (name + '_config_NOT_RELEASED.yml')
        config.write_text(yaml.serialize(root))
        trees[name] = plain(root)
        arms[name] = dict(train_scale=item['train_scale'], config_template=str(config), RGB_gate_template=str(gate))
    delta = difference(trees['HD'], trees['TRAIN4K'])
    allowed = {'.experiment_name', '.output_dir', '.pipeline.datamanager.data',
               '.pipeline.datamanager.dataparser.data', '.pipeline.final_color_gate',
               '.pipeline.final_color_gate_sha256', '.pipeline.final_grid_artifact_dir', '.pipeline.final_rgb_train_scale'}
    assert {row['key'] for row in delta} <= allowed, delta
    write(out / 'config_comparison.json', dict(differences=delta,
          model_optimizer_scheduler_geometry_sampler_settings_identical=True,
          exact_source_step=100000, LR_preserved=1e-4, initial_dynamic_rays=6825,
          attempted_updates_per_chunk=5000, actual_Adam_updates_recorded_separately=True))
    bound = dict(technical['bound_files'])
    bound.update(old['bound_files'])
    additional = [V / 'root_release.json', Q / 'complete.json', Q / 'selected_parent.json', Q / 'parity.json',
                  Q / 'resource_summary.json', Q / 'zero_update_visual_review/visual_review.json',
                  P / 'complete.json', P / 'state.json', Path(baseline['evaluation']),
                  Path(source['checkpoint']), Path(source['runtime']), Path(source['grid']),
                  Path(source['grid']).with_suffix('.json')]
    additional += list(Q.glob('*/receipt.json'))
    # Bind actual RGB/valid source bytes, not only transform descriptors.
    for dataset in ['data_HD_identity', 'data_TRAIN4K']:
        descriptor = F / 'adapter' / dataset / 'transforms.json'
        additional.append(descriptor)
        frames = json.loads(descriptor.read_text())['frames']
        for frame in frames:
            for key in ['file_path', 'mask_path']:
                if key in frame:
                    value = Path(frame[key])
                    additional.append(value if value.is_absolute() else descriptor.parent / value)
    additional += [p for p in out.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    for path in additional:
        expected = sha(path)
        if str(path) in bound:
            assert bound[str(path)] == expected, path
        bound[str(path)] = expected
    for path, expected in bound.items():
        assert sha(path) == expected, path
    release = dict(production_launch_authorized=False, restoration_probe_authorized=False,
        anchor_step=100000, interval=5000, arm_order=['HD', 'TRAIN4K'], arms=arms,
        source=source, baseline=baseline, output=str(production), bound_files=bound,
        python_path=[str(code)] + spec['python_path'], hostname=socket.gethostname(), python=sys.executable,
        torch_version=technical['torch_version'], cuda_version=technical['cuda_version'],
        GPU_lock=technical['GPU_lock'], minimum_available_RAM_GiB=20, abort_below_available_RAM_GiB=2,
        resource_wait_timeout_seconds=600, worker=str(code / 'run_marshawn_native_resolution_chunk.py'),
        controller=str(code / 'run_marshawn_native_resolution_pair.py'),
        preserved_HD_history=parent['all_HD_checkpoint_controls'],
        **{key: old[key] for key in ['evaluator', 'renderer', 'ready', 'box', 'v2_manifest']})
    write(out / 'release_candidate.json', release)
    write(out / 'review_manifest.json', dict(bound_files=bound,
          release_candidate_sha256=sha(out / 'release_candidate.json'), bound_file_count=len(bound),
          technical_checkpoints_excluded_from_ancestry=True,
          prior_generations_unchanged=True, production_launch_authorized=False))
    print(json.dumps(dict(plan=str(out), bound_files=len(bound), release_candidate_sha256=sha(out / 'release_candidate.json'))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True)
    main(p.parse_args())
