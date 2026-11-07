"""External exact-spec map factory around unchanged quarter-step/.1 science."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(a):
    policy = json.loads(a.policy.read_text())
    arm = policy['arms'][a.variant]
    spec_path = Path(arm['spec'])
    assert sha(spec_path) == arm['spec_sha256']
    assert sha(policy['parent_release']) == policy['parent_release_sha256']
    assert sha(__file__) == policy['factory_sha256']
    assert sha(Path(__file__).with_name('marshawn_gpu_isolation.py')) == policy['isolation_sha256']
    root = Path(policy['science_root'])
    manifest = Path(policy['science_manifest'])
    assert sha(manifest) == policy['science_manifest_sha256']
    inventory = json.loads(manifest.read_text())['files_sha256']
    assert len(inventory) == 1495
    for name, digest in inventory.items():
        if sha(root / name) != digest:
            raise ValueError('Frozen science changed: ' + name)
    spec = json.loads(spec_path.read_text())
    assert sha(Path(spec['data']) / 'ready.json') == arm['ready_sha256']
    if a.dry_run:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if a.lease_receipt is None:
            raise ValueError('Actual supervising global GPU lease required')
        from marshawn_gpu_isolation import require_controller_lease
        lease = require_controller_lease(a.lease_receipt)
    sys.path[:0] = [str(root), str(root / 'LookCloser/scripts')]
    import train_marshawn_semantic_admission as base
    assert Path(base.__file__).resolve() == (root / 'LookCloser/scripts/train_marshawn_semantic_admission.py').resolve()
    original_validate = base.validate_spec

    def validate_exact(value):
        if value != spec:
            raise ValueError('Factory accepts only the sealed exact map spec')
        assert value['frequency_threshold'] == arm['frequency_threshold']
        # The inherited validator describes the original .95 map provenance.
        # Validate every other scientific setting with it, then return the
        # unchanged real spec; never relabel .93 metadata or its runtime input.
        reference = copy.deepcopy(value)
        reference['frequency_threshold'] = .95
        original_validate(reference)
        return value

    base.validate_spec = validate_exact
    validate_exact(spec)
    ready = base.validate_data(Path(spec['data']))
    assert ready['inventory']['frequency_recipe']['ssim_threshold'] == arm['frequency_threshold']
    if a.dry_run:
        import torch
        import yaml
        parent_path = Path(policy['parent_spec'])
        assert sha(parent_path) == policy['parent_spec_sha256']
        parent = json.loads(parent_path.read_text())
        original_validate(parent)
        before = base.build_config(parent)
        after = base.build_config(spec)
        def normalized(config, source):
            config = copy.deepcopy(config)
            assert config.output_dir == Path(source['output'])
            assert config.pipeline.datamanager.data == Path(source['data'])
            assert config.pipeline.datamanager.dataparser.data == Path(source['data'])
            assert config.pipeline.model.semantic_data == source['data']
            config.output_dir = Path('<OUTPUT>')
            config.pipeline.datamanager.data = Path('<DATA>')
            config.pipeline.datamanager.dataparser.data = Path('<DATA>')
            config.pipeline.model.semantic_data = '<DATA>'
            return yaml.dump(config)
        left, right = normalized(before, parent), normalized(after, spec)
        if left != right:
            import difflib
            raise ValueError('Scientific config differs:\n' + ''.join(difflib.unified_diff(left.splitlines(True), right.splitlines(True))))
        assert after.load_checkpoint is None and after.machine.seed == 42
        assert not torch.cuda.is_initialized()
        result = dict(passed=True, CPU_only=True, GPU_used=False, variant=a.variant,
            factory_sha256=sha(__file__), policy_sha256=sha(a.policy), spec_sha256=sha(spec_path),
            ready_sha256=sha(Path(spec['data']) / 'ready.json'), science_files_verified=1495,
            normalized_config_equal=True, normalized_config_sha256=hashlib.sha256(right.encode()).hexdigest(),
            only_config_differences=['data paths', 'output paths'], seed=42, cold_checkpoint=None,
            frequency_threshold=spec['frequency_threshold'], mandatory_end=106316, iteration_limit=None,
            plateau=base.PlateauTracker().to_dict(),
            FR_switch=after.pipeline.feature_reweighting_switch_step,
            FR_after=after.pipeline.feature_reweighting_after_switch,
            source_manifest_sha256=sha(manifest), training_launched=False)
        if a.output is None or a.output.exists():
            raise ValueError('New CPU proof output required')
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
        return
    from run_marshawn_semantic_release import verify
    release = verify(spec_path)
    if release.get('external_factory_policy_sha256') != sha(a.policy):
        raise ValueError('Exact external factory release required')
    import torch
    from marshawn_gpu_isolation import verify_cuda
    gpu = verify_cuda(lease['expected_physical_uuid'], torch)
    torch.cuda.set_per_process_memory_fraction(10 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=True)
    provenance = dict(factory_sha256=sha(__file__), policy_sha256=sha(a.policy),
        source_science_manifest_sha256=sha(manifest), spec_sha256=sha(spec_path),
        lease=lease, GPU=gpu, allocator_cap_GiB=10,
        scientific_source_modified=False, model_or_sampler_hooks_installed=False,
        config_override='Exact map provenance validation only; actual spec/metadata retained')
    receipt = output / ('external_factory_resume.json' if (output / 'latest.json').exists() else 'external_factory_cold.json')
    receipt.write_text(json.dumps(provenance, indent=2) + '\n')
    base.train(spec_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--variant', choices=['lr_decay', 'threshold093'], required=True)
    p.add_argument('--lease-receipt', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output', type=Path)
    main(p.parse_args())
