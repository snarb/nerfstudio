"""CPU source/config audit and actual frozen ARM arithmetic probes; never CUDA."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type) or callable(value):
        return value.__module__ + '.' + value.__qualname__
    if dataclasses.is_dataclass(value):
        return {f.name: normalize(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for key in set(a) | set(b):
            result.update(differences(a.get(key), b.get(key), prefix + '.' + key))
        return result
    return {} if a == b else {prefix: dict(before=a, after=b)}


def main(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(root / 'code/LookCloser/scripts'), str(root / 'code')]
    import torch
    import nerfacc
    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.model_components.lookcloser_samplers import FrequencyAwareVolumetricSampler
    from marshawn_corrected_arm_warm_factory import ARMS, build_config, build_effective_config
    from train_marshawn_corrected_arm_warm import validate_spec
    from types import SimpleNamespace
    read = lambda path: json.loads(Path(path).read_text())
    manifest = read(root / 'source_manifest.json')
    assert manifest['unchanged_original_files'] == 1495 and not manifest['changed_original_files']
    for name, digest in manifest['files_sha256'].items():
        assert sha(root / 'code' / name) == digest, name
    parent, anchor = read(root / 'anchor/spec.json'), read(root / 'anchor_audit.json')
    reference = normalize(build_config(parent, anchor, root / 'anchor/checkpoint.ckpt', root / 'out'))
    configs = {}
    for phase in ('zero', 'technical', 'production'):
        for arm in ARMS:
            spec = read(root / 'specs' / (phase + '_' + arm) / 'spec.json')
            validate_spec(spec)
            cfg = build_effective_config(parent, anchor, root / 'anchor/checkpoint.ckpt', root / 'out', arm)
            delta = differences(reference, normalize(cfg))
            assert set(delta) == ({'.pipeline.model.corrected_arm_allocator'} if ARMS[arm] else set()), delta
            configs[phase + '_' + arm] = delta
            bad = copy.deepcopy(spec)
            bad['arm'] = 'legacyuncap0'
            try:
                validate_spec(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Cap factor silently accepted')
    class Occupancy:
        def sampling(self, **kwargs):
            return torch.tensor([0, 0]), torch.tensor([0., 1.]), torch.tensor([1., 2.])
    class Frequency:
        aabb_size_buf = torch.ones(3)
        def query(self, positions):
            return torch.zeros(len(positions))
        def level_to_freq(self, levels):
            return torch.full_like(levels, 1. / (2. * .6))
    rays = RayBundle(origins=torch.zeros(1, 3), directions=torch.tensor([[1., 0., 0.]]), pixel_area=torch.ones(1, 1))
    sampler = FrequencyAwareVolumetricSampler(Occupancy(), Frequency())
    original_pack = nerfacc.pack_info
    def cpu_pack(indices, num_rays):
        assert not indices.is_cuda
        return sampler._pack_info_from_sorted_indices(indices, num_rays)
    nerfacc.pack_info = cpu_pack
    cases = {}
    try:
        for name, corrected, cap in (('legacy1024', False, 1024), ('corrected1024', True, 1024),
                                     ('legacyuncap0', False, 0), ('legacy3', False, 3), ('corrected3', True, 3)):
            samples, ids, stats = sampler(rays, render_step_size=.5, near_plane=0., far_plane=2.,
                alpha_thre=0., early_stop_eps=0., cone_angle=0., adaptive_min_step_size=1e-6,
                adaptive_max_step_size=100., adaptive_min_frequency_level=0., adaptive_max_frequency_level=None,
                adaptive_interval_level_mode='midpoint', max_steps_per_ray=cap, corrected_allocator=corrected)
            cases[name] = dict(starts=samples.frustums.starts.reshape(-1).tolist(),
                ends=samples.frustums.ends.reshape(-1).tolist(), samples=int(stats.num_samples))
    finally:
        nerfacc.pack_info = original_pack
    assert cases['legacy1024'] == cases['legacyuncap0']
    assert cases['legacy1024']['starts'] != cases['corrected1024']['starts']
    assert cases['legacy3']['ends'][-1] < 2. and cases['corrected3']['ends'][-1] == 2.
    assert cases['corrected3']['samples'] == 3
    assert not torch.cuda.is_initialized() and nerfacc.pack_info is original_pack
    result = dict(passed=True, CPU_only=True, CUDA_initialized=False, READY=False,
        source1495_exact=True, configuration_differences=configs,
        arithmetic_probe=cases, actual_frozen_forward_executed=True,
        CPU_pack_metadata_adapter_restored=True, native_CUDA_traversal_not_tested=True,
        zero_update_source_parity='PENDING', actual_Adam32_plus32='PENDING',
        external_guard_and_physical_UUID_binding='PENDING_ROOT_REVIEW',
        GPU_started=False, releases_created=False)
    (root / 'CPU_readiness.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, configs=6, actual_frozen_forward_cases=5, READY=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
