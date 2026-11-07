"""CPU-only verification of the isolated W32 continuation package, not release."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(root, base):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(root/'code/LookCloser/scripts'), str(root/'code')]
    import numpy as np
    import torch
    import marshawn_margin_continuation_policy as runtime
    from marshawn_source_point_admission import SourcePointAdmission
    from marshawn_source_margin_admission import GeometricSourcePointAdmission

    torch.set_num_threads(4)
    assert Path(runtime.__file__).resolve() == (root/'code/LookCloser/scripts/marshawn_margin_continuation_policy.py').resolve()
    inventory = read(root/'source_manifest.json')['files_sha256']
    assert len(inventory) == 1506
    assert sha(root/'source_manifest.json') == sha(base/'source_manifest.json')
    for name, digest in inventory.items():
        assert sha(root/'code'/name) == digest == sha(base/'code'/name), name
    anchor = read(root/'anchor_audit.json')
    for name, digest in anchor['source_hashes'].items():
        assert sha(name) == digest, name
    assert anchor['fullstate_hashes'] == read(base/'anchor_audit.json')['fullstate_hashes']
    assert anchor['actual_Adam'] == {'fields': [126262.0]}
    policy = read(root/'external_guard_margin_v1/policy.json')
    for name, digest in policy['prepared_bindings'].items():
        assert sha(root/name) == digest, name
    for specpath in (root/'specs').glob('*/spec.json'):
        spec = read(specpath)
        for path_key, digest_key in [('parent_checkpoint','parent_checkpoint_sha256'),
                                    ('parent_spec','parent_spec_sha256'),
                                    ('anchor_audit','anchor_audit_sha256'),
                                    ('margin_manifest','margin_manifest_sha256'),
                                    ('zero_reference','zero_reference_sha256')]:
            assert sha(spec[path_key]) == spec[digest_key], (specpath, path_key)
        assert not Path(spec['release']).exists(), 'Preparation must not authorize GPU work'
        reference = read(spec['zero_reference'])
        for name, digest in reference['artifacts_sha256'].items():
            assert sha(name) == digest, name
    manifest_path = root/'margin/manifest.json'
    manifest = read(manifest_path)
    assert manifest['source_geometry_support_width_native_pixels'] == 32
    assert manifest['original_RGB_loss_width'] == 64
    assert manifest['source_key'] == 'semantic_gate_pad1'
    assert sha(manifest['probe_positions']) == manifest['probe_positions_sha256']
    source = np.load(manifest['source_grid'], allow_pickle=False)
    old = np.load(base/'margin/source_grid.npz', allow_pickle=False)
    gate = source['semantic_gate_pad1']
    assert gate.sum() == 91703 and old['semantic_gate_pad1'].sum() == 105049
    assert np.array_equal(source['aabb'], old['aabb'])
    assert not np.any(gate & ~old['semantic_gate_pad1'])
    # Confirm the actual installed generic runtime queries W32 rather than the
    # prior W64 source; independently instantiate the audited lookup as reference.
    indices = np.argwhere(old['semantic_gate_pad1'])
    centers = source['aabb'][0] + (indices+.5)*(source['aabb'][1]-source['aabb'][0])/128
    positions = torch.from_numpy(centers)

    class Field:
        _semantic_admission_installed = True

        def _normalize_positions(self, points):
            return points, torch.ones(points.shape[:-1], dtype=torch.bool), points.shape[:-1], points

    model = SimpleNamespace(field=Field(), device=torch.device('cpu'))
    rng_before = torch.get_rng_state().clone()
    installed = runtime.MarginPolicy(model, None, manifest_path, sha(manifest_path), 'source_margin002')
    actual = model.field._normalize_positions(positions)[1]
    expected = GeometricSourcePointAdmission(torch.from_numpy(gate), source['aabb'],
        margin=.002, epsilon=2e-7, base_class=SourcePointAdmission)(positions)
    assert torch.equal(actual, expected) and torch.equal(torch.get_rng_state(), rng_before)
    assert actual.sum() == gate.sum() and (~actual).sum() == 13346
    assert installed.binding['source_grid_sha256'] == manifest['source_grid_sha256']
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, CPU_only=True, GPU_started=False, source_files=1506,
        all_science_bytes_identical=True, anchor_seven_hashes_identical=True,
        anchor_checkpoint_sha256=anchor['checkpoint_sha256'], actual_Adam=anchor['actual_Adam'],
        root_policy_sha256=sha(root/'external_guard_margin_v1/policy.json'),
        source_grid_sha256=sha(manifest['source_grid']), source_grid_cells=int(gate.sum()),
        prior_W64_cells=105049, tested_W64_centers=len(positions),
        actual_W32_selector_retains=int(actual.sum()), actual_W32_selector_excludes=int((~actual).sum()),
        actual_runtime_matches_audited_lookup=True, global_torch_RNG_unchanged=True,
        CPU_config_sampler_resume_sha256=sha(root/'CPU_readiness.json'),
        original_RGB_width=64, no_actual_root_releases=True,
        imported_runtime=dict(path=runtime.__file__, sha256=sha(runtime.__file__)))
    (root/'CPU_binding_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    args = parser.parse_args()
    verify(args.root, args.base)
