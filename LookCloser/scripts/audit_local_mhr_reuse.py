"""Compare a generic completion replay with a separately sealed legacy study.

This establishes implementation equivalence, not a fresh time or RGB render.
"""
from pathlib import Path
import argparse
import numpy as np
from run_local_mhr_completion import ARM, require, read, save, sha, check_seal


def compare_arrays(a, b):
    x, y = np.load(a), np.load(b)
    require(x.files == y.files, f'Array keys differ: {a}')
    for key in x.files:
        np.testing.assert_array_equal(x[key], y[key])
    return x.files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True)
    a = p.parse_args(); root = a.output.resolve(); reference = a.reference.resolve()
    require(root != reference and not (root / 'equivalence.json').exists(), 'Use distinct new output')
    checked = {}; check_seal(reference, checked)
    audit = read(root / 'audit.json'); require(audit['status'] == 'passed', 'Generic audit incomplete')
    for name, digest in audit['inventory'].items():
        require(sha(root / name) == digest, f'Changed generic artifact: {name}')
    config = read(root / 'config.json')
    require(sha(root / 'config.json') == audit['config_sha256'], 'Config changed after audit')
    for path, digest in config['input_hashes'].items():
        require(sha(path) == digest, f'Changed configured input: {path}')
        checked[path] = digest
    for path in [a.report, a.tests, Path(__file__)]:
        checked[str(path.resolve())] = sha(path)
    require(read(reference / 'request.json')['frame'] == config['frame'], 'Reference time differs')
    require(read(reference / 'request.json')['production_mesh_sha256'] == sha(config['spec']['mesh']), 'Reference base differs')
    arrays = {}; meshes = {}
    for name in ['domain_evidence.npz', 'proposal_evidence.npz']:
        rel = Path('candidates') / ARM / name
        arrays[str(rel)] = compare_arrays(root / rel, reference / rel)
    for name in ['admission.npz', 'certificates.npz', 'strict/evidence.npz', 'interpolated/evidence.npz']:
        rel = Path('admission') / ARM / name
        arrays[str(rel)] = compare_arrays(root / rel, reference / rel)
    for rel in [Path('candidates') / ARM / 'local_raw.ply'] + [Path('admission') / ARM / branch / 'mesh.ply' for branch in ['strict', 'interpolated']]:
        require(sha(root / rel) == sha(reference / rel), f'Mesh bytes differ: {rel}')
        meshes[str(rel)] = sha(root / rel)
    require(read(root / 'admission' / ARM / 'certificates.json') == read(reference / 'admission' / ARM / 'certificates.json'), 'Certificate notes differ')
    reference_visual = read(reference / 'visual_review.json')
    for path, digest in reference_visual['viewed_images'].items():
        require(sha(path) == digest, 'Reference visual changed')
    save(root / 'equivalence.json', dict(status='passed', frame=config['frame'], arrays_exact=arrays, meshes_byte_exact=meshes,
        reference_root=str(reference), reference_seal_sha256=sha(reference / 'final_seal.json'),
        reference_review_sha256=sha(reference / 'visual_review.json'), reference_reviewed_images=reference_visual['viewed_images'],
        original_review_not_rerendered=True, fresh_temporal_transfer=False, new_visual_acceptance=False,
        generic_audit_sha256=sha(root / 'audit.json'), reference_bindings_rehashed=len(checked),
        script_sha256=sha(__file__), production_modified=False, checked_bindings=checked,
        report_path=str(a.report.resolve()), tests_path=str(a.tests.resolve()),
        inventory={str(f.relative_to(root)): sha(f) for f in sorted(root.rglob('*')) if f.is_file()}))
    print('exact equivalence:', len(arrays), 'array archives and', len(meshes), 'PLY files', flush=True)


if __name__ == '__main__':
    main()
