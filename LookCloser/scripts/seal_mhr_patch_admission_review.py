"""Bind the completed numeric audit and separate posthoc visual diagnostics."""
from pathlib import Path
from study_multiview_face_prior import read, save, sha

ROOT = Path('/mnt/data/dec5_mhr_local_patch_admission')
PRODUCERS = {
    'native_clay_review': 'review_mhr_depth_admitted_patches.py',
    'target_admission_diagnosis': 'diagnose_mhr_target_admission.py',
    'mask_veto_witnesses': 'inspect_mhr_mask_veto_witnesses.py',
}


def main():
    target = ROOT / 'final_seal.json'
    assert not target.exists(), 'Preserve the existing seal'
    audit = read(ROOT / 'audit.json')
    assert audit['status'] == 'passed'
    checked = {}

    def check(path, expected):
        path = Path(path)
        assert sha(path) == expected, str(path)
        checked[str(path)] = expected

    for name, digest in audit['inventory'].items():
        check(ROOT / name, digest)
    for directory, script in PRODUCERS.items():
        receipt = read(ROOT / directory / 'result.json')
        check(Path(__file__).with_name(script), receipt['script_sha256'])
        for path, digest in receipt.get('input_hashes', {}).items():
            check(path, digest)
        for path, digest in receipt.get('source_rgb_hashes', {}).items():
            check(path, digest)
        for row in receipt.get('files', []):
            check(row['path'], row['sha256'])
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
    report = Path(__file__).resolve().parents[1] / 'experiments/dec5_mhr_local_patch_admission.md'
    save(target, dict(status='passed', script_sha256=sha(__file__), checked_external_bindings=checked,
                     inventory=files, report_path=str(report), report_sha256=sha(report),
                     main_visual_review=['G004_B005_1210FG', 'M004_B005_12109O', 'E004_B005_1210I7', 'requested_hole'],
                     independent_visual_review=['requested_hole', 'G004_B005_1210FG', 'mask/C004_E005_1210X7', 'mask/E004_D005_1210L4'],
                     verdict='fail_goal_no_hole_repair', production_accepted=False))
    print('seal passed', len(files), 'files;', len(checked), 'checked bindings')


if __name__ == '__main__':
    main()
