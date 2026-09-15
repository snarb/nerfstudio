"""Hash-bind the rejected, prior-only offset control and its exact replay."""
from pathlib import Path
import hashlib
import numpy as np
import study_mhr_zero_margin as study
from study_multiview_face_prior import read,save,sha


def main():
    root=study.ROOT;final=study.FINAL;target=root/'final_seal.json';assert not target.exists()
    checked={}
    def check(path,digest):
        path=Path(path);assert path.is_file() and sha(path)==digest,str(path);checked[str(path.resolve())]=digest
    q=read(final/'protocol.json');audit=read(final/'audit.json');state=read(final/'continuation_result.json')
    assert audit['status']=='passed' and audit['exact_all_iterates_replayed'] and state['first10_exact']
    assert state['stop_reason']=='hard_cap_not_converged'
    proof=q['zero_margin_adapter']
    for prefix in ['wrapper','frozen_fit','frozen_continuation']:
        check(proof[prefix+'_path'],proof[prefix+'_sha256'])
    for key in ['optimizer','diagnostics','observer']:
        p=proof[key];generated=study.replace_exact(p['original_source'],p['replacements'])
        assert generated==p['generated_source']
        assert hashlib.sha256(generated.encode()).hexdigest()==p['generated_sha256']
        assert hashlib.sha256(p['original_source'].encode()).hexdigest()==p['original_sha256']
    for p,h in audit['checked_bindings'].items():check(p,h)
    e=q['evidence'];check(e['actual_source_mesh'],e['actual_source_mesh_sha256'])
    check(e['mask_path'],e['mask_sha256']);check(Path(e['mask_path']).with_name('cameras.json'),e['mask_names_sha256'])
    check(e['override_path'],e['override_sha256'])
    check(Path(e['override_path']).with_name('request.json'),e['independent_mask_override_request_sha256'])
    check(Path(e['override_path']).with_name('result.json'),e['mask_override_result_sha256'])
    check('/mnt/data/dec5_jaw_measured_mask_control/001193/request.json',e['source_request_sha256'])
    dense=Path(e['depth_receipt']['dense'])
    transforms=Path(e['depth_receipt']['transforms'])
    check(transforms,sha(transforms))
    mapping={r['physical_camera']:r['file_path'] for r in read(transforms)['frames']}
    for name,digest in e['depth_receipt']['depth_sha256'].items():
        check(dense/'stereo/depth_maps'/(mapping[name]+'.geometric.bin'),digest)
    sources={
        final/'review_v2/result.json':'review_mhr_silhouette_conformance.py',
        final/'locality/result.json':'probe_mhr_silhouette_locality.py',
        final/'safety_review/result.json':'review_mhr_silhouette_continuation_safety.py',
        root/'comparison/result.json':'audit_mhr_zero_margin.py',
        root/'topology_localization/result.json':'inspect_mhr_zero_margin_topology.py',
        final/'audit.json':'audit_mhr_silhouette_convergence.py'}
    for receipt,producer in sources.items():
        result=read(receipt);check(Path(__file__).with_name(producer),result['script_sha256'])
        for p,h in result.get('input_hashes',{}).items():check(p,h)
        for p in result.get('files',[]):check(p['path'],p['sha256'])
    safety=read(root/'safety_wrapper.json')
    check(Path(__file__).with_name('review_mhr_zero_margin_safety.py'),safety['wrapper_sha256'])
    check(safety['frozen_helper_path'],safety['frozen_helper_sha256'])
    check(final/'safety_review/result.json',safety['safety_receipt_sha256'])
    assert safety['residual_stats']['margin0']['strict_binary_mask_pass']==14
    assert read(final/'safety_review/result.json')['strict_unmodified_binary_mask_pass']==44
    a=np.load(final/'fit.npz');np.testing.assert_array_equal(a['vertices'][~a['active']],a['baseline'][~a['active']])
    assert not (root/'candidates').exists() and not (root/'admission').exists()
    repo=Path(__file__).resolve().parents[1]
    report=repo/'experiments/dec5_mhr_silhouette_zero_margin.md';tests=repo/'tests/test_mhr_zero_margin.py'
    check(report,sha(report));check(tests,sha(tests));check(__file__,sha(__file__))
    save(target,dict(status='passed',checked_bindings=checked,
        inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
        report_path=str(report),report_sha256=sha(report),tests_path=str(tests),tests_sha256=sha(tests),
        script_sha256=sha(__file__),exact_100_iterates=True,first10_exact=True,
        verdict='rejected_fit_only_topology_regression',patch_generated=False,production_modified=False,
        actual_native_review=['comparison/residual_prior_clay.png','comparison/requested_hole_prior_only.png',
            'comparison/G004_B005_1210FG_clay.png','comparison/C004_E005_1210X7_clay.png',
            'fit100/review_v2/C004_E005_1210X7_projection.png',
            'topology_localization/G004_B005_1210FG_unsafe_wire.png','topology_localization/C004_E005_1210X7_unsafe_wire.png',
            'fit100/safety_review/C004_E005_1210X7.png','fit100/safety_review/G004_B005_1210FG.png']))
    print('seal passed',len(checked),'bindings',flush=True)


if __name__=='__main__':main()
