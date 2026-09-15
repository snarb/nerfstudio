"""Seal continuation evidence without promoting the folded full prior."""
from pathlib import Path
from continue_mhr_silhouette_convergence import ROOT
from study_multiview_face_prior import read,save,sha


def main():
    target=ROOT/'final_seal.json';assert not target.exists()
    q=read(ROOT/'protocol.json');audit=read(ROOT/'audit.json');state=read(ROOT/'continuation_result.json')
    assert audit['status']=='passed' and audit['exact_all_iterates_replayed'] and state['first10_exact']
    checked={}
    def check(path,digest):assert sha(path)==digest,str(path);checked[str(path)]=digest
    check(q['continuation']['wrapper_path'],q['continuation']['wrapper_sha256'])
    check(q['continuation']['frozen_producer_path'],q['continuation']['frozen_producer_sha256'])
    for path,digest in audit['checked_bindings'].items():check(path,digest)
    check(Path(__file__).with_name('audit_mhr_silhouette_convergence.py'),audit['script_sha256'])
    check(ROOT/'audit_evidence.npz',audit['evidence_sha256'])
    wrapper=read(ROOT/'review_wrapper.json')
    check(Path(__file__).with_name('review_mhr_silhouette_convergence.py'),wrapper['script_sha256'])
    for path,digest in wrapper['helpers'].items():check(path,digest)
    safety=read(ROOT/'safety_review/result.json')
    check(Path(__file__).with_name('review_mhr_silhouette_continuation_safety.py'),safety['script_sha256'])
    for path,digest in safety['input_hashes'].items():check(path,digest)
    for row in safety['files']:check(row['path'],row['sha256'])
    assert safety['strict_unmodified_binary_mask_pass']==safety['queries']==44
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_silhouette_convergence.md'
    tests=Path(__file__).resolve().parents[1]/'tests/test_mhr_silhouette_convergence.py'
    save(target,dict(status='passed',script_sha256=sha(__file__),checked_bindings=checked,
        inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()},
        report_path=str(report),report_sha256=sha(report),tests_path=str(tests),tests_sha256=sha(tests),
        main_inspected=['requested_hole_prior_only','C_E_clay','G_B_projection'],
        independent_inspected=['C_E_projection','C_E_clay','requested_hole_prior_only','safety_C_E','safety_G_B'],
        stop_reason=state['stop_reason'],verdict='promising_local_prior_but_global_folds_and_not_converged',
        original_geometry_unchanged=True,patch_generated=False,production_accepted=False))
    print('seal passed',len(checked),'bindings')


if __name__=='__main__':main()
