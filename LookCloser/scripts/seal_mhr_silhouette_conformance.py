"""Final immutable evidence snapshot for a bounded, unaccepted prior fit."""
from pathlib import Path
import platform
import numpy as np
import scipy
import open3d
from fit_mhr_silhouette_conformance import ROOT
from study_multiview_face_prior import read, save, sha


def main():
    target=ROOT/'final_seal.json';assert not target.exists()
    audit=read(ROOT/'audit.json');assert audit['status']=='passed' and audit['exact_fit_replay']
    checked={}
    def check(path,digest):
        assert sha(path)==digest,str(path);checked[str(path)]=digest
    for path,digest in audit['checked_bindings'].items():check(path,digest)
    check(Path(__file__).with_name('audit_mhr_silhouette_conformance.py'),audit['script_sha256'])
    check(ROOT/'audit_evidence.npz',audit['evidence_sha256'])
    for directory,producer in [('review_v2','review_mhr_silhouette_conformance.py'),('locality','probe_mhr_silhouette_locality.py')]:
        receipt=read(ROOT/directory/'result.json')
        check(Path(__file__).with_name(producer),receipt['script_sha256'])
        for path,digest in receipt['input_hashes'].items():check(path,digest)
    review=read(ROOT/'review_v2/result.json')
    check(Path(__file__).with_name('check_mhr_conformance_crossings.py'),review['crossing_helper_sha256'])
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_silhouette_conformance.md'
    save(target,dict(status='passed',script_sha256=sha(__file__),checked_bindings=checked,
        inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()},
        report_path=str(report),report_sha256=sha(report),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,open3d=open3d.__version__),
        inspected=['C_E_clay','C_E_projection','G_B_clay','requested_hole_prior_only'],
        verdict='partial_cross_view_improvement_but_local_silhouette_failure',outer_converged=False,
        original_mesh_unchanged=True,patch_generated=False,production_accepted=False))
    print('seal passed',len(checked),'bindings')


if __name__=='__main__':main()
