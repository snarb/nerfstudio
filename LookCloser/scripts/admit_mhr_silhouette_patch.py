"""Frozen admission mathematics, explicitly rebound to the final silhouette prior."""
from pathlib import Path
import argparse
import admit_mhr_local_patch_depth as admission
from build_mhr_silhouette_patch_candidates import ROOT as CANDIDATES,PRIOR,ARM
from study_multiview_face_prior import read,save,sha

OUT=Path('/mnt/data/dec5_mhr_silhouette_patch_admission')
CONTROL=Path('/mnt/data/dec5_mhr_local_patch_admission')


def configure():
    admission.CANDIDATES=CANDIDATES
    admission.OUT=OUT
    admission.PRIOR=CANDIDATES/'prior'
    admission.ARMS=[ARM]


def binding():
    q=read(CONTROL/'request.json');producer=Path(admission.__file__)
    assert sha(producer)==q['script_sha256']
    for p,h in q['helpers'].items():assert sha(producer.with_name(p))==h,p
    for p in [CANDIDATES/'prior/initial.npz',CANDIDATES/'prior'/ARM/'fit.npz']:
        assert p.resolve()==(PRIOR/'fit.npz').resolve()
        assert sha(p)==sha(PRIOR/'fit.npz')
    return dict(wrapper_path=str(Path(__file__).resolve()),wrapper_sha256=sha(__file__),
        frozen_admission_path=str(producer.resolve()),frozen_admission_sha256=sha(producer),
        control_request_sha256=sha(CONTROL/'request.json'),candidate_root=str(CANDIDATES),output_root=str(OUT),
        actual_prior_root=str(PRIOR),actual_prior_fit_sha256=sha(PRIOR/'fit.npz'),
        actual_prior_topology_sha256=sha(PRIOR/'review_v2/silhouette_topology.npz'),actual_prior_seal_sha256=sha(PRIOR/'final_seal.json'),
        alias_root=str(CANDIDATES/'prior'),alias_files_are_exact_final_prior=True,admission_math_unchanged=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    proof=binding();configure()
    if args.audit:
        assert read(OUT/'request.json')['final_prior_binding']==proof
        import audit_mhr_local_patch_depth as audit
        audit.main()
    else:
        original=admission.save
        def extended(path,value):
            if Path(path)==OUT/'request.json':value=dict(value,final_prior_binding=proof)
            original(path,value)
        admission.save=extended
        try:admission.run()
        finally:admission.save=original


if __name__=='__main__':main()
