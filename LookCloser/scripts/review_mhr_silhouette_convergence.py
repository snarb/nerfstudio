"""Rebind frozen native/topology/locality diagnostics to the continuation only."""
from pathlib import Path
import fit_mhr_silhouette_conformance as fit
from continue_mhr_silhouette_convergence import ROOT, CONTROL
from study_multiview_face_prior import read, save, sha


def main():
    proofs={}
    for producer,receipt in [('review_mhr_silhouette_conformance.py','review_v2/result.json'),
                             ('probe_mhr_silhouette_locality.py','locality/result.json')]:
        path=Path(__file__).with_name(producer)
        expected=read(CONTROL/receipt)['script_sha256']
        assert sha(path)==expected
        proofs[str(path)]=expected
    save(ROOT/'review_wrapper.json',dict(script_sha256=sha(__file__),helpers=proofs,
        root=str(ROOT),control=str(CONTROL),protocol_sha256=sha(ROOT/'protocol.json'),
        only_paths_changed=True,target_used_posthoc_only=True))
    fit.ROOT=ROOT
    import review_mhr_silhouette_conformance as review
    import probe_mhr_silhouette_locality as locality
    review.main()
    locality.main()


if __name__=='__main__':main()
