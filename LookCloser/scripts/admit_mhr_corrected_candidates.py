"""Apply unchanged measured-depth admission to explicitly sealed corrected priors.

This is the fixed 001193 experiment adapter, not general frame support or
production approval. Run only after seal/build/probe; semantic-only output is
never substituted for measured admission.
"""
import argparse
from pathlib import Path
import probe_mhr_conic_candidates as probe
from study_multiview_face_prior import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    args = parser.parse_args()
    prior, root = args.prior.resolve(), args.candidate_root.resolve()
    assert len(root.parts) >= 4 and root != prior
    assert root not in prior.parents and prior not in root.parents
    assert read(prior / 'protocol.json')['frame'] == '001193'
    seal = read(prior / 'final_seal.json')
    assert seal['status'] == 'passed'
    for p, h in seal['checked_bindings'].items():
        assert sha(p) == h, p
    for p, h in seal['inventory'].items():
        assert sha(prior / p) == h, p
    probe.PRIOR = prior; probe.ROOT = root; probe.configure()
    control = probe.control
    proof = control.binding()
    assert read(root / 'request.json') == proof
    assert not (root / 'admission').exists()
    posthoc = read(root / 'posthoc_probe/result.json')
    assert posthoc['no_depth_admission_performed']
    control.configure()
    admission = control.admission
    original = admission.save

    def annotated(path, value):
        if Path(path) == admission.OUT / 'request.json':
            value = dict(value, production_base_binding=proof,
                correction_adapter_sha256=sha(__file__),
                corrected_prior_seal_sha256=sha(prior / 'final_seal.json'),
                numerical_admission_unchanged=True)
        original(path, value)

    admission.save = annotated
    try:
        admission.run()
    finally:
        admission.save = original


if __name__ == '__main__':
    main()
