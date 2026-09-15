"""Run unchanged native and hole-locality reviewers on separate conformance fits."""
import argparse
from pathlib import Path
from conform_mhr_measured_surface import ROOT,ARMS
from build_train_hair_semantics import read,sha,write


def main(action):
    scripts=['review_mhr_local_head_prior.py','probe_mhr_local_head_prior.py']
    request=dict(script_sha256=sha(__file__),reviewer_hashes={name:sha(Path(__file__).with_name(name)) for name in scripts},
        protocol_sha256=sha(ROOT/'protocol.json'),fit_summary_sha256=sha(ROOT/'fit_summary.json'),
        target_used_for_fit=False,original_geometry_changed=False)
    if (ROOT/'review_request.json').exists():assert read(ROOT/'review_request.json')==request
    else:write(ROOT/'review_request.json',request)
    if action=='review':
        import review_mhr_local_head_prior as worker
    else:
        import probe_mhr_local_head_prior as worker
    worker.OUT=ROOT
    worker.main(list(ARMS))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['review','probe'])
    main(parser.parse_args().action)
