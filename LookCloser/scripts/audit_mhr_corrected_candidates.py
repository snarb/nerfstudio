"""Replay corrected-prior admission independently of concurrently rendered RGB.

Only the output inventory excludes rgb* subtrees. All numerical depth/semantic
and certificate replay code remains frozen. RGB is verified by its own review.
"""
import argparse
import importlib
import inspect
from pathlib import Path
import probe_mhr_conic_candidates as probe
from study_multiview_face_prior import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-root', type=Path, required=True); args = p.parse_args()
    root = args.candidate_root.resolve()
    prior = Path(read(root / 'candidates/request.json')['final_prior_root']).resolve()
    assert len(root.parts) >= 4 and root != prior
    probe.PRIOR = prior; probe.ROOT = root; probe.configure()
    assert read(root / 'request.json') == probe.control.binding()
    probe.control.configure()
    audit = importlib.import_module('audit_mhr_local_patch_depth')
    assert not (root / 'admission/audit.json').exists()
    source = inspect.getsource(audit.main)
    old = "if p.is_file() and p.name!='audit.json'"
    assert source.count(old) == 1
    source = source.replace(old, old + " and not p.relative_to(OUT).parts[0].startswith('rgb')")
    original_save = audit.save

    def annotated(path, value):
        if Path(path) == root / 'admission/audit.json':
            value = dict(value, geometry_only_inventory_rgb_review_separate=True,
                adapter_sha256=sha(__file__), generated_audit_source=source,
                original_audit_sha256=sha(audit.__file__))
        original_save(path, value)

    ns = dict(audit.__dict__, save=annotated)
    exec(compile(source, '<geometry-only-admission-audit>', 'exec'), ns)
    ns['main']()


if __name__ == '__main__': main()
