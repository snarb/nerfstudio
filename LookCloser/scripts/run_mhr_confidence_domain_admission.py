"""Run the frozen admission protocol on a separately declared proposal domain.

Only input/output paths and provenance are changed; admission helpers are intact.
"""
from pathlib import Path
import argparse
import admit_mhr_local_patch_depth as admission
from study_multiview_face_prior import read, save, sha

CANDIDATES = Path('/mnt/data/dec5_mhr_local_confidence_domain')
OUT = Path('/mnt/data/dec5_mhr_local_confidence_admission')
CONTROL = Path('/mnt/data/dec5_mhr_local_patch_admission')


def binding():
    frozen = read(CONTROL / 'request.json')
    producer = Path(admission.__file__)
    assert sha(producer) == frozen['script_sha256']
    for name, digest in frozen['helpers'].items():
        assert sha(producer.with_name(name)) == digest, name
    candidate = read(CANDIDATES / 'request.json')
    assert candidate['actual_opposite_edge_must_be_open'] is False
    assert candidate['target_camera_or_rgb_used'] is False
    assert candidate['raw_overlapping_surface_not_acceptable'] is True
    return dict(wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
                frozen_producer_path=str(producer.resolve()), frozen_producer_sha256=sha(producer),
                original_control=str(CONTROL), original_control_request_sha256=sha(CONTROL / 'request.json'),
                candidate_root=str(CANDIDATES), candidate_request_sha256=sha(CANDIDATES / 'request.json'),
                output_root=str(OUT), algorithm_and_thresholds_unchanged=True,
                domain_difference='nearest-opposite-open-edge constraint removed before unchanged admission')


def configure():
    admission.CANDIDATES = CANDIDATES
    admission.OUT = OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    proof = binding()
    configure()
    if args.audit:
        assert read(OUT / 'request.json')['control_wrapper'] == proof
        # Import after configuration: the frozen replay binds these globals.
        import audit_mhr_local_patch_depth as audit
        audit.main()
    else:
        original_save = admission.save

        def save_with_binding(path, data):
            if Path(path) == OUT / 'request.json':
                data = dict(data, control_wrapper=proof)
            original_save(path, data)

        # Provenance-only request extension; no geometry or evidence helper change.
        admission.save = save_with_binding
        try:
            admission.run()
        finally:
            admission.save = original_save


if __name__ == '__main__':
    main()
