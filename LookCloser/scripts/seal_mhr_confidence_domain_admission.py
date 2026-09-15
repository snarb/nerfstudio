"""Verify the domain-only protocol difference and seal both-branch evidence."""
from pathlib import Path
from run_mhr_confidence_domain_admission import OUT, CANDIDATES, CONTROL, binding
from study_multiview_face_prior import read, save, sha


def main():
    target = OUT / 'final_seal.json'
    assert not target.exists()
    request, control = read(OUT / 'request.json'), read(CONTROL / 'request.json')
    assert request['control_wrapper'] == binding()
    delta = {'candidate_request_sha256', 'candidate_result_sha256', 'control_wrapper'}
    assert {k: v for k, v in request.items() if k not in delta} == {k: v for k, v in control.items() if k not in delta}
    candidate = read(CANDIDATES / 'request.json')
    old = read(Path(candidate['parent_open_edge_control']) / 'request.json')
    for key in ['maximum_boundary_vertex_distance', 'maximum_edge', 'maximum_original_surface_distance',
                'minimum_centroid_distance', 'minimum_normal_dot', 'neutral_y_range_cm', 'source_mesh_sha256', 'input_hashes']:
        assert candidate[key] == old[key], key
    assert candidate['actual_opposite_edge_must_be_open'] is False
    assert old['actual_opposite_edge_must_be_open'] is True
    checked = {}

    def check(path, digest):
        path = Path(path)
        assert sha(path) == digest, str(path)
        checked[str(path)] = digest

    audit = read(OUT / 'audit.json')
    assert audit['status'] == 'passed'
    for path, digest in audit['inventory'].items():
        check(OUT / path, digest)
    for arm in request['arms']:
        for path, digest in read(CANDIDATES / arm / 'result.json')['hashes'].items():
            check(CANDIDATES / arm / path, digest)
    for directory, producer in [('native_clay_review', 'review_mhr_depth_admitted_patches.py'),
                                ('branch_difference', 'review_mhr_admission_branch_difference.py'),
                                ('branch_difference_locations', 'localize_mhr_branch_difference.py')]:
        receipt = read(OUT / directory / 'result.json')
        check(Path(__file__).with_name(producer), receipt['script_sha256'])
        for path, digest in receipt['input_hashes'].items():
            check(path, digest)
        for row in receipt['files']:
            check(row['path'], row['sha256'])
    original_review = read(OUT / 'native_clay_review/result.json')
    for row in original_review['statistics']:
        if row['camera'] == 'requested_hole_posthoc':
            assert row['original_missing'] == row['remaining_missing'] == 44
    branches = read(OUT / 'branch_difference/result.json')
    for row in branches['statistics']:
        if row['camera'] == 'requested_hole':
            assert row['fixed_roi']['changed_pixels'] == 0
    report = Path(__file__).resolve().parents[1] / 'experiments/dec5_mhr_local_confidence_admission.md'
    save(target, dict(status='passed', protocol_difference_only_candidate_domain=True,
                      checked_bindings=checked, script_sha256=sha(__file__),
                      report_path=str(report), report_sha256=sha(report),
                      inventory={str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
                      main_visual_review=['native/G_B', 'native/M_B', 'native/E_B', 'native/requested_hole'],
                      independent_visual_review=['native/G_B', 'native/requested_hole', 'branch/smooth400_G_B',
                                                 'locations/smooth025_component1', 'locations/smooth400_component1', 'locations/smooth400_component2'],
                      verdict='fail_goal_no_hole_repair', production_accepted=False))
    print('seal passed', len(checked), 'checked bindings')


if __name__ == '__main__':
    main()
