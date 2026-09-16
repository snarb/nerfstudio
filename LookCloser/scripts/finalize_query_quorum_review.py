"""Seal already inspected bounded controls; never promotes them to production.

Visual observations below are the main agent's actual image review, not an
automatic image-quality classifier. Raw pending computation records remain
immutable; this separate receipt resolves the human/LLM review status.
"""
from pathlib import Path
from study_multiview_face_prior import read, save, sha
from study_query_support_quorum import ROOT, FRAME
from combine_query_quorum_texture_guard import ROOT as TEXTURE
from review_measured_free_surface import VIEWS


def main():
    root = ROOT / FRAME
    output = root / 'final_review.json'
    assert not output.exists()
    checked = {}

    def verify(path, digest):
        path = Path(path)
        assert sha(path) == digest, str(path)
        checked[str(path)] = digest

    def bind(path):
        verify(path, sha(path))

    q, r = read(root / 'request.json'), read(root / 'result.json')
    verify(root / 'request.json', r['request_sha256'])
    verify(q['mesh'], q['mesh_sha256'])
    verify(Path(__file__).with_name('study_query_support_quorum.py'), q['script_sha256'])
    for p, digest in q['helpers'].items(): verify(p, digest)
    for name, digest in r['hashes'].items(): verify(root / name, digest)
    audit = read(root / 'independent_audit.json')
    verify(root / 'request.json', audit['request_sha256'])
    verify(root / 'result.json', audit['result_sha256'])
    assert audit['all_removed_samples_replayed'] and audit['vertices_and_triangle_subset_verified']
    assert audit['removed_triangles'] == 300 and not audit['quality_approval']
    bind(root / 'independent_audit.json'); bind(root / 'audit_adapter.json')
    review = read(root / 'review/result.json')
    for p, digest in review['input_hashes'].items(): verify(p, digest)
    for p, digest in review['images'].items(): verify(root / 'review' / p, digest)
    bind(root / 'review/result.json')
    sheets = read(root / 'review/side_effect_sheets.json')
    verify(root / 'review/result.json', sheets['review_sha256'])
    for s in sheets['records']: verify(s['path'], s['sha256'])
    adapter = read(TEXTURE / 'adapter.json')
    for key, name in [('wrapper_sha256', 'combine_query_quorum_texture_guard.py'),
                      ('producer_sha256', 'study_measured_texture_visibility.py'),
                      ('reviewer_sha256', 'review_measured_texture_visibility.py')]:
        verify(Path(__file__).with_name(name), adapter[key])
    verify(root / 'result.json', adapter['geometry_result_sha256'])
    bind(TEXTURE / 'adapter.json')
    records = []
    for view in VIEWS:
        folder = TEXTURE / FRAME / view
        verify(root / 'rgb' / view / 'request.json', adapter['baseline_requests'][view])
        result = read(folder / 'review/result.json')
        assert result['mesh_and_target_depth_exact']
        assert result['actual_sources_contradicting_stable_far'][1] == 0
        for p, digest in result['bindings'].items(): verify(p, digest)
        for p, digest in result['hashes'].items(): verify(folder / 'review' / p, digest)
        verify(folder / 'guard_audit.json', result['guard_audit_sha256'])
        bind(folder / 'review/result.json')
        records.append({k: result[k] for k in ['view', 'changed_rgb', 'new_black_rgb',
            'actual_sources_contradicting_stable_far', 'diagnostic_pixels', 'diagnostic_contradicting_sources']})
    blue = root / 'blue_wedge_support'
    for folder, mapping in [(blue, 'outputs'), (blue / 'semantic_witnesses', 'images')]:
        result = read(folder / 'result.json')
        for p, digest in result['input_hashes'].items(): verify(p, digest)
        for p, digest in result[mapping].items(): verify(folder / p, digest)
        bind(folder / 'result.json')
    actual_viewed = [blue / 'regions.png']
    for view in VIEWS:
        actual_viewed += [root / 'review' / view / 'lipstick_native.png',
                          root / 'review' / view / 'all_side_effects_native.png']
        actual_viewed += [TEXTURE / FRAME / view / 'review' / (name + '.png') for name in
                          ['head_native', 'head_new_black', 'lipstick_native', 'lipstick_new_black']]
    actual_viewed += sorted((blue / 'semantic_witnesses').glob('*.png'))
    assert len(actual_viewed) == 24
    for p in actual_viewed: bind(p)
    notes = {
        'moving': 'Combined color guard gives little visible change; hand/tube contour remains irregular. No matched GT for this virtual camera.',
        'H004_C005_1210SZ': 'Quorum trims the false flap above the nail but does not restore the clean GT gap; texture guard changes little visibly.',
        'K004_B005_1210DS': 'Quorum trims the flap, not the main blue wedge. Guard largely replaces cloth-blue wedge with skin-colored surface; wrong protrusion and jagged edge remain.',
        'head': 'Combined native head panels retain prior crown hole/fringe and chin/neck contour defects. No broad new head change is apparent in these still comparisons; not a temporal safety claim.',
        'semantic': 'All five native witnesses viewed. H/C, K/B, I/C, J/C projections lie in the visible room gap by barrel/finger; J/A lies on clothing behind it. Close measured depths in the first four do not establish a real foreground surface.'}
    bind(__file__)
    save(output, dict(reviewer='root LLM actual native image inspection',
        geometry_visual_status='fail', combined_visual_status='fail',
        semantic_visual_status='reviewed_diagnostic_only', notes=notes,
        actual_viewed_images={str(p): sha(p) for p in actual_viewed},
        checked_inputs=checked, checked_count=len(checked), source_guard_records=records,
        production_promoted=False, delivery_changed=False, artifact_free=False,
        geometry_or_rgb_uses_corrected_diagnostic_core=False,
        full_frame_quality_metrics_computed=False, temporal_transfer_tested=False))
    print('verified', len(checked), 'bindings; 24 images reviewed; both controls fail final visual goal', flush=True)


if __name__ == '__main__':
    main()
