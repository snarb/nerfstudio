"""Seal measured partial progress, explicitly not whole-video acceptance."""
from pathlib import Path
import numpy as np
from PIL import Image
from study_multiview_face_prior import read, save, sha

ROOT = Path('/mnt/data/dec5_mhr_sampling_candidates')
PRIOR = Path('/mnt/data/dec5_mhr_sampling_surface')
DEST = Path('/mnt/data/dec5_mhr_sampling_final_review')


def main():
    assert not DEST.exists()
    checked = {}

    def check(path, digest):
        assert sha(path) == digest, str(path)
        checked[str(path)] = digest

    seal = read(PRIOR / 'final_seal.json')
    assert seal['status'] == 'passed'
    for p, h in seal['checked_bindings'].items(): check(p, h)
    for p, h in seal['inventory'].items(): check(PRIOR / p, h)
    for arm, count in [('surface', 40), ('vertex32', 11)]:
        root = Path('/mnt/data/dec5_mhr_sampling_' + arm)
        audit = read(root / 'guard_audit.json')
        assert len(audit['records']) == count and not audit['new_all_intersections_observed']
        for p, h in audit['checked_bindings'].items(): check(p, h)
    admission = ROOT / 'admission'; audit = read(admission / 'audit.json')
    assert audit['status'] == 'passed' and audit['geometry_only_inventory_rgb_review_separate']
    for p, h in audit['inventory'].items(): check(admission / p, h)
    q = read(admission / 'request.json')
    check(q['inputs']['actual_source_mesh'], q['inputs']['actual_source_mesh_sha256'])
    for folder in [ROOT / 'posthoc_probe', Path('/mnt/data/dec5_mhr_sampling_surface_cohort'),
                   Path('/mnt/data/dec5_mhr_sampling_residual_diagnosis')]:
        r = read(folder / 'result.json')
        for p, h in r['input_hashes'].items(): check(p, h)
        check(folder / 'evidence.npz', r['evidence_sha256'])
    review = admission / 'rgb_review'; r = read(review / 'result.json')
    for p, h in r['input_hashes'].items(): check(p, h)
    for p, h in r['outputs'].items(): check(review / p, h)
    for view in ['F004_E', 'old_moving']:
        requests, images, depths = [], [], []
        for variant in ['baseline', 'interpolated']:
            folder = admission / 'rgb' / view / variant / 'frames/001193'
            receipt = read(folder / 'complete.json')
            check(folder.parent.parent / 'request.json', receipt['request_sha256'])
            for p, h in receipt['hashes'].items(): check(folder / p, h)
            requests.append(read(folder.parent.parent / 'request.json'))
            images.append(np.asarray(Image.open(folder / 'frame.png')))
            depths.append(np.load(folder / 'target_depth.npz')['depth'])
        for key in ['recipe', 'source_rows']:
            assert requests[0][key] == requests[1][key]
        assert requests[0]['inventory'][0]['camera'] == requests[1]['inventory'][0]['camera']
        assert all(np.isfinite(d).all() for d in depths)
        assert not ((depths[0] > 0) & (depths[1] <= 0)).any()
        black = (images[0].max(2) > 0) & (images[1].max(2) == 0)
        assert int(black.sum()) == (0 if view == 'F004_E' else 3)
    xy = np.load('/mnt/data/dec5_mhr_production_patch_001193/residual_hole/evidence.npz')['portrait_xy']
    native = admission / 'rgb/F004_E/interpolated/frames/001193'
    depth = np.rot90(np.load(native / 'target_depth.npz')['depth'])
    rgb = np.asarray(Image.open(native / 'frame.png'))
    x, y = xy.T
    assert int((depth[y,x] > 0).sum()) == 20
    assert int(((depth[y,x] > 0) & (rgb[y,x].max(1) > 0)).sum()) == 20
    video = Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_v1/presentation/video.mp4')
    check(video, '447bf60f513c495ea9af997007f529a9b414d54d27a995511ecacc36b024353e')
    viewed = [review / f for f in ['F004_E_native.png', 'old_moving_native.png',
        'F004_E_overview.png', 'old_moving_overview.png', 'old_moving_new_black_0.png',
        'old_moving_new_black_1.png', 'old_moving_new_black_2.png']]
    clay = Path('/mnt/data/dec5_mhr_sampling_review')
    viewed += [clay / f for f in ['residual_clay.png', 'C004_E005_1210X7.png',
                                'G004_B005_1210FG.png', 'M004_B005_12109O.png']]
    DEST.mkdir()
    save(DEST / 'visual_review.json', dict(reviewer='parent_LLM', status='partial_not_promoted',
        viewed={str(p): sha(p) for p in viewed},
        notes='Native F/E hole visibly shrinks but ten fixed rays remain missing. No new black RGB in F/E. '
              'Moving view retains ragged hair/neck boundaries; three new one-pixel black changes appear '
              'at clothing, neck margin and hair. No obvious broad face/colour regression in the two '
              'reviewed overviews. Inherited whole-prior face folds remain. Not a complete hole repair '
              'or a dynamic all-frame video review.',
        diagnostic_output_resolution=[1080,1920], delivered_6k_video_unchanged=True))
    save(DEST / 'result.json', dict(status='verified_partial_progress', production_accepted=False,
        remaining_fixed_hole_rays=10, corrected_colored_rays=20, checked_bindings=checked,
        admission_audit_sha256=sha(admission / 'audit.json'),
        visual_review_sha256=sha(DEST / 'visual_review.json'), script_sha256=sha(__file__)))
    print('Verified partial repair; bindings', len(checked), 'remaining rays 10; video unchanged', flush=True)


if __name__ == '__main__': main()
