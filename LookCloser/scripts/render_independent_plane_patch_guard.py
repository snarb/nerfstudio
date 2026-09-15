"""Matched current-texture RGB gate for the independent patch qualifier."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from review_jaw_repair_transfer import verified_image, panel
from build_independent_plane_patch_guard import ROOT as BASE, LOWER, FRAME
from render_forearm_layer_qualified_guard import ROOT as COLOR, MOVIE, VIEWS


def render(worker, workers, variant='guarded'):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid disjoint partition')
    implementation = install();engine.torch.set_num_threads(2)
    ROOT = BASE if variant == 'guarded' else BASE / 'unguarded_diagnostic'
    q = read(ROOT / 'qualification_request.json')
    for path, digest in q['scripts'].items():
        assert sha(path) == digest
    result = read(ROOT / 'geometry/result.json')
    assert result['mesh_sha256'] == sha(ROOT / 'geometry/mesh.ply')
    if variant == 'guarded':
        assert result['guard_passed'], 'Unconverged measured-depth guard; do not render as passed'
    else:
        assert q['guards_disabled_explicit'] and result['diagnostic_only'] and not result['guard_passed']
    assert read(ROOT / 'qualification_result.json')['geometry_result_sha256'] == sha(ROOT / 'geometry/result.json')
    for i, view in enumerate(VIEWS):
        if i % workers != worker:
            continue
        parent = LOWER / 'rgb' / view / 'candidate'
        request = deepcopy(engine.verify_request(parent))
        assert len(request['inventory']) == 1 and request['inventory'][0]['frame_id'] == FRAME
        request['inventory'][0].update(mesh=str(ROOT / 'geometry/mesh.ply'), mesh_sha256=result['mesh_sha256'])
        request.update(qualification_request_sha256=sha(ROOT / 'qualification_request.json'),
            qualification_result_sha256=sha(ROOT / 'qualification_result.json'),
            effective_guard=variant, effective_guard_changed=True,
            measured_depth_guard_disabled=(variant != 'guarded'),
            strict_guard_parent_request_sha256=sha(parent / 'request.json'),
            source_quality_implementation_sha256=implementation,
            full_video_candidate=False, artifact_free_approval=False, partial_diagnostic_only=True)
        request['script_hashes'][Path(__file__).name] = sha(__file__)
        dest = ROOT / 'rgb' / view;dest.mkdir(parents=True, exist_ok=True);(dest / 'frames').mkdir(exist_ok=True)
        if (dest / 'request.json').exists() and read(dest / 'request.json') != request:
            raise ValueError('Changed render request')
        atomic_json(dest / 'request.json', request)
        engine.render(dest, [FRAME])


def review(variant='guarded'):
    ROOT = BASE if variant == 'guarded' else BASE / 'unguarded_diagnostic'
    records = []
    for view in VIEWS:
        roots = [MOVIE if view == 'moving' else LOWER / 'rgb' / view / 'baseline',
                 COLOR / 'rgb' / view, ROOT / 'rgb' / view]
        images, receipt, depth = [], [], []
        for root in roots:
            im, r = verified_image(root, FRAME);images.append(im);receipt.append(r)
            depth.append(np.rot90(np.load(root / 'frames' / FRAME / 'target_depth.npz')['depth']))
        for key in ['camera', 'source_cameras', 'fixed_exposure']:
            assert all(r[key] == receipt[0][key] for r in receipt)
        labels = ['production', 'four-view color qualifier',
                  'independent plane-patch qualifier' if variant == 'guarded' else 'NO final depth veto / diagnostic']
        upper = int(np.any(images[-1][:1200] != images[0][:1200], axis=2).sum())
        if view != 'moving':
            gt = LOWER / FRAME / (view+'.png') if view.startswith('E') else Path('/mnt/data/dec5_wrist_observations') / FRAME / (view+'.png')
            images.insert(0, np.array(Image.open(gt)));labels.insert(0, 'real train GT')
        paths = []
        detail = (90, 1650, 430, 1920) if view == 'moving' else ((40, 1650, 300, 1920) if view.startswith('H') else (80, 1650, 380, 1920))
        for label, box in [('detail', detail), ('overview', (0, 1320, 700, 1920))]:
            path = ROOT / 'review' / (view+'_'+label+'.png')
            panel(path, images, labels, box);paths.append(dict(path=str(path), sha256=sha(path)))
        records.append(dict(view=view, panels=paths, upper_1200_rows_changed_pixels=upper,
            rgb_changes_vs_color=int(np.any(images[-1] != images[-2], axis=2).sum()),
            new_depth_vs_color=int(((depth[-2] == 0) & (depth[-1] > 0)).sum()),
            lost_depth_vs_color=int(((depth[-2] > 0) & (depth[-1] == 0)).sum()),
            new_black_vs_color=int(((images[-1].max(2) == 0) & (images[-2].max(2) > 0)).sum()),
            counts_not_anatomical_quality_metrics=True, visual_status='pending'))
    atomic_json(ROOT / 'review/result.json', dict(records=records, script_sha256=sha(__file__), full_frame_quality_metrics=False))
    print(records, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['render', 'review'])
    parser.add_argument('--variant', choices=['guarded', 'unguarded_diagnostic'], default='guarded')
    parser.add_argument('--worker', type=int, default=0);parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.action == 'render':
        render(args.worker, args.workers, args.variant)
    else:
        review(args.variant)
