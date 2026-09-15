"""Three-view paired RGB gate: raw reference-only vs raw multiview admission."""
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from build_multiview_forearm_admission import ROOT, PREVIOUS, LOWER, FRAME
from render_forearm_layer_qualified_guard import VIEWS
from review_jaw_repair_transfer import verified_image, panel


def render(worker, workers):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    assert workers > 0 and 0 <= worker < workers
    implementation = install();engine.torch.set_num_threads(2)
    q = read(ROOT / 'request.json');r = read(ROOT / 'result.json')
    assert r['request_sha256'] == sha(ROOT / 'request.json') and r['mesh_sha256'] == sha(ROOT / 'mesh.ply')
    for p, digest in q['scripts'].items():
        assert sha(p) == digest
    for i, view in enumerate(VIEWS):
        if i % workers != worker:
            continue
        parent = PREVIOUS / 'unguarded_diagnostic/rgb' / view
        request = deepcopy(engine.verify_request(parent))
        request['inventory'][0].update(mesh=str(ROOT / 'mesh.ply'), mesh_sha256=r['mesh_sha256'])
        for key in ['qualification_request_sha256', 'qualification_result_sha256', 'strict_guard_parent_request_sha256']:
            request.pop(key, None)
        request.update(multiview_admission_request_sha256=sha(ROOT / 'request.json'),
            multiview_admission_result_sha256=sha(ROOT / 'result.json'), parent_rgb_request_sha256=sha(parent / 'request.json'),
            effective_guard='multiview_admission_without_final_pm_veto', source_quality_implementation_sha256=implementation,
            measured_depth_guard_disabled=True, full_video_candidate=False, artifact_free_approval=False)
        request['script_hashes'][Path(__file__).name] = sha(__file__)
        dest = ROOT / 'rgb' / view;dest.mkdir(parents=True, exist_ok=True);(dest / 'frames').mkdir(exist_ok=True)
        if (dest / 'request.json').exists() and read(dest / 'request.json') != request:
            raise ValueError('Changed RGB request')
        atomic_json(dest / 'request.json', request);engine.render(dest, [FRAME])


def review():
    records = []
    for view in VIEWS:
        control = PREVIOUS / 'unguarded_diagnostic/rgb' / view
        a, ar = verified_image(control, FRAME);b, br = verified_image(ROOT / 'rgb' / view, FRAME)
        for key in ['camera', 'source_cameras', 'fixed_exposure']:
            assert ar[key] == br[key]
        images, labels = [a, b], ['raw reference-only admission', 'raw multiview admission']
        if view != 'moving':
            gt = LOWER / FRAME / (view+'.png') if view.startswith('E') else Path('/mnt/data/dec5_wrist_observations') / FRAME / (view+'.png')
            images.insert(0, np.array(Image.open(gt)));labels.insert(0, 'real train GT')
        box = (90, 1650, 430, 1920) if view == 'moving' else ((40, 1650, 300, 1920) if view.startswith('H') else (80, 1650, 380, 1920))
        paths = []
        for label, crop in [('detail', box), ('overview', (0, 1320, 700, 1920))]:
            path = ROOT / 'review' / (view+'_'+label+'.png');panel(path, images, labels, crop)
            paths.append(dict(path=str(path), sha256=sha(path)))
        za = np.rot90(np.load(control / 'frames' / FRAME / 'target_depth.npz')['depth'])
        zb = np.rot90(np.load(ROOT / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth'])
        records.append(dict(view=view, panels=paths, new_depth_pixels=int(((za == 0) & (zb > 0)).sum()),
            lost_depth_pixels=int(((za > 0) & (zb == 0)).sum()),
            new_black_pixels=int(((a.max(2) > 0) & (b.max(2) == 0)).sum()),
            upper_1200_changed=int(np.any(a[:1200] != b[:1200], axis=2).sum()),
            counts_not_anatomical_metrics=True, visual_status='pending'))
    atomic_json(ROOT / 'review/result.json', dict(records=records, script_sha256=sha(__file__), full_frame_quality_metrics=False))
    print(records, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('action', choices=['render', 'review'])
    p.add_argument('--worker', type=int, default=0);p.add_argument('--workers', type=int, default=1);a = p.parse_args()
    if a.action == 'render':
        render(a.worker, a.workers)
    else:
        review()
