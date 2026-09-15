"""Matched hard-source ownership pilot on the multiview-admitted forearm."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from component_texture_owner import components, choose_owners
from build_multiview_forearm_admission import ROOT as BASE, FRAME
from render_forearm_layer_qualified_guard import VIEWS
from review_jaw_repair_transfer import verified_image, panel

ROOT = Path('/mnt/data/dec5_component_coherent_forearm')


def render(worker, workers):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid partition')
    implementation = install();engine.torch.set_num_threads(2)
    q = read(BASE / 'request.json');r = read(BASE / 'result.json')
    assert sha(BASE / 'mesh.ply') == r['mesh_sha256']
    old = o3d.io.read_triangle_mesh(q['source_mesh']);mesh = o3d.io.read_triangle_mesh(str(BASE / 'mesh.ply'))
    old_count = len(old.triangles);t = np.asarray(mesh.triangles);tv = np.asarray(mesh.vertices)[t]
    regions = components(t, old_count)
    areas = .5*np.linalg.norm(np.cross(tv[:, 1]-tv[:, 0], tv[:, 2]-tv[:, 0]), axis=1)
    original_selector = engine.select_surface_sources
    for i, view in enumerate(VIEWS):
        if i % workers != worker:
            continue
        parent = BASE / 'rgb' / view;request = deepcopy(engine.verify_request(parent))
        request.update(component_owner=dict(original_triangles=old_count, minimum_faces=100, minimum_coverage=.8,
            coverage_slack=.01, supported_faces_only=True, original_face_labels_preserved=True,
            parent_request_sha256=sha(parent / 'request.json'), source_quality_implementation_sha256=implementation,
            geometry_changed=False, calibration_changed=False, average_rgb=False),
            source_quality_implementation_sha256=implementation, full_video_candidate=False, artifact_free_approval=False)
        for name in [Path(__file__).name, 'component_texture_owner.py']:
            request['script_hashes'][name] = sha(Path(__file__).with_name(name))
        dest = ROOT / view;dest.mkdir(parents=True, exist_ok=True);(dest / 'frames').mkdir(exist_ok=True)
        if (dest / 'request.json').exists() and read(dest / 'request.json') != request:
            raise ValueError('Changed component-owner request')
        atomic_json(dest / 'request.json', request)
        def selector(rgb, quality, triangles, **kwargs):
            np.testing.assert_array_equal(triangles, t)
            baseline, graph = original_selector(rgb, quality, triangles, **kwargs)
            selected, records = choose_owners(baseline, quality, regions, areas, old_count)
            atomic_json(dest / 'owner_selection.json', dict(request_sha256=sha(dest / 'request.json'), regions=records,
                baseline_graph=graph, final_labels_not_minimizers_of_baseline_graph=True))
            graph = dict(graph, post_graph_component_owner=True,
                         energy_scope='baseline graph labels before component ownership', component_regions=records)
            return selected, graph
        engine.select_surface_sources = selector
        try:
            engine.render(dest, [FRAME])
        finally:
            engine.select_surface_sources = original_selector


def review():
    records = []
    for view in VIEWS:
        before, ar = verified_image(BASE / 'rgb' / view, FRAME);after, br = verified_image(ROOT / view, FRAME)
        for key in ['camera', 'source_cameras', 'fixed_exposure', 'mesh_sha256']:
            assert ar[key] == br[key]
        za = np.load(BASE / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth']
        zb = np.load(ROOT / view / 'frames' / FRAME / 'target_depth.npz')['depth']
        np.testing.assert_array_equal(za, zb)
        box = (90, 1650, 430, 1920) if view == 'moving' else ((40, 1650, 300, 1920) if view.startswith('H') else (80, 1650, 380, 1920))
        paths = []
        for label, crop in [('detail', box), ('overview', (0, 1320, 700, 1920))]:
            path = ROOT / 'review' / (view+'_'+label+'.png');panel(path, [before, after], ['per-face graph sources', 'coverage-first component owner'], crop)
            paths.append(dict(path=str(path), sha256=sha(path)))
        records.append(dict(view=view, panels=paths, depth_array_identical=True,
            changed_rgb_pixels=int(np.any(before != after, axis=2).sum()), upper_1200_changed=int(np.any(before[:1200] != after[:1200], axis=2).sum()),
            new_black_pixels=int(((before.max(2) > 0) & (after.max(2) == 0)).sum()), visual_status='pending'))
    atomic_json(ROOT / 'review/result.json', dict(records=records, full_frame_quality_metrics=False, script_sha256=sha(__file__)))
    print(records, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('action', choices=['render', 'review'])
    p.add_argument('--worker', type=int, default=0);p.add_argument('--workers', type=int, default=1);a = p.parse_args()
    if a.action == 'render':
        render(a.worker, a.workers)
    else:
        review()
