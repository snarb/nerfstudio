"""Target-visible hard patch owners, with a matched optional old/new region join."""
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from build_multiview_forearm_admission import ROOT as BASE, FRAME
from render_forearm_layer_qualified_guard import VIEWS
from component_texture_owner import choose_owners
from nearby_texture_regions import texture_regions, visible_face_weights
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from review_jaw_repair_transfer import verified_image, panel

ROOT = Path('/mnt/data/dec5_visible_patch_owner')


def render(mode, worker, workers):
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid partition')
    implementation = install();engine.torch.set_num_threads(2)
    q = read(BASE / 'request.json');result = read(BASE / 'result.json')
    assert result['mesh_sha256'] == sha(BASE / 'mesh.ply')
    old = o3d.io.read_triangle_mesh(q['source_mesh']);mesh = o3d.io.read_triangle_mesh(str(BASE / 'mesh.ply'))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    regions = texture_regions(vertices, triangles, len(old.triangles), join_original=(mode == 'joined'))
    scene = scene_for(vertices, triangles);original_selector = engine.select_surface_sources
    for i, view in enumerate(VIEWS):
        if i % workers != worker:
            continue
        parent = BASE / 'rgb' / view;request = deepcopy(engine.verify_request(parent));entry = request['inventory'][0]
        depth, ids, _ = camera_depth(scene, entry['camera'])
        spec = entry['source_masks'];maskroot = Path(spec['root'])
        for filename, key in [('masks.npz', 'masks_sha256'), ('cameras.json', 'cameras_sha256')]:
            assert sha(maskroot / filename) == spec[key]
        names = read(maskroot / 'cameras.json')
        if entry['camera']['physical_camera'] in names:
            mask = np.load(maskroot / 'masks.npz')['masks'][names.index(entry['camera']['physical_camera'])]
            depth = depth.copy();depth[mask == 0] = np.inf
        saved = np.load(parent / 'frames' / FRAME / 'target_depth.npz')['depth']
        np.testing.assert_array_equal(np.isfinite(depth), saved > 0)
        np.testing.assert_allclose(depth[np.isfinite(depth)], saved[saved > 0], atol=2e-6)
        weights = visible_face_weights(ids, depth, len(triangles), regions)
        request.update(visible_patch_owner=dict(mode=mode, parent_request_sha256=sha(parent / 'request.json'),
            original_triangles=len(old.triangles), joined_original_faces=int((regions[:len(old.triangles)] >= 0).sum()),
            join_distance=.0015, normal_cosine=.8, minimum_faces=100, minimum_coverage=.8, coverage_slack=.01,
            area_weights='target-visible pixel counts; zero for faces outside the region',
            source_rgb_not_target_rgb=True, geometry_changed=False, rgb_averaging=False),
            full_video_candidate=False, artifact_free_approval=False, source_quality_implementation_sha256=implementation)
        for name in [Path(__file__).name, 'component_texture_owner.py', 'nearby_texture_regions.py']:
            request['script_hashes'][name] = sha(Path(__file__).with_name(name))
        dest = ROOT / mode / view;dest.mkdir(parents=True, exist_ok=True);(dest / 'frames').mkdir(exist_ok=True)
        if (dest / 'request.json').exists() and read(dest / 'request.json') != request:
            raise ValueError('Changed visible-owner request')
        atomic_json(dest / 'request.json', request)
        np.savez_compressed(dest / 'region_weights.npz', region=regions, weights=weights)
        def selector(rgb, quality, t, **kwargs):
            np.testing.assert_array_equal(t, triangles)
            baseline, graph = original_selector(rgb, quality, t, **kwargs)
            selected, records = choose_owners(baseline, quality, regions, weights, 0)
            np.testing.assert_array_equal(selected[regions < 0], baseline[regions < 0])
            atomic_json(dest / 'owner_selection.json', dict(request_sha256=sha(dest / 'request.json'), regions=records,
                baseline_graph=graph, changed_original_faces=int((selected[:len(old.triangles)] != baseline[:len(old.triangles)]).sum()),
                unchanged_labels_outside_regions=True, final_labels_not_baseline_graph_minimizer=True))
            return selected, dict(graph, post_graph_visible_owner=True, energy_scope='before region-owner override')
        engine.select_surface_sources = selector
        try:
            engine.render(dest, [FRAME])
        finally:
            engine.select_surface_sources = original_selector


def review():
    records = []
    for view in VIEWS:
        im, reference = verified_image(BASE / 'rgb' / view, FRAME);images = [im]
        depth = np.load(BASE / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth']
        for mode in ['added', 'joined']:
            after, r = verified_image(ROOT / mode / view, FRAME)
            for key in ['camera', 'source_cameras', 'fixed_exposure', 'mesh_sha256']:
                assert reference[key] == r[key]
            np.testing.assert_array_equal(depth, np.load(ROOT / mode / view / 'frames' / FRAME / 'target_depth.npz')['depth'])
            records.append(dict(view=view, mode=mode, changed_pixels=int(np.any(after != im, axis=2).sum()),
                upper_1200_changed=int(np.any(after[:1200] != im[:1200], axis=2).sum()),
                new_black=int(((im.max(2) > 0) & (after.max(2) == 0)).sum()), depth_identical=True, visual_status='pending'))
            images.append(after)
        box = (90, 1650, 430, 1920) if view == 'moving' else ((40, 1650, 300, 1920) if view.startswith('H') else (80, 1650, 380, 1920))
        for name, crop in [('detail', box), ('overview', (0, 1320, 700, 1920))]:
            panel(ROOT / 'review' / (view+'_'+name+'.png'), images, ['per-face graph', 'visible added-region owner', 'visible old+new region owner'], crop)
    atomic_json(ROOT / 'review/result.json', dict(records=records, full_frame_quality_metrics=False, script_sha256=sha(__file__)))
    print(records, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('action', choices=['render', 'review'])
    p.add_argument('--mode', choices=['added', 'joined'], default='added')
    p.add_argument('--worker', type=int, default=0);p.add_argument('--workers', type=int, default=1);a = p.parse_args()
    if a.action == 'render':
        render(a.mode, a.worker, a.workers)
    else:
        review()
