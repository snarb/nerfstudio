"""Locate residual ray misses in successively earlier stereo proposal domains.

No mesh repair: relaxed layers are diagnostic upper bounds, not accepted anatomy.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.ndimage import label, find_objects, binary_dilation, binary_erosion
from joint_temporal_texture import read, sha, atomic_json
from review_jaw_repair_transfer import verified_image, panel
from review_foundation_hand_geometry import grid_triangles
from study_foundation_anchor_bias import sample
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from build_independent_plane_patch_guard import ROOT as PARENT, LOWER, FRAME

ROOT = Path('/mnt/data/dec5_forearm_proposal_gap_diagnosis')


def run():
    ROOT.mkdir(exist_ok=False)
    raw = PARENT / 'unguarded_diagnostic';control = read(LOWER / 'empty_ray/request.json')
    fields = np.load(LOWER / 'bias/point_fields.npz');prefix = control['reference']+'_offset_diagnostic_'
    xyz, depth = fields[prefix+'xyz'], fields[prefix+'depth']
    stage = read(LOWER / FRAME / 'request.json')
    pair = next(p for p in stage['pairs'] if Path(p['directory']).name == control['reference'])
    source = Path(pair['directory']);cal = np.load(source / 'calibration.npz')
    predpath = LOWER / FRAME / 'inference' / source.name / 'prediction.npz';pred = np.load(predpath)
    assert sha(predpath) == read(predpath.parent / 'complete.json')['prediction_sha256']
    yy, xx = np.indices(depth.shape)
    disparity = float(cal['cropped_intrinsic'][0, 0]*cal['baseline'])/depth+float(cal['disparity_offset'])
    right = sample(binary_erosion(cal['right_mask'].astype(bool), iterations=2).astype(float),
                   np.column_stack([xx.ravel()-disparity.ravel(), yy.ravel()])).reshape(xx.shape) > .999
    masked = fields[prefix+'valid'] & binary_erosion(cal['left_mask'].astype(bool), iterations=2) & right
    earlier = np.load(LOWER / 'empty_ray/proposal.npz')
    agreement = earlier['agreement'];eligible = np.load(LOWER / 'foreground/proposal.npz')['eligible']
    assert not (eligible & ~agreement).any() and not (agreement & ~masked).any()
    domains = dict(eligible=eligible, cross_pair_agreement=agreement, reference_masked=masked,
                   reference_without_semantic_mask=pred['consistent'] & pred['valid_source_domain'])
    inputs = [source / 'calibration.npz', predpath, LOWER / 'bias/point_fields.npz',
              LOWER / 'empty_ray/proposal.npz', LOWER / 'foreground/proposal.npz', raw / 'qualification_request.json']
    request = dict(frame=FRAME, source_hashes={str(p): sha(p) for p in inputs},
        raw_mesh_sha256=sha(raw / 'geometry/mesh.ply'), source_mesh=control['source_mesh'],
        source_mesh_sha256=control['source_mesh_sha256'], reference=control['reference'],
        layer_order=list(domains), maximum_edge=.002, diagnostic_only=True, no_geometry_publication=True,
        source_frame_and_camera_unchanged=True, heldout_used=False, script_sha256=sha(__file__))
    assert sha(control['source_mesh']) == control['source_mesh_sha256']
    atomic_json(ROOT / 'request.json', request)
    original = o3d.io.read_triangle_mesh(control['source_mesh']);ov, ot = np.asarray(original.vertices), np.asarray(original.triangles)
    queries = {};records = [];layer_stats = []
    for view, box in [('H004_A005_1210M6', (40, 1650, 300, 1920)), ('moving', (90, 1650, 430, 1920))]:
        im, receipt = verified_image(raw / 'rgb' / view, FRAME)
        x0, y0, x1, y1 = box
        z = np.rot90(np.load(raw / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth'])[y0:y1, x0:x1]
        black = im[y0:y1, x0:x1].max(2) == 0;labels, _ = label(black);components = []
        for i, sl in enumerate(find_objects(labels), 1):
            ys, xs = sl;m = labels == i
            if m.sum() >= 30 and xs.start > 0 and xs.stop < black.shape[1] and ys.start > 0 and ys.stop < black.shape[0]:
                ring = binary_dilation(m, iterations=3) & ~m & (z > 0)
                boundary = z[ring]
                components.append(dict(label=i, pixels=int(m.sum()), bbox=[xs.start+x0, ys.start+y0, xs.stop+x0, ys.stop+y0],
                    original_depth_valid=int((m & (z > 0)).sum()), ring_depth_median=float(np.median(boundary)) if len(boundary) else None,
                    layers={}))
        queries[view] = dict(camera=receipt['camera'], box=box, original_depth=z, image=im, labels=labels, components=components, renders=[])
    for name, mask in domains.items():
        good = mask & np.isfinite(xyz).all(2) & np.isfinite(depth) & (depth > 0)
        v, t = grid_triangles(xyz, good, maximum_edge=.002)
        scene = scene_for(np.concatenate([ov, v]), np.concatenate([ot, t+len(ov)]))
        layer_stats.append(dict(layer=name, grid_points=int(good.sum()), triangles=len(t)))
        for view, query in queries.items():
            d, ids, _ = camera_depth(scene, query['camera']);x0, y0, x1, y1 = query['box']
            d = np.rot90(d)[y0:y1, x0:x1];ids = np.rot90(ids)[y0:y1, x0:x1]
            finite = np.isfinite(d)
            if name == 'eligible':
                np.testing.assert_array_equal(finite, query['original_depth'] > 0)
                np.testing.assert_allclose(d[finite], query['original_depth'][finite], atol=2e-6)
            rgb = np.zeros((*d.shape, 3), np.uint8);rgb[finite] = [130, 130, 130];rgb[finite & (ids >= len(ot))] = [200, 145, 90]
            for component in query['components']:
                m = query['labels'] == component['label'];hits = finite & m
                median = component['ring_depth_median']
                component['layers'][name] = dict(hits=int(hits.sum()),
                    added_hits=int((hits & (ids >= len(ot))).sum()),
                    median_hit_depth=float(np.median(d[hits])) if hits.any() else None,
                    within_001_of_boundary=int((hits & (np.abs(d-median) <= .01)).sum()) if median is not None else None)
                rgb[m & ~finite] = [180, 30, 30]
                rgb[hits] = [30, 200, 90]
            query['renders'].append(rgb)
            np.savez_compressed(ROOT / (view+'_'+name+'.npz'), depth=d, triangle=ids)
    for view, query in queries.items():
        x0, y0, x1, y1 = query['box'];im = query['image'][y0:y1, x0:x1]
        path = ROOT / (view+'.png');panel(path, [im]+query['renders'], ['raw consensus RGB']+list(domains), (0, 0, x1-x0, y1-y0))
        records.append(dict(view=view, components=query['components'], panel=str(path), panel_sha256=sha(path)))
    atomic_json(ROOT / 'result.json', dict(request_sha256=sha(ROOT / 'request.json'), layers=layer_stats, records=records,
        visibility_not_surface_correctness=True, geometry_changed=False, visual_status='pending'))
    print(layer_stats, [(r['view'], [(c['pixels'], {k:v['hits'] for k,v in c['layers'].items()}) for c in r['components']]) for r in records], flush=True)


if __name__ == '__main__':
    run()
