"""Same lower-row proposal, independently photographed competing-layer vetoes."""
from pathlib import Path
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json
from calibrated_depth_witness import load_images
from study_confidence_depth_prior import unproject, support
from bake_joint_temporal_mesh import camera_depth
from study_foundation_lower_forearm import ROOT as LOWER, FRAME
from study_forearm_independent_patch_evidence import ROOT as DIAG, select_witnesses
from contrastive_plane_guard import score_events, best_plane_decision

ROOT = Path('/mnt/data/dec5_independent_plane_patch_guard')


def run():
    import build_foundation_foreground_patch as engine
    import study_forearm_plane_transfer_v3 as real
    ROOT.mkdir(exist_ok=False)
    diag = read(DIAG / 'request.json')
    for path, digest in diag['scripts'].items():
        assert sha(path) == digest
    # The changed rule is explicitly exploratory and checked on ALL saved
    # controls before any new geometry is built, not tuned per camera.
    controls = dict(count=0, rejected=0); exploratory = dict(conflict=0, rejected=0)
    for path in sorted(DIAG.glob('*/evidence.npz')):
        a = np.load(path);events = read(path.parent / 'events.json')['events']
        q = {k: a[k][[1, 3]] for k in ['near', 'far', 'available', 'query_std']}
        decision = best_plane_decision(**q)['reject_far']
        for e, reject in zip(events, decision):
            if e['kind'] == 'measured_control':
                controls['count'] += 1; controls['rejected'] += int(reject)
            if e['kind'] == 'conflict':
                exploratory['conflict'] += 1; exploratory['rejected'] += int(reject)
    assert controls['count'] >= 500 and controls['rejected'] == 0
    images, _, rgb = load_images(FRAME);assert rgb == diag['source_rgb_receipt']
    real.configure();rows, dl, depth_receipt = real.v2.v1.load_real(FRAME)
    assert depth_receipt == diag['source_depth_receipt']
    depths = {r['physical_camera']: d for r, d in zip(rows, dl)}
    proposal = np.load(LOWER / 'foreground/proposal.npz')
    center = np.median(proposal['added_vertices'], axis=0)
    excluded = set(diag['excluded_from_photo_validation'])
    witnesses = {r['physical_camera']: select_witnesses(r, rows, excluded, center) for r in rows}
    request = dict(frame=FRAME, exploratory_rule_not_independent_final_validation=True,
        diagnostic_request_sha256=sha(DIAG / 'request.json'), diagnostic_result_sha256=sha(DIAG / 'result.json'),
        source_rgb_receipt=rgb, source_depth_receipt=depth_receipt,
        proposal_sha256=sha(LOWER / 'foreground/proposal.npz'),
        controls=controls, exploratory_sample=exploratory,
        rule=dict(radius=15, plane_models=['query_parallel', 'proposal_tangent'],
                  compare_best_plane_for_each_depth=True, ncc=.65, margin=.2, min_query_std=4,
                  near_witnesses=3, maximum_far_witnesses=1, exclude_all_prior_cameras=sorted(excluded),
                  fractional_native_raycast_must_hit_added_surface=True, fractional_native_depth_difference_max=.001),
        source_selection_center=center.tolist(), witnesses={k: [r['physical_camera'] for r in v] for k, v in witnesses.items()},
        scripts={str(Path(__file__).with_name(n).resolve()): sha(Path(__file__).with_name(n)) for n in
            [Path(__file__).name, 'contrastive_plane_guard.py', 'plane_patch_evidence.py',
             'study_forearm_independent_patch_evidence.py', 'build_foundation_foreground_patch.py', 'guard_jaw_measured_depth.py']},
        effective_guard_changed=True, heldout_used=False, production_updated=False,
        base_inner_request_guard_sha_is_provenance_only=True)
    atomic_json(ROOT / 'qualification_request.json', request)
    state = {};stats = []
    original_scene_for = engine.scene_for
    def capture_scene(vertices, triangles):
        p = vertices[triangles]
        normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        state['normals'] = normals
        return original_scene_for(vertices, triangles)
    def guard(scene, camera, observed, all_rows, all_depths, original_count, total_count, offset):
        actual = dict(camera)
        if offset == 0:
            actual['cx'] += .5; actual['cy'] += .5
        d, ids, _ = camera_depth(scene, actual)
        y, x = np.nonzero(np.isfinite(d) & (ids >= original_count) & (ids < total_count))
        qx, qy = np.rint(x+offset).astype(int), np.rint(y+offset).astype(int)
        inside = (qx < 1920) & (qy < 1080)
        x, y, qx, qy = x[inside], y[inside], qx[inside], qy[inside]
        obs = observed[qy, qx];farther = np.isfinite(obs) & (obs > 0) & (obs > d[y, x]+.003)
        x, y, qx, qy, obs = x[farther], y[farther], qx[farther], qy[farther], obs[farther]
        raw = len(x)
        far = unproject(camera, qx, qy, obs);votes, _ = support(far, camera, all_rows, all_depths)
        trusted = votes >= 3
        x, y, qx, qy, far = x[trusted], y[trusted], qx[trusted], qy[trusted], far[trusted]
        rejected = np.zeros(len(x), bool)
        ndepth, nids = d[y, x].copy(), ids[y, x].copy()
        admissible = np.ones(len(x), bool)
        if offset == .5 and len(x):
            center = np.asarray(camera['transform_matrix'])[:3, 3]
            ray = unproject(camera, qx, qy, np.ones(len(x)))-center
            cast = scene.cast_rays(o3d.core.Tensor(np.column_stack([np.broadcast_to(center, ray.shape), ray]).astype(np.float32)))
            ndepth, nids = cast['t_hit'].numpy(), cast['primitive_ids'].numpy()
            admissible = np.isfinite(ndepth) & (nids >= original_count) & (nids < total_count) & (np.abs(ndepth-d[y, x]) <= .001)
        which = np.flatnonzero(admissible)
        ws = witnesses[camera['physical_camera']]
        if len(which) and len(ws) >= 3:
            near = unproject(camera, qx[which], qy[which], ndepth[which])
            score = score_events(camera, ws, images, depths, near, far[which], state['normals'][nids[which]])
            rejected[which] = score['reject_far']
        keep = ~rejected
        stats.append(dict(camera=camera['physical_camera'], offset=offset, raw_far=raw, original_trusted=len(x),
                          native_comparable=len(which), disqualified=int(rejected.sum()), remaining=int(keep.sum())))
        print('photo guard', camera['physical_camera'], offset, len(x), int(rejected.sum()), flush=True)
        return np.unique(ids[y[keep], x[keep]]).astype(int), int(keep.sum()), raw
    engine.ROOT = ROOT / 'geometry';engine.CONTROL = LOWER / 'empty_ray';engine.BIAS = LOWER / 'bias'
    engine.scene_for = capture_scene;engine.measured_pixel_veto = guard
    engine.run()
    assert sha(ROOT / 'geometry/proposal.npz') == request['proposal_sha256']
    atomic_json(ROOT / 'qualification_result.json', dict(request_sha256=sha(ROOT / 'qualification_request.json'),
        geometry_result_sha256=sha(ROOT / 'geometry/result.json'), checks=stats,
        production_updated=False, visual_status='pending'))


if __name__ == '__main__':
    run()
