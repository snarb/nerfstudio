"""Isolated counterfactual: query support quorum versus any near observation.

All four triangle samples must have fewer than3 actual query-near views AND
at least6 stable/corroborated farther views before deletion. This deliberately
changes the conservative near-protection policy; it is not a proven safe rule.
No masks/ROI select deletion, no vertices move, no production artifact changes.
"""
import argparse
from pathlib import Path
import multiprocessing
import numpy as np
from study_multiview_face_prior import read, save, sha

SOURCE = Path('/mnt/data/dec5_measured_free_center_pruning/000995')
ROOT = Path('/mnt/data/dec5_query_support_quorum')
FRAME = '000995'


def removable(near, stable, corroborated):
    near, stable, corroborated = map(np.asarray, (near, stable, corroborated))
    if near.shape != stable.shape or near.shape != corroborated.shape or near.ndim != 2 or near.shape[1] != 4:
        raise ValueError('Expected triangle x four-sample counts')
    return (near < 3).all(1) & (stable >= 6).all(1) & (corroborated >= 6).all(1)


def geometry():
    import open3d as o3d
    from review_full_block_transfer import ROOT as DEPTH_ROOT
    from study_confidence_depth_prior import load_real, project_integer, support, unproject
    from diagnose_jaw_measured_depth import observed_at
    from prune_measured_free_surface import near_tap_evidence
    q = read(SOURCE / 'request.json'); r = read(SOURCE / 'result.json')
    assert r['request_sha256'] == sha(SOURCE / 'request.json')
    for name, digest in r['hashes'].items(): assert sha(SOURCE / name) == digest, name
    assert sha(q['mesh']) == q['mesh_sha256'] and q['parameters']['near_native_radius'] == 0
    assert q['parameters']['near_tolerance'] == .0015
    root = ROOT / FRAME; assert not ROOT.exists()
    rows, depths, receipt = load_real(DEPTH_ROOT, FRAME); assert receipt == q['depth_receipt']
    from joint_temporal_texture import HELD_CAMERAS
    assert len(rows) == 62 and len({x['physical_camera'] for x in rows}) == 62
    assert not {x['physical_camera'] for x in rows} & HELD_CAMERAS
    cached = np.load(SOURCE / 'evidence.npz'); points = cached['points']; samples = cached['sample_indices']
    near = np.zeros(len(points), np.uint8)
    for row, depth in zip(rows, depths):
        uv, z = project_integer(row, points)
        near += near_tap_evidence(depth, uv, z, tolerance=.0015, radius=0)
    np.testing.assert_array_equal(near, cached['near_counts'])
    stable = cached['stable_far_counts']; candidates = np.flatnonzero((near[samples] < 3).all(1) & (stable[samples] >= 6).all(1))
    selected = points[samples[candidates]].reshape(-1,3); trusted = np.zeros((62,len(candidates),4), bool)
    root.mkdir(parents=True)
    request = dict(frame=FRAME, mesh=q['mesh'], mesh_sha256=q['mesh_sha256'],
        source_request_sha256=sha(SOURCE / 'request.json'), source_result_sha256=sha(SOURCE / 'result.json'),
        source_evidence_sha256=sha(SOURCE / 'evidence.npz'), depth_receipt=receipt,
        policy=dict(query_near_protection_minimum=3, near_tolerance=.0015, near_native_radius=0,
            min_stable_far=6, min_corroborated_far=6, other_views_per_far_observation=3),
        scope='all_triangles_all_four_samples', changed_policy_not_proven_safe=True,
        target_used=False, mask_used=False, production_changed=False,
        script_sha256=sha(__file__), helpers={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in
            ['prune_measured_free_surface.py','study_confidence_depth_prior.py','diagnose_jaw_measured_depth.py','joint_temporal_texture.py']})
    save(root / 'request.json', request)
    for ci,(row,depth) in enumerate(zip(rows,depths)):
        xy,z,obs,valid = observed_at(selected,row,depth)
        ids = np.flatnonzero(valid & (obs > z + np.maximum(.005,.01*z)))
        if len(ids):
            votes,_ = support(unproject(row,xy[ids,0],xy[ids,1],obs[ids]),row,rows,depths)
            trusted[ci].reshape(-1)[ids] = votes >= 3
        if (ci+1)%10 == 0: print('far corroboration',ci+1,'/62',flush=True)
    removed = candidates[removable(near[samples[candidates]],stable[samples[candidates]],trusted.sum(0))]
    mesh = o3d.io.read_triangle_mesh(q['mesh']); v,t = np.asarray(mesh.vertices),np.asarray(mesh.triangles)
    np.testing.assert_array_equal(samples[:,:3],t); np.testing.assert_array_equal(points[:len(v)],v)
    keep = np.ones(len(t),bool); keep[removed]=False
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
    out.compute_vertex_normals(); assert o3d.io.write_triangle_mesh(str(root / 'mesh.ply'),out)
    loaded=o3d.io.read_triangle_mesh(str(root / 'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(loaded.vertices),v); np.testing.assert_array_equal(np.asarray(loaded.triangles),t[keep])
    np.savez_compressed(root / 'evidence.npz',sample_indices=samples,points=points,near_counts=near,
        stable_far_counts=stable,candidates=candidates,trusted_far_by_camera=trusted,removed_triangle_ids=removed)
    save(root / 'result.json',dict(request_sha256=sha(root/'request.json'),candidate_triangles=len(candidates),
        removed_triangles=len(removed),additional_to_any_near=int((~np.isin(removed,cached['removed_triangle_ids'])).sum()),
        all_previous_removals_retained=bool(np.isin(cached['removed_triangle_ids'],removed).all()),
        vertices_unchanged=True,production_changed=False,visual_status='pending',
        hashes={n:sha(root/n) for n in ['mesh.ply','evidence.npz']}))
    print('removed',len(removed),'candidates',len(candidates),flush=True)


def render_view(view):
    import review_measured_free_surface as renderer
    renderer.ROOT=ROOT
    renderer.render(FRAME,view)


def render():
    import review_measured_free_surface as renderer
    renderer.ROOT=ROOT; renderer.prepare(FRAME)
    with multiprocessing.get_context('spawn').Pool(3,maxtasksperchild=1) as pool:
        pool.map(render_view,renderer.VIEWS)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['geometry','render'])
    globals()[p.parse_args().stage]()
