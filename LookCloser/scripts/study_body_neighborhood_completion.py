"""Append-only body prior on the protected forearm baseline; no movie mutation."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from joint_temporal_texture import read, sha, atomic_json
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from body_surface_neighborhood import select_seeds
from local_surface_certificate import certify
from study_jaw_repair_transfer import mask_votes
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_train_confidence import footprint_veto
from diagnose_jaw_measured_depth import barycentric_samples
from guard_jaw_measured_depth import initial_admission, measured_pixel_veto

ROOT = Path('/mnt/data/dec5_body_neighborhood_completion')
SOURCE = Path('/mnt/data/dec5_protected_constrained_forearm')
PARENT = Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
SETTINGS = dict(sample_count=300000, seed=17, poisson_depth=10, poisson_scale=1.05,
                poisson_linear_fit=True, threads=4, body_max_x=-.03,
                maximum_surface_distance=.006, maximum_boundary_distance=.006,
                maximum_triangle_edge=.0015, minimum_centroid_distance=.00002,
                nearest_boundary_barycentric_threshold=.03, minimum_normal_dot=.25,
                seed_radius=.006, sectors=8, per_sector=3,
                minimum_seed_depth_views=3, maximum_seed_poisson_distance=.0005,
                maximum_loo_p90=.0005, maximum_predicted_offset=.0005,
                free_separation=.003, free_other_views=3, guard_max_rounds=8)
SCRIPTS = ['study_body_neighborhood_completion.py', 'body_surface_neighborhood.py',
           'local_surface_certificate.py', 'study_jaw_repair_transfer.py',
           'study_jaw_depth_footprint.py', 'study_jaw_train_confidence.py',
           'diagnose_jaw_measured_depth.py', 'guard_jaw_measured_depth.py',
           'study_confidence_depth_prior.py', 'study_forearm_plane_transfer.py',
           'study_forearm_plane_transfer_v2.py', 'study_forearm_plane_transfer_v3.py']


def load_inputs(frame):
    import study_forearm_plane_transfer_v3 as source
    source.configure()
    rows, depths, hashes = source.v2.v1.load_real(frame)
    old = SOURCE / frame
    req = read(old / 'request.json'); result = read(old / 'geometry_result.json')
    assert result['request_sha256'] == sha(old / 'request.json')
    assert result['observed_guard_passed']
    assert hashes == req['source_depth_sha256']
    assert len(rows) == 62 and len({r['physical_camera'] for r in rows}) == 62
    assert not ({r['physical_camera'] for r in rows} & {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'})
    for name, digest in result['hashes'].items():
        assert sha(old / name) == digest
    assert all(d.shape == (1080, 1920) and np.isfinite(d).all() for d in depths)
    entry = next(r for r in read(PARENT / 'request.json')['inventory'] if r['frame_id'] == frame)
    maskroot = Path(entry['source_masks']['root'])
    assert sha(maskroot / 'masks.npz') == entry['source_masks']['masks_sha256']
    return rows, depths, hashes, entry, maskroot


def prepare(frame):
    root = ROOT / frame; root.mkdir(parents=True, exist_ok=False)
    rows, depths, hashes, entry, maskroot = load_inputs(frame)
    source = SOURCE / frame / 'guarded.ply'
    request = dict(frame=frame, source_mesh=str(source), source_mesh_sha256=sha(source),
        source_result_sha256=sha(source.parent / 'geometry_result.json'), source_depth_sha256=hashes,
        source_masks=str(maskroot), source_mask_sha256=sha(maskroot / 'masks.npz'),
        mask_names_sha256=sha(maskroot / 'cameras.json'), settings=SETTINGS,
        scripts={name:sha(Path(__file__).with_name(name)) for name in SCRIPTS},
        heldout_used=False, target_camera_used_in_geometry=False, original_surface_preserved=True,
        geometry_is_inferred_prior=True, temporal_observations_used=False, production_accepted=False,
        open3d_version=o3d.__version__)
    atomic_json(root / 'request.json', request)
    mesh = o3d.io.read_triangle_mesh(str(source)); mesh.compute_vertex_normals(); mesh.compute_triangle_normals()
    v = np.asarray(mesh.vertices); t = np.asarray(mesh.triangles); nv, nt = len(v), len(t)
    # Whole mesh supplies boundary context; only below-head proposals may survive.
    o3d.utility.random.seed(SETTINGS['seed'])
    cloud = mesh.sample_points_uniformly(number_of_points=SETTINGS['sample_count'], use_triangle_normal=True)
    o3d.io.write_point_cloud(str(root / 'oriented_samples.ply'), cloud)
    atomic_json(root / 'progress.json', dict(stage='poisson', unix_time=time.time()))
    raw, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,
        depth=SETTINGS['poisson_depth'], scale=SETTINGS['poisson_scale'], linear_fit=True, n_threads=4)
    raw.compute_vertex_normals(); rv = np.asarray(raw.vertices); rt = np.asarray(raw.triangles)
    assert np.isfinite(rv).all()
    o3d.io.write_triangle_mesh(str(root / 'poisson_raw.ply'), raw)
    scene = scene_for(v, t)
    closest = scene.compute_closest_points(o3d.core.Tensor(rv.astype(np.float32)))
    cp = closest['points'].numpy(); ci = closest['primitive_ids'].numpy()
    distance = np.linalg.norm(rv - cp, axis=1)
    edges, count = np.unique(np.sort(t[:, [[0,1], [1,2], [2,0]]].reshape(-1,2), axis=1), axis=0, return_counts=True)
    boundary_vertices = np.unique(edges[count == 1])
    bd = cKDTree(v[boundary_vertices]).query(rv)[0]
    dot = np.sum(np.asarray(raw.vertex_normals) * np.asarray(mesh.triangle_normals)[ci], axis=1)
    good = (distance <= .006) & (bd <= .006) & (rv[:,0] < -.03) & (dot >= .25)
    centers = rv[rt].mean(1); cc = scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
    cd = np.linalg.norm(centers - cc['points'].numpy(), axis=1)
    bary = np.column_stack([1 - cc['primitive_uvs'].numpy().sum(1), cc['primitive_uvs'].numpy()])
    near = np.isin(t[cc['primitive_ids'].numpy()], boundary_vertices).any(1)
    lengths = np.linalg.norm(rv[rt] - rv[rt[:, [1,2,0]]], axis=2).max(1)
    keep = good[rt].all(1) & (cd >= .00002) & (bary.min(1) <= .03) & near & (lengths <= .0015)
    used = np.unique(rt[keep]); remap = np.full(len(rv), -1, int); remap[used] = np.arange(len(used))
    mv = np.concatenate([v, rv[used]]); proposals = remap[rt[keep]] + nv
    np.savez_compressed(root / 'proposal.npz', vertices=mv, proposals=proposals, raw_vertex_ids=used,
        raw_triangle_ids=np.flatnonzero(keep), closest_points=cp[used], surface_distance=distance[used],
        boundary_distance=bd[used], normal_dot=dot[used], density=np.asarray(density)[used])
    masks = np.load(maskroot / 'masks.npz')['masks']; names = read(maskroot / 'cameras.json')
    ms, mo = mask_votes(mv, proposals, rows, masks, names)
    semantic = np.flatnonzero((ms >= 2) & (mo == 0)); pp = proposals[semantic]
    print(frame, 'raw', len(rt), 'local', len(proposals), 'semantic', len(pp), flush=True)
    if not len(pp):
        raise ValueError('No semantically admitted proposals; retain diagnostic workspace')
    points = barycentric_samples(mv[pp]); votes, refs = train_reference_votes(points.reshape(-1,3), rows, depths)
    free = footprint_veto(points.reshape(-1,3), rows, depths).reshape(62,-1,10)
    strict = initial_admission(votes.reshape(-1,10), free, ms[semantic], mo[semantic])
    query_ids = np.unique(pp); q = mv[query_ids]
    normals = np.asarray(raw.vertex_normals)[used[query_ids - nv]]
    seed_ids = np.unique(np.concatenate([np.asarray(x, int) for x in cKDTree(v).query_ball_point(q, .006)]))
    sv, _ = train_reference_votes(v[seed_ids], rows, depths)
    rawscene = scene_for(rv, rt)
    nearest = rawscene.compute_closest_points(o3d.core.Tensor(v[seed_ids].astype(np.float32)))['points'].numpy()
    discrepancy = np.linalg.norm(v[seed_ids] - nearest, axis=1); valid = (sv >= 3) & (discrepancy <= .0005)
    seeds = v[seed_ids[valid]]; sn = np.asarray(mesh.vertex_normals)[seed_ids[valid]]
    neighborhoods = cKDTree(seeds).query_ball_point(q, .006)
    certificate = np.zeros(len(q), bool); notes = []; neighbors = []
    for i, ids in enumerate(neighborhoods):
        ids = np.asarray(ids, int)
        selected = ids[select_seeds(q[i], normals[i], seeds[ids], sn[ids])]
        certificate[i], note = certify(q[i], normals[i], seeds[selected])
        notes.append(note); neighbors.append(selected.tolist())
    lookup = np.zeros(len(mv), bool); lookup[query_ids] = certificate
    prior = lookup[pp].all(1) & ~free.any(axis=(0,2))
    selected = semantic[strict | prior]; triangles = np.concatenate([t, proposals[selected]])
    initial = len(selected)
    np.savez_compressed(root / 'admission.npz', mask_support=ms, mask_outside=mo, semantic_ids=semantic,
        votes=votes.reshape(-1,10), references=refs.reshape(-1,10), free=free, strict=strict,
        seed_ids=seed_ids, seed_votes=sv, seed_poisson_distance=discrepancy, valid_seed_mask=valid,
        query_ids=query_ids, query_normals=normals, certificate=certificate, prior=prior)
    atomic_json(root / 'certificates.json', dict(notes=notes, seed_neighbors=neighbors))
    print(frame, 'seeds', len(seeds), 'certified vertices', int(certificate.sum()), 'initial triangles', initial, flush=True)
    rounds = []
    for iteration in range(SETTINGS['guard_max_rounds']):
        current = scene_for(mv, triangles); remove = set(); checks = []
        for ci, (row, depth) in enumerate(zip(rows, depths)):
            for offset in [0, .5]:
                implicated, n, raw_n = measured_pixel_veto(current, row, depth, rows, depths, nt, len(triangles), offset)
                remove.update(implicated.tolist())
                checks.append(dict(camera=row['physical_camera'], offset=offset, trusted_free_pixels=n, raw_far_pixels=raw_n))
            if (ci + 1) % 10 == 0:
                atomic_json(root / 'progress.json', dict(stage='native_guard', iteration=iteration, cameras=ci+1, unix_time=time.time()))
        rounds.append(dict(removed=len(remove), checks=checks)); print(frame, 'guard', iteration, len(remove), flush=True)
        if not remove: break
        take = np.ones(len(triangles), bool); take[list(remove)] = False
        assert take[:nt].all(); selected = selected[take[nt:]]; triangles = triangles[take]
    if rounds[-1]['removed']:
        raise ValueError('Guard did not converge; not publishable')
    final = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mv), o3d.utility.Vector3iVector(triangles)); final.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(root / 'mesh.ply'), final)
    np.savez_compressed(root / 'retained.npz', proposal_ids=selected)
    checks = []; current = scene_for(mv, triangles)
    for label, camera in [('moving', entry['camera']), ('train_H_A', next(r for r in rows if r['physical_camera'] == 'H004_A005_1210M6'))]:
        d, ids, _ = camera_depth(current, camera); od, _, _ = camera_depth(scene, camera)
        visible = np.isfinite(d); lighting = np.abs(np.asarray(final.triangle_normals) @ [.3,.4,.866])
        rgb = np.zeros((*d.shape,3), np.uint8); rgb[visible] = (60 + 170 * lighting[ids[visible],None]).astype(np.uint8)
        rgb[visible & (ids >= nt)] = [255,60,40]
        Image.fromarray(np.rot90(rgb)).save(root / (label + '_added.png'))
        checks.append(dict(view=label, newly_visible=int((visible & ~np.isfinite(od)).sum()),
            original_now_missing=int((~visible & np.isfinite(od)).sum()), new_surface_visible=int((visible & (ids >= nt)).sum())))
    atomic_json(root / 'result.json', dict(request_sha256=sha(root / 'request.json'), added=len(selected),
        original_vertices=nv, original_triangles=nt, raw_proposals=len(proposals), semantic=len(semantic),
        verified_seeds=len(seeds), certified_vertices=int(certificate.sum()), initial=initial, rounds=rounds,
        views=checks, observed_guard_passed=True, production_accepted=False, visual_status='pending',
        hashes={n:sha(root/n) for n in ['oriented_samples.ply','poisson_raw.ply','proposal.npz','admission.npz',
                                      'certificates.json','retained.npz','mesh.ply','moving_added.png','train_H_A_added.png']}))
    print(frame, 'finished', len(selected), checks, flush=True)


def render(frame):
    import render_smooth_temporal_mesh_video as renderer
    from study_native_texture_footprint import install
    implementation = install(); renderer.torch.set_num_threads(2)
    root = ROOT / frame; result = read(root / 'result.json'); source = read(root / 'request.json')
    assert result['request_sha256'] == sha(root / 'request.json') and result['observed_guard_passed']
    assert sha(root / 'mesh.ply') == result['hashes']['mesh.ply']
    from joint_temporal_texture import cameras
    rows, _, _ = cameras(frame); parent = renderer.verify_request(PARENT)
    for view in ['moving', 'H004_A005_1210M6']:
        for variant in ['baseline', 'repaired']:
            request = deepcopy(parent); request['inventory'] = [r for r in request['inventory'] if r['frame_id'] == frame]
            row = request['inventory'][0]
            if view != 'moving': row['camera'] = next(r for r in rows if r['physical_camera'] == view)
            mesh = root / 'mesh.ply' if variant == 'repaired' else Path(source['source_mesh'])
            row.update(mesh=str(mesh), mesh_sha256=sha(mesh))
            request.update(partial_diagnostic_only=True, full_video_candidate=False,
                geometry_result_sha256=sha(root / 'result.json'), same_footprint_for_both_geometry_variants=True,
                native_footprint_implementation_sha256=implementation)
            for name in SCRIPTS + ['study_native_texture_footprint.py', 'native_texture_footprint.py']:
                request['script_hashes'][name] = sha(Path(__file__).with_name(name))
            dest = root / 'rgb' / view / variant; dest.mkdir(parents=True, exist_ok=False); (dest/'frames').mkdir()
            atomic_json(dest / 'request.json', request); renderer.render(dest, [frame])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['prepare','render'])
    p.add_argument('--frame', choices=['001029','001033','001037'], required=True); a = p.parse_args()
    {'prepare':prepare, 'render':render}[a.action](a.frame)
