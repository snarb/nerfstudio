"""One fixed train-only silhouette-aware lower-neck conformance experiment."""
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import lsmr
from conform_mhr_measured_surface import uniform_laplacian, barycentric_matrix
from admit_mhr_local_patch_depth import inputs, Scene2
from study_multiview_face_prior import read, save, sha
from triangulate_face_prior import quantiles

ROOT = Path('/mnt/data/dec5_mhr_silhouette_conformance')
SOURCE = Path('/mnt/data/dec5_mhr_measured_conformance')
RECIPE = dict(outer_iterations=10, maximum_step=.001, maximum_lsmr_iterations=600,
              data_sigma=.001, laplacian_sigma=.0005, magnitude_sigma=.006,
              association_max_distance=.006, normal_cosine_minimum=.25,
              silhouette_weight=4., silhouette_sigma_pixels=2., boundary_tolerance_pixels=2.,
              outside_robust_scale_pixels=8., active_neutral_y=[135.,153.],
              lsmr_atol=1e-8, lsmr_btol=1e-8, fit_target_or_eval_input=False,
              silhouette_normalization='fixed number of fit cameras times active vertices',
              data_normalization='equal mean weight for associated face and neck anchor groups',
              final_iterate_only=True, patch_admission_performed=False)


def project_jacobian(points, row):
    pose = np.asarray(row['transform_matrix'], float)
    q = (points - pose[:3, 3]) @ pose[:3, :3]
    z = -q[:, 2]
    safe = np.where(abs(q[:, 2]) > 1e-10, q[:, 2], -1e-10)
    uv = np.c_[-row['fl_x'] * q[:, 0] / safe + row['cx'] - .5,
               row['fl_y'] * q[:, 1] / safe + row['cy'] - .5]
    jac = np.zeros((len(points), 2, 3))
    jac[:, 0, 0] = -row['fl_x'] / safe
    jac[:, 0, 2] = row['fl_x'] * q[:, 0] / safe**2
    jac[:, 1, 1] = row['fl_y'] / safe
    jac[:, 1, 2] = -row['fl_y'] * q[:, 1] / safe**2
    return uv, z, jac @ pose[:3, :3].T


def sample_sdf(sdf, uv):
    """Bilinear value and its exact derivative within the sampled cell."""
    xy = np.floor(uv).astype(int)
    xy[:, 0] = xy[:, 0].clip(0, sdf.shape[1] - 2)
    xy[:, 1] = xy[:, 1].clip(0, sdf.shape[0] - 2)
    x, y = xy.T
    dx, dy = (uv - xy).T
    a, b, c, d = sdf[y, x], sdf[y, x+1], sdf[y+1, x], sdf[y+1, x+1]
    value = a*(1-dx)*(1-dy) + b*dx*(1-dy) + c*(1-dx)*dy + d*dx*dy
    gradient = np.c_[(b-a)*(1-dy)+(d-c)*dy, (c-a)*(1-dx)+(d-b)*dx]
    return value, gradient


def silhouette_samples(vertices, rows, sdfs):
    records = []
    for row, sdf in zip(rows, sdfs):
        uv, z, jac = project_jacobian(vertices, row)
        available = (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < sdf.shape[1]-3) & (uv[:, 1] > 2) & (uv[:, 1] < sdf.shape[0]-3)
        ids = np.flatnonzero(available)
        value, gradient = sample_sdf(sdf, uv[ids])
        records.append((ids, value, np.einsum('ni,nij->nj', gradient, jac[ids])))
    return records


def silhouette_stats(vertices, rows, sdfs):
    records = silhouette_samples(vertices, rows, sdfs)
    values = np.concatenate([r[1] for r in records])
    outside = np.maximum(values - RECIPE['boundary_tolerance_pixels'], 0)
    return dict(available_samples=len(values), outside_samples=int((outside > 0).sum()),
                outside_fraction=float(np.mean(outside > 0)), excess_pixels=quantiles(outside),
                excess_nonzero_pixels=quantiles(outside[outside > 0]),
                per_camera=[dict(camera=row['physical_camera'], available=len(record[1]),
                                 outside=int((record[1] > 2).sum()), max_excess=float(np.maximum(record[1]-2, 0).max(initial=0)))
                            for row, record in zip(rows, records)])


def optimize(base, triangles, neutral, points, normals, neck, rows, sdfs):
    """Receives only fitting anchors and cameras; reserved evidence is not passed."""
    import open3d as o3d
    active = (neutral[:, 1] > 135) & (neutral[:, 1] < 153)
    ids = np.flatnonzero(active)
    n, count = len(base), len(ids)
    head_tri = triangles[(neutral[triangles, 1] > 140).all(1)]
    lap = uniform_laplacian(triangles, n)[ids][:, ids]
    smooth = sparse.kron(lap/(RECIPE['laplacian_sigma']*np.sqrt(count)), sparse.eye(3), format='csr')
    magnitude = sparse.eye(count*3, format='csr')/(RECIPE['magnitude_sigma']*np.sqrt(count))
    current = base.copy()
    history = []
    for outer in range(RECIPE['outer_iterations']):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current), o3d.utility.Vector3iVector(head_tri))
        mesh.compute_triangle_normals()
        nearest = Scene2(current, head_tri).compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
        tid, uv = nearest['primitive_ids'].numpy(), nearest['primitive_uvs'].numpy()
        bary = np.c_[1-uv.sum(1), uv]
        distance = np.linalg.norm(nearest['points'].numpy()-points, axis=1)
        dot = np.sum(np.asarray(mesh.triangle_normals)[tid]*normals, axis=1)
        use = (distance <= .006) & (dot >= .25)
        assert (use & neck).sum() >= 50 and (use & ~neck).sum() >= 50
        assoc = barycentric_matrix(head_tri[tid[use]], bary[use], n)
        robust = 1/np.sqrt(1+(distance[use]/.001)**2)
        groups = np.where(neck[use], (use & neck).sum(), (use & ~neck).sum())
        weight = np.sqrt(robust/(2*groups))/.001
        data = sparse.kron(sparse.diags(weight) @ assoc[:, ids], sparse.eye(3), format='csr')
        data_rhs = ((points[use]-assoc@current)*weight[:, None]).ravel()
        ri, ci, vv, rr = [], [], [], []
        cursor = 0
        for qids, values, gradients in silhouette_samples(current[ids], rows, sdfs):
            excess = np.maximum(values-2, 0)
            take = excess > 0
            local = qids[take]
            robust = 1/np.sqrt(1+(excess[take]/8.)**2)
            weights = np.sqrt(4.*robust/(len(rows)*count))/2.
            ri.extend(np.repeat(np.arange(cursor, cursor+len(local)), 3))
            ci.extend((3*local[:, None]+np.arange(3)).ravel())
            vv.extend((gradients[take]*weights[:, None]).ravel())
            rr.extend(-excess[take]*weights)
            cursor += len(local)
        sil = sparse.coo_matrix((vv, (ri, ci)), shape=(cursor, count*3)).tocsr()
        displacement = (current[ids]-base[ids]).ravel()
        system = sparse.vstack([data, smooth, magnitude, sil], format='csr')
        rhs = np.r_[data_rhs, -smooth@displacement, -magnitude@displacement, rr]
        solution = lsmr(system, rhs, atol=1e-8, btol=1e-8, maxiter=RECIPE['maximum_lsmr_iterations'])
        step = solution[0].reshape(-1, 3)
        maximum = np.linalg.norm(step, axis=1).max()
        step *= min(1., RECIPE['maximum_step']/max(maximum, 1e-20))
        assert np.isfinite(step).all()
        current[ids] += step
        np.testing.assert_array_equal(current[~active], base[~active])
        record = dict(outer=outer, associated_face=int((use & ~neck).sum()), associated_neck=int((use & neck).sum()),
                      associated_active=int((np.asarray(assoc[:, ids].sum(1)).ravel() > 0).sum()),
                      silhouette_rows=cursor, lsmr_stop=int(solution[1]), lsmr_iterations=int(solution[2]),
                      unconstrained_maximum_step=float(maximum), applied_maximum_step=float(np.linalg.norm(step, axis=1).max()),
                      maximum_displacement=float(np.linalg.norm(current-base, axis=1).max()),
                      train_silhouette=silhouette_stats(current[ids], rows, sdfs))
        history.append(record)
        print('iteration', outer, 'silhouette outside', record['train_silhouette']['outside_samples'], 'max displacement', record['maximum_displacement'], flush=True)
        save(ROOT/'progress.json', dict(history=history))
    return current, history


def prepare():
    parent = read(SOURCE/'protocol.json')
    for path, digest in parent['input_hashes'].items():
        assert sha(path) == digest, path
    prior = read(SOURCE/'smooth100/result.json')
    assert prior['fit_sha256'] == sha(SOURCE/'smooth100/fit.npz')
    assert prior['protocol_sha256'] == sha(SOURCE/'protocol.json')
    cq, rows, depths, masks, names, evidence = inputs()
    del depths
    anchors = np.load(SOURCE/'anchors.npz')
    anchor_rows = read(SOURCE/'anchors.json')['cameras']
    assert [r['physical_camera'] for r in rows] == [r['physical_camera'] for r in anchor_rows]
    validation = np.array([any(r['physical_camera'].startswith(p) for p in parent['validation_prefixes']) for r in rows])
    np.testing.assert_array_equal(validation, anchors['validation'])
    assert validation.sum() == 8 and (~validation).sum() == 54
    return parent, rows, masks, names, evidence, validation


def main():
    import open3d as o3d
    started = time.monotonic()
    assert not ROOT.exists()
    parent, rows, masks, names, evidence, validation = prepare()
    ROOT.mkdir()
    paths = [SOURCE/'smooth100/fit.npz', SOURCE/'smooth100/result.json', SOURCE/'protocol.json', SOURCE/'initial.npz', SOURCE/'anchors.npz', SOURCE/'anchors.json']
    helpers = ['conform_mhr_measured_surface.py', 'admit_mhr_local_patch_depth.py', 'joint_temporal_texture.py', 'study_multiview_face_prior.py']
    save(ROOT/'protocol.json', dict(frame='001193', recipe=RECIPE, input_hashes={str(p):sha(p) for p in paths},
         original_mesh=parent['original_mesh'], original_mesh_sha256=parent['original_mesh_sha256'],
         fit_cameras=[r['physical_camera'] for r, v in zip(rows, validation) if not v],
         validation_cameras=[r['physical_camera'] for r, v in zip(rows, validation) if v],
         evidence=evidence, script_sha256=sha(__file__), helpers={h:sha(Path(__file__).with_name(h)) for h in helpers},
         original_and_override_have_all62_provenance=True, target_used=False, production_accepted=False))
    assert sha(parent['original_mesh']) == parent['original_mesh_sha256']
    initial = np.load(SOURCE/'initial.npz')
    base = np.load(SOURCE/'smooth100/fit.npz')['vertices']
    triangles, neutral = initial['triangles'], initial['neutral']
    active = (neutral[:, 1] > 135) & (neutral[:, 1] < 153)
    sdfs = []
    for index, row in enumerate(rows):
        mask = masks[names.index(row['physical_camera'])].astype(bool)
        sdfs.append((distance_transform_edt(~mask)-distance_transform_edt(mask)).astype(np.float32))
        if (index+1) % 10 == 0: print('sdf', index+1, flush=True)
    trainrows, trainsdfs = [r for r,v in zip(rows,validation) if not v], [s for s,v in zip(sdfs,validation) if not v]
    valrows, valsdfs = [r for r,v in zip(rows,validation) if v], [s for s,v in zip(sdfs,validation) if v]
    obs = np.load(SOURCE/'anchors.npz')
    train = ~validation[obs['camera']]
    before = dict(train=silhouette_stats(base[active], trainrows, trainsdfs), validation=silhouette_stats(base[active], valrows, valsdfs))
    current, history = optimize(base, triangles, neutral, obs['points'][train], obs['normals'][train], obs['neck'][train], trainrows, trainsdfs)
    after = dict(train=silhouette_stats(current[active], trainrows, trainsdfs), validation=silhouette_stats(current[active], valrows, valsdfs))
    head_tri = triangles[(neutral[triangles, 1] > 140).all(1)]
    stats, geometry = {}, {}
    for name, vertices in [('baseline', base), ('silhouette', current)]:
        nearest = Scene2(vertices, head_tri).compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)))
        delta = nearest['points'].numpy()-obs['points']
        distance = np.linalg.norm(delta, axis=1)
        plane = np.sum(delta*obs['normals'], axis=1)
        stats[name] = {}
        for split, selected in [('train', train), ('validation', ~train)]:
            for group, g in [('face', ~obs['neck']), ('neck_candidate', obs['neck'])]:
                take = selected & g
                stats[name][split+'_'+group] = dict(surface_distance=quantiles(distance[take]), absolute_point_plane=quantiles(abs(plane[take])))
        geometry[name] = dict(distance=distance, point_plane=plane, nearest_triangle=nearest['primitive_ids'].numpy(), nearest_uv=nearest['primitive_uvs'].numpy())
    np.testing.assert_array_equal(current[~active], base[~active])
    np.savez_compressed(ROOT/'fit.npz', vertices=current, baseline=base, triangles=triangles, neutral=neutral,
                        active=active, displacement=current-base, **{n+'_'+k:v for n,d in geometry.items() for k,v in d.items()})
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(current), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(ROOT/'prior_only.ply'), mesh)
    save(ROOT/'result.json', dict(protocol_sha256=sha(ROOT/'protocol.json'), before=before, after=after,
         anchor_stats=stats, history=history, seconds=time.monotonic()-started, active_vertices=int(active.sum()),
         hashes={n:sha(ROOT/n) for n in ['fit.npz','prior_only.ply']}, outside_active_exact=True,
         original_geometry_unchanged=True, topology_review_pending=True, prior_only=True, production_accepted=False))
    print('terminal', time.monotonic()-started, flush=True)


if __name__ == '__main__': main()
