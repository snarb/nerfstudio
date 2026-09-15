"""Camera-independent local prior proposals, NOT depth-approved repairs.

Retain original geometry exactly; append only finely sampled lower-head/neck
prior facets near actual open mesh edges. No target camera participates.
"""
from pathlib import Path
import argparse
import numpy as np
from build_train_hair_semantics import read, sha, write

ROOT = Path('/mnt/data/dec5_mhr_local_patch_candidates')
PRIOR = Path('/mnt/data/dec5_mhr_measured_conformance')
REVIEW = Path('/mnt/data/dec5_mhr_conformance_independent_review')
ARMS = ['smooth025', 'smooth100', 'smooth400']


def boundary_opposites(triangles):
    """For each vertex, whether its opposite triangle edge is an open edge."""
    t = np.asarray(triangles)
    edges = np.sort(t[:, [[1, 2], [2, 0], [0, 1]]], axis=2)
    _, inverse, counts = np.unique(edges.reshape(-1, 2), axis=0,
                                   return_inverse=True, return_counts=True)
    return (counts[inverse] == 1).reshape(-1, 3)


def near_actual_boundary(bary, opposite_open, tolerance=.03):
    return ((np.asarray(bary) <= tolerance) & np.asarray(opposite_open)).any(axis=1)


def subdivide(vertices, triangles, parent_ids, max_edge=.00075):
    """Uniform midpoint rounds preserve shared edges and source facet identity."""
    v, t, parents = vertices.copy(), triangles.copy(), parent_ids.copy()
    rounds = 0
    while len(t) and np.linalg.norm(v[t[:, [0, 1, 2]]] - v[t[:, [1, 2, 0]]], axis=2).max() > max_edge:
        if rounds >= 8:
            raise ValueError('Unexpected prior scale; subdivision cap reached')
        edges, inverse = np.unique(np.sort(t[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1),
                                   axis=0, return_inverse=True)
        mid = inverse.reshape(-1, 3) + len(v)
        v = np.concatenate([v, v[edges].mean(axis=1)])
        a, b, c = t.T; ab, bc, ca = mid.T
        t = np.stack([np.c_[a, ab, ca], np.c_[ab, b, bc],
                      np.c_[ca, bc, c], np.c_[ab, bc, ca]], axis=1).reshape(-1, 3)
        parents = np.repeat(parents, 4)
        rounds += 1
    return v, t, parents, rounds


def build(root):
    import open3d as o3d
    from scipy.spatial import cKDTree
    from diffusion_mesh_repair import scene_for
    proto = read(PRIOR/'protocol.json')
    assert sha(proto['original_mesh']) == proto['original_mesh_sha256']
    bindings = [PRIOR/'protocol.json', PRIOR/'initial.npz', REVIEW/'result.json']
    for arm in ARMS:
        bindings += [PRIOR/arm/'fit.npz', PRIOR/arm/'result.json', REVIEW/(arm+'_geometry.npz')]
        assert sha(PRIOR/arm/'fit.npz') == read(PRIOR/arm/'result.json')['fit_sha256']
    root.mkdir(exist_ok=False)
    request = dict(frame='001193', arms=ARMS, source_mesh=proto['original_mesh'],
        source_mesh_sha256=proto['original_mesh_sha256'], input_hashes={str(p):sha(p) for p in bindings},
        neutral_y_range_cm=[135,153], maximum_original_surface_distance=.002,
        maximum_boundary_vertex_distance=.003, maximum_edge=.00075,
        minimum_normal_dot=.25, minimum_centroid_distance=.00002,
        boundary_barycentric_threshold=.03, actual_opposite_edge_must_be_open=True,
        target_camera_or_rgb_used=False, source_geometry_unchanged=True,
        inferred_not_measured=True, depth_admission_performed=False, script_sha256=sha(__file__))
    write(root/'request.json', request)
    initial = np.load(PRIOR/'initial.npz'); neutral=initial['neutral']; pt=initial['triangles']
    old = o3d.io.read_triangle_mesh(proto['original_mesh']); old.compute_triangle_normals()
    ov=np.asarray(old.vertices); ot=np.asarray(old.triangles); on=np.asarray(old.triangle_normals)
    scene=scene_for(ov,ot); opposites=boundary_opposites(ot)
    edges=ot[:,[[1,2],[2,0],[0,1]]][opposites]; tree=cKDTree(ov[np.unique(edges)])
    summaries=[]
    for arm in ARMS:
        folder=root/arm; folder.mkdir()
        fit=np.load(PRIOR/arm/'fit.npz'); independent=np.load(REVIEW/(arm+'_geometry.npz'))
        band=((neutral[pt,1]>=135)&(neutral[pt,1]<=153)).all(1)
        bad=fit['normal_reversed_triangles'].copy()
        bad[np.unique(independent['self_intersection_pairs'])]=True
        ids=np.flatnonzero(band&~bad)
        used=np.unique(pt[ids]); mapping=np.full(len(neutral),-1,int); mapping[used]=np.arange(len(used))
        v,t,parents,rounds=subdivide(fit['vertices'][used],mapping[pt[ids]],ids)
        prior=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t)); prior.compute_vertex_normals()
        cp=scene.compute_closest_points(o3d.core.Tensor(v.astype(np.float32)))
        closest=cp['points'].numpy(); nearest_ids=cp['primitive_ids'].numpy()
        distance=np.linalg.norm(v-closest,axis=1); bd=tree.query(v)[0]
        dot=np.sum(np.asarray(prior.vertex_normals)*on[nearest_ids],axis=1)
        good=(distance<=.002)&(bd<=.003)&(dot>=.25)
        cc=scene.compute_closest_points(o3d.core.Tensor(v[t].mean(1).astype(np.float32)))
        uv=cc['primitive_uvs'].numpy(); bary=np.c_[1-uv.sum(1),uv]
        center_distance=np.linalg.norm(v[t].mean(1)-cc['points'].numpy(),axis=1)
        edge=near_actual_boundary(bary,opposites[cc['primitive_ids'].numpy()])
        keep=good[t].all(1)&edge&(center_distance>=.00002)
        added_used=np.unique(t[keep]); remap=np.full(len(v),-1,int); remap[added_used]=np.arange(len(added_used))
        vv=np.concatenate([ov,v[added_used]]); pp=remap[t[keep]]+len(ov); tt=np.concatenate([ot,pp])
        candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt))
        candidate.compute_vertex_normals(); assert o3d.io.write_triangle_mesh(str(folder/'local_raw.ply'),candidate)
        reread=o3d.io.read_triangle_mesh(str(folder/'local_raw.ply'))
        np.testing.assert_array_equal(np.asarray(reread.vertices)[:len(ov)],ov)
        np.testing.assert_array_equal(np.asarray(reread.triangles)[:len(ot)],ot)
        np.savez_compressed(folder/'proposal_evidence.npz',proposals=pp,parent_triangle_ids=parents[keep],
            closest_points=closest[added_used],closest_triangle_ids=nearest_ids[added_used],
            distance=distance[added_used],boundary_distance=bd[added_used],normal_dot=dot[added_used],
            centroid_barycentric=bary[keep],centroid_nearest_triangle=cc['primitive_ids'].numpy()[keep],
            centroid_distance=center_distance[keep])
        result=dict(arm=arm,request_sha256=sha(root/'request.json'),original_vertices=len(ov),
            original_triangles=len(ot),anatomical_triangles=int(band.sum()),excluded_unsafe_in_band=int((band&bad).sum()),
            subdivision_rounds=rounds,subdivided_triangles=len(t),all_vertices_local=int(good[t].all(1).sum()),
            local_actual_boundary=int((good[t].all(1)&edge).sum()),added_vertices=len(added_used),proposals=len(pp),
            original_prefix_exact=True,depth_guard_passed=False,production_accepted=False,visual_status='pending',
            hashes={n:sha(folder/n) for n in ['local_raw.ply','proposal_evidence.npz']})
        write(folder/'result.json',result); summaries.append(result); print(arm,result,flush=True)
    write(root/'result.json',dict(request_sha256=sha(root/'request.json'),arms=summaries,production_accepted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=ROOT)
    build(parser.parse_args().output)
