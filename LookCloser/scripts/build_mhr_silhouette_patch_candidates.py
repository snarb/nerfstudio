"""One generic, append-only local candidate domain from the final silhouette prior."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from build_mhr_local_patch_candidates import subdivide,boundary_opposites
from admit_mhr_local_patch_depth import Scene2
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_silhouette_patch_candidates')
PRIOR=Path('/mnt/data/dec5_mhr_silhouette_convergence')
ARM='silhouette100'


def build(root=ROOT):
    import open3d as o3d
    protocol=read(PRIOR/'protocol.json');seal=read(PRIOR/'final_seal.json');assert seal['status']=='passed'
    for path,digest in seal['inventory'].items():assert sha(PRIOR/path)==digest,path
    for path,digest in seal['checked_bindings'].items():assert sha(path)==digest,path
    assert sha(protocol['original_mesh'])==protocol['original_mesh_sha256']
    fit=np.load(PRIOR/'fit.npz');v0,pt,neutral=fit['vertices'],fit['triangles'],fit['neutral']
    topology=np.load(PRIOR/'review_v2/silhouette_topology.npz')
    inherited=Path('/mnt/data/dec5_mhr_measured_conformance/smooth100/fit.npz')
    bad=np.load(inherited)['normal_reversed_triangles'].copy()
    bad[topology['reversed_triangles']]=True;bad[np.unique(topology['intersections'])]=True
    band=((neutral[pt,1]>=135)&(neutral[pt,1]<=153)).all(1)
    ids=np.flatnonzero(band&~bad)
    root.mkdir(exist_ok=False)
    paths=[PRIOR/'protocol.json',PRIOR/'fit.npz',PRIOR/'final_seal.json',PRIOR/'review_v2/result.json',PRIOR/'review_v2/silhouette_topology.npz',inherited]
    request=dict(frame='001193',arms=[ARM],source_mesh=protocol['original_mesh'],source_mesh_sha256=protocol['original_mesh_sha256'],
        input_hashes={str(p):sha(p) for p in paths},script_sha256=sha(__file__),
        helpers={n:sha(Path(__file__).with_name(n)) for n in ['build_mhr_local_patch_candidates.py','admit_mhr_local_patch_depth.py']},
        neutral_y_range_cm=[135,153],maximum_edge=.00075,maximum_original_surface_distance=.002,
        maximum_boundary_vertex_distance=.003,minimum_normal_dot=.25,minimum_centroid_distance=.00002,
        actual_opposite_edge_must_be_open=False,all_crossing_and_reversed_parents_excluded=True,
        target_camera_or_rgb_used=False,source_geometry_unchanged=True,raw_overlapping_surface_not_acceptable=True,
        inferred_not_measured=True,depth_admission_performed=False,final_prior_root=str(PRIOR))
    save(root/'request.json',request)
    # Exact-file links let the frozen admission helper load the NEW final prior.
    priorroot=root/'prior';(priorroot/ARM).mkdir(parents=True)
    (priorroot/'initial.npz').symlink_to(PRIOR/'fit.npz')
    (priorroot/ARM/'fit.npz').symlink_to(PRIOR/'fit.npz')
    folder=root/ARM;folder.mkdir()
    old=o3d.io.read_triangle_mesh(protocol['original_mesh']);old.compute_triangle_normals()
    ov,ot=np.asarray(old.vertices),np.asarray(old.triangles);scene=Scene2(ov,ot)
    edges=ot[:,[[1,2],[2,0],[0,1]]][boundary_opposites(ot)]
    tree=cKDTree(ov[np.unique(edges)])
    used=np.unique(pt[ids]);remap=np.full(len(v0),-1,int);remap[used]=np.arange(len(used))
    v,t,parents,rounds=subdivide(v0[used],remap[pt[ids]],ids,max_edge=.00075)
    print('subdivision',rounds,len(t),flush=True)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t));mesh.compute_vertex_normals()
    cp=scene.compute_closest_points(o3d.core.Tensor(v.astype(np.float32)));nearest=cp['points'].numpy();tid=cp['primitive_ids'].numpy()
    distance=np.linalg.norm(v-nearest,axis=1);bd=tree.query(v)[0]
    dot=np.sum(np.asarray(mesh.vertex_normals)*np.asarray(old.triangle_normals)[tid],axis=1)
    good=(distance<=.002)&(bd<=.003)&(dot>=.25)
    centers=v[t].mean(1);cc=scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
    gap=np.linalg.norm(centers-cc['points'].numpy(),axis=1);local=good[t].all(1);keep=local&(gap>=.00002)
    selected=np.unique(t[keep]);mapping=np.full(len(v),-1,int);mapping[selected]=np.arange(len(selected))
    vv=np.concatenate([ov,v[selected]]);pp=mapping[t[keep]]+len(ov);tt=np.concatenate([ot,pp])
    candidate=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt));candidate.compute_vertex_normals()
    assert o3d.io.write_triangle_mesh(str(folder/'local_raw.ply'),candidate)
    reread=o3d.io.read_triangle_mesh(str(folder/'local_raw.ply'))
    np.testing.assert_array_equal(np.asarray(reread.vertices)[:len(ov)],ov);np.testing.assert_array_equal(np.asarray(reread.triangles)[:len(ot)],ot)
    np.savez_compressed(folder/'proposal_evidence.npz',proposals=pp,parent_triangle_ids=parents[keep],closest_points=nearest[selected],
        closest_triangle_ids=tid[selected],distance=distance[selected],boundary_distance=bd[selected],normal_dot=dot[selected],centroid_distance=gap[keep])
    np.savez_compressed(folder/'domain_evidence.npz',subdivided_vertices=v,subdivided_triangles=t,parent_ids=parents,good_vertex=good,
        centroid_gap=gap,local_before_centroid_gate=local,retained=keep,unsafe_parent=bad)
    result=dict(arm=ARM,request_sha256=sha(root/'request.json'),original_vertices=len(ov),original_triangles=len(ot),
        anatomical_triangles=int(band.sum()),excluded_unsafe_in_band=int((band&bad).sum()),subdivision_rounds=rounds,
        subdivided_triangles=len(t),all_vertices_local=int(local.sum()),centroid_gate_rejected_local=int((local&~keep).sum()),
        added_vertices=len(selected),proposals=len(pp),original_prefix_exact=True,depth_guard_passed=False,production_accepted=False,
        hashes={p:sha(folder/p) for p in ['local_raw.ply','proposal_evidence.npz','domain_evidence.npz']})
    save(folder/'result.json',result);save(root/'result.json',dict(request_sha256=sha(root/'request.json'),arms=[result],production_accepted=False))
    print(result,flush=True)


if __name__=='__main__':build()
