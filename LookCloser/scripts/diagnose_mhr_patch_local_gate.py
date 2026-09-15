"""Explain raw extraction misses at pre-existing post-hoc probe points."""
import numpy as np
from scipy.spatial import cKDTree
from build_mhr_local_patch_candidates import ROOT, PRIOR, REVIEW, ARMS, subdivide, boundary_opposites, near_actual_boundary
from build_train_hair_semantics import read, sha, write


def main():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    r=read(ROOT/'request.json'); initial=np.load(PRIOR/'initial.npz'); pt=initial['triangles']; n=initial['neutral']
    old=o3d.io.read_triangle_mesh(r['source_mesh']); old.compute_triangle_normals()
    ov=np.asarray(old.vertices); ot=np.asarray(old.triangles); on=np.asarray(old.triangle_normals)
    scene=scene_for(ov,ot); opposite=boundary_opposites(ot)
    tree=cKDTree(ov[np.unique(ot[:,[[1,2],[2,0],[0,1]]][opposite])])
    out=ROOT/'gate_diagnosis'; out.mkdir(exist_ok=False); summary=[]; inputs={}
    for arm in ARMS:
        fp=PRIOR/arm/'fit.npz'; gp=REVIEW/(arm+'_geometry.npz')
        qp=PRIOR/'probe_smooth025_smooth100_smooth400'/(arm+'.npz')
        for p in [fp,gp,qp]: inputs[str(p)]=sha(p)
        fit=np.load(fp); geometry=np.load(gp); queries=np.load(qp)['target_prior_points']
        safe=((n[pt,1]>=135)&(n[pt,1]<=153)).all(1)&~fit['normal_reversed_triangles']
        safe[np.unique(geometry['self_intersection_pairs'])]=False
        ids=np.flatnonzero(safe); used=np.unique(pt[ids]); mapping=np.full(len(n),-1,int); mapping[used]=np.arange(len(used))
        v,t,parents,_=subdivide(fit['vertices'][used],mapping[pt[ids]],ids)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t)); mesh.compute_vertex_normals()
        local=scene_for(v,t); near=local.compute_closest_points(o3d.core.Tensor(queries.astype(np.float32)))
        tids=near['primitive_ids'].numpy(); verts=t[tids]; qv=v[verts]
        cp=scene.compute_closest_points(o3d.core.Tensor(qv.reshape(-1,3).astype(np.float32)))
        distance=np.linalg.norm(qv.reshape(-1,3)-cp['points'].numpy(),axis=1).reshape(-1,3)
        bd=tree.query(qv.reshape(-1,3))[0].reshape(-1,3)
        dot=np.sum(np.asarray(mesh.vertex_normals)[verts].reshape(-1,3)*on[cp['primitive_ids'].numpy()],axis=1).reshape(-1,3)
        centers=qv.mean(1); cc=scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
        uv=cc['primitive_uvs'].numpy(); bary=np.c_[1-uv.sum(1),uv]; op=opposite[cc['primitive_ids'].numpy()]
        edge=near_actual_boundary(bary,op); center_distance=np.linalg.norm(centers-cc['points'].numpy(),axis=1)
        gates={'in_selected_band':np.linalg.norm(queries-near['points'].numpy(),axis=1)<1e-6,
            'vertex_surface_distance':(distance<=.002).all(1),'vertex_boundary_distance':(bd<=.003).all(1),
            'vertex_normal_agreement':(dot>=.25).all(1),'actual_open_edge':edge,'minimum_centroid_distance':center_distance>=.00002}
        allpass=np.logical_and.reduce(list(gates.values()))
        item=dict(arm=arm,queries=len(queries),gates={k:int(vv.sum()) for k,vv in gates.items()},
            all_pass=int(allpass.sum()),minimum_vertex_normal_cosine=float(dot.min()),
            maximum_vertex_surface_distance=float(distance.max()),maximum_boundary_distance=float(bd.max()),
            gates_are_not_independent=True)
        np.savez_compressed(out/(arm+'.npz'),query_points=queries,triangle_ids=tids,parent_ids=parents[tids],
            distance=distance,boundary_distance=bd,normal_dot=dot,barycentric=bary,opposite_open=op,
            centroid_distance=center_distance,all_pass=allpass,**gates)
        summary.append(item); print(item,flush=True)
    write(out/'result.json',dict(arms=summary,input_hashes=inputs,script_sha256=sha(__file__),
        build_request_sha256=sha(ROOT/'request.json'),diagnostic_only=True,geometry_changed=False,
        arrays={a:sha(out/(a+'.npz')) for a in ARMS}))


if __name__=='__main__': main()
