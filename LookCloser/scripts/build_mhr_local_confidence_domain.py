"""Broader local proposal domain for subsequent measured-confidence admission.

Same prior/locality constraints as the open-edge control, without its nearest
open-edge projection requirement. These overlapping raw proposals MUST NOT be
rendered as an accepted repair; subsequent measured-depth/seed-hull gates decide.
"""
from pathlib import Path
import numpy as np
from build_train_hair_semantics import read, sha, write
from build_mhr_local_patch_candidates import PRIOR, REVIEW, ARMS, ROOT as CONTROL, subdivide, boundary_opposites

ROOT=Path('/mnt/data/dec5_mhr_local_confidence_domain')


def main():
    import open3d as o3d
    from scipy.spatial import cKDTree
    from diffusion_mesh_repair import scene_for
    control=read(CONTROL/'request.json')
    for p,h in control['input_hashes'].items(): assert sha(p)==h,p
    assert sha(control['source_mesh'])==control['source_mesh_sha256']
    ROOT.mkdir(exist_ok=False)
    request=dict(control)
    request.update(parent_open_edge_control=str(CONTROL),parent_request_sha256=sha(CONTROL/'request.json'),
        script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('build_mhr_local_patch_candidates.py')),
        actual_opposite_edge_must_be_open=False,domain='all_local_facets_pending_independent_confidence',
        rationale='Nearest original point may be inside an adjacent facet even for a missing-ray prior point.',
        depth_admission_performed=False,required_next_step='unchanged measured-depth or certified local interpolation plus native free-space veto',
        raw_overlapping_surface_not_acceptable=True)
    write(ROOT/'request.json',request)
    initial=np.load(PRIOR/'initial.npz'); n=initial['neutral']; pt=initial['triangles']
    old=o3d.io.read_triangle_mesh(control['source_mesh']); old.compute_triangle_normals()
    ov=np.asarray(old.vertices); ot=np.asarray(old.triangles); normals=np.asarray(old.triangle_normals)
    scene=scene_for(ov,ot); opposite=boundary_opposites(ot)
    tree=cKDTree(ov[np.unique(ot[:,[[1,2],[2,0],[0,1]]][opposite])]); summaries=[]
    for arm in ARMS:
        folder=ROOT/arm; folder.mkdir(); fit=np.load(PRIOR/arm/'fit.npz')
        bad=fit['normal_reversed_triangles'].copy(); bad[np.unique(np.load(REVIEW/(arm+'_geometry.npz'))['self_intersection_pairs'])]=True
        ids=np.flatnonzero(((n[pt,1]>=135)&(n[pt,1]<=153)).all(1)&~bad)
        used=np.unique(pt[ids]); remap=np.full(len(n),-1,int); remap[used]=np.arange(len(used))
        v,t,parents,rounds=subdivide(fit['vertices'][used],remap[pt[ids]],ids)
        prior=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t)); prior.compute_vertex_normals()
        cp=scene.compute_closest_points(o3d.core.Tensor(v.astype(np.float32))); closest=cp['points'].numpy(); nearest_ids=cp['primitive_ids'].numpy()
        distance=np.linalg.norm(v-closest,axis=1); bd=tree.query(v)[0]
        dot=np.sum(np.asarray(prior.vertex_normals)*normals[nearest_ids],axis=1)
        good=(distance<=.002)&(bd<=.003)&(dot>=.25)
        centers=v[t].mean(1); cc=scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
        center_distance=np.linalg.norm(centers-cc['points'].numpy(),axis=1)
        keep=good[t].all(1)&(center_distance>=.00002)
        used=np.unique(t[keep]); remap=np.full(len(v),-1,int); remap[used]=np.arange(len(used))
        vv=np.concatenate([ov,v[used]]); pp=remap[t[keep]]+len(ov); tt=np.concatenate([ot,pp])
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vv),o3d.utility.Vector3iVector(tt)); mesh.compute_vertex_normals()
        assert o3d.io.write_triangle_mesh(str(folder/'local_raw.ply'),mesh)
        reread=o3d.io.read_triangle_mesh(str(folder/'local_raw.ply'))
        np.testing.assert_array_equal(np.asarray(reread.vertices)[:len(ov)],ov); np.testing.assert_array_equal(np.asarray(reread.triangles)[:len(ot)],ot)
        np.savez_compressed(folder/'proposal_evidence.npz',proposals=pp,parent_triangle_ids=parents[keep],
            closest_points=closest[used],closest_triangle_ids=nearest_ids[used],distance=distance[used],
            boundary_distance=bd[used],normal_dot=dot[used],centroid_distance=center_distance[keep])
        result=dict(arm=arm,request_sha256=sha(ROOT/'request.json'),original_vertices=len(ov),original_triangles=len(ot),
            subdivision_rounds=rounds,subdivided_triangles=len(t),proposals=len(pp),added_vertices=len(used),
            original_prefix_exact=True,production_accepted=False,depth_guard_passed=False,
            hashes={p:sha(folder/p) for p in ['local_raw.ply','proposal_evidence.npz']})
        write(folder/'result.json',result); summaries.append(result); print(arm,len(pp),'local proposals',flush=True)
    write(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),arms=summaries,production_accepted=False))


if __name__=='__main__': main()
