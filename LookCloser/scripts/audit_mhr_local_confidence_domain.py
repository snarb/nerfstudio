"""Replay wider local proposal extraction and prove open-edge control inclusion."""
from pathlib import Path
import argparse
import numpy as np
from build_mhr_local_confidence_domain import ROOT, CONTROL, PRIOR, REVIEW, ARMS
from build_mhr_local_patch_candidates import subdivide, boundary_opposites
from build_train_hair_semantics import read, sha, write


def triangle_rows(vertices,triangles):
    return np.ascontiguousarray(vertices[triangles],dtype=np.float64).reshape(len(triangles),9).view('V72').ravel()


def verify():
    import open3d as o3d
    from scipy.spatial import cKDTree
    from diffusion_mesh_repair import scene_for
    r=read(ROOT/'request.json')
    assert r['script_sha256']==sha(Path(__file__).with_name('build_mhr_local_confidence_domain.py'))
    assert r['helper_sha256']==sha(Path(__file__).with_name('build_mhr_local_patch_candidates.py'))
    assert r['parent_request_sha256']==sha(CONTROL/'request.json') and not r['actual_opposite_edge_must_be_open']
    for p,h in r['input_hashes'].items(): assert sha(p)==h,p
    assert sha(r['source_mesh'])==r['source_mesh_sha256']
    old=o3d.io.read_triangle_mesh(r['source_mesh']); old.compute_triangle_normals()
    ov=np.asarray(old.vertices); ot=np.asarray(old.triangles); on=np.asarray(old.triangle_normals)
    scene=scene_for(ov,ot); edge=boundary_opposites(ot)
    tree=cKDTree(ov[np.unique(ot[:,[[1,2],[2,0],[0,1]]][edge])])
    initial=np.load(PRIOR/'initial.npz'); n=initial['neutral']; pt=initial['triangles']; summary=[]
    for arm in ARMS:
        result=read(ROOT/arm/'result.json'); assert result['request_sha256']==sha(ROOT/'request.json')
        for p,h in result['hashes'].items(): assert sha(ROOT/arm/p)==h,p
        fit=np.load(PRIOR/arm/'fit.npz'); bad=fit['normal_reversed_triangles'].copy()
        bad[np.unique(np.load(REVIEW/(arm+'_geometry.npz'))['self_intersection_pairs'])]=True
        ids=np.flatnonzero(((n[pt,1]>=135)&(n[pt,1]<=153)).all(1)&~bad)
        used=np.unique(pt[ids]); remap=np.full(len(n),-1,int); remap[used]=np.arange(len(used))
        v,t,parents,rounds=subdivide(fit['vertices'][used],remap[pt[ids]],ids)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t)); mesh.compute_vertex_normals()
        cp=scene.compute_closest_points(o3d.core.Tensor(v.astype(np.float32)))
        dist=np.linalg.norm(v-cp['points'].numpy(),axis=1); bd=tree.query(v)[0]
        dot=np.sum(np.asarray(mesh.vertex_normals)*on[cp['primitive_ids'].numpy()],axis=1)
        center=v[t].mean(1); cc=scene.compute_closest_points(o3d.core.Tensor(center.astype(np.float32)))
        keep=((dist<=.002)&(bd<=.003)&(dot>=.25))[t].all(1)&(np.linalg.norm(center-cc['points'].numpy(),axis=1)>=.00002)
        mesh=o3d.io.read_triangle_mesh(str(ROOT/arm/'local_raw.ply')); vv=np.asarray(mesh.vertices); tt=np.asarray(mesh.triangles)
        np.testing.assert_array_equal(vv[:len(ov)],ov); np.testing.assert_array_equal(tt[:len(ot)],ot)
        np.testing.assert_array_equal(vv[tt[len(ot):]],v[t[keep]])
        ev=np.load(ROOT/arm/'proposal_evidence.npz'); np.testing.assert_array_equal(ev['parent_triangle_ids'],parents[keep])
        np.testing.assert_array_equal(ev['proposals'],tt[len(ot):]); assert int(keep.sum())==result['proposals']
        cr=read(CONTROL/arm/'result.json'); cm=o3d.io.read_triangle_mesh(str(CONTROL/arm/'local_raw.ply'))
        assert sha(CONTROL/arm/'local_raw.ply')==cr['hashes']['local_raw.ply']
        cv=np.asarray(cm.vertices); ct=np.asarray(cm.triangles)[len(ot):]
        assert np.isin(triangle_rows(cv,ct),triangle_rows(vv,tt[len(ot):])).all()
        summary.append(dict(arm=arm,replayed_proposals=int(keep.sum()),old_control_facets_present=len(ct),
                            exact_original_prefix=True,normal_and_full_domain_replayed=True))
    return summary


def main(action):
    result=verify()
    if action=='seal':
        files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
        files += [Path(__file__).resolve(),Path(__file__).with_name('build_mhr_local_confidence_domain.py').resolve(),
                  Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_local_confidence_domain.md']
        write(ROOT/'artifact_manifest.json',dict(bindings={str(p):sha(p) for p in files},replay=result,
            status='raw_domain_verified_not_accepted_geometry'))
        print('sealed',len(files),'bindings',result,flush=True)
    else:
        for p,h in read(ROOT/'artifact_manifest.json')['bindings'].items(): assert sha(p)==h,p
        print('full local domain replay/hash audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['seal','check']); main(p.parse_args().action)
