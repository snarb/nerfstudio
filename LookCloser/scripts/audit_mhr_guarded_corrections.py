"""Rehash fit inputs and independently replay every saved geometry guard."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from check_mhr_conformance_crossings import transverse_crossings
from study_multiview_face_prior import read,save,sha


def pairs(v,t):
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    all_pairs=np.asarray(mesh.get_self_intersecting_triangles(),int).reshape(-1,2)
    strict=all_pairs[transverse_crossings(v[t[all_pairs[:,0]]],v[t[all_pairs[:,1]]])] if len(all_pairs) else all_pairs
    return set(map(tuple,np.sort(all_pairs,axis=1))),set(map(tuple,np.sort(strict,axis=1)))


def cross(v,t):return np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])


def audit(root):
    dest=root/'guard_audit.json';assert not dest.exists()
    q,r=read(root/'protocol.json'),read(root/'result.json');assert r['protocol_sha256']==sha(root/'protocol.json')
    bindings={}
    def check(path,digest):
        assert sha(path)==digest,str(path);bindings[str(path)]=digest
    for p,h in q['input_hashes'].items():check(p,h)
    for p,h in r['hashes'].items():check(root/p,h)
    for kind in ['contact_correction','bounded_contact_correction']:
        if kind in q:
            for p,h in q[kind]['helper_hashes'].items():check(p,h)
            if 'qp_backend' in q[kind]:
                for p,h in q[kind]['qp_backend']['helper_hashes'].items():check(p,h)
    f=np.load(root/'fit.npz');base,t,original,active=f['baseline'],f['triangles'],f['original_reference'],f['active']
    bc=cross(base,t);area=np.linalg.norm(bc,axis=1);oc=cross(original,t);good=np.sum(oc*bc,axis=1)>0
    allowed_all,allowed_strict=pairs(base,t);records=[]
    files=sorted((root/'iterates').glob('*.npz'));assert len(files)==len(r['history'])==len(r['guards'])
    previous=base
    for path,h,g in zip(files,r['history'],r['guards']):
        v=np.load(path)['vertices'];assert np.isfinite(v).all()
        np.testing.assert_array_equal(v[~active],base[~active]);c=cross(v,t);a=np.linalg.norm(c,axis=1)
        ratio=a/area;cos=np.sum(c*bc,axis=1)/np.maximum(a*area,1e-30)
        assert ratio.min()>=q['settings']['minimum_relative_area']-1e-12
        assert cos.min()>=q['settings']['minimum_normal_cosine']-1e-12
        assert (np.sum(c[good]*oc[good],axis=1)>0).all()
        displacement=float(np.linalg.norm(v-base,axis=1).max());step=float(np.linalg.norm(v-previous,axis=1).max())
        assert displacement<=q['settings']['maximum_displacement']+1e-12
        np.testing.assert_allclose(step,h['applied_maximum_step'],rtol=1e-6,atol=1e-12)
        ap,sp=pairs(v,t);assert not sp-allowed_strict
        records.append(dict(iterate=path.name,step=step,maximum_displacement=displacement,
            new_all_intersection_pairs=len(ap-allowed_all),new_strict_pairs=len(sp-allowed_strict),
            minimum_relative_area=float(ratio.min()),minimum_normal_cosine=float(cos.min())))
        check(path,sha(path));previous=v
    np.testing.assert_array_equal(previous,f['vertices'])
    mesh=o3d.io.read_triangle_mesh(str(root/'prior_only.ply'))
    np.testing.assert_array_equal(np.asarray(mesh.vertices),previous);np.testing.assert_array_equal(np.asarray(mesh.triangles),t)
    for path in root.rglob('*'):
        if path.is_file():bindings[str(path)]=sha(path)
    save(dest,dict(status='passed_geometry_guard_replay_not_anatomical_acceptance',
        records=records,checked_bindings=bindings,script_sha256=sha(__file__),
        new_all_intersections_observed=any(r['new_all_intersection_pairs'] for r in records),
        continuous_collision_freedom_not_proved=True,coplanar_overlap_not_excluded_by_transverse_guard=True,
        target_used_in_fit=q['target_used'],production_accepted=False))
    print(root.name,'verified iterates',len(records),'bindings',len(bindings),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();audit(a.root)
