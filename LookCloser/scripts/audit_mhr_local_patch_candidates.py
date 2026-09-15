"""Verify retained local proposals and seal raw diagnostic evidence."""
from pathlib import Path
import argparse
import numpy as np
from build_mhr_local_patch_candidates import ROOT, PRIOR, REVIEW, ARMS, boundary_opposites, near_actual_boundary
from build_train_hair_semantics import read, sha, write


def verify():
    import open3d as o3d
    from scipy.spatial import cKDTree
    from diffusion_mesh_repair import scene_for
    r=read(ROOT/'request.json')
    assert r['script_sha256']==sha(Path(__file__).with_name('build_mhr_local_patch_candidates.py'))
    assert sha(r['source_mesh'])==r['source_mesh_sha256']
    for p,h in r['input_hashes'].items(): assert sha(p)==h,p
    old=o3d.io.read_triangle_mesh(r['source_mesh']); old.compute_triangle_normals()
    v=np.asarray(old.vertices); t=np.asarray(old.triangles); normal=np.asarray(old.triangle_normals)
    scene=scene_for(v,t); opposite=boundary_opposites(t)
    tree=cKDTree(v[np.unique(t[:,[[1,2],[2,0],[0,1]]][opposite])])
    initial=np.load(PRIOR/'initial.npz'); n=initial['neutral']; pt=initial['triangles']
    for arm in ARMS:
        folder=ROOT/arm; result=read(folder/'result.json')
        assert result['request_sha256']==sha(ROOT/'request.json')
        for p,h in result['hashes'].items(): assert sha(folder/p)==h,p
        mesh=o3d.io.read_triangle_mesh(str(folder/'local_raw.ply')); vv=np.asarray(mesh.vertices); tt=np.asarray(mesh.triangles)
        np.testing.assert_array_equal(vv[:len(v)],v); np.testing.assert_array_equal(tt[:len(t)],t)
        added=vv[len(v):]; p=tt[len(t):]; evidence=np.load(folder/'proposal_evidence.npz')
        np.testing.assert_array_equal(p,evidence['proposals']); assert len(p)==result['proposals']
        assert np.isfinite(vv).all() and (p>=len(v)).all()
        parents=evidence['parent_triangle_ids']; neutral_tri=n[pt[parents]]
        assert ((neutral_tri[:,:,1]>=135)&(neutral_tri[:,:,1]<=153)).all()
        assert not np.load(PRIOR/arm/'fit.npz')['normal_reversed_triangles'][parents].any()
        unsafe=np.unique(np.load(REVIEW/(arm+'_geometry.npz'))['self_intersection_pairs'])
        assert not np.isin(parents,unsafe).any()
        nearest=scene.compute_closest_points(o3d.core.Tensor(added.astype(np.float32)))
        cp=nearest['points'].numpy(); distances=np.linalg.norm(added-cp,axis=1); bd=tree.query(added)[0]
        np.testing.assert_array_equal(cp,evidence['closest_points'])
        np.testing.assert_allclose(distances,evidence['distance'],atol=1e-12)
        np.testing.assert_allclose(bd,evidence['boundary_distance'],atol=1e-12)
        assert (distances<=.002).all() and (bd<=.003).all() and (evidence['normal_dot']>=.25).all()
        # Replay normals from the full selected/subdivided prior would be required
        # to validate producer normal values independently; this audit binds them
        # and validates geometric retention, not a separate normal reconstruction.
        assert np.linalg.norm(vv[p[:,[0,1,2]]]-vv[p[:,[1,2,0]]],axis=2).max()<=.00075
        centers=vv[p].mean(1); cc=scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
        uv=cc['primitive_uvs'].numpy(); bary=np.c_[1-uv.sum(1),uv]
        assert near_actual_boundary(bary,opposite[cc['primitive_ids'].numpy()]).all()
        assert (np.linalg.norm(centers-cc['points'].numpy(),axis=1)>=.00002).all()
        np.testing.assert_array_equal(bary,evidence['centroid_barycentric'])
    for sub,script in [('raw_review','review_mhr_local_patch_candidates.py'),('gate_diagnosis','diagnose_mhr_patch_local_gate.py')]:
        result=read(ROOT/sub/'result.json')
        assert result['script_sha256']==sha(Path(__file__).with_name(script))
        for p,h in result['input_hashes'].items(): assert sha(p)==h,p
        for item in result.get('files',[]): assert sha(item['path'])==item['sha256']
        if sub=='gate_diagnosis':
            for arm,h in result['arrays'].items(): assert sha(ROOT/sub/(arm+'.npz'))==h


def main(action):
    verify()
    if action=='seal':
        names=['G004_A005_121071.png','G004_B005_1210FG.png','M004_A005_1210WZ.png',
               'M004_B005_12109O.png','E004_B005_1210I7.png','H004_C005_1210SZ.png','requested_hole.png']
        write(ROOT/'visual_review.json',dict(reviewer='main LLM',
            inspected={str(ROOT/'raw_review'/n):sha(ROOT/'raw_review'/n) for n in names},
            verdict='fail_raw_candidate_not_production',
            notes='All three raw variants retain upper-face detail but add ragged lower-neck fringe sheets/islands. Requested hole remains44/44 for025/100,29/44 for400. No RGB repair or depth approval demonstrated.',
            depth_admission_is_separate=True,production_promoted=False))
        scripts=['build_mhr_local_patch_candidates.py','review_mhr_local_patch_candidates.py',
                 'diagnose_mhr_patch_local_gate.py',Path(__file__).name]
        files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
        files += [Path(__file__).with_name(n).resolve() for n in scripts]
        files += [Path(__file__).resolve().parents[1]/n for n in ['tests/test_mhr_local_patch_candidates.py',
                    'experiments/dec5_mhr_local_patch_candidates.md']]
        files += [Path('/mnt/data/dec5_mhr_local_patch_candidates_tests.log')]
        write(ROOT/'artifact_manifest.json',dict(bindings={str(p):sha(p) for p in files},
            status='raw_candidate_diagnostics_verified_not_accepted_repair'))
        print('sealed',len(files),'bindings',flush=True)
    else:
        for p,h in read(ROOT/'artifact_manifest.json')['bindings'].items(): assert sha(p)==h,p
        print('local proposal geometry/hash audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['seal','check']); main(p.parse_args().action)
