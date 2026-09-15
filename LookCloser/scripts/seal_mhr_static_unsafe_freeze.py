"""Seal one rejected static restriction; independently replay global topology."""
from pathlib import Path
import hashlib
import numpy as np
import study_mhr_static_unsafe_freeze as study
from study_multiview_face_prior import read,save,sha


def main():
    import open3d as o3d
    from check_mhr_conformance_crossings import transverse_crossings
    root=study.ROOT;final=study.FINAL;target=root/'final_seal.json';assert not target.exists()
    checked={}
    def check(path,digest):
        path=Path(path);assert path.is_file() and sha(path)==digest,str(path);checked[str(path.resolve())]=digest
    parent=read(study.zero.ROOT/'final_seal.json');assert parent['status']=='passed'
    for p,h in parent['inventory'].items():check(study.zero.ROOT/p,h)
    for p,h in parent['checked_bindings'].items():check(p,h)
    q=read(final/'protocol.json');audit=read(final/'audit.json');wrapper=read(root/'audit_wrapper.json')
    assert audit['status']=='passed' and wrapper['status']=='passed'
    assert audit['exact_all_iterates_replayed'] and wrapper['all_100_fixed_vertices_exact']
    assert wrapper['normalization_count']==1566 and wrapper['restricted_laplacian_shape']==[4698,4338]
    for p,h in audit['checked_bindings'].items():check(p,h)
    proof=q['static_freeze_adapter']
    for prefix in ['wrapper','zero_helper','frozen_fit','frozen_continuation']:
        check(proof[prefix+'_path'],proof[prefix+'_sha256'])
    adapters=[proof['optimizer'],proof['observer'],wrapper['empty_crossing_set_audit_adapter']]
    safety=read(root/'safety_wrapper.json');adapters.append(safety['crop_adapter'])
    for item in adapters:
        assert study.zero.replace_exact(item['original_source'],item['replacements'])==item['generated_source']
        for key in ['original','generated']:
            assert hashlib.sha256(item[key+'_source'].encode()).hexdigest()==item[key+'_sha256']
    check(Path(__file__).with_name('audit_mhr_static_unsafe_freeze.py'),wrapper['wrapper_sha256'])
    check(Path(__file__).with_name('review_mhr_static_freeze_safety.py'),safety['wrapper_sha256'])
    check(safety['frozen_helper_path'],safety['frozen_helper_sha256'])
    sources={final/'review_v2/result.json':'review_mhr_silhouette_conformance.py',
        final/'locality/result.json':'probe_mhr_silhouette_locality.py',
        final/'safety_review_empty_safe/result.json':'review_mhr_silhouette_continuation_safety.py',
        root/'comparison/result.json':'review_mhr_static_unsafe_freeze.py',
        root/'restriction_tradeoff.json':'explain_mhr_static_freeze.py',
        final/'audit.json':'audit_mhr_silhouette_convergence.py'}
    for receipt,producer in sources.items():
        value=read(receipt);check(Path(__file__).with_name(producer),value['script_sha256'])
        for p,h in value.get('input_hashes',{}).items():check(p,h)
        for row in value.get('files',[]):check(row['path'],row['sha256'])
    a=np.load(final/'fit.npz');v,t=a['vertices'],a['triangles'];frozen,_,_=study.selection()
    np.testing.assert_array_equal(v[frozen],a['baseline'][frozen]);np.testing.assert_array_equal(v[~a['active']],a['baseline'][~a['active']])
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t))
    pairs=np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1,2)
    strict=pairs[transverse_crossings(v[t[pairs[:,0]]],v[t[pairs[:,1]]])]
    topology=np.load(final/'review_v2/silhouette_topology.npz')
    np.testing.assert_array_equal(pairs,topology['intersections']);np.testing.assert_array_equal(strict,topology['strict_pairs'])
    cross=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
    b=a['baseline'];basecross=np.cross(b[t[:,1]]-b[t[:,0]],b[t[:,2]]-b[t[:,0]])
    np.testing.assert_array_equal(np.flatnonzero(np.sum(cross*basecross,axis=1)<=0),topology['reversed_triangles'])
    result=read(root/'comparison/result.json');assert not result['admission_permitted']
    assert result['posthoc']['freeze']['topology']==dict(strict_pairs=163,new_strict_pairs=0,normal_changes_over90=100)
    assert not (root/'candidates').exists() and not (root/'admission').exists()
    repo=Path(__file__).resolve().parents[1];report=repo/'experiments/dec5_mhr_static_unsafe_freeze.md';tests=repo/'tests/test_mhr_static_unsafe_freeze.py'
    for p in [report,tests,Path(__file__).resolve()]:check(p,sha(p))
    save(target,dict(status='passed',checked_bindings=checked,
        inventory={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
        report_path=str(report),report_sha256=sha(report),tests_path=str(tests),tests_sha256=sha(tests),script_sha256=sha(__file__),
        global_topology_exactly_replayed=True,static_frozen_vertices=120,free_vertices=1446,all_original_rows=1566,
        verdict='rejected_fit_only_normal_and_silhouette_regression',patch_generated=False,production_modified=False,
        actually_inspected=['comparison/residual_prior_clay.png','comparison/requested_hole_prior_only.png',
            'comparison/C004_E005_1210X7_clay.png','comparison/G004_B005_1210FG_clay.png','comparison/M004_B005_12109O_clay.png',
            'fit100/safety_review_empty_safe/C004_E005_1210X7.png','fit100/safety_review_empty_safe/G004_B005_1210FG.png']))
    print('seal passed',len(checked),'bindings; global topology replayed',flush=True)


if __name__=='__main__':main()
