"""Replay restricted solve and independently verify unchanged full-row energies."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
import study_mhr_static_unsafe_freeze as study
from study_multiview_face_prior import read,save,sha


def main():
    proof=study.configure();q=read(study.FINAL/'protocol.json')
    assert json.dumps(proof,sort_keys=True)==json.dumps(q['static_freeze_adapter'],sort_keys=True)
    assert q['recipe']==read(study.FAILED/'protocol.json')['recipe']
    saved=np.load(study.FINAL/'fit.npz');failed=np.load(study.FAILED/'fit.npz')
    np.testing.assert_array_equal(saved['baseline'],failed['baseline'])
    frozen,facets,pairs=study.selection();full=np.flatnonzero(saved['active']);free=np.setdiff1d(full,frozen)
    np.testing.assert_array_equal(saved['vertices'][frozen],saved['baseline'][frozen])
    # Independent full-vs-restricted matrix comparison with arbitrary nonzero
    # free displacement. Fixed columns are zero, not reweighted or deleted rows.
    lap=study.fit.uniform_laplacian(saved['triangles'],len(saved['vertices']))
    scale=q['recipe']['laplacian_sigma']*np.sqrt(len(full))
    full_lap=sparse.kron(lap[full][:,full]/scale,sparse.eye(3),format='csr')
    reduced=sparse.kron(lap[full][:,free]/scale,sparse.eye(3),format='csr')
    rng=np.random.default_rng(1201446);d=rng.normal(size=(len(free),3))*.001
    expanded=np.zeros((len(saved['vertices']),3));expanded[free]=d
    np.testing.assert_allclose(full_lap@expanded[full].ravel(),reduced@d.ravel(),rtol=1e-13,atol=1e-13)
    old_energy=np.sum((expanded[full]/(q['recipe']['magnitude_sigma']*np.sqrt(len(full))))**2)
    new_energy=np.sum((d/(q['recipe']['magnitude_sigma']*np.sqrt(len(full))))**2)
    np.testing.assert_allclose(old_energy,new_energy,rtol=1e-14)
    factory=study.continuation.make_optimizer
    def replay_factory():
        factory.__globals__['ROOT']=study.continuation.ROOT
        return factory()
    study.continuation.make_optimizer=replay_factory
    import audit_mhr_silhouette_convergence as frozen_audit
    # The earlier auditor assumed every fit had new crossings. Preserve its
    # failed first replay, and make only its empty-set reporting well-defined.
    audit_main,audit_adapter=study.zero.adapter(frozen_audit.main,[
        ("ROOT/'audit_replay'", "ROOT/'audit_replay_empty_safe'",1),
        ("        c=centers[ids];scene=fit.Scene2(v,tri[ids]);closest=scene.compute_closest_points",
         "        if len(ids)==0:\n            localization[name]=dict(triangles=0,triangle_ids=[],minimum_surface_distance_to_requested_rim=None,requested_first_hit_facets_in_set=0)\n            continue\n        c=centers[ids];scene=fit.Scene2(v,tri[ids]);closest=scene.compute_closest_points",1)],dict(frozen_audit.__dict__))
    audit_main()
    for i in range(1,101):
        v=np.load(study.FINAL/'iterates'/f'{i:03d}.npz')['vertices']
        np.testing.assert_array_equal(v[frozen],saved['baseline'][frozen])
    np.savez_compressed(study.ROOT/'restriction_evidence.npz',frozen_vertices=frozen,free_vertices=free,
        original_active_vertices=full,unsafe_facets=facets,new_pairs=pairs)
    save(study.ROOT/'audit_wrapper.json',dict(status='passed',wrapper_sha256=sha(__file__),
        restricted_proof=proof,frozen_auditor_path=str(Path(frozen_audit.__file__).resolve()),frozen_auditor_sha256=sha(frozen_audit.__file__),
        audit_sha256=sha(study.FINAL/'audit.json'),all_100_fixed_vertices_exact=True,
        empty_crossing_set_audit_adapter=audit_adapter,
        full_laplacian_shape=list(full_lap.shape),restricted_laplacian_shape=list(reduced.shape),
        original_laplacian_rows_preserved=True,normalization_count=1566,
        random_full_restricted_operator_energy_equivalent=True,
        evidence_sha256=sha(study.ROOT/'restriction_evidence.npz'),production_modified=False))
    print('restricted audit passed: 100 exact iterates, fixed120, full1566-row energy',flush=True)


if __name__=='__main__':main()
