"""Attribute the static restriction's residual, without another exclusion set."""
import numpy as np
import study_mhr_static_unsafe_freeze as study
from study_multiview_face_prior import save,sha


def main():
    a=np.load(study.FINAL/'fit.npz');q=np.load(study.ROOT/'comparison/cohorts.npz')
    frozen,_,_=study.selection();full=np.flatnonzero(a['active']);is_fixed=np.isin(full,frozen)
    np.testing.assert_array_equal(q['freeze'][:,is_fixed],q['base'][:,is_fixed])
    stats={}
    for split,selected in [('fit',~q['validation']),('reserved',q['validation'])]:
        total=np.maximum(q['freeze'][q['common']&selected[:,None]],0).sum()
        stats[split]={}
        for label,selection in [('fixed',is_fixed),('free',~is_fixed)]:
            choose=q['common']&selected[:,None]&selection[None,:];values=np.maximum(q['freeze'][choose],0)
            stats[split][label]=dict(samples=len(values),outside=int((values>0).sum()),mean_positive_sdf=float(values.mean()),
                total_positive_sdf=float(values.sum()),fraction_of_total_positive_sdf=float(values.sum()/total))
    topo=np.load(study.FINAL/'review_v2/silhouette_topology.npz');tri=a['triangles'][topo['reversed_triangles']]
    incident=np.isin(tri,frozen).sum(1)
    save(study.ROOT/'restriction_tradeoff.json',dict(script_sha256=sha(__file__),fixed_sdf_exactly_matches_original_base=True,
        fixed_free_cohorts=stats,normal_change_faces_by_number_fixed_vertices={str(i):int((incident==i).sum()) for i in range(4)},
        input_hashes={str(p):sha(p) for p in [study.FINAL/'fit.npz',study.ROOT/'comparison/cohorts.npz',study.FINAL/'review_v2/silhouette_topology.npz']},
        further_exclusion_search_performed=False))
    print(stats,flush=True)


if __name__=='__main__':main()
