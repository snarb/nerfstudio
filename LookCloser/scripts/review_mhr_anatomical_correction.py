"""Compare on shared vertex IDs when the anatomical active domain changes."""
from pathlib import Path
import inspect
import numpy as np
import review_mhr_guarded_corrections as review
from study_multiview_face_prior import save,sha


if __name__=='__main__':
    root=Path('/mnt/data/dec5_mhr_anatomical_review')
    arms={'margin2':Path('/mnt/data/dec5_mhr_silhouette_convergence'),
        'contacts':Path('/mnt/data/dec5_mhr_contact_correction_v2'),
        'anatomical':Path('/mnt/data/dec5_mhr_anatomical_correction')}
    for p in arms.values():assert (p/'result.json').exists(),f'Fit is not terminal: {p}'
    common=np.logical_and.reduce([np.load(p/'fit.npz')['active'] for p in arms.values()])
    source=inspect.getsource(review.main)
    old="v,t,active=data['vertices'],data['triangles'],data['active']"
    assert source.count(old)==1
    generated=source.replace(old,old+'\n        active=COMMON_ACTIVE')
    ns=dict(review.__dict__,ROOT=root,ARMS=arms,COMMON_ACTIVE=common)
    exec(compile(generated,'<shared_anatomical_vertex_cohort_review>','exec'),ns)
    ns['main']()
    save(root/'runtime_config.json',dict(arms={k:str(v) for k,v in arms.items()},output=str(root),
        common_vertex_ids=np.flatnonzero(common).tolist(),generated_source=generated,
        wrapper_sha256=sha(__file__),review_helper_sha256=sha(review.__file__),
        fit_inputs_unchanged=True,posthoc_ray_tests_not_admission=True))
