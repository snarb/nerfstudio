"""Posthoc fixed-cohort/native review of the exact-ball correction control."""
from pathlib import Path
import review_mhr_guarded_corrections as review
from study_multiview_face_prior import save,sha


if __name__=='__main__':
    review.ROOT=Path('/mnt/data/dec5_mhr_certified_conic_review')
    review.ARMS={'margin2':Path('/mnt/data/dec5_mhr_silhouette_convergence'),
        'contacts':Path('/mnt/data/dec5_mhr_contact_correction_v2'),
        'conic':Path('/mnt/data/dec5_mhr_certified_conic_correction')}
    for root in review.ARMS.values():assert (root/'result.json').exists(),root
    review.main()
    save(review.ROOT/'runtime_config.json',dict(arms={k:str(v) for k,v in review.ARMS.items()},
        output=str(review.ROOT),wrapper_sha256=sha(__file__),review_helper_sha256=sha(review.__file__),
        fit_inputs_unchanged=True,posthoc_ray_tests_not_admission=True))
