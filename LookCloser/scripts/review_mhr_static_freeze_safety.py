"""Visibility-filtered native evidence for remaining normal changes, no patch."""
from pathlib import Path
import study_mhr_static_unsafe_freeze as study
from study_multiview_face_prior import save,sha


def main():
    study.configure();study.fit.ROOT=study.FINAL
    import review_mhr_silhouette_continuation_safety as frozen
    # There are no new strict crossings. Choose the diagnostic crop from the
    # remaining >90-degree normal-change facets; their display stays amber.
    worker,proof=study.zero.adapter(frozen.main,[
        ('centers=v[tri[bad]].mean(1)','centers=v[tri[reversed_ids]].mean(1)',1),
        ("np.asarray(audit['topology_localization']['new_strict_crossing']['triangle_ids'])",
         "np.asarray(audit['topology_localization']['new_strict_crossing']['triangle_ids'],dtype=int)",1),
        ("ROOT/'safety_review'", "ROOT/'safety_review_empty_safe'",1)],dict(frozen.__dict__))
    worker()
    save(study.ROOT/'safety_wrapper.json',dict(wrapper_sha256=sha(__file__),frozen_helper_path=str(Path(frozen.__file__).resolve()),
        frozen_helper_sha256=sha(frozen.__file__),crop_adapter=proof,
        safety_receipt_sha256=sha(study.FINAL/'safety_review_empty_safe/result.json'),
        native_crop_from_normal_changes_not_target=True,production_modified=False))


if __name__=='__main__':main()
