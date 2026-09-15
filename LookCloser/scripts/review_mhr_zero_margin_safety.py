"""Exact binary-mask posthoc checks and visibility-filtered unsafe-facet crops."""
from pathlib import Path
import numpy as np
import study_mhr_zero_margin as study
from study_multiview_face_prior import read,save,sha
from study_jaw_repair_transfer import mask_votes


def main():
    study.configure();study.fit.ROOT=study.FINAL
    import review_mhr_silhouette_continuation_safety as frozen
    frozen.main()
    _,rows,masks,names,evidence,_=study.fit.prepare()
    residual=np.load(study.ROOT/'comparison/residual.npz');stats={};arrays={}
    for label in ['margin2','margin0']:
        p=residual[label+'_points'];index=np.repeat(np.arange(len(p))[:,None],3,axis=1)
        support,outside=mask_votes(p,index,rows,masks,names)
        stats[label]=dict(queries=len(p),strict_binary_mask_pass=int(((support>=2)&(outside==0)).sum()),
            minimum_support=int(support.min()),maximum_outside=int(outside.max()))
        arrays[label+'_support']=support;arrays[label+'_outside']=outside
    np.savez_compressed(study.ROOT/'residual_binary_masks.npz',**arrays)
    save(study.ROOT/'safety_wrapper.json',dict(wrapper_sha256=sha(__file__),frozen_helper_path=str(Path(frozen.__file__).resolve()),
        frozen_helper_sha256=sha(frozen.__file__),safety_receipt_sha256=sha(study.FINAL/'safety_review/result.json'),
        residual_input_sha256=sha(study.ROOT/'comparison/residual.npz'),residual_evidence_sha256=sha(study.ROOT/'residual_binary_masks.npz'),
        residual_stats=stats,posthoc_only=True,production_modified=False))
    print(stats,flush=True)


if __name__=='__main__':main()
