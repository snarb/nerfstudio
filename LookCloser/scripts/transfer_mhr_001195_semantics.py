"""MediaPipe-only entry point: do not import Torch-dependent geometry wrappers."""
from transfer_mhr_001195 import ROOT,HEAD,FRAME,verify
from study_multiview_face_prior import save,sha
import study_mhr_local_head_prior as model


if __name__=='__main__':
    verify();model.OUT=HEAD;model.FRAME=FRAME
    model.semantic()
    save(ROOT/'semantic_complete.json',dict(config_sha256=sha(ROOT/'config.json'),
        wrapper_sha256=sha(__file__),producer_sha256=sha(model.__file__),
        change='minimal imports for isolated MediaPipe environment; identical semantic function',
        failed_full_wrapper_log='/mnt/data/dec5_mhr_transfer_001195_semantic.log'))
