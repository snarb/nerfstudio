"""Independent complete deterministic replay of generic candidate extraction."""
from pathlib import Path
import numpy as np
from build_mhr_silhouette_patch_candidates import build, ROOT, ARM
from admit_mhr_silhouette_patch import OUT, CONTROL, binding
from study_multiview_face_prior import read,save,sha


def main():
    replay=OUT/'candidate_replay';build(replay)
    assert read(replay/'request.json')==read(ROOT/'request.json')
    comparisons={}
    for filename in ['domain_evidence.npz','proposal_evidence.npz']:
        a=np.load(ROOT/ARM/filename);b=np.load(replay/ARM/filename)
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        comparisons[filename]=a.files
    assert sha(ROOT/ARM/'local_raw.ply')==sha(replay/ARM/'local_raw.ply')
    q=read(OUT/'request.json');old=read(CONTROL/'request.json')
    assert q['final_prior_binding']==binding()
    changed={'candidate_request_sha256','candidate_result_sha256','arms','final_prior_binding'}
    assert {k:v for k,v in q.items() if k not in changed}=={k:v for k,v in old.items() if k not in changed}
    save(OUT/'candidate_audit.json',dict(status='passed',exact_array_replay=comparisons,raw_ply_byte_exact=True,
        admission_math_inputs_unchanged=True,script_sha256=sha(__file__),
        input_hashes={str(ROOT/'request.json'):sha(ROOT/'request.json'),str(ROOT/'result.json'):sha(ROOT/'result.json'),
            str(OUT/'request.json'):sha(OUT/'request.json'),str(CONTROL/'request.json'):sha(CONTROL/'request.json')},
        replay_root=str(replay)))
    print('candidate replay passed')


if __name__=='__main__':main()
