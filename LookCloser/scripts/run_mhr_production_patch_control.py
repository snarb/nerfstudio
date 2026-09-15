"""Re-evaluate inferred local completion on the actual frozen video mesh.

Only the candidate/admission base changes from the raw mesh to its already
carved/repaired production counterpart. No production file is written.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import numpy as np
from study_multiview_face_prior import read, save, sha
import build_mhr_silhouette_patch_candidates as builder
import admit_mhr_local_patch_depth as admission

ROOT=Path('/mnt/data/dec5_mhr_production_patch_001193')
CANDIDATES=ROOT/'candidates'
OUT=ROOT/'admission'
PARENT=Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
FRAME='001193'
ARM='silhouette100'


def binding():
    q=read(PARENT/'request.json');row=next(r for r in q['inventory'] if r['frame_id']==FRAME)
    for name in ['mesh','metadata']:assert sha(row[name])==row[name+'_sha256']
    raw=read(builder.PRIOR/'protocol.json')
    assert Path(row['metadata'])==Path(raw['original_mesh']).with_suffix('.json')
    assert sha(raw['original_mesh'])==raw['original_mesh_sha256']
    depth_input=read(admission.SOURCE/'request.json')
    assert depth_input['source_mesh_sha256']==row['mesh_sha256']
    seal=read('/mnt/data/dec5_mhr_silhouette_patch_admission/final_seal.json')
    assert seal['status']=='passed'
    return dict(frame=FRAME,parent_request_path=str(PARENT/'request.json'),parent_request_sha256=sha(PARENT/'request.json'),
        production_mesh=row['mesh'],production_mesh_sha256=row['mesh_sha256'],metadata=row['metadata'],metadata_sha256=row['metadata_sha256'],
        raw_mesh_not_used_as_base=raw['original_mesh'],raw_mesh_sha256=raw['original_mesh_sha256'],
        prior_protocol_sha256=sha(builder.PRIOR/'protocol.json'),prior_fit_sha256=sha(builder.PRIOR/'fit.npz'),
        raw_admission_seal_sha256=sha('/mnt/data/dec5_mhr_silhouette_patch_admission/final_seal.json'),
        normalization_metadata_identical=True,original_carving_preserved_in_base=True,
        wrapper_sha256=sha(__file__),builder_sha256=sha(builder.__file__),admission_sha256=sha(admission.__file__),
        candidate_math_unchanged=True,admission_math_unchanged=True,production_modified=False)


def build(destination):
    proof=binding();original_read,original_save=builder.read,builder.save
    calls=[]
    def rebind(path):
        value=original_read(path)
        if Path(path)==builder.PRIOR/'protocol.json':
            calls.append(str(path));value=deepcopy(value)
            value.update(original_mesh=proof['production_mesh'],original_mesh_sha256=proof['production_mesh_sha256'])
        return value
    def annotate(path,value):
        if Path(path)==destination/'request.json':value=dict(value,production_base_binding=proof)
        original_save(path,value)
    builder.read,builder.save=rebind,annotate
    try:builder.build(destination)
    finally:builder.read,builder.save=original_read,original_save
    assert len(calls)==1, calls


def configure():
    admission.CANDIDATES=CANDIDATES;admission.OUT=OUT
    admission.PRIOR=CANDIDATES/'prior';admission.ARMS=[ARM]
    for p in [admission.PRIOR/'initial.npz',admission.PRIOR/ARM/'fit.npz']:
        assert p.resolve()==(builder.PRIOR/'fit.npz').resolve()
    frozen=read('/mnt/data/dec5_mhr_local_patch_admission/request.json')
    assert sha(admission.__file__)==frozen['script_sha256']
    for n,h in frozen['helpers'].items():assert sha(Path(admission.__file__).with_name(n))==h


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['build','admit','audit'])
    args=parser.parse_args();proof=binding();ROOT.mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():assert read(ROOT/'request.json')==proof
    else:save(ROOT/'request.json',proof)
    if args.stage=='build':build(CANDIDATES);return
    configure()
    if args.stage=='admit':
        original=admission.save
        def annotate(path,value):
            if Path(path)==OUT/'request.json':value=dict(value,production_base_binding=proof)
            original(path,value)
        admission.save=annotate
        try:admission.run()
        finally:admission.save=original
        return
    assert read(OUT/'request.json')['production_base_binding']==proof
    replay=ROOT/'candidate_replay';build(replay)
    assert read(replay/'request.json')==read(CANDIDATES/'request.json')
    for name in ['domain_evidence.npz','proposal_evidence.npz']:
        a=np.load(CANDIDATES/ARM/name);b=np.load(replay/ARM/name)
        assert a.files==b.files
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    assert sha(CANDIDATES/ARM/'local_raw.ply')==sha(replay/ARM/'local_raw.ply')
    import audit_mhr_local_patch_depth as audit
    audit.main()
    save(ROOT/'audit.json',dict(production_base_binding=proof,candidate_arrays_replayed=True,
        candidate_ply_byte_exact=True,admission_audit_sha256=sha(OUT/'audit.json'),production_modified=False))


if __name__=='__main__':main()
