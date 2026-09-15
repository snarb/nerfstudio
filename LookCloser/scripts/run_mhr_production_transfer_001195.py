"""Frozen 001195 prior/admission on the current production base, opt-in only."""
from pathlib import Path
from copy import deepcopy
import argparse
import importlib
import sys
import numpy as np
import transfer_mhr_001195 as transfer
import patch_mhr_transfer_001195 as adapters
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_production_patch_001195')
CANDIDATES=ROOT/'candidates'
OUT=ROOT/'admission'
FRAME='001195'
ARM='silhouette100'
PARENT=Path('/mnt/data/dec5_cinematic_wide_spiral_v3/wide_spiral_free')
PRIOR=transfer.FINAL


def binding():
    transfer.verify()
    parent=read(PARENT/'request.json');row=next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    for key in ['mesh','metadata']:assert sha(row[key])==row[key+'_sha256']
    q=read(PRIOR/'protocol.json');assert q['frame']==FRAME
    assert Path(row['metadata'])==Path(q['original_mesh']).with_suffix('.json')
    assert sha(q['original_mesh'])==q['original_mesh_sha256']
    source=read(Path('/mnt/data/dec5_jaw_measured_mask_control')/FRAME/'request.json')
    assert source['source_mesh_sha256']==row['mesh_sha256']
    seal=read(transfer.ROOT/'final_seal.json');assert seal['status']=='passed'
    return dict(frame=FRAME,parent_request_path=str(PARENT/'request.json'),parent_request_sha256=sha(PARENT/'request.json'),
        production_mesh=row['mesh'],production_mesh_sha256=row['mesh_sha256'],metadata=row['metadata'],metadata_sha256=row['metadata_sha256'],
        raw_mesh_not_used_as_base=q['original_mesh'],raw_mesh_sha256=q['original_mesh_sha256'],
        prior_protocol_sha256=sha(PRIOR/'protocol.json'),prior_fit_sha256=sha(PRIOR/'fit.npz'),
        prior_topology_sha256=sha(PRIOR/'review_v2/silhouette_topology.npz'),prior_seal_sha256=sha(PRIOR/'final_seal.json'),
        raw_transfer_seal_sha256=sha(transfer.ROOT/'final_seal.json'),normalization_metadata_identical=True,
        wrapper_sha256=sha(__file__),frame_adapter_path=str(Path(adapters.__file__).resolve()),frame_adapter_sha256=sha(adapters.__file__),
        candidate_math_unchanged=True,admission_math_unchanged=True,fresh_fit_performed=False,
        added_faces_copied_from_previous_arm=False,production_modified=False)


def build(destination=CANDIDATES):
    import build_mhr_silhouette_patch_candidates as builder
    proof=binding()
    def rebound_read(path):
        value=read(path)
        if Path(path)==PRIOR/'protocol.json':
            value=deepcopy(value);value.update(original_mesh=proof['production_mesh'],original_mesh_sha256=proof['production_mesh_sha256'])
        return value
    def annotated(path,value):
        if Path(path)==destination/'request.json':value=dict(value,production_base_binding=proof)
        save(path,value)
    worker,adapter=adapters.adapted(builder,'build',[
        ("inherited=Path('/mnt/data/dec5_mhr_measured_conformance/smooth100/fit.npz')","inherited=CONFORM/'smooth100/fit.npz'",1),
        ("frame='001193'","frame=FRAME",1)],dict(ROOT=destination,PRIOR=PRIOR,ARM=ARM,CONFORM=transfer.CONFORM,FRAME=FRAME,
            read=rebound_read,save=annotated))
    name='candidate_adapter.json' if destination==CANDIDATES else 'candidate_adapter_replay.json'
    save(ROOT/name,dict(adapter,production_base_binding=proof));worker(destination)


def configure():
    transfer.configure()
    import admit_mhr_local_patch_depth as admission
    admission.CANDIDATES=CANDIDATES;admission.OUT=OUT;admission.PRIOR=CANDIDATES/'prior';admission.ARMS=[ARM]
    for p in [admission.PRIOR/'initial.npz',admission.PRIOR/ARM/'fit.npz']:assert p.resolve()==(PRIOR/'fit.npz').resolve()
    return admission


def production_review_module():
    configure()
    import run_mhr_production_patch_control as old
    old.ROOT=ROOT;old.OUT=OUT;old.CANDIDATES=CANDIDATES;old.PARENT=PARENT;old.FRAME=FRAME;old.ARM=ARM
    old.binding=binding;old.configure=configure
    old.builder.PRIOR=PRIOR
    review=importlib.import_module('review_mhr_production_patch_control')
    import review_mhr_silhouette_patch_rgb as rgb
    rgb.FRAME=FRAME
    return review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build','admit','audit','clay','render','rgb_review','side_effects'])
    parser.add_argument('--views',nargs='+',default=['old_moving','F004_E','M004_B','C004_E']);args=parser.parse_args()
    proof=binding();ROOT.mkdir(exist_ok=True)
    if (ROOT/'request.json').exists():assert read(ROOT/'request.json')==proof
    else:save(ROOT/'request.json',proof)
    if args.stage=='build':build();return
    admission=configure()
    if args.stage=='admit':
        original=admission.save
        def annotated(path,value):
            if Path(path)==OUT/'request.json':value=dict(value,production_base_binding=proof)
            original(path,value)
        admission.save=annotated
        try:admission.run()
        finally:admission.save=original
        return
    if args.stage=='audit':
        replay=ROOT/'candidate_replay';build(replay)
        assert read(replay/'request.json')==read(CANDIDATES/'request.json')
        for name in ['domain_evidence.npz','proposal_evidence.npz']:
            a=np.load(CANDIDATES/ARM/name);b=np.load(replay/ARM/name);assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        assert sha(CANDIDATES/ARM/'local_raw.ply')==sha(replay/ARM/'local_raw.ply')
        importlib.import_module('audit_mhr_local_patch_depth').main()
        save(ROOT/'audit.json',dict(status='passed',production_base_binding=proof,candidate_arrays_replayed=True,
            candidate_ply_byte_exact=True,admission_audit_sha256=sha(OUT/'audit.json'),wrapper_sha256=sha(__file__)))
        return
    review=production_review_module()
    if args.stage=='clay':review.clay()
    elif args.stage=='render':review.render(args.views)
    elif args.stage=='rgb_review':review.rgb_review(args.views)
    else:
        import localize_mhr_silhouette_patch_occlusion as occlusion
        worker,adapter=adapters.adapted(occlusion,'main',[("'frames/001193'","'frames'/FRAME",1)],dict(OUT=OUT,FRAME=FRAME))
        occlusion.main=worker
        import localize_mhr_production_patch_side_effects as effects
        worker2,adapter2=adapters.adapted(effects,'main',[("    residual_mask_attribution()","    # No new target-defined residual search in this bounded transfer.",1)],
            dict(ROOT=ROOT,OUT=OUT,FRAME=FRAME))
        save(ROOT/'side_effects_wrapper.json',dict(wrapper_sha256=sha(__file__),occlusion_adapter=adapter,side_effects_adapter=adapter2,
            production_base_binding=proof,target_used_posthoc_only=True));worker2()
    save(ROOT/(args.stage+'_wrapper.json'),dict(wrapper_sha256=sha(__file__),production_base_binding=proof,
        frozen_review_path=str(Path(review.__file__).resolve()),frozen_review_sha256=sha(review.__file__),views=args.views,
        frame_globals_explicitly_rebound=True,current_policy=True,production_modified=False))


if __name__=='__main__':main()
