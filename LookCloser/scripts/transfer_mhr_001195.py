"""Explicit frame-only rebinding of the frozen MHR measured/silhouette pipeline."""
from pathlib import Path
import argparse
import importlib
import numpy as np
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_transfer_001195')
FRAME='001195'
CANONICAL=ROOT/'canonical_pose'
HEAD=ROOT/'head'
CONFORM=ROOT/'measured_conformance'
TEN=ROOT/'silhouette10'
FINAL=ROOT/'silhouette100'
RGB=Path('/mnt/data/dec5_multiview_face_prior')
OLD=Path('/mnt/data/dec5_mhr_local_head_prior')
PRODUCERS=['study_canonical_face_prior.py','study_mhr_local_head_prior.py','fit_mhr_local_head_prior.py',
    'fit_mhr_named_articulation.py','conform_mhr_measured_surface.py','fit_mhr_silhouette_conformance.py',
    'continue_mhr_silhouette_convergence.py','admit_mhr_local_patch_depth.py','review_mhr_silhouette_conformance.py',
    'probe_mhr_silhouette_locality.py','review_mhr_local_head_prior.py']


def configure():
    import study_canonical_face_prior as canonical
    import study_mhr_local_head_prior as head
    import conform_mhr_measured_surface as conform
    import admit_mhr_local_patch_depth as admission
    import fit_mhr_silhouette_conformance as sil
    import continue_mhr_silhouette_convergence as continuation
    canonical.OUT=CANONICAL;canonical.FRAME=FRAME
    head.OUT=HEAD;head.FRAME=FRAME;head.CANONICAL=CANONICAL
    conform.ROOT=CONFORM;conform.PARENT=HEAD;conform.ARMS={'smooth100':1.}
    admission.FRAME=FRAME;admission.SOURCE=Path('/mnt/data/dec5_jaw_measured_mask_control')/FRAME
    admission.CANDIDATES=ROOT/'evidence_geometry'
    sil.ROOT=TEN;sil.SOURCE=CONFORM
    continuation.ROOT=FINAL;continuation.CONTROL=TEN
    # Execution only: all inherited on-demand scenes use the same ray convention.
    import diffusion_mesh_repair
    diffusion_mesh_repair.scene_for=admission.Scene2
    return canonical,head,conform,sil,continuation


def init():
    ROOT.mkdir(exist_ok=False)
    paths=[RGB/FRAME/'input.json',RGB/'inference.json',OLD/'protocol.json',
        Path('/mnt/data/dec5_mhr_silhouette_convergence/final_seal.json'),
        Path('/mnt/data/dec5_jaw_measured_mask_control')/FRAME/'request.json']
    save(ROOT/'config.json',dict(frame=FRAME,root=str(ROOT),initialization='fresh canonical similarity from this frame measured landmarks; no previous fitted pose',
        stages=dict(canonical=str(CANONICAL),head=str(HEAD),conformance=str(CONFORM),silhouette10=str(TEN),silhouette100=str(FINAL)),
        inputs={str(p):sha(p) for p in paths},producers={p:sha(Path(__file__).with_name(p)) for p in PRODUCERS},
        wrapper_sha256=sha(__file__),same_frozen_hyperparameters=True,conformance_arm='smooth100',
        validation_prefixes=read(OLD/'protocol.json')['validation_prefixes'],cpu_threads=2,
        target_or_heldout_used_for_fit=False,production_changed=False))
    source=read(RGB/FRAME/'input.json');p=ROOT/'evidence_geometry';p.mkdir()
    save(p/'request.json',dict(frame=FRAME,source_mesh=source['mesh'],source_mesh_sha256=source['mesh_sha256'],
        evidence_only_no_candidates=True))
    save(p/'result.json',dict(request_sha256=sha(p/'request.json'),evidence_only_no_candidates=True))
    canonical,_,_,_,_=configure();CANONICAL.mkdir()
    old=Path('/mnt/data/dec5_canonical_face_prior')
    for name in ['canonical.npz','canonical_face_model.obj','LICENSE']:(CANONICAL/name).symlink_to(old/name)
    q=read(old/'protocol.json');q.update(frame=FRAME,source_input_sha256=sha(RGB/FRAME/'input.json'),
        original_mesh=source['mesh'],original_mesh_sha256=source['mesh_sha256'])
    save(CANONICAL/'protocol.json',q)
    canonical.stage()
    world,parameters,stats,errors=canonical.fit_once('similarity')
    dest=CANONICAL/'similarity';dest.mkdir();np.savez_compressed(dest/'fit.npz',vertices=world,parameters=parameters,**errors)
    save(dest/'result.json',dict(stats,fit_sha256=sha(dest/'fit.npz'),protocol_sha256=sha(CANONICAL/'protocol.json')))
    _,head,_,_,_=configure();head.init()
    print('fresh frame initialization complete',stats,flush=True)


def verify():
    q=read(ROOT/'config.json');assert sha(__file__)==q['wrapper_sha256']
    for p,h in q['inputs'].items():assert sha(p)==h,p
    for p,h in q['producers'].items():assert sha(Path(__file__).with_name(p))==h,p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','semantic','anchors','fit','articulation','conform','silhouette10','silhouette100','review']);args=parser.parse_args()
    if args.stage=='init':init();return
    verify();canonical,head,conform,sil,continuation=configure()
    if args.stage in ['semantic','anchors']:getattr(head,args.stage)()
    elif args.stage=='fit':importlib.import_module('fit_mhr_local_head_prior').run()
    elif args.stage=='articulation':importlib.import_module('fit_mhr_named_articulation').main()
    elif args.stage=='conform':
        original=conform.write
        def write(path,value):
            if Path(path)==CONFORM/'protocol.json':value=dict(value,frame=FRAME,transfer_config_sha256=sha(ROOT/'config.json'))
            original(path,value)
        conform.write=write;conform.init();conform.fit()
    elif args.stage in ['silhouette10','silhouette100']:
        original=sil.save
        def save_frame(path,value):
            if Path(path).name=='protocol.json':value=dict(value,frame=FRAME,transfer_config_sha256=sha(ROOT/'config.json'))
            original(path,value)
        sil.save=save_frame
        if args.stage=='silhouette10':sil.main()
        else:continuation.main()
    elif args.stage=='review':
        sil.ROOT=FINAL
        review=importlib.import_module('review_mhr_silhouette_conformance');review.main()
    save(ROOT/(args.stage+'_complete.json'),dict(stage=args.stage,config_sha256=sha(ROOT/'config.json'),wrapper_sha256=sha(__file__)))


if __name__=='__main__':main()
