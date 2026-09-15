"""Explicit head/neck-domain control; guards also reject nontransverse contacts."""
from pathlib import Path
import inspect
import fit_mhr_guarded_correction as parent
import fit_mhr_constrained_correction as qp
import fit_mhr_conic_correction as driver
import guard_mhr_anatomical_correction as anatomical
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_mhr_anatomical_correction')
ANCHOR_PROTOCOL=Path('/mnt/data/dec5_mhr_local_head_prior/protocol.json')
original_optimizer=qp.original_optimizer


def domain_optimizer(guard):
    worker,stats=original_optimizer(guard)
    proof=read(ROOT/'optimizer_source.json');source=proof['generated_source']
    old='active = (neutral[:, 1] > 135) & (neutral[:, 1] < 153)'
    assert source.count(old)==1
    source=source.replace(old,'active = anatomical_domain(neutral)')
    ns=dict(worker.__globals__,anatomical_domain=anatomical.anatomical_domain)
    exec(compile(source,'<anatomical_correction_optimizer>','exec'),ns)
    save(ROOT/'optimizer_source.json',dict(proof,generated_source=source,
        anatomical_domain_helper_sha256=sha(anatomical.__file__),neutral_abs_x_max_cm=12))
    return ns['optimize'],stats


if __name__=='__main__':
    assert read(ANCHOR_PROTOCOL)['initial_model_domain_abs_x']==12
    main_source=inspect.getsource(parent.main)
    old='active=(neutral[:,1]>135)&(neutral[:,1]<153)';assert main_source.count(old)==1
    generated_main=main_source.replace(old,'active=anatomical_domain(neutral)')
    parent.__dict__['anatomical_domain']=anatomical.anatomical_domain
    exec(compile(generated_main,'<anatomical_correction_main>','exec'),parent.__dict__)
    qp.original_optimizer=domain_optimizer
    source=Path(driver.__file__).read_text()
    changes={"ROOT=Path('/mnt/data/dec5_mhr_conic_correction')":f'ROOT=Path({str(ROOT)!r})',
        'import conic_surface_step as backend':'import certified_conic_surface_step as backend',
        'parent.StepGuard=qp.ConstraintGuard':'parent.StepGuard=ANATOMICAL_GUARD',
        'qp_backend=backend.provenance(),maximum_constraint_generation_rounds=9,':
        'qp_backend=backend.provenance(),anatomical_domain=DOMAIN_PROOF,maximum_constraint_generation_rounds=9,'}
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    proof=dict(neutral_y_cm=[135,153],neutral_abs_x_max_cm=12,
        width_from_existing_anchor_protocol=True,normalization_uses_reduced_active_count=True,
        new_all_intersection_pairs_forbidden=True,discrete_not_continuous_guard=True,
        input_hashes={str(p):sha(p) for p in [ANCHOR_PROTOCOL,Path(__file__),Path(anatomical.__file__)]},
        original_main_source=main_source,generated_main_source=generated_main,generated_driver_source=source)
    exec(compile(source,'<anatomical_conic_driver>','exec'),dict(__name__='__main__',
        __file__=driver.__file__,DOMAIN_PROOF=proof,ANATOMICAL_GUARD=anatomical.AllContactGuard))
