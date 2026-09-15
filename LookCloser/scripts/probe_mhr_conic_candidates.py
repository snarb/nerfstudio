"""Seal the audited prior and probe frozen subdivided proposals, without admission.

The seal approves reproducibility/geometry guards only, not anatomical quality.
The old frozen margin-two CLI is NOT bypassed or falsely labelled compatible.
This explicitly binds the new prior to the same actual production base.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
import fit_mhr_guarded_correction as fit
import run_mhr_production_patch_control as control
from study_multiview_face_prior import read,save,sha

PRIOR=Path('/mnt/data/dec5_mhr_certified_conic_correction')
ROOT=Path('/mnt/data/dec5_mhr_conic_candidates')
REVIEW=Path('/mnt/data/dec5_mhr_certified_conic_review')
ARM='certified_conic'


def seal():
    dest=PRIOR/'final_seal.json';assert not dest.exists()
    q=read(PRIOR/'protocol.json');r=read(PRIOR/'result.json');a=read(PRIOR/'guard_audit.json')
    assert r['protocol_sha256']==sha(PRIOR/'protocol.json')
    assert not q['target_used'] and not q['production_accepted']
    assert len(a['records'])==len(r['history'])>0 and not a['new_all_intersections_observed']
    checked={**a['checked_bindings'],**q['input_hashes'],**r['solver_outputs']}
    checked[str(Path(fit.__file__))]=q['script_sha256']
    contact=q['contact_correction']
    if 'execution_wrapper_sha256' in contact:
        checked[str(Path(__file__).with_name('run_mhr_certified_conic_correction.py'))]=contact['execution_wrapper_sha256']
    if 'anatomical_domain' in contact:checked.update(contact['anatomical_domain']['input_hashes'])
    rr=read(REVIEW/'result.json');checked.update(rr['input_hashes'])
    for p,h in rr['hashes'].items():checked[str(REVIEW/p)]=h
    for p,h in checked.items():assert sha(p)==h,p
    reviewed=['residual_clay.png','C004_E005_1210X7.png','G004_B005_1210FG.png','M004_B005_12109O.png']
    for p in REVIEW.iterdir():
        if p.is_file():checked[str(p)]=sha(p)
    save(dest,dict(status='passed',scope='checked_fit_and_discrete_geometry_guards_only',
        production_accepted=False,anatomical_full_prior_accepted=False,not_an_admission_seal=True,
        visual_review=dict(status='reviewed_not_promoted',reviewer='parent_LLM',
            notes='Viewed all four comparisons: local neck change, no obvious new large fold; inherited facial folds remain. No RGB patch rendered.',
            viewed={str(REVIEW/p):sha(REVIEW/p) for p in reviewed}),
        checked_bindings=checked,script_sha256=sha(__file__),
        inventory={str(p.relative_to(PRIOR)):sha(p) for p in PRIOR.rglob('*') if p.is_file()}))


def configure():
    control.ROOT=ROOT;control.CANDIDATES=ROOT/'candidates';control.OUT=ROOT/'admission';control.ARM=ARM
    control.builder.PRIOR=PRIOR;control.builder.ARM=ARM
    original=control.binding
    def binding():
        return dict(original(),new_prior_recipe_explicit=True,prior_compatibility_with_margin2_cli=False,
            proposal_only_not_admission=True,conic_adapter_sha256=sha(__file__),
            prior_geometry_seal_sha256=sha(PRIOR/'final_seal.json'))
    control.binding=binding


def build():
    assert not ROOT.exists();ROOT.mkdir()
    save(ROOT/'request.json',control.binding())
    control.build(ROOT/'candidates')


def probe():
    from study_jaw_repair_transfer import mask_votes
    from bake_joint_temporal_mesh import camera_depth
    from admit_mhr_local_patch_depth import Scene2
    control.configure();cq,rows,depths,masks,names,binding=control.admission.inputs();del depths
    folder=ROOT/'candidates'/ARM
    cr=read(folder/'result.json')
    for p,h in cr['hashes'].items():assert sha(folder/p)==h,p
    raw=o3d.io.read_triangle_mesh(str(folder/'local_raw.ply'));v,t=np.asarray(raw.vertices),np.asarray(raw.triangles)
    proposals=np.load(folder/'proposal_evidence.npz')['proposals'];old=t[:cr['original_triangles']]
    support,outside=mask_votes(v,proposals,rows,masks,names);keep=(support>=2)&(outside==0)
    base=Path('/mnt/data/dec5_mhr_production_patch_001193')
    evidence=base/'residual_hole/evidence.npz';camera_path=base/'admission/rgb/F004_E/baseline/frames/001193/result.json'
    xy=np.load(evidence)['portrait_xy'];camera=read(camera_path)['camera']
    records={};arrays=dict(portrait_xy=xy,mask_support=support,mask_outside=outside,semantic_keep=keep)
    for name,tri in [('production_base',old),('raw',t),('semantic_only',np.concatenate([old,proposals[keep]]))]:
        d,ids,_=camera_depth(Scene2(v,tri),camera);d=np.rot90(d)[xy[:,1],xy[:,0]]
        records[name]=dict(residual_hits=int(np.isfinite(d).sum()),residual_misses=int((~np.isfinite(d)).sum()))
        arrays[name+'_depth']=d
    dest=ROOT/'posthoc_probe';dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'evidence.npz',**arrays)
    save(dest/'result.json',dict(records=records,proposals=len(proposals),semantic_pass=int(keep.sum()),
        no_depth_admission_performed=True,semantic_only_not_publishable=True,production_accepted=False,
        inputs=binding,input_hashes={str(p):sha(p) for p in [ROOT/'request.json',folder/'result.json',
            folder/'local_raw.ply',evidence,camera_path,Path(__file__)]},evidence_sha256=sha(dest/'evidence.npz')))
    print(records,'semantic',int(keep.sum()),'of',len(proposals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['seal','build','probe'])
    p.add_argument('--prior',type=Path,default=PRIOR);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--review',type=Path,default=REVIEW);args=p.parse_args()
    PRIOR=args.prior.resolve();ROOT=args.output.resolve();REVIEW=args.review.resolve()
    assert len(ROOT.parts)>=4 and ROOT!=PRIOR and ROOT!=REVIEW
    assert ROOT not in PRIOR.parents and PRIOR not in ROOT.parents
    if args.stage=='seal':seal()
    else:
        configure()
        if args.stage=='build':build()
        else:probe()
