"""Freeze native witness diagnosis and the reviewed, rejected alignment canary."""
import argparse
from pathlib import Path
import time
import subprocess
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from diagnose_body_rgb_veto import ROOT as DIAG
from align_poisson_to_measured_depth import ROOT
from review_jaw_repair_transfer import verified_image

MANIFEST=ROOT/'artifact_manifest.json'
FRAME='001037'


def run(check=False):
    if check:
        files=read(MANIFEST)['files']
        for p,h in files.items():
            if sha(p)!=h:raise ValueError('Changed artifact: '+p)
        print('Verified',len(files),'hashes');return
    if MANIFEST.exists():raise ValueError('Already frozen')
    processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    for name in ['align_poisson_to_measured_depth.py','audit_observed_poisson_alignment.py','diagnose_body_rgb_veto.py']:
        if any('scripts/'+name+' ' in line for line in processes.splitlines()):raise ValueError('Worker still active')
    diag=DIAG/FRAME;dr=read(diag/'result.json');out=ROOT/FRAME
    assert sha(diag/'events.npz')==dr['events_sha256']
    assert not any(dr['profiles_control_image_pixel_differences'].values()) and dr['profile_control_decision_differences']==0
    cases={c['panel']:c['panel_sha256'] for c in dr['cases']};assert len(cases)==6
    for p,h in cases.items():assert sha(p)==h
    atomic_json(diag/'visual_review.json',dict(reviewer='main_agent',status='reviewed_diagnostic_not_geometry_approval',
        inspected=cases,notes='Retained examples include real skin-depth contradictions; other cases are cloth/skin or ambiguous. Gain control images identical.'))
    req=read(out/'request.json');result=read(out/'result.json');audit=read(out/'independent_audit.json')
    assert result['request_sha256']==sha(out/'request.json') and audit['mesh_sha256']==sha(out/'mesh.ply')
    assert audit['original_prefix_exact'] and audit['depth_equations_and_solves_recomputed'] and audit['direct_admission_recomputed']
    assert len(audit['native_checks'])==124 and all(r['trusted_free_pixels']==0 for r in audit['native_checks'])
    for n,h in result['hashes'].items():assert sha(out/n)==h
    review=read(out/'review/result.json');panels=[out/(v+'_added.png') for v in ['moving','train_H_A']]
    external={}
    for record in review['records']:
        view=record['view']
        for variant in ['baseline','repaired']:
            location=out/'rgb'/view/variant;verified_image(location,FRAME)
            q=read(location/'request.json');assert q['same_footprint_for_both_geometry_variants']
            for n,h in q['script_hashes'].items():
                p=Path(__file__).with_name(n);assert sha(p)==h;external[str(p)]=h
        assert record['changed_upper_1400']==0
        for metric in record['metrics']:assert all(np.isfinite(metric[k]) for k in ['psnr','ssim','lpips'])
        panels += [out/'review'/(view+'_'+kind+'.png') for kind in ['hand','torso']]
    atomic_json(out/'visual_review.json',dict(reviewer='main_agent',status='fail',
        inspected={str(p):sha(p) for p in panels},video_changed=False,
        notes='Only small forearm coverage gain. Broken hand and wrist persist; new clothing patches remain fragmented. Not video-ready.'))
    review['visual_status']='reviewed_fail_hand_video_gate';atomic_json(out/'review/result.json',review)
    for record in [dr,req['rgb_receipt']]:external.update(record['source_rgb_hashes'])
    external.update(req['source_depth_sha256']);external.update(dr['source_depth_sha256'])
    for p,h in external.items():assert sha(p)==h
    scripts=['calibrated_depth_witness.py','diagnose_body_rgb_veto.py','regularized_depth_displacement.py',
             'align_poisson_to_measured_depth.py','audit_observed_poisson_alignment.py','freeze_body_rgb_alignment.py']
    paths=[Path(__file__).with_name(n) for n in scripts]
    repo=Path(__file__).parents[1]
    paths += [repo/'tests'/n for n in ['test_calibrated_depth_witness.py','test_regularized_depth_displacement.py','test_observed_poisson_depth_equations.py']]
    paths += [repo/'experiments/dec5_body_rgb_veto_and_alignment.md',Path(req['source_mesh'])]
    from study_body_neighborhood_completion import ROOT as BODY
    paths += [p for p in (BODY/FRAME).glob('*') if p.is_file()]
    paths += [Path('/mnt/data')/n for n in ['dec5_body_rgb_veto_001037.log','dec5_body_rgb_veto_001037_v2.log',
        'dec5_observed_poisson_alignment_001037.log','dec5_observed_poisson_alignment_001037_rgb.log',
        'dec5_observed_poisson_alignment_001037_audit.log','dec5_observed_poisson_alignment_001037_review.log',
        'dec5_observed_poisson_alignment_tests.log']]
    atomic_json(ROOT/'supervision_final.json',dict(unix_time=time.time(),workers_alive=False,video_rerendered=False,
        diagnostic_failure_retained=str(DIAG/'001037_diagnostic_error'),tests_passed=17))
    files={str(p):sha(p) for root in [ROOT,DIAG] for p in root.rglob('*') if p.is_file() and p!=MANIFEST}
    files.update(external);files.update({str(p):sha(p) for p in paths})
    atomic_json(MANIFEST,dict(files=files,status='reviewed_diagnosis_alignment_fails_video_gate',
        frame=FRAME,goal_complete=False,video_changed=False,native_checks=124,tests_passed=17))
    print('Frozen',len(files),'hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
