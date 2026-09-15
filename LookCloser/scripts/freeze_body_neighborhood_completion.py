"""Freeze reviewed body controls and their negative hand result, not a movie."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from study_body_neighborhood_completion import ROOT,SOURCE,PARENT
from study_body_single_depth_seed import ROOT as WEAK
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
    workers=['scripts/study_body_neighborhood_completion.py ', 'scripts/study_body_single_depth_seed.py ',
             'scripts/audit_body_neighborhood_completion.py ', 'scripts/audit_body_single_depth_seed.py ',
             'scripts/review_body_neighborhood_completion.py ']
    if any(w in line for w in workers for line in processes.splitlines()):raise ValueError('Workers remain active')
    external=set();baselines={};all_metrics=[]
    for root in [ROOT,WEAK]:
        folder=root/FRAME;req=read(folder/'request.json');result=read(folder/'result.json');audit=read(folder/'independent_audit.json')
        assert result['request_sha256']==sha(folder/'request.json') and result['observed_guard_passed']
        assert audit['mesh_sha256']==sha(folder/'mesh.ply') and audit['original_prefix_exact']
        assert len(audit['native_checks'])==124 and all(r['trusted_free_pixels']==0 for r in audit['native_checks'])
        for n,h in result['hashes'].items():assert sha(folder/n)==h
        for p,h in req['source_depth_sha256'].items():assert sha(p)==h;external.add(Path(p))
        for n,h in req['scripts'].items():
            p=Path(__file__).with_name(n);assert sha(p)==h;external.add(p)
        external.add(Path(req['source_mesh']))
        for p in [Path(req['source_masks'])/'masks.npz',Path(req['source_masks'])/'cameras.json']:
            external.add(p)
        review=read(folder/'review/result.json');panels=[]
        for record in review['records']:
            view=record['view']
            for variant in ['baseline','repaired']:
                location=folder/'rgb'/view/variant;im,r=verified_image(location,FRAME)
                request=read(location/'request.json');assert request['same_footprint_for_both_geometry_variants']
                for n,h in request['script_hashes'].items():
                    p=Path(__file__).with_name(n);assert sha(p)==h;external.add(p)
                if variant=='baseline':
                    if view in baselines:np.testing.assert_array_equal(im,baselines[view])
                    else:baselines[view]=im
            for m in record['metrics']:
                assert all(np.isfinite(m[k]) for k in ['psnr','ssim','lpips'])
            all_metrics+=record['metrics']
            panels += [folder/'review'/(view+'_'+kind+'.png') for kind in ['hand','torso']]
        if root==ROOT:
            panels += [folder/(view+'_added.png') for view in ['moving','train_H_A']]
            panels += [folder/'admission_diagnosis'/(name+'.png') for name in ['train_forearm_stages','train_hand_stages']]
        atomic_json(folder/'visual_review.json',dict(reviewer='main_agent',status='fail',
            inspected={str(p):sha(p) for p in panels},
            notes='Clothing/cuff gain; conspicuous broken hand, wrist/forearm holes and source seams remain. Not a video-ready repair.',
            video_changed=False,coverage_gain_is_not_anatomical_validation=True))
        review['visual_status']='reviewed_fail_hand_video_gate';atomic_json(folder/'review/result.json',review)
    assert len(all_metrics)==4
    atomic_json(ROOT/'supervision_final.json',dict(unix_time=time.time(),relevant_workers_alive=False,
        reconstruction_and_render_terminal=True,checker_initial_error_retained=True,
        video_request_sha256=sha(PARENT/'request.json'),full_video_rerendered=False))
    names=['body_surface_neighborhood.py','study_body_neighborhood_completion.py','audit_body_neighborhood_completion.py',
           'study_body_single_depth_seed.py','audit_body_single_depth_seed.py','review_body_neighborhood_completion.py',
           'diagnose_body_prior_admission.py','explain_body_seed_gate.py','freeze_body_neighborhood_completion.py']
    external.update(Path(__file__).with_name(n) for n in names)
    repo=Path(__file__).parents[1];external.update([repo/'experiments/dec5_body_neighborhood_completion.md',
        repo/'tests/test_body_surface_neighborhood.py',PARENT/'request.json'])
    external.update(p for p in (SOURCE/FRAME).glob('*.json'))
    logs=['dec5_body_neighborhood_001037.log','dec5_body_neighborhood_001037_rgb.log',
          'dec5_body_neighborhood_001037_audit.log','dec5_body_neighborhood_001037_audit_v2.log',
          'dec5_body_neighborhood_001037_review.log','dec5_body_neighborhood_001037_diagnosis.log',
          'dec5_body_single_depth_001037.log','dec5_body_single_depth_001037_rgb.log',
          'dec5_body_single_depth_001037_audit.log','dec5_body_single_depth_001037_review.log',
          'dec5_body_neighborhood_tests.log']
    external.update(Path('/mnt/data')/n for n in logs)
    files={str(p):sha(p) for root in [ROOT,WEAK] for p in root.rglob('*') if p.is_file() and p!=MANIFEST}
    files.update({str(p):sha(p) for p in external})
    atomic_json(MANIFEST,dict(files=files,frames=[FRAME],variants=2,native_checks=248,focused_tests_passed=11,
        status='reviewed_clothing_gain_hand_failure',goal_complete=False,video_changed=False))
    print('Frozen',len(files),'hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
