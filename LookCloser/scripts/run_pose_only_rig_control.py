#!/usr/bin/env python3
"""Fit a shared pose-only rig and gate it on fixed independent train times.

This opt-in diagnostic never launches dense reconstruction or reads held-camera
RGB. It preserves all original intrinsics, completes the query similarity gauge,
and emits an eligibility receipt, not an accepted surface-rendering recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from complete_rig_similarity_gauge import complete_query_gauge

SCRIPTS = Path(__file__).resolve().parent
INTRINSICS = ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'p1', 'p2', 'camera_model')
THRESHOLDS = dict(minimum_median_improvement_pixels=.02, minimum_improved_pair_fraction=.60,
                  maximum_rotation_degrees=.6, maximum_center_shift_world=.25,
                  require_each_held_time_median_and_p90_improvement=True)


def evaluate_gate(scores, candidate_label, manifest, baseline_label):
    summaries={r['calibration']:r for r in scores['summaries']}
    baseline=summaries[baseline_label];candidate=summaries[candidate_label]
    rows=scores['results'];checks={}
    def keys(label):
        return [(r['frame_id'],r['left_camera'],r['right_camera'],r['points']) for r in rows[label]]
    if keys(baseline_label)!=keys(candidate_label) or len(set(keys(baseline_label)))!=len(keys(baseline_label)):
        raise ValueError('Candidate must retain the identical unique held pair inventory')
    checks['overall_median']=baseline['pair_block_median']-candidate['pair_block_median']>=THRESHOLDS['minimum_median_improvement_pixels']
    checks['overall_p90']=candidate['pair_block_p90']<baseline['pair_block_p90']
    checks['pair_fraction']=candidate['fraction_pairs_improved']>=THRESHOLDS['minimum_improved_pair_fraction']
    checks['converged']='Termination: CONVERGENCE' in manifest['solver_report']
    checks['rotation_bound']=max(r['rotation_degrees'] for r in manifest['camera_changes'])<=THRESHOLDS['maximum_rotation_degrees']
    checks['translation_bound']=max(r['center_shift_world'] for r in manifest['camera_changes'])<=THRESHOLDS['maximum_center_shift_world']
    per_time=[]
    for time in scores['held_frames']:
        old=np.array([r['block_median_absolute_error'] for r in rows[baseline_label] if r['frame_id']==time])
        new=np.array([r['block_median_absolute_error'] for r in rows[candidate_label] if r['frame_id']==time])
        if not len(old) or old.shape!=new.shape or not np.isfinite(np.r_[old,new]).all():
            raise ValueError('Invalid held-time errors')
        checks['held_'+time]=bool(np.median(new)<np.median(old) and np.quantile(new,.9)<np.quantile(old,.9))
        per_time.append(dict(frame_id=time,pairs=len(old),median_before_after=[float(np.median(old)),float(np.median(new))],
                             p90_before_after=[float(np.quantile(old,.9)),float(np.quantile(new,.9))]))
    return dict(eligible_for_dense_control=all(checks.values()),checks=checks,per_time=per_time,
                baseline={k:v for k,v in baseline.items() if k!='camera_summary'},
                candidate={k:v for k,v in candidate.items() if k!='camera_summary'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['fit-tracks','held-tracks','original-template','reference-focal','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve existing pose-only control; no marker-only resume')
    fit=json.loads(args.fit_tracks.read_text());held=json.loads(args.held_tracks.read_text())
    if set(fit['frames'])&set(held['frames']):raise ValueError('Fit and held temporal frames overlap')
    if fit['calibration_sha256']!=sha256(args.original_template):raise ValueError('Changed original template')
    names=fit['physical_cameras'];forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if len(names)!=len(set(names)) or len(names)!=62 or set(names)&forbidden or held['physical_cameras']!=names:
        raise ValueError('Need the same 62 unique train-only camera identities')
    scripts=[SCRIPTS/n for n in ['run_pose_only_rig_control.py','refine_multitime_camera_rig.py','score_multitime_rig_holdout.py',
        'complete_rig_similarity_gauge.py','audit_source_epipolar_residuals.py','audit_spatial_temporal_residuals.py']]
    files=[args.fit_tracks,args.held_tracks,args.original_template,args.reference_focal,*scripts]
    args.output.mkdir(parents=True)
    atomic_json(args.output/'request.json',dict(fit_frames=fit['frames'],held_frames=held['frames'],mode='poses',regularized=True,
        train_camera_count=62,thresholds=THRESHOLDS,uses_eval_rgb=False,uses_semantic_masks=False,changes_intrinsics=False,
        query_gauge='Apply the recorded common post-BA similarity to previously untouched query camera poses',
        input_hashes={str(p.resolve()):sha256(p) for p in files},python=sys.executable))
    def run(stage,command):
        atomic_json(args.output/'progress.json',dict(stage=stage,status='running',timestamp=datetime.now(timezone.utc).isoformat()))
        with (args.output/(stage+'.log')).open('w') as stream:
            subprocess.run([sys.executable,*map(str,command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(stage+' complete',flush=True)
    rig=args.output/'rig_regularized'
    run('fit',[SCRIPTS/'refine_multitime_camera_rig.py','--tracks',args.fit_tracks,'--output',rig,'--mode','poses','--regularized'])
    original=json.loads(args.original_template.read_text());refined=json.loads((rig/'transforms.json').read_text())
    manifest=json.loads((rig/'manifest.json').read_text());before={f['physical_camera']:f for f in original['frames']}
    for f in refined['frames']:
        for key in INTRINSICS:
            if f.get(key,original.get(key))!=before[f['physical_camera']].get(key,original.get(key)):
                raise ValueError('Pose-only control changed an intrinsic: '+key)
    completed=complete_query_gauge(original,refined,manifest)
    calibration=args.output/'calibration.json';atomic_json(calibration,completed)
    scores_path=args.output/'held_scores.json'
    run('held_score',[SCRIPTS/'score_multitime_rig_holdout.py','--held-tracks',args.held_tracks,'--calibrations',
        args.original_template,calibration,args.reference_focal,'--fit-frames',*fit['frames'],'--output',scores_path])
    scores=json.loads(scores_path.read_text())
    gate=evaluate_gate(scores,str(calibration),manifest,str(args.original_template))
    atomic_json(args.output/'selection_before_render.json',dict(gate,thresholds=THRESHOLDS,
        timestamp=datetime.now(timezone.utc).isoformat(),selection_uses_eval_rgb=False,selection_uses_render_metrics=False,
        request_sha256=sha256(args.output/'request.json'),calibration_sha256=sha256(calibration),held_scores_sha256=sha256(scores_path),
        intrinsics_identical_to_original=True,shared_across_all_time_frames=True,query_gauge_completed=True,
        accepted_surface_recipe=False))
    atomic_json(args.output/'progress.json',dict(stage='held_gate',status='complete',eligible=gate['eligible_for_dense_control']))
    print(json.dumps(gate,indent=2),flush=True)


if __name__=='__main__':main()
