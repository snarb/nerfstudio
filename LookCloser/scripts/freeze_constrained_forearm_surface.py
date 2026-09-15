"""Freeze the rejected three-time constrained-surface transfer and its evidence."""
from pathlib import Path
import argparse,shutil,subprocess,time
from joint_temporal_texture import read,sha,atomic_json
from freeze_forearm_secondary_reference import verify

DATA=Path('/mnt/data');HERE=Path(__file__).resolve().parent
OLD=DATA/'dec5_constrained_forearm_surface';ROOT=DATA/'dec5_constrained_forearm_surface_guard16'
REVIEW=DATA/'dec5_constrained_forearm_guard16_review';PILOT=DATA/'dec5_constrained_forearm_surface_review'
EXTRA=DATA/'dec5_constrained_forearm_additional_heldout';MANIFEST=ROOT/'artifact_manifest.json'
FRAMES=['001029','001033','001037']


def run(check=False):
    if check:
        m=read(MANIFEST);verify(m['retained_hashes']);verify(m['external_hashes'])
        print('Verified',len(m['retained_hashes']),'retained and',len(m['external_hashes']),'external files',flush=True);return
    if MANIFEST.exists():raise ValueError('Already frozen; use --check')
    process=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    patterns=['scripts/study_coherent_forearm_replacement.py --','scripts/study_confidence_boundary_completion.py render',
        'scripts/audit_coherent_forearm_replacement.py --','scripts/diagnose_constrained_surface_regression.py --',
        'scripts/review_forearm_additional_heldout.py render','scripts/review_constrained_forearm_surface.py score']
    if any(p in line for line in process.splitlines() for p in patterns):raise ValueError('Relevant workers still active')
    ancestor=DATA/'dec5_coherent_forearm_replacement/artifact_manifest.json';old=read(ancestor)
    verify(old['retained_hashes']);verify(old['external_hashes']);external=dict(old['external_hashes']);external[str(ancestor)]=sha(ancestor)
    snapshots=ROOT/'config';snapshots.mkdir(exist_ok=True);reviews={};producers=[]
    for frame in FRAMES:
        folder=ROOT/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
        audit=read(folder/'independent_audit.json');loss=read(folder/'production_loss_diagnosis.json')
        if request['guard_max_rounds']!=16 or 'feasible_depth_constraints' not in request:raise ValueError('Mixed transfer recipes')
        if result['request_sha256']!=sha(folder/'request.json') or not result['observed_guard_passed'] or result['rounds'][-1]['removed_triangles']!=0:raise ValueError('Unverified final geometry')
        if audit['geometry_result_sha256']!=sha(folder/'geometry_result.json') or not audit['feasible_depth_constraints_replayed'] or audit['checks']!=124 or audit['qualified_veto_pixels']!=0:raise ValueError('Missing independent audit')
        if loss['geometry_result_sha256']!=sha(folder/'geometry_result.json') or not loss['production_prefix_verified']:raise ValueError('Changed production-loss diagnosis')
        verify(request['source_depth_sha256']);external.update(request['source_depth_sha256'])
        external.update(result['color_guard_provenance']['source_rgb_hashes'])
        verify({str(folder/n):h for n,h in result['hashes'].items()})
        if sha(folder/'transferred.ply')!=sha(OLD/frame/'transferred.ply'):raise ValueError('Guard budget changed proposed geometry')
        if frame!='001029' and sha(folder/'guarded.ply')!=sha(OLD/frame/'guarded.ply'):raise ValueError('Changed already-converged geometry')
        panels={str(REVIEW/frame/(n+'_comparison.png')):sha(REVIEW/frame/(n+'_comparison.png')) for n in ['moving','H004_A005_1210M6']}
        notes={'001029':'Strong regression: previously good skin develops dense black speckling in both views.',
            '001033':'Some coverage gained, but vertical speckling and texture fidelity regress; wrist/hand remain broken.',
            '001037':'Partial coverage gain; wrist/hand holes and seams remain. Not artifact-free.'}[frame]
        reviews[frame]=dict(status='fail',notes=notes,actually_inspected_panels=panels)
        for parent in [OLD,ROOT]:
            r=read(parent/frame/'request.json')
            for name,h in r['scripts'].items():
                candidates=[HERE/name,parent/'config'/name,OLD/'config'/name]
                source=next((p for p in candidates if p.is_file() and sha(p)==h),None)
                if source is None:raise ValueError('Missing exact producer: '+name)
                target=snapshots/(h+'_'+name)
                if target.exists() and sha(target)!=h:raise ValueError('Changed source archive')
                if not target.exists():shutil.copyfile(source,target)
            producers.append(dict(request=str(parent/frame/'request.json'),sha256=sha(parent/frame/'request.json')))
    if read(OLD/'001029/geometry_result.json')['observed_guard_passed']:raise ValueError('Original budget failure was overwritten')
    reuse=read(ROOT/'001037/rgb_reuse.json');verify(reuse['reused_hashes']);external.update(reuse['reused_hashes'])
    if reuse['new_result_sha256']!=sha(ROOT/'001037/geometry_result.json') or not reuse['mesh_bytes_exact']:raise ValueError('Invalid reuse receipt')
    metrics=read(REVIEW/'metrics.json');verify(metrics['prediction_hashes']);external.update(metrics['prediction_hashes'])
    if len(metrics['rows'])!=6 or metrics['scope']!='fixed manual train forearm skin; not heldout face or full-frame':raise ValueError('Unexpected metric scope or inventory')
    extent=read(REVIEW/'rgb_change_extent.json')
    if len(extent['records'])!=6 or any(r['changed_rgb_pixels_top_1400_rows'] for r in extent['records']):raise ValueError('Unexpected upper-image RGB change')
    atomic_json(REVIEW/'visual_review.json',dict(frames=reviews,status='rejected_for_temporal_transfer',production_accepted=False,
        full_video_rerendered=False,head_region_rgb_unchanged=True,main_face_csv_changed=False))
    atomic_json(PILOT/'visual_review.json',dict(frame='001037',status='fail',partial_local_gain=True,
        notes='Pilot allowed the transfer test only, not production acceptance; wrist/hand defects remain.',
        panels={str(PILOT/'001037'/(n+'_comparison.png')):sha(PILOT/'001037'/(n+'_comparison.png')) for n in ['moving','H004_A005_1210M6']}))
    gt=read(EXTRA/'001037/gt_receipt.json');gt_hashes={r['gt']:r['gt_sha256'] for r in gt['records']};verify(gt_hashes)
    external.update({r['source']:r['source_sha256'] for r in gt['records']})
    atomic_json(EXTRA/'001037/visual_review.json',dict(status='reviewed_not_suitable_for_full_forearm_validation',
        actually_inspected_gt=gt_hashes,notes='J/D clips the forearm at the image edge; L/B has no useful full forearm view.',
        predictions_rendered=0,metrics_computed=False,geometry_input=False))
    tests=DATA/'dec5_constrained_forearm_tests_final.log'
    if '20 passed' not in tests.read_text():raise ValueError('Focused tests incomplete')
    names=['audit_coherent_forearm_replacement.py','diagnose_constrained_surface_regression.py','review_constrained_forearm_surface.py',
        'review_forearm_additional_heldout.py','review_confidence_boundary_completion.py','study_confidence_boundary_completion.py',
        'study_early_texture_prior.py','render_smooth_temporal_mesh_video.py','wide_dynamic_camera_flight.py',Path(__file__).name]
    for name in names:
        target=snapshots/name
        if target.exists() and sha(target)!=sha(HERE/name):raise ValueError('Changed current snapshot')
        if not target.exists():shutil.copyfile(HERE/name,target)
    report=HERE.parent/'experiments/dec5_constrained_forearm_surface.md'
    external.update({str(p):sha(p) for p in DATA.glob('dec5_constrained_forearm*.log') if 'freeze' not in p.name})
    external.update({str(report):sha(report),str(tests):sha(tests)});verify(external)
    atomic_json(ROOT/'final_process_check.json',dict(unix_time=time.time(),workers=[],scope='final_observed_state',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip(),
        free_bytes=shutil.disk_usage(DATA).free))
    retained={str(p):sha(p) for root in [OLD,ROOT,REVIEW,PILOT,EXTRA] for p in sorted(root.rglob('*')) if p.is_file()}
    atomic_json(MANIFEST,dict(status='completed_rejected_transfer',frames=FRAMES,visual_pass=0,visual_fail=3,tests_passed=20,
        independent_native_ray_checks=372,production_changed=False,full_video_rerendered=False,producers=producers,
        retained_hashes=retained,external_hashes=external))
    print('Frozen',len(retained),'files; transfer rejected; published movie unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.check)
