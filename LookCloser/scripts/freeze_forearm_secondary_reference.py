"""Freeze the reviewed, unpromoted secondary-reference canary and probe inputs."""
from pathlib import Path
import argparse,json,shutil,subprocess,time
from joint_temporal_texture import read,sha,atomic_json
from probe_forearm_reference_coverage import BASE,OUT as PROBE
from study_forearm_secondary_reference import OUT
import study_forearm_plane_transfer_v3 as prior

DATA=Path('/mnt/data');HERE=Path(__file__).resolve().parent
REVIEW=DATA/'dec5_forearm_secondary_reference_review'
MANIFEST=OUT/'artifact_manifest.json'


def verify(hashes):
    for filename,expected in hashes.items():
        if sha(filename)!=expected:raise ValueError('Changed file: '+filename)


def run(check=False):
    if check:
        result=read(MANIFEST);verify(result['retained_hashes']);verify(result['external_hashes'])
        print('Verified',len(result['retained_hashes']),'retained and',len(result['external_hashes']),'external files',flush=True);return
    if MANIFEST.exists():raise ValueError('Already frozen; use --check')
    processes=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    workers=[s for s in processes.splitlines() if any(n in s for n in [
        'scripts/study_forearm_secondary_reference.py --','scripts/study_confidence_boundary_completion.py render',
        'scripts/probe_forearm_reference_coverage.py --','scripts/audit_forearm_secondary_reference.py --'])]
    if workers:raise ValueError('Relevant workers still active')
    testlog=DATA/'dec5_forearm_secondary_tests_full.log'
    if '20 passed' not in testlog.read_text():raise ValueError('Focused tests did not pass')
    ancestor=DATA/'dec5_forearm_admission_order_manifest.json';old=read(ancestor)
    verify(old['retained_hashes']);external={str(ancestor):sha(ancestor)}
    movie=DATA/'dec5_phase30_early_texture_dynamic_150'
    published={str(movie/'video.mp4'):'ce1ff5f7fd612a46e39b6ae89d10aee7534f895ea7e7887d722dc10903bccd40',
        str(movie/'frames.zip'):'5e2d078a2e249c90a9a4b3d4112a7652bda533d45422c91832c7355480c2f7b6'}
    verify(published);external.update(published)
    snapshots=OUT/'config';snapshots.mkdir(exist_ok=True);scripts=set()
    prior.configure();v1=prior.v2.v1
    probe_reviews={}
    for frame in ['001029','001033','001037']:
        folder=PROBE/frame;request=read(folder/'request.json');result=read(folder/'result.json')
        if result['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed probe request')
        verify(request['depth_sha256']);external.update(request['depth_sha256'])
        for n,h in request['scripts'].items():
            if sha(HERE/n)!=h:raise ValueError('Changed probe producer')
        scripts.update(request['scripts'])
        for record in result['records']:
            for suffix,key in [('.png','panel_sha256'),('.npz','arrays_sha256')]:
                if sha(folder/(record['camera']+suffix))!=record[key]:raise ValueError('Changed probe output')
        panels={str(folder/(n+'.png')):sha(folder/(n+'.png')) for n in v1.NAMES}
        probe_reviews[frame]=dict(status='reviewed_diagnostic_not_geometry',panel_sha256=panels,
            notes='Side/cuff margins and isolated skin points; no evidence of a complete wrist/hand reconstruction.')
        source=prior.OUT/frame;spec=read(source/'input.json')
        inputs=[source/'input.json',source/'analysis.json',source/'diagnostic.npz',Path(spec['metadata']),
            prior.OUT/'controls'/frame/'staged63/transforms.json',BASE/frame/'guarded.ply',
            BASE/frame/'geometry_result.json',Path('/mnt/data/dec5_forearm_multiview_anchors')/frame/'result.json']
        inputs.extend(source/'rgb'/(n+'.png') for n in v1.NAMES)
        external.update({str(p):sha(p) for p in inputs})
        rows,_,_=v1.cameras(frame)
        atomic_json(snapshots/(frame+'_camera_and_mask_values.json'),dict(train_cameras=rows,
            skin_polygons=v1.POLYGONS[frame],captured_at_freeze=True))
    atomic_json(PROBE/'visual_review.json',dict(frames=probe_reviews,all_nine_native_panels_actually_inspected=True))
    request=read(OUT/'001037/request.json');result=read(OUT/'001037/geometry_result.json')
    audit=read(OUT/'001037/independent_audit.json')
    if not result['observed_guard_passed'] or audit['checks']!=124 or audit['qualified_veto_pixels']!=0:
        raise ValueError('Incomplete geometry audit')
    if audit['geometry_result_sha256']!=sha(OUT/'001037/geometry_result.json'):raise ValueError('Changed audited geometry')
    for n,h in request['scripts'].items():
        if sha(HERE/n)!=h:raise ValueError('Changed assembly producer')
    scripts.update(request['scripts'])
    metrics=read(REVIEW/'metrics.json');verify(metrics['prediction_hashes']);external.update(metrics['prediction_hashes'])
    if metrics['scope']!='fixed manual train forearm skin; not heldout face or full-frame':raise ValueError('Wrong metric scope')
    panels={str(REVIEW/'001037'/(n+'_comparison.png')):sha(REVIEW/'001037'/(n+'_comparison.png')) for n in ['moving',v1.NAMES[1]]}
    atomic_json(REVIEW/'visual_review.json',dict(frame='001037',status='fail',production_accepted=False,
        comparison='partial_local_gain_not_complete_repair',actually_inspected_native_panels=panels,
        notes='Side skin coverage improves slightly; broad wrist/forearm voids, fragmented hand and patch color seams remain in both views. No production promotion.',
        fixed_train_metrics_not_heldout=True,transfer_meshes_not_run=['001029','001033']))
    scripts.update(['audit_forearm_secondary_reference.py',Path(__file__).name,'study_forearm_plane_transfer.py',
        'study_forearm_plane_transfer_v2.py','study_forearm_plane_transfer_v3.py','study_forearm_confidence_prior.py',
        'study_confidence_depth_prior.py','joint_temporal_texture.py','study_confidence_boundary_completion.py',
        'review_confidence_boundary_completion.py','render_smooth_temporal_mesh_video.py','study_early_texture_prior.py'])
    for n in scripts:
        dest=snapshots/n
        if dest.exists() and sha(dest)!=sha(HERE/n):raise ValueError('Changed source snapshot')
        if not dest.exists():shutil.copyfile(HERE/n,dest)
    report=HERE.parent/'experiments/dec5_forearm_secondary_reference.md'
    logfiles=[p for p in DATA.glob('dec5_forearm_secondary*.log') if 'freeze' not in p.name]
    external.update({str(p):sha(p) for p in logfiles+[report,testlog]})
    check_record=dict(unix_time=time.time(),scope='final_state_not_reconstructed_historical_checks',workers=[],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip(),
        free_bytes=shutil.disk_usage(DATA).free)
    atomic_json(OUT/'final_process_check.json',check_record)
    hashes={str(p):sha(p) for root in [OUT,PROBE,REVIEW] for p in sorted(root.rglob('*')) if p.is_file()}
    atomic_json(MANIFEST,dict(status='completed_canary_not_production_accepted',canary_frame='001037',
        diagnostic_frames=['001029','001033','001037'],visual_status='fail',tests_passed=20,
        heldout_face_metrics_computed=False,main_campaign_csv_changed=False,full_video_rerendered=False,
        retained_hashes=hashes,external_hashes=external))
    print('Frozen',len(hashes),'files; canary unpromoted; movie unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.check)
