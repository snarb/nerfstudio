"""Freeze a reviewed but unpromoted coherent-replacement canary."""
from pathlib import Path
import argparse,shutil,subprocess,time
from joint_temporal_texture import read,sha,atomic_json,ROOT as COLOR_ROOT
from freeze_forearm_secondary_reference import verify

DATA=Path('/mnt/data');HERE=Path(__file__).resolve().parent
OUT=DATA/'dec5_coherent_forearm_replacement';REVIEW=DATA/'dec5_coherent_forearm_replacement_review'
MANIFEST=OUT/'artifact_manifest.json'


def run(check=False):
    if check:
        m=read(MANIFEST);verify(m['retained_hashes']);verify(m['external_hashes'])
        print('Verified',len(m['retained_hashes']),'retained and',len(m['external_hashes']),'external files',flush=True);return
    if MANIFEST.exists():raise ValueError('Already frozen; use --check')
    process=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    patterns=['scripts/study_coherent_forearm_replacement.py --','scripts/study_confidence_boundary_completion.py render',
        'scripts/audit_coherent_forearm_replacement.py --','scripts/diagnose_coherent_forearm_residual.py --','scripts/probe_forearm_mask_depth_intervals.py --']
    if any(p in line for line in process.splitlines() for p in patterns):raise ValueError('Relevant worker still active')
    tests=DATA/'dec5_coherent_forearm_tests.log'
    if '16 passed' not in tests.read_text():raise ValueError('Focused tests missing')
    ancestor=DATA/'dec5_forearm_secondary_reference/artifact_manifest.json';old=read(ancestor)
    verify(old['retained_hashes']);verify(old['external_hashes']);external={str(ancestor):sha(ancestor)}
    # Bind old inputs themselves, not just the old manifest's current bytes.
    external.update(old['external_hashes'])
    frame='001037';folder=OUT/frame;request=read(folder/'request.json');result=read(folder/'geometry_result.json')
    audit=read(folder/'independent_audit.json');diagnostic=read(folder/'residual_attribution.json')
    if result['request_sha256']!=sha(folder/'request.json') or not result['observed_guard_passed']:raise ValueError('Incomplete canary')
    if audit['geometry_result_sha256']!=sha(folder/'geometry_result.json') or audit['checks']!=124 or audit['qualified_veto_pixels']!=0:raise ValueError('Changed audited geometry')
    if not diagnostic['saved_render_depth_verified'] or diagnostic['geometry_result_sha256']!=sha(folder/'geometry_result.json'):raise ValueError('Invalid residual diagnosis')
    verify({str(folder/n):h for n,h in result['hashes'].items()});verify(request['source_depth_sha256'])
    external.update(request['source_depth_sha256']);external[request['source_mesh']]=request['source_mesh_sha256']
    verify(result['color_guard_provenance']['source_rgb_hashes']);external.update(result['color_guard_provenance']['source_rgb_hashes'])
    for n,k in [('parameters.npz','profiles_sha256'),('exposure.json','exposure_sha256')]:
        p=COLOR_ROOT/n
        if sha(p)!=result['color_guard_provenance'][k]:raise ValueError('Changed color calibration')
        external[str(p)]=sha(p)
    metrics=read(REVIEW/'metrics.json');verify(metrics['prediction_hashes']);external.update(metrics['prediction_hashes'])
    if metrics['scope']!='fixed manual train forearm skin; not heldout face or full-frame':raise ValueError('Wrong metric scope')
    panels=[REVIEW/frame/(n+'_comparison.png') for n in ['moving','H004_A005_1210M6']]
    panels.append(folder/'residual_attribution.png')
    interval=folder/'mask_depth_intervals';probe=read(interval/'result.json')
    if probe['residual_attribution_sha256']!=sha(folder/'residual_attribution.json') or probe['arrays_sha256']!=sha(interval/'samples.npz'):raise ValueError('Changed interval probe')
    for r in probe['panels']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed projection panel')
        panels.append(Path(r['path']))
    atomic_json(REVIEW/'visual_review.json',dict(frame=frame,status='fail',production_accepted=False,
        actually_inspected_panels={str(p):sha(p) for p in panels},
        notes='Smoother forearm interior, but more black ROI pixels and persistent wrist/hand defects. Reject video promotion. Feasible-depth overlays trace side/cuff margins; they are not reconstructed anatomy.',
        next_test='bounded coherent solve with unchanged-mask feasible depth sets and measured pins',full_video_rerendered=False))
    names=set(request['scripts'])|{Path(__file__).name,'audit_coherent_forearm_replacement.py','diagnose_coherent_forearm_residual.py',
        'probe_forearm_mask_depth_intervals.py','study_confidence_boundary_completion.py','review_confidence_boundary_completion.py',
        'wide_dynamic_camera_flight.py','study_early_texture_prior.py','render_smooth_temporal_mesh_video.py',
        'joint_temporal_texture.py','study_confidence_depth_prior.py','freeze_forearm_secondary_reference.py'}
    snapshots=OUT/'config';snapshots.mkdir(exist_ok=True)
    for name in names:
        source=HERE/name
        if name in request['scripts'] and sha(source)!=request['scripts'][name]:raise ValueError('Changed producer source')
        dest=snapshots/name
        if dest.exists() and sha(dest)!=sha(source):raise ValueError('Changed existing snapshot')
        if not dest.exists():shutil.copyfile(source,dest)
    report=HERE.parent/'experiments/dec5_coherent_forearm_replacement.md'
    logs=[p for p in DATA.glob('dec5_coherent_forearm*.log') if 'freeze' not in p.name]
    external.update({str(p):sha(p) for p in logs+[tests,report]})
    atomic_json(OUT/'final_process_check.json',dict(unix_time=time.time(),workers=[],scope='final_observed_state',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip(),
        disk_free_bytes=shutil.disk_usage(DATA).free))
    hashes={str(p):sha(p) for root in [OUT,REVIEW] for p in sorted(root.rglob('*')) if p.is_file()}
    atomic_json(MANIFEST,dict(status='completed_control_not_production_accepted',visual_status='fail',frame=frame,tests_passed=16,
        full_video_rerendered=False,heldout_face_metrics_computed=False,retained_hashes=hashes,external_hashes=external))
    print('Frozen',len(hashes),'files; canary unpromoted',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();run(a.check)
