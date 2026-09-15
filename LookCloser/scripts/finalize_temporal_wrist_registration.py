"""Audit/freeze temporal registration evidence without claiming a repaired mesh."""
from pathlib import Path
import argparse, shutil, subprocess, time
import numpy as np
import torch
from joint_temporal_texture import read, sha, atomic_json, cameras, ROOT as COLOR, HELD_CAMERAS
from temporal_rigid_patch import transfer_points, fit_rigid
from study_wrist_observations import OUTPUT as OBS, NAMES
from freeze_forearm_secondary_reference import verify

DATA=Path('/mnt/data'); HERE=Path(__file__).resolve().parent
DIRECT=DATA/'dec5_temporal_wrist_registration'; CHAIN=DATA/'dec5_temporal_wrist_chain_registration'
AFFINE=DATA/'dec5_temporal_wrist_affine'; MANIFEST=CHAIN/'artifact_manifest.json'


def run(check=False):
    if check:
        m=read(MANIFEST);verify(m['retained_hashes']);verify(m['external_hashes'])
        print('Verified',len(m['retained_hashes']),'retained and',len(m['external_hashes']),'external files',flush=True);return
    if MANIFEST.exists():raise ValueError('Already frozen; use --check')
    ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    if any('python scripts/'+name in line for line in ps.splitlines() for name in ['study_wrist_observations.py','study_temporal_wrist_registration.py','fit_temporal_wrist_affine.py']):
        raise ValueError('Worker still active')
    external={}; config=CHAIN/'config';config.mkdir(exist_ok=True)
    for frame in ['001029','001031','001033','001035','001037']:
        request=read(OBS/frame/'request.json');result=read(OBS/frame/'result.json')
        if result['request_sha256']!=sha(OBS/frame/'request.json'):raise ValueError('Changed observation request')
        for record in result['records']:
            if record['camera']['physical_camera'] in HELD_CAMERAS:raise ValueError('Heldout data leak')
            verify({record['image']:record['image_sha256']})
            external[record['camera']['file_path']]=record['source_sha256']
        external[str(COLOR/'camera_profiles.json')]=request['profiles_sha256'];external[str(COLOR/'exposure.json')]=request['exposure_sha256']
    for root in [DIRECT,CHAIN]:
        request=read(root/'request.json');result=read(root/'flow_result.json')
        if result['request_sha256']!=sha(root/'request.json'):raise ValueError('Changed flow request')
        checkpoint=Path(torch.hub.get_dir())/'checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
        external[str(checkpoint)]=request['flow_weights_sha256']
        for record in result['records']:verify({record['path']:record['sha256']})
        for name,h in request['scripts'].items():
            source=next((p for p in [HERE/name,DIRECT/'config'/name] if p.exists() and sha(p)==h),None)
            if source is None:raise ValueError('Missing producer '+name)
            dest=config/(h+'_'+name)
            if dest.exists() and sha(dest)!=h:raise ValueError('Changed snapshot')
            if not dest.exists():shutil.copyfile(source,dest)
    result=read(CHAIN/'registration_result.json');data=np.load(CHAIN/'correspondences.npz')
    verify({str(CHAIN/'correspondences.npz'):result['array_sha256']})
    for name in ['source_mesh','source_metadata','target_metadata']:external[result[name]]=result[name+'_sha256']
    expected=transfer_points(data['source_points'],read(result['source_metadata']),read(result['target_metadata']))
    np.testing.assert_allclose(expected,data['common_gauge_points'],atol=1e-12,rtol=0)
    rows,_,_=cameras('001037');lookup={r['physical_camera']:r for r in rows}
    _,_,errors=fit_rigid(expected,[lookup[n] for n in NAMES],data['point_indices'],data['camera_indices'],data['observations'],data['parameters'],list(range(5)))
    np.testing.assert_allclose(errors,data['errors'],atol=1e-3,rtol=0)
    affine=read(AFFINE/'result.json');req=read(AFFINE/'request.json')
    if req['parent_result_sha256']!=sha(CHAIN/'registration_result.json') or affine['request_sha256']!=sha(AFFINE/'request.json'):raise ValueError('Changed affine ancestry')
    verify({str(AFFINE/'fit.npz'):affine['fit_sha256']})
    seen=[OBS/f/'six_train_views_native.png' for f in ['001029','001037']]
    seen += [CHAIN/'review'/(n+'_reprojection.png') for n in NAMES]
    seen += [AFFINE/(n+'_comparison.png') for n in NAMES]
    atomic_json(CHAIN/'visual_review.json',dict(status='temporal_registration_partial_not_mesh_acceptance',
        actually_inspected_images={str(p):sha(p) for p in seen},
        notes='Later hand visibly blurred; chained tracks follow palm. H/A discrepancy persists in rigid and affine controls.',
        production_accepted=False,geometry_changed=False,full_video_changed=False))
    atomic_json(CHAIN/'independent_audit.json',dict(common_point_gauge_replayed=True,
        rigid_refit_from_saved_correspondences_replayed=True,error_atol_pixels=.001,
        correspondence_sha256=sha(CHAIN/'correspondences.npz'),script_sha256=sha(__file__),
        raw_flow_reinferred=False,mesh_repair_claimed=False))
    if '4 passed' not in (DATA/'dec5_temporal_wrist_chain_tests.log').read_text():raise ValueError('Tests incomplete')
    for name in ['fit_temporal_wrist_affine.py','probe_temporal_wrist_camera_consensus.py',Path(__file__).name]:
        dest=config/name
        if dest.exists() and sha(dest)!=sha(HERE/name):raise ValueError('Changed snapshot')
        if not dest.exists():shutil.copyfile(HERE/name,dest)
    external.update({str(p):sha(p) for p in DATA.glob('dec5_temporal_wrist*.log') if 'freeze' not in p.name})
    external.update({str(p):sha(p) for p in DATA.glob('dec5_wrist_observations_*.log')})
    report=HERE.parent/'experiments/dec5_temporal_wrist_registration.md';external[str(report)]=sha(report)
    verify(external)
    atomic_json(CHAIN/'final_process_check.json',dict(unix_time=time.time(),workers=[],free_bytes=shutil.disk_usage(DATA).free,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()))
    retained={str(p):sha(p) for root in [DIRECT,CHAIN,AFFINE,OBS] for p in sorted(root.rglob('*')) if p.is_file()}
    atomic_json(MANIFEST,dict(status='completed_registration_study_not_geometry_repair',tests_passed=4,
        heldout_rgb_used=False,new_video=False,retained_hashes=retained,external_hashes=external))
    print('Frozen',len(retained),'files; registration only, no repaired mesh/video',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
