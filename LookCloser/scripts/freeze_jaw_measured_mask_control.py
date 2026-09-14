"""Retain the measured-mask ablation and ancestry without production promotion."""
from pathlib import Path
import argparse
import shutil
import subprocess
import time
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image,HELD
from study_jaw_repair_transfer import FRAMES

OUT=Path('/mnt/data/dec5_jaw_measured_mask_control')
MASK=Path('/mnt/data/dec5_measured_foreground_override')
DIAG=Path('/mnt/data/dec5_jaw_veto_measured_evidence')
BASE=Path('/mnt/data/dec5_jaw_repair_transfer')


def verify(hashes):
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Changed retained artifact: '+p)


def run(check=False):
    manifest=OUT/'artifact_manifest.json'
    if check:
        data=read(manifest);verify(data['retained_hashes']);verify(data['unchanged_movie'])
        # Reused RGB is covered transitively by its frozen parent inventory.
        verify(read(BASE/'artifact_manifest.json')['retained_hashes'])
        print('Measured-mask artifacts and reused RGB ancestry pass');return
    if manifest.exists():raise ValueError('Already frozen; use --check')
    prior=read(BASE/'artifact_manifest.json');verify(prior['retained_hashes']);verify(prior['unchanged_movie'])
    for frame in FRAMES:
        result=read(OUT/frame/'result.json');request=read(OUT/frame/'request.json');audit=read(OUT/frame/'independent_audit.json')
        if audit['result_sha256']!=sha(OUT/frame/'result.json') or not audit['semantic_replay_exact']:raise ValueError('Missing semantic audit')
        if len(audit['checks'])!=124 or any(r['qualified_veto_pixels'] for r in audit['checks']):raise ValueError('Incomplete ray audit')
        verify({str(OUT/frame/p):h for p,h in result['hashes'].items()})
        mr=read(MASK/frame/'result.json');verify({str(MASK/frame/p):h for p,h in mr['hashes'].items()})
        if request['mask_override']['result_sha256']!=sha(MASK/frame/'result.json'):raise ValueError('Changed mask linkage')
        a=np.load(MASK/frame/'evidence.npz');xy=a['query_xy'];expected=np.zeros((1080,1920),bool)
        good=a['other_qualified_foreground']>=3;expected[xy[good,1],xy[good,0]]=True
        if not np.array_equal(expected,a['seeds']):raise ValueError('Seeds differ from observed evidence')
        if frame!='001193' and sha(OUT/frame/'mesh.ply')!=sha(BASE/frame/'mesh.ply'):raise ValueError('Changed reused mesh')
        for view in ['moving','F004_E005_1210FP']:
            for variant in ['baseline','repaired']:verified_image(OUT/'rgb'/frame/view/variant,frame)
    for variant in ['previous','updated']:verified_image(OUT/'veto_camera'/variant,'001193')
    verified_image(OUT/'heldout','001193');verified_image(HELD,'001193')
    visual=read(OUT/'visual_review.json');visual['viewed_sha256']={p:sha(p) for p in visual['actually_viewed']}
    atomic_json(OUT/'visual_review.json',visual)
    scripts=['diagnose_jaw_veto_depth.py','build_measured_foreground_override.py','study_jaw_repair_transfer.py',
        'audit_jaw_repair_transfer.py','review_jaw_measured_mask_control.py','review_jaw_repair_transfer.py',
        'diagnose_jaw_transfer_support.py',Path(__file__).name]
    config=OUT/'config';config.mkdir(exist_ok=True)
    for name in scripts:
        src=Path(__file__).with_name(name);dst=config/name
        if dst.exists() and sha(dst)!=sha(src):raise ValueError('Changed source snapshot')
        if not dst.exists():shutil.copyfile(src,dst)
    ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    workers=[r for r in ps.splitlines() if 'python scripts/' in r and any(n in r for n in scripts[:-1])]
    if workers:raise ValueError('Relevant workers still running')
    tests=Path('/mnt/data/dec5_measured_mask_tests.log')
    if '25 passed' not in tests.read_text():raise ValueError('Missing tests')
    atomic_json(OUT/'final_check.json',dict(unix_time=time.time(),relevant_workers=[],tests_passed=25,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip(),
        disk_free_bytes=shutil.disk_usage(OUT).free))
    retained={str(p):sha(p) for root in [OUT,MASK,DIAG] for p in sorted(root.rglob('*')) if p.is_file()}
    retained[str(BASE/'artifact_manifest.json')]=sha(BASE/'artifact_manifest.json')
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_jaw_measured_mask.md';retained[str(report)]=sha(report)
    for pattern in ['dec5_jaw_measured_mask*.log','dec5_measured_mask*.log','dec5_jaw_veto_depth*.log']:
        retained.update({str(p):sha(p) for p in Path('/mnt/data').glob(pattern) if 'freeze' not in p.name})
    atomic_json(manifest,dict(status='partial_local_gain_not_production_promoted',artifact_free=False,
        frames=FRAMES,retained_hashes=retained,unchanged_movie=prior['unchanged_movie'],
        previous_artifact_manifest_sha256=sha(BASE/'artifact_manifest.json'),
        face_metrics_only=True,main_csv_unchanged=True,tests_passed=25,source_texture_masks_unchanged=True))
    print('Frozen',len(retained),'files; no production promotion',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
