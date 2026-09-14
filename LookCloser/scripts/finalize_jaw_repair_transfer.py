"""Freeze reviewed jaw controls without claiming production acceptance."""
from pathlib import Path
import argparse
import shutil
import subprocess
import time
from joint_temporal_texture import read,sha,atomic_json
from study_jaw_repair_transfer import OUT,FRAMES,SCRIPTS
from review_jaw_repair_transfer import verified_image,HELD


def verify(hashes):
    for name,h in hashes.items():
        if sha(name)!=h:raise ValueError('Changed artifact: '+name)


def run(output,check=False):
    manifest=output/'artifact_manifest.json'
    if check:
        data=read(manifest);verify(data['retained_hashes']);verify(data['unchanged_movie'])
        print('Retained jaw artifacts and published movie hashes pass');return
    if manifest.exists():raise ValueError('Already frozen; use --check')
    reviews=read(output/'visual_review.json')
    if reviews['status']!='reviewed_known_artifacts_not_production_promoted':raise ValueError('Missing actual review')
    for frame in FRAMES:
        folder=output/frame;request=read(folder/'request.json');result=read(folder/'result.json')
        for name,h in request['scripts'].items():
            if sha(Path(__file__).with_name(name))!=h:raise ValueError('Changed producer source')
        verify({str(folder/p):h for p,h in result['hashes'].items()})
        audit=read(folder/'independent_audit.json')
        if audit['result_sha256']!=sha(folder/'result.json') or len(audit['checks'])!=124:
            raise ValueError('Wrong independent audit')
        if any(c['qualified_veto_pixels'] for c in audit['checks']):raise ValueError('Ray veto remained')
        for view in ['moving','F004_E005_1210FP']:
            for variant in ['baseline','repaired']:verified_image(output/'rgb'/frame/view/variant,frame)
    verified_image(HELD,'001193');verified_image(output/'heldout','001193')
    metrics=read(output/'heldout/metrics.json')
    if metrics['full_frame_metrics']:raise ValueError('Unexpected full-frame metrics')
    reviewed={str(output/p):sha(output/p) for p in reviews['actually_viewed']}
    reviews['reviewed_sha256']=reviewed;atomic_json(output/'visual_review.json',reviews)
    config=output/'config';config.mkdir(exist_ok=True)
    scripts=SCRIPTS+['review_jaw_repair_transfer.py','audit_jaw_repair_transfer.py',
        'diagnose_jaw_transfer_support.py','inspect_jaw_transfer_mask_disagreement.py',Path(__file__).name]
    for name in scripts:
        source=Path(__file__).with_name(name);target=config/name
        if target.exists() and sha(target)!=sha(source):raise ValueError('Changed snapshot')
        if not target.exists():shutil.copyfile(source,target)
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_jaw_repair_transfer.md'
    testlog=Path('/mnt/data/dec5_jaw_transfer_tests.log')
    if '16 passed' not in testlog.read_text():raise ValueError('Missing passing tests')
    movie=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
    unchanged={str(movie/'video.mp4'):'ce1ff5f7fd612a46e39b6ae89d10aee7534f895ea7e7887d722dc10903bccd40',
        str(movie/'frames.zip'):'5e2d078a2e249c90a9a4b3d4112a7652bda533d45422c91832c7355480c2f7b6'}
    verify(unchanged)
    ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    live=[r for r in ps.splitlines() if 'python scripts/' in r and any(n in r for n in [
        'study_jaw_repair_transfer.py','review_jaw_repair_transfer.py','inspect_jaw_transfer_mask_disagreement.py'])]
    if live:raise ValueError('Workers still active')
    atomic_json(output/'final_check.json',dict(unix_time=time.time(),relevant_workers=[],
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip(),
        disk_free_bytes=shutil.disk_usage(output).free,tests_passed=16))
    retained={str(p):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    retained[str(report)]=sha(report)
    retained.update({str(p):sha(p) for p in Path('/mnt/data').glob('dec5_jaw_transfer_*.log') if 'finalize' not in p.name})
    atomic_json(manifest,dict(status='four_time_transfer_complete_not_production_promoted',frames=FRAMES,
        artifact_free=False,production_accepted=False,retained_hashes=retained,unchanged_movie=unchanged,
        main_campaign_csv_unchanged=True,tests_passed=16,next_issue='separate unsupported shape from ambiguous silhouette-mask rejection'))
    print('Frozen',len(retained),'artifacts; all workers terminal; not production accepted',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--check',action='store_true')
    a=p.parse_args();run(a.output,a.check)
