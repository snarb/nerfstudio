"""Audit completed forearm controls, including one intentional config overwrite."""
from pathlib import Path
import argparse
from joint_temporal_texture import read,sha,atomic_json


def audit(root):
    digest=sha(root/'request.json');complete=read(root/'complete.json')
    if complete['request_sha256']!=digest:raise ValueError('Control request mismatch')
    for name,expected in complete['hashes'].items():
        if sha(root/name)!=expected:raise ValueError('Control output mismatch')
    cfg=str(root/'pipeline/dense/stereo/patch-match.cfg')
    successor=read(root/'stages/patch-config.json')
    if successor['request_sha256']!=digest:raise ValueError('Config stage request mismatch')
    overwritten=[];verified=0
    for stage in sorted((root/'stages').glob('*.json')):
        record=read(stage)
        if record['request_sha256']!=digest:raise ValueError('Stage request mismatch')
        for path,expected in record['retained_hashes'].items():
            current=sha(path)
            if current!=expected:
                # Undistortion creates COLMAP's default auto-source config;
                # the explicit62/12 stage deliberately replaces that file.
                if not (stage.name=='undistort.json' and path==cfg and
                        successor['retained_hashes'].get(path)==current):
                    raise ValueError(f'Unexplained changed artifact {stage.name}: {path}')
                overwritten.append(dict(path=path,earlier_stage=stage.name,earlier_sha256=expected,
                    replacing_stage='patch-config.json',current_sha256=current))
            verified+=1
    for variant in ['fuse-original','fuse-full-block']:
        out=root/(variant+'_render');receipt=read(out/'frames/001033/complete.json')
        if receipt['request_sha256']!=sha(out/'request.json'):raise ValueError('Render request mismatch')
        for name,expected in receipt['hashes'].items():
            if sha(out/'frames/001033'/name)!=expected:raise ValueError('Render artifact mismatch')
    qc=read(root/'depth_qc.json')
    if qc['maps']!=62 or qc['shape']!=[1080,1920]:raise ValueError('Depth inventory mismatch')
    review=read(root/'visual_review.json')
    if review.get('status')!='rejected_as_forearm_repair' or len(review.get('evidence',[]))<2:
        raise ValueError('Explicit matched RGB/clay rejection review required')
    for evidence in review['evidence']:
        if sha(evidence['path'])!=evidence['sha256']:raise ValueError('Stale visual review')
    result=dict(status='diagnostic_complete_candidate_not_promoted',depth_qc=qc,
        checked_stage_artifact_entries=verified,explicitly_superseded_artifacts=overwritten,
        matched_render_receipts_verified=True,full_frame_quality_metrics=False,
        request_sha256=digest,visual_review_sha256=sha(root/'visual_review.json'),script_sha256=sha(__file__),
        report_sha256=sha(Path(__file__).resolve().parents[1]/'experiments/dec5_temporal_missing_surface.md'))
    atomic_json(root/'final_audit.json',result);print(result['status'],verified,'stage entries checked',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_depth_control_001033'))
    audit(p.parse_args().root)
