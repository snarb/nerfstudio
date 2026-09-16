"""Seal inspected partial controls without promoting them or rewriting receipts."""
from pathlib import Path
from study_multiview_face_prior import read,save,sha

ROOTS=[Path('/mnt/data/dec5_face_interior_visibility90_001123'),
       Path('/mnt/data/dec5_face_consensus_visibility_001123')]
NOTES=Path('/mnt/data/dec5_face_visibility_recovery_review.json')


def main():
    notes=read(NOTES); assert notes['visual_status']=='fail' and notes['production_promoted'] is False
    expected=set()
    for root in ROOTS:expected.update(str(p) for p in (root/'review').glob('*.png'))
    expected.update(str(p) for p in ROOTS[0].glob('*mask.png'))
    expected.add(str(ROOTS[0]/'mask_overview.png'))
    nose=Path('/mnt/data/dec5_nose_source_visibility_001123')
    expected.update(str(p) for p in (nose/'witnesses').glob('*.png'))
    assert set(notes['viewed_images'])==expected and len(expected)==19
    hashes={str(NOTES):sha(NOTES)}
    diag=read(nose/'result.json')
    for p,h in diag['input_hashes'].items():assert sha(p)==h;hashes[p]=h
    assert sha(nose/'evidence.npz')==diag['evidence_sha256']
    for root in ROOTS:
        r=read(root/'result.json');q=read(root/'request.json');audit=read(root/'independent_audit.json')
        assert audit['result_sha256']==sha(root/'result.json')
        assert audit['request_sha256']==sha(root/'request.json')==r['request_sha256']
        assert audit['changed_points_replayed']==r['new_source_points']
        for p,h in q['input_hashes'].items():assert sha(p)==h;hashes[p]=h
        for name,h in r['hashes'].items():assert sha(root/name)==h;hashes[str(root/name)]=h
        for p,h in read(root/'review/audit.json')['images'].items():assert sha(p)==h;hashes[p]=h
        assert not (root/'visual_review.json').exists()
        save(root/'visual_review.json',dict(visual_status='fail',reason='Partial nose-seam improvement, visible residual; not artifact-free.',
             reviewed_notes=str(NOTES),notes_sha256=sha(NOTES),result_sha256=sha(root/'result.json'),
             production_promoted=False,mesh_repair_claimed=False))
    sem=Path('/mnt/data/dec5_train_face_support_001123');sq=read(sem/'request.json');sr=read(sem/'complete.json')
    assert sr['request_sha256']==sha(sem/'request.json')
    assert len(sr['outputs'])==len({x['camera'] for x in sr['outputs']})==62
    assert sha(sq['model']['file'])==sq['model']['sha256']
    hashes[sq['model']['file']]=sq['model']['sha256']
    for x in sr['outputs']:assert sha(x['path'])==x['sha256'];hashes[x['path']]=x['sha256']
    for folder in [*ROOTS,nose,sem]:
        for p in folder.rglob('*'):
            if p.is_file():hashes[str(p)]=sha(p)
    for name in ['diagnose_nose_source_visibility.py','infer_train_face_support.py',
        'study_face_interior_visibility.py','run_face_interior_visibility_90.py',
        'run_face_consensus_visibility.py','review_face_interior_visibility.py',
        'review_face_consensus_visibility.py','audit_face_visibility_recovery.py',
        'seal_face_visibility_recovery.py','joint_temporal_texture.py','native_texture_footprint.py',
        'view_consistent_source_quality.py','temporal_texture_view_prior.py',
        'diagnose_gap_texture_admission.py','calibrated_depth_witness.py']:
        p=Path(__file__).with_name(name).resolve();hashes[str(p)]=sha(p)
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_face_interior_visibility.md'
    hashes[str(report)]=sha(report)
    output=Path('/mnt/data/dec5_face_visibility_recovery_manifest.json');assert not output.exists()
    save(output,dict(hashes=hashes,viewed_images=19,visual_status='fail',production_promoted=False,
        scope='HD single-frame partial renderer control, not6K video or mesh repair',
        all_workers_terminal=True))
    for p,h in read(output)['hashes'].items():assert sha(p)==h,p
    print('sealed',len(hashes),'hash bindings; visual fail retained',flush=True)


if __name__=='__main__':main()
