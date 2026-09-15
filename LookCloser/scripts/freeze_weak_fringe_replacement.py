"""Seal two-time geometry/RGB controls, not temporal production approval."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from study_weak_fringe_replacement import ROOT,INSET,MASKS,MOVIE,FRAMES
from review_jaw_repair_transfer import verified_image

SCRIPTS=['study_weak_fringe_replacement.py','audit_weak_fringe_replacement.py','review_weak_fringe_replacement.py']


def run():
    ps=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    live=[p for p in ps if any('python scripts/'+n in p for n in SCRIPTS) and '/bin/bash' not in p]
    assert not live,live
    audit=read(ROOT/'geometry_audit.json');assert len(audit['records'])==4
    review=read(ROOT/'review/result.json');assert len(review['records'])==12
    hashes={}
    for frame in FRAMES:
        q=read(ROOT/frame/'request.json')
        for p,h in q['scripts'].items():assert sha(p)==h,p;hashes[p]=h
        hashes[q['source_mesh']]=q['source_mesh_sha256'];assert sha(q['source_mesh'])==q['source_mesh_sha256']
        maskq=read(MASKS/frame/'request.json')
        for p,h in maskq['rgb_receipt']['source_rgb_hashes'].items():assert sha(p)==h,p;hashes[p]=h
        receipt=q['depth_receipt'];mapping={r['physical_camera']:r for r in read(receipt['transforms'])['frames']}
        for name,h in receipt['depth_sha256'].items():
            p=Path(receipt['dense'])/'stereo/depth_maps'/(mapping[name]['file_path']+'.geometric.bin')
            assert sha(p)==h,p;hashes[str(p)]=h
        for p in [MASKS/frame/'masks.npz',MASKS/frame/'cameras.json',MASKS/frame/'request.json',Path(receipt['transforms']),
                  INSET/frame/'guarded/mesh.ply',INSET/'review'/frame/'native_unmasked_gt.png']:
            hashes[str(p)]=sha(p)
        for arm in ['remove_only','replace']:
            record=next(r for r in audit['records'] if r['frame']==frame and r['arm']==arm)
            assert record['mesh_sha256']==sha(ROOT/frame/arm/'mesh.ply')
            assert record['all_removed_samples_below_two_depth_votes'] and record['background_integral_replay']
            if arm=='replace':assert len(record['native_ray_checks'])==124 and all(r['trusted_free_pixels']==0 for r in record['native_ray_checks'])
        for view in ['moving','native_unmasked']:
            roots=[MOVIE if view=='moving' else INSET/'rgb'/frame/view/'baseline',INSET/'rgb'/frame/view/'completion',
                   ROOT/'rgb'/frame/view/'remove_only',ROOT/'rgb'/frame/view/'replace']
            records=[];requests=[]
            for root in roots:
                im,r=verified_image(root,frame);records.append(r);requests.append(read(root/'request.json'))
                assert im.shape==(1920,1080,3) and np.isfinite(im).all()
                hashes[str(root/'request.json')]=sha(root/'request.json')
                for n,h in read(root/'frames'/frame/'complete.json')['hashes'].items():hashes[str(root/'frames'/frame/n)]=h
            for r in records:
                for k in ['camera','source_cameras','fixed_exposure']:assert r[k]==records[0][k]
            entries=[next(e for e in q['inventory'] if e['frame_id']==frame) for q in requests]
            for e in entries:assert e['source_masks']==entries[0]['source_masks']
            for q in requests:
                for k in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert q[k]==requests[0][k]
                if view=='native_unmasked':assert q['native_target_mask_disabled']
    inspected=[ROOT/'review'/f/(view+'_'+part+'.png') for f in FRAMES for view in ['moving','native_unmasked']
               for part in ['removal_crown','replacement_crown','jaw']]
    atomic_json(ROOT/'visual_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        inspected_images={str(p):sha(p) for p in inspected},status='partial_fringe_improvement_not_promoted',
        notes='All twelve matched panels inspected. Detached crown fringe is less conspicuous in moving views after '
              'removal; large side-view crown opening and ragged/tan hair edges remain. Added inset shell partly '
              'fills native gaps but does not restore the whole silhouette. No conspicuous new face/jaw defect in '
              'these crops. Native frozen face-skin regions have zero added/lost depth or black pixels. '
              'This is not held-out metric validation or whole-video acceptance.',
        production_updated=False,full_video_approved=False))
    atomic_json(ROOT/'completion.json',dict(geometry_audit_sha256=sha(ROOT/'geometry_audit.json'),
        visual_review_sha256=sha(ROOT/'visual_review.json'),workers_terminal=True,production_updated=False,
        quality_metrics_computed=False))
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name!='artifact_manifest.json':hashes[str(p)]=sha(p)
    for n in [*SCRIPTS,Path(__file__).name]:
        p=Path(__file__).resolve().with_name(n);hashes[str(p)]=sha(p)
    for p in Path('/mnt/data').glob('dec5_weak_fringe_replacement*.log'):
        if 'freeze' not in p.name:hashes[str(p)]=sha(p)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_updated=False,visual_approval=False))
    print('Sealed',len(hashes),'hashes',flush=True)


def check():
    hashes=read(ROOT/'artifact_manifest.json')['hashes']
    for p,h in hashes.items():assert sha(p)==h,p
    print('Rechecked',len(hashes),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();check() if a.check else run()
