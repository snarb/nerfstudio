"""Seal reviewed inset-shell diagnostics, never promote them to production."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from probe_inset_head_completion import ROOT,MASKS,RAW,SOURCE,FRAMES,MOVIE
from review_jaw_repair_transfer import verified_image

SCRIPTS=['probe_inset_head_completion.py','guard_inset_head_completion.py',
         'render_inset_head_completion.py','audit_inset_head_completion.py']


def run():
    ps=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    live=[p for p in ps if any('python scripts/'+s in p for s in SCRIPTS) and '/bin/bash' not in p]
    assert not live,live
    hashes={};audit=read(ROOT/'geometry_audit.json');assert len(audit['records'])==2
    reviewed=read(ROOT/'review/result.json');assert len(reviewed['records'])==4
    for frame in FRAMES:
        a=next(r for r in audit['records'] if r['frame']==frame)
        assert len(a['native_ray_checks'])==124 and all(x['trusted_free_pixels']==0 for x in a['native_ray_checks'])
        assert a['mesh_sha256']==sha(ROOT/frame/'guarded/mesh.ply')
        maskreq=read(MASKS/frame/'request.json')
        for p,h in maskreq['rgb_receipt']['source_rgb_hashes'].items():
            assert sha(p)==h,p;hashes[p]=h
        receipt=maskreq['depth_receipt'];cal=read(receipt['transforms']);mapping={r['physical_camera']:r for r in cal['frames']}
        for name,h in receipt['depth_sha256'].items():
            p=Path(receipt['dense'])/'stereo/depth_maps'/(mapping[name]['file_path']+'.geometric.bin')
            assert sha(p)==h,p;hashes[str(p)]=h
        for p in [MASKS/frame/'request.json',MASKS/frame/'result.json',MASKS/frame/'masks.npz',
                  RAW/frame/'poisson_raw.ply',SOURCE/frame/'request.json',Path(receipt['transforms'])]:
            hashes[str(p)]=sha(p)
        for view in ['moving','native_unmasked']:
            roots=[MOVIE if view=='moving' else ROOT/'rgb'/frame/view/'baseline',ROOT/'rgb'/frame/view/'completion']
            rs=[];qs=[]
            for root in roots:
                im,r=verified_image(root,frame);rs.append(r);q=read(root/'request.json');qs.append(q)
                assert im.shape==(1920,1080,3) and np.isfinite(im).all()
                hashes[str(root/'request.json')]=sha(root/'request.json')
                for n,h in read(root/'frames'/frame/'complete.json')['hashes'].items():hashes[str(root/'frames'/frame/n)]=h
            for k in ['camera','source_cameras','fixed_exposure']:assert rs[0][k]==rs[1][k]
            for k in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert qs[0][k]==qs[1][k]
            entries=[next(e for e in q['inventory'] if e['frame_id']==frame) for q in qs]
            assert entries[0]['source_masks']==entries[1]['source_masks']
            assert rs[1]['mesh_sha256']==a['mesh_sha256']
            if view=='native_unmasked':assert all(q['native_target_mask_disabled'] for q in qs)
    inspected=[ROOT/'review'/f/(view+'_crown.png') for f in FRAMES for view in ['moving','native_unmasked']]
    inspected += [ROOT/'review'/f/'native_unmasked_jaw.png' for f in FRAMES]
    inspected += [ROOT/f/'native_train_comparison.png' for f in FRAMES]
    inspected += [ROOT/'001123/inset_001000/native_train.png',ROOT/'rgb/001123/native_unmasked/completion/frames/001123/frame.png']
    atomic_json(ROOT/'visual_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        inspected_images={str(p):sha(p) for p in inspected},status='partial_native_crown_improvement_not_promoted',
        notes='Native crown gaps partly fill, more clearly at001123. Upper arch/opening and detached fringe remain. '
              'Moving crown comparisons show no substantial repair. Native face/jaw has no conspicuous new defect '
              'in inspected crops, but no heldout/temporal quality approval. Brown/ragged crown-edge texture remains.',
        production_updated=False,full_video_approved=False))
    atomic_json(ROOT/'completion.json',dict(geometry_audit_sha256=sha(ROOT/'geometry_audit.json'),
        review_result_sha256=sha(ROOT/'review/result.json'),visual_review_sha256=sha(ROOT/'visual_review.json'),
        workers_terminal=True,production_updated=False,quality_metrics_computed=False))
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name!='artifact_manifest.json':hashes[str(p)]=sha(p)
    for n in [*SCRIPTS,Path(__file__).name]:
        p=Path(__file__).resolve().with_name(n);hashes[str(p)]=sha(p)
    for p in Path('/mnt/data').glob('dec5_inset_head_completion*.log'):
        if 'freeze' not in p.name:hashes[str(p)]=sha(p)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_updated=False,visual_approval=False))
    print('Sealed',len(hashes),'hashes',flush=True)


def check():
    hashes=read(ROOT/'artifact_manifest.json')['hashes']
    for p,h in hashes.items():assert sha(p)==h,p
    print('Rechecked',len(hashes),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    check() if a.check else run()
