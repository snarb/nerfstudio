"""Seal a manual diagnostic review; check never rewrites producer receipts."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from joint_temporal_texture import read, sha, atomic_json
from diagnose_matched_forearm_radiometry import ROOT as FIRST
from probe_forearm_radiometry_shift import ROOT as SECOND


def verify_inputs():
    hashes = {}
    for root in [FIRST,SECOND]:
        complete = read(root/'complete.json');q = read(root/'request.json');r = read(root/'result.json')
        assert r['request_sha256'] == sha(root/'request.json')
        for path,digest in {**q.get('hashes',{}),**complete['hashes']}.items():
            assert sha(path) == digest, path
            hashes[path] = digest
        hashes[str(root/'complete.json')] = sha(root/'complete.json')
    r = read(FIRST/'result.json')
    assert len(r['replay']) == 6 and max(x['max_uint8_error'] for x in r['replay']) <= 1
    assert sum(x['selected_not_four_tap_visible'] for x in r['replay']) == 3
    r = read(SECOND/'result.json')
    assert len(r['records']) == 4
    for record in r['records']:
        assert record['count'] >= 100 and len(record['trials']) == 81
        assert len({tuple(x['offset']) for x in record['trials']}) == 81
        assert record['best_color_shift']['median_absolute_rgb_difference'] <= record['baseline']['median_absolute_rgb_difference']
    return hashes


def freeze():
    hashes = verify_inputs()
    images = list(FIRST.glob('*_native.png')) + list(FIRST.glob('*_same_surface.png')) + list(SECOND.glob('*.png'))
    assert len(images) == 12
    # Authored only after the main agent inspected these exact twelve images.
    review = dict(utc=datetime.now(timezone.utc).isoformat(),diagnostic_status='reviewed',
        inspected_images={str(p):sha(p) for p in images}, production_updated=False,
        notes='A broad brightness difference persists in raw fixed-exposure same-surface projections. Small shifts do not remove I/B versus J/B disagreement. Native images contain different smooth shading; this experiment does not separate BRDF, local shape, or spatial camera response. No correction or geometry improvement is accepted.',
        numerical_replay_caveat='All selected RGB replay within one uint8 level. Three source pixels differ at the four-tap visibility threshold with CPU float64 versus CUDA sampling; not claimed exact visibility parity.',
        status_scope='Overrides producer pending visual status for diagnostic review only, not full-frame quality approval.')
    atomic_json(SECOND/'visual_review.json',review)
    for p in [SECOND/'visual_review.json',Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'tests/test_matched_forearm_radiometry.py']:
        hashes[str(p)] = sha(p)
    ps = subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    names = ['diagnose_matched_forearm_radiometry.py','probe_forearm_radiometry_shift.py']
    live = [line for line in ps.splitlines() if any('python scripts/'+n in line for n in names) and '/bin/bash' not in line]
    assert not live
    atomic_json(SECOND/'audit.json',dict(utc=datetime.now(timezone.utc).isoformat(),terminal=True,
        live_workers=live,paired_sources=4,offset_trials=324,images_reviewed=12,
        full_frame_quality_metrics=False,production_updated=False,exact_visibility_replay=False))
    hashes[str(SECOND/'audit.json')] = sha(SECOND/'audit.json')
    for p in [Path('/mnt/data/dec5_matched_forearm_radiometry.log'),Path('/mnt/data/dec5_forearm_radiometry_shift.log')]:
        hashes[str(p)] = sha(p)
    atomic_json(SECOND/'artifact_manifest.json',dict(hashes=hashes,production_updated=False))
    print('Frozen',len(hashes),'hashes',flush=True)


def check():
    verify_inputs()
    hashes=read(SECOND/'artifact_manifest.json')['hashes']
    for p,digest in hashes.items():
        assert sha(p)==digest,p
    print('Rechecked',len(hashes),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    check() if a.check else freeze()
