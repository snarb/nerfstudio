"""Independent multiview-admission replay and partial visual-result freeze."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import shutil
import subprocess
import numpy as np
import open3d as o3d
from scipy.ndimage import label
from joint_temporal_texture import read, sha, atomic_json
from build_multiview_forearm_admission import ROOT, PREVIOUS, LOWER, FRAME, missing_in_train_views
from diagnose_forearm_proposal_gaps import ROOT as DIAG
from diffusion_mesh_repair import scene_for
from review_jaw_repair_transfer import verified_image
from render_forearm_layer_qualified_guard import VIEWS


def freeze():
    q = read(ROOT / 'request.json');r = read(ROOT / 'result.json');a = np.load(ROOT / 'admission.npz')
    assert r['request_sha256'] == sha(ROOT / 'request.json') and r['admission_sha256'] == sha(ROOT / 'admission.npz')
    assert r['mesh_sha256'] == sha(ROOT / 'mesh.ply') and sha(q['source_mesh']) == q['source_mesh_sha256']
    inputs = dict(q['scripts']);inputs[q['source_mesh']] = q['source_mesh_sha256']
    for path, digest in inputs.items():
        assert sha(path) == digest
    old = o3d.io.read_triangle_mesh(q['source_mesh']);new = o3d.io.read_triangle_mesh(str(ROOT / 'mesh.ply'))
    np.testing.assert_array_equal(np.asarray(new.vertices)[:len(old.vertices)], np.asarray(old.vertices))
    np.testing.assert_array_equal(np.asarray(new.triangles)[:len(old.triangles)], np.asarray(old.triangles))
    assert not (a['old'] & ~a['new']).any() and not (a['new'] & ~a['agreement']).any()
    assert (a['votes'][a['new'] & ~a['old']] >= 2).all()
    assert sha(LOWER / 'bias/point_fields.npz') == q['point_fields_sha256']
    base = read(LOWER / 'empty_ray/request.json');fields = np.load(LOWER / 'bias/point_fields.npz')
    xyz = fields[base['reference']+'_offset_diagnostic_xyz']
    votes, available = missing_in_train_views(xyz[a['agreement']], q['cameras'], scene_for(np.asarray(old.vertices), np.asarray(old.triangles)))
    np.testing.assert_array_equal(votes, a['votes'][a['agreement']]);np.testing.assert_array_equal(available, a['available'][a['agreement']])
    diagnosis = read(DIAG / 'result.json');dq = read(DIAG / 'request.json')
    assert diagnosis['request_sha256'] == sha(DIAG / 'request.json')
    assert dq['script_sha256'] == sha(Path(__file__).with_name('diagnose_forearm_proposal_gaps.py'))
    for path, digest in dq['source_hashes'].items():
        assert sha(path) == digest;inputs[path] = digest
    records = [];inspected = [DIAG / 'H004_A005_1210M6.png', DIAG / 'moving.png']
    for view in VIEWS:
        control = PREVIOUS / 'unguarded_diagnostic/rgb' / view
        before, ar = verified_image(control, FRAME);after, br = verified_image(ROOT / 'rgb' / view, FRAME)
        rq = read(ROOT / 'rgb' / view / 'request.json')
        assert rq['multiview_admission_request_sha256'] == sha(ROOT / 'request.json')
        assert rq['multiview_admission_result_sha256'] == sha(ROOT / 'result.json')
        assert rq['measured_depth_guard_disabled'] and not rq['artifact_free_approval']
        for key in ['camera', 'source_cameras', 'fixed_exposure']:
            assert ar[key] == br[key]
        assert after.shape == (1920, 1080, 3) and np.isfinite(after).all()
        assert np.array_equal(before[:1200], after[:1200])
        za = np.rot90(np.load(control / 'frames' / FRAME / 'target_depth.npz')['depth'])
        zb = np.rot90(np.load(ROOT / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth'])
        rec = dict(view=view, upper_1200_rows_unchanged=True, lost_depth_pixels=int(((za > 0) & (zb == 0)).sum()), components=[])
        if view != 'E004_D005_1210L4':
            box = (40, 1650, 300, 1920) if view.startswith('H') else (90, 1650, 430, 1920)
            x0, y0, x1, y1 = box;labels, _ = label(before[y0:y1, x0:x1].max(2) == 0)
            group = next(g for g in diagnosis['records'] if g['view'] == view)
            for c in group['components']:
                m = labels == c['label'];assert int(m.sum()) == c['pixels']
                rec['components'].append(dict(bbox=c['bbox'], before_missing=int((m & (za[y0:y1, x0:x1] == 0)).sum()),
                    after_missing=int((m & (zb[y0:y1, x0:x1] == 0)).sum()),
                    after_black=int((m & (after[y0:y1, x0:x1].max(2) == 0)).sum())))
        records.append(rec);inspected.append(ROOT / 'review' / (view+'_detail.png'))
    inspected.append(ROOT / 'review/moving_overview.png')
    atomic_json(ROOT / 'audit.json', dict(records=records, train_visibility_votes_replayed=True, source_arrays_preserved=True,
        final_pm_guard_applied=False, component_counts_are_diagnostic_not_anatomical_metrics=True, full_frame_quality_metrics=False))
    atomic_json(ROOT / 'visual_review.json', dict(status='partial_wrist_gap_improvement_not_full_video_approval',
        inspected_images={str(p): sha(p) for p in inspected},
        local_wrist_gate='Substantial reduction of internal wrist holes in H/A and moving view; E/D scattered skin speckles reduced.',
        remaining='Hand distortion and holes, cuff-side truncation, exposed old/new layer boundary and two-tone forearm texture remain.',
        whole_frame_status='fail', geometry_inferred=True, no_final_pm_veto=True, production_updated=False, full_video_approved=False))
    names = ['build_multiview_forearm_admission.py', 'render_multiview_forearm_admission.py', 'diagnose_forearm_proposal_gaps.py']
    ps = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    live = [s for s in ps.splitlines() if any('python scripts/'+n in s for n in names) and '/bin/bash' not in s]
    assert not live
    atomic_json(ROOT / 'terminal_check.json', dict(utc=datetime.now(timezone.utc).isoformat(), live_workers=live,
        free_bytes=shutil.disk_usage(ROOT).free,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()))
    hashes = dict(inputs)
    for folder in [ROOT, DIAG]:
        hashes.update({str(p): sha(p) for p in folder.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    for name in names + [Path(__file__).name]:
        p = Path(__file__).resolve().with_name(name);hashes[str(p)] = sha(p)
    for p in list(Path('/mnt/data').glob('dec5_multiview_forearm_admission_*.log')) + [Path('/mnt/data/dec5_forearm_proposal_gap_diagnosis.log')]:
        if 'freeze' not in p.name:
            hashes[str(p)] = sha(p)
    atomic_json(ROOT / 'artifact_manifest.json', dict(hashes=hashes, full_video_approved=False, production_updated=False))
    print('Frozen', len(hashes), 'hashes', records, flush=True)


def verify():
    hashes = read(ROOT / 'artifact_manifest.json')['hashes']
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed artifact: '+path)
    print('Rechecked', len(hashes), 'hashes', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--check', action='store_true');a = p.parse_args()
    if a.check:
        verify()
    elif (ROOT / 'artifact_manifest.json').exists():
        raise ValueError('Already frozen; use --check')
    else:
        freeze()
