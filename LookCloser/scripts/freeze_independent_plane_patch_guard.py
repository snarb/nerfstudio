"""Replay scores, verify geometry/RGB receipts, and record negative visual gates."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np
import open3d as o3d
from scipy.ndimage import label, find_objects
from joint_temporal_texture import read, sha, atomic_json, ROOT as COLOR_FIT
from review_jaw_repair_transfer import verified_image
from plane_patch_evidence import evidence_decision
from contrastive_plane_guard import best_plane_decision
from build_independent_plane_patch_guard import ROOT, DIAG, LOWER, FRAME
from render_forearm_layer_qualified_guard import VIEWS, MOVIE


def freeze():
    inputs = {};diagnostic = read(DIAG / 'request.json');dq = read(DIAG / 'result.json')
    assert dq['request_sha256'] == sha(DIAG / 'request.json')
    totals = dict(conflict=0, conflict_rejected=0, measured_control=0, measured_control_rejected=0, legacy=0, legacy_rejected=0)
    exploratory = dict(totals)
    for path in sorted(DIAG.glob('*/evidence.npz')):
        q = read(path.parent / 'events.json');a = np.load(path)
        assert q['evidence_sha256'] == sha(path) and q['request_sha256'] == sha(DIAG / 'request.json')
        assert len(q['witnesses']) == len(set(q['witnesses']))
        assert path.parent.name not in q['witnesses']
        assert not set(q['witnesses']) & set(diagnostic['excluded_from_photo_validation'])
        arrays = {k: a[k] for k in ['near', 'far', 'available', 'query_std']}
        scores = evidence_decision(**arrays)
        for k, values in scores.items():
            np.testing.assert_array_equal(values, a[k])
        second = best_plane_decision(**{k: v[[1, 3]] for k, v in arrays.items()})
        for i, event in enumerate(q['events']):
            k = event['kind'];totals[k] += 1;totals[k+'_rejected'] += int(scores['reject_far'][i])
            exploratory[k] += 1;exploratory[k+'_rejected'] += int(second['reject_far'][i])
    assert totals == dq['totals']
    assert exploratory['measured_control_rejected'] == 0
    for key in ['scripts', 'source_depth_receipt']:
        inputs.update(diagnostic[key])
    rgb = diagnostic['source_rgb_receipt'];inputs.update(rgb['source_rgb_hashes'])
    for name, key in [('parameters.npz', 'parameters_sha256'), ('camera_profiles.json', 'profiles_sha256'), ('exposure.json', 'exposure_sha256')]:
        inputs[str(COLOR_FIT / name)] = rgb[key]
    old_request = read(LOWER / 'foreground/request.json')
    old = o3d.io.read_triangle_mesh(old_request['source_mesh'])
    inputs[old_request['source_mesh']] = old_request['source_mesh_sha256']
    comparisons = [];geom = [];inspected = list((DIAG / 'review').glob('*.png'));residuals = []
    for variant in ['guarded', 'unguarded_diagnostic']:
        base = ROOT if variant == 'guarded' else ROOT / variant
        q = read(base / 'qualification_request.json');r = read(base / 'qualification_result.json')
        g = read(base / 'geometry/result.json')
        inputs.update(q['scripts'])
        assert r['request_sha256'] == sha(base / 'qualification_request.json')
        assert r['geometry_result_sha256'] == sha(base / 'geometry/result.json')
        assert g['mesh_sha256'] == sha(base / 'geometry/mesh.ply')
        mesh = o3d.io.read_triangle_mesh(str(base / 'geometry/mesh.ply'))
        np.testing.assert_array_equal(np.asarray(mesh.vertices)[:len(old.vertices)], np.asarray(old.vertices))
        np.testing.assert_array_equal(np.asarray(mesh.triangles)[:len(old.triangles)], np.asarray(old.triangles))
        rec = dict(variant=variant, retained_added_triangles=len(mesh.triangles)-len(old.triangles), original_prefix_preserved=True)
        if variant == 'guarded':
            assert g['guard_passed'] and g['rounds'][-1]['removed'] == 0
            last = r['checks'][-124:]
            assert len({(s['camera'], s['offset']) for s in last}) == 124
            assert all(s['remaining'] == 0 for s in last)
            rec.update(original_final_trusted_observations=sum(s['original_trusted'] for s in last),
                       disqualified_final_observations=sum(s['disqualified'] for s in last), rounds=len(g['rounds']))
        else:
            assert not g['guard_passed'] and not g['guard_applied'] and q['guards_disabled_explicit']
        geom.append(rec)
        for view in VIEWS:
            im, receipt = verified_image(base / 'rgb' / view, FRAME)
            control = MOVIE if view == 'moving' else LOWER / 'rgb' / view / 'baseline'
            before, previous = verified_image(control, FRAME)
            rq = read(base / 'rgb' / view / 'request.json')
            assert rq['qualification_request_sha256'] == sha(base / 'qualification_request.json')
            assert rq['qualification_result_sha256'] == sha(base / 'qualification_result.json')
            assert rq['measured_depth_guard_disabled'] == (variant != 'guarded')
            for key in ['camera', 'source_cameras', 'fixed_exposure']:
                assert receipt[key] == previous[key]
            assert im.shape == (1920, 1080, 3) and np.isfinite(im).all()
            assert np.array_equal(before[:1200], im[:1200])
            comparisons.append(dict(variant=variant, view=view, status='fail', upper_1200_rows_unchanged=True,
                notes=('No consistent benefit over color qualifier; holes, dark pepper and layer seam remain.' if variant == 'guarded'
                       else 'Most lower forearm peppering is gone, but wrist holes and two-tone layer seam remain. Final measured-depth guard deliberately absent.')))
            inspected.append(base / 'review' / (view+'_detail.png'))
            if variant == 'unguarded_diagnostic' and view in ['H004_A005_1210M6', 'moving']:
                box = (40, 1650, 300, 1920) if view.startswith('H') else (90, 1650, 430, 1920)
                x0, y0, x1, y1 = box
                z = np.rot90(np.load(base / 'rgb' / view / 'frames' / FRAME / 'target_depth.npz')['depth'])[y0:y1, x0:x1]
                black = im[y0:y1, x0:x1].max(2) == 0;labels, _ = label(black);components = []
                for i, sl in enumerate(find_objects(labels), 1):
                    yy, xx = sl;mask = labels == i;count = int(mask.sum())
                    if count >= 5 and xx.start > 0 and xx.stop < black.shape[1] and yy.start > 0 and yy.stop < black.shape[0]:
                        components.append(dict(pixels=count, bbox=[xx.start+x0, yy.start+y0, xx.stop+x0, yy.stop+y0],
                                               depth_valid_pixels=int((mask & (z > 0)).sum())))
                residuals.append(dict(view=view, crop=box, interior_black_components=components,
                                      diagnostic_components_not_gt_anatomical_mask=True))
    for path, digest in inputs.items():
        assert sha(path) == digest, path
    atomic_json(ROOT / 'audit.json', dict(score_replay=totals, exploratory_rule_replay=exploratory, geometry=geom,
        comparisons=comparisons, residuals=residuals, full_frame_quality_metrics=False,
        source_depth_maps_verified=len(diagnostic['source_depth_receipt']),
        diagnostic_camera_groups=len(dq['records']), full_video_approved=False))
    atomic_json(ROOT / 'visual_review.json', dict(status='fail', inspected_images={str(p): sha(p) for p in inspected},
        comparisons=comparisons, production_updated=False, full_video_approved=False,
        patch_diagnosis='Several near patches match skin/cloth context better than farther ones in excluded-from-prior cameras; palm alternatives remain similar. Patch planes and boundary context are not exact depth truth.',
        next_geometry_question='Raw proposal still has true wrist ray misses. Locate missing proposal coverage and old/new seam; stronger final-veto relaxation alone cannot remove them.'))
    names = ['study_forearm_independent_patch_evidence.py', 'build_independent_plane_patch_guard.py',
             'build_forearm_consensus_ceiling.py', 'render_independent_plane_patch_guard.py']
    ps = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    live = [s for s in ps.splitlines() if any('python scripts/'+n in s for n in names) and '/bin/bash' not in s]
    assert not live
    logs = list(Path('/mnt/data').glob('dec5_independent_patch_*.log'))
    logs += [Path('/mnt/data/dec5_forearm_independent_patch_evidence.log'), Path('/mnt/data/dec5_independent_plane_patch_guard_build.log')]
    failures = {str(p): [s for s in p.read_text().splitlines() if any(t in s for t in ['Traceback', 'CUDA out of memory', 'CUDA error'])] for p in logs}
    assert not any(failures.values())
    atomic_json(ROOT / 'terminal_check.json', dict(utc=datetime.now(timezone.utc).isoformat(), live_workers=live,
        errors=failures, free_bytes=shutil.disk_usage(ROOT).free,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()))
    hashes = dict(inputs)
    for base in [ROOT, DIAG]:
        hashes.update({str(p): sha(p) for p in base.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'})
    for name in names + ['plane_patch_evidence.py', 'contrastive_plane_guard.py', Path(__file__).name]:
        p = Path(__file__).resolve().with_name(name);hashes[str(p)] = sha(p)
    for p in logs:
        hashes[str(p)] = sha(p)
    atomic_json(ROOT / 'artifact_manifest.json', dict(hashes=hashes, production_updated=False, full_video_approved=False))
    print('Frozen', len(hashes), 'hashes;', geom, flush=True)


def verify():
    hashes = read(ROOT / 'artifact_manifest.json')['hashes']
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed artifact: '+path)
    print('Rechecked', len(hashes), 'hashes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--check', action='store_true');args = parser.parse_args()
    if args.check:
        verify()
    elif (ROOT / 'artifact_manifest.json').exists():
        raise ValueError('Already frozen; use --check')
    else:
        freeze()
