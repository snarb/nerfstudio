"""Audit the opt-in layer guard; no publication or geometry acceptance implied."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read, sha, atomic_json, cameras
from review_jaw_repair_transfer import verified_image
from study_confidence_depth_prior import unproject
from study_forearm_layer_qualified_guard import ROOT, LOWER, FRAME
from render_forearm_layer_qualified_guard import VIEWS


def footprint_audit():
    """Check every remap contributor for the largest diagnostic patch/search.

    Rectified black padding must not become photographed patch evidence. Check
    the four enclosing remap pixels against the actual 1080x1920 portrait,
    including one native pixel of safety, not the coarse semantic arm mask.
    """
    source = LOWER / FRAME / 'E004_E_F004_E'
    cal = np.load(source / 'calibration.npz')
    row = next(r for r in cameras(FRAME)[0] if r['physical_camera'] == 'F004_E005_1210FP')
    group = next(r for r in read(LOWER / 'veto_diagnosis/result.json')['records'] if r['camera'] == row['physical_camera'])
    e2 = np.eye(4)
    e2[:3, :3] = cal['R2'] @ cal['E2'][:3, :3]
    e2[:3, 3] = cal['R2'] @ cal['E2'][:3, 3]
    records = []
    for i, event in enumerate(group['samples']):
        x, y = event['native_xy']
        points = unproject(row, np.array([x, x]), np.array([y, y]),
                           np.array([event['proposed_depth'], event['observed_depth']]))
        uv = []
        for e, k in [(cal['rectified_extrinsic'], cal['cropped_intrinsic']), (e2, cal['P2'][:, :3])]:
            p = points @ e[:3, :3].T + e[:3, 3]
            p = p @ k.T
            uv.append(p[:, :2] / p[:, 2:])
        ds = uv[0][:, 0] - uv[1][:, 0]
        sweep = np.arange(np.floor(ds.min()-64), np.ceil(ds.max()+64)+.5, .5)
        y, x = np.mgrid[-15:16, -15:16]
        offsets = np.column_stack([x.ravel(), y.ravel()])
        # Include exact hypotheses, which need not lie on the .5-pixel sweep.
        left_centers = np.vstack([uv[1][:1] + np.column_stack([sweep, np.zeros(len(sweep))]), uv[0]])
        for side, centers in [('right', uv[1][:1]), ('left', left_centers)]:
            xy = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
            base = np.floor(xy).astype(int)
            contributors = np.concatenate([base + d for d in [(0, 0), (1, 0), (0, 1), (1, 1)]])
            mx, my = cal[side+'_map_x'], cal[side+'_map_y']
            assert (contributors >= 0).all()
            assert (contributors[:, 0] < mx.shape[1]).all() and (contributors[:, 1] < mx.shape[0]).all()
            a = mx[contributors[:, 1], contributors[:, 0]]
            b = my[contributors[:, 1], contributors[:, 0]]
            valid = np.isfinite(a) & np.isfinite(b) & (a >= 1) & (a <= 1078) & (b >= 1) & (b <= 1918)
            assert valid.all(), 'NCC includes unavailable photographed footprint'
            records.append(dict(sample=i, side=side, contributor_count=len(a), all_available=True))
    return records


def freeze():
    q = read(ROOT / 'qualification_request.json')
    result = read(ROOT / 'qualification_result.json')
    for path, digest in q['script_hashes'].items():
        assert sha(path) == digest
    assert result['request_sha256'] == sha(ROOT / 'qualification_request.json')
    assert result['geometry_result_sha256'] == sha(ROOT / 'geometry/result.json')
    assert q['lower_stage_request_sha256'] == sha(LOWER / FRAME / 'request.json')
    assert q['original_guard_request_sha256'] == sha(LOWER / 'foreground/request.json')
    assert q['raw_proposal_sha256'] == sha(ROOT / 'geometry/proposal.npz') == sha(LOWER / 'foreground/proposal.npz')
    gq = read(ROOT / 'geometry/request.json')
    gr = read(ROOT / 'geometry/result.json')
    assert gr['mesh_sha256'] == sha(ROOT / 'geometry/mesh.ply')
    assert gr['request_sha256'] == sha(ROOT / 'geometry/request.json')
    assert len(result['checks']) == len(gr['rounds']) * 124
    last = result['checks'][-124:]
    assert len({(c['camera'], c['offset']) for c in last}) == 124
    assert all(c['remaining'] == 0 for c in last)
    assert all(c['original_trusted'] == c['disqualified'] + c['remaining'] for c in result['checks'])
    source = o3d.io.read_triangle_mesh(gq['source_mesh'])
    mesh = o3d.io.read_triangle_mesh(str(ROOT / 'geometry/mesh.ply'))
    strict = o3d.io.read_triangle_mesh(str(LOWER / 'foreground/mesh.ply'))
    assert sha(gq['source_mesh']) == gq['source_mesh_sha256']
    np.testing.assert_array_equal(np.asarray(mesh.vertices)[:len(source.vertices)], np.asarray(source.vertices))
    np.testing.assert_array_equal(np.asarray(mesh.triangles)[:len(source.triangles)], np.asarray(source.triangles))
    np.testing.assert_array_equal(np.asarray(mesh.vertices), np.asarray(strict.vertices))
    candidate_faces = set(map(tuple, np.asarray(mesh.triangles)[len(source.triangles):]))
    strict_faces = set(map(tuple, np.asarray(strict.triangles)[len(source.triangles):]))
    comparisons = []
    for view in VIEWS:
        old, ar = verified_image(LOWER / 'rgb' / view / 'candidate', FRAME)
        new, br = verified_image(ROOT / 'rgb' / view, FRAME)
        rq = read(ROOT / 'rgb' / view / 'request.json')
        assert rq['qualification_request_sha256'] == sha(ROOT / 'qualification_request.json')
        assert rq['qualification_result_sha256'] == sha(ROOT / 'qualification_result.json')
        for key in ['camera', 'source_cameras', 'fixed_exposure']:
            assert ar[key] == br[key]
        assert new.shape == (1920, 1080, 3) and np.isfinite(new).all()
        assert np.array_equal(new[:1200], old[:1200])
        comparisons.append(dict(view=view, changed_rgb_pixels=int(np.any(old != new, axis=2).sum()), upper_1200_rows_unchanged=True))
    diag = read(ROOT / 'ambiguity/result.json')
    assert diag['script_sha256'] == sha(Path(__file__).with_name('diagnose_forearm_stereo_ambiguity.py'))
    for path, digest in diag['source_hashes'].items():
        assert sha(path) == digest
    assert diag['event_source_sha256'] == sha(LOWER / 'veto_diagnosis/result.json')
    assert len(diag['records']) == 5
    for record in diag['records']:
        for estimate in record['estimates']:
            assert np.isfinite([estimate[k] for k in ['query_rgb_std', 'prior_ncc', 'pm_ncc', 'best_ncc']]).all()
    footprint = footprint_audit()
    inspected = [ROOT / 'review' / (v+'_detail.png') for v in VIEWS]
    inspected += [ROOT / 'review/moving_overview.png', ROOT / 'ambiguity/ncc_curves.png']
    atomic_json(ROOT / 'visual_review.json', dict(status='fail',
        inspected_images={str(p): sha(p) for p in inspected},
        verdicts=[dict(view=v, status='fail', notes=n) for v, n in zip(VIEWS, [
            'Slight extra arm coverage; severe wrist cutout, peppered arm and hard depth/texture transition remain.',
            'Some pits smaller; scattered blue/dark surface defects remain versus real train RGB.',
            'Additional brown arm area, but major black wrist/forearm break and peppering remain.'])],
        diagnostic_verdict='Forearm sample NCC favors nearer hypothesis over far PM, but small windows are low-texture and larger windows include context; not proof of correct depth.',
        full_video_approved=False, production_updated=False, camera_path_changed=False))
    atomic_json(ROOT / 'audit.json', dict(comparisons=comparisons,
        original_mesh_arrays_preserved=True, retained_added_triangles=gr['retained_triangles'],
        strict_added_faces_missing=len(strict_faces-candidate_faces), newly_retained_faces=len(candidate_faces-strict_faces),
        final_original_trusted_ray_observations=sum(c['original_trusted'] for c in last),
        final_disqualified_ray_observations=sum(c['disqualified'] for c in last),
        final_qualified_veto_ray_observations=0, original_guard_unchanged=False,
        ncc_source_footprint_checks=footprint, full_frame_quality_metrics=False,
        effective_algorithm_request=str(ROOT / 'qualification_request.json')))
    ps = subprocess.check_output(['ps', '-eo', 'pid,etime,args'], text=True)
    names = ['study_forearm_layer_qualified_guard.py', 'render_forearm_layer_qualified_guard.py', 'diagnose_forearm_stereo_ambiguity.py']
    live = [s for s in ps.splitlines() if any('python scripts/'+n in s for n in names) and '/bin/bash' not in s]
    assert not live
    atomic_json(ROOT / 'terminal_check.json', dict(utc=datetime.now(timezone.utc).isoformat(), live_owned_workers=live,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip(),
        free_bytes=shutil.disk_usage(ROOT).free))
    hashes = {str(p): sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
    hashes.update(q['script_hashes'])
    hashes.update(diag['source_hashes'])
    for n in names + [Path(__file__).name]:
        p = Path(__file__).resolve().with_name(n)
        hashes[str(p)] = sha(p)
    for p in [LOWER / FRAME / 'request.json', LOWER / 'foreground/request.json', LOWER / 'foreground/mesh.ply',
              LOWER / 'veto_diagnosis/result.json', Path(gq['source_mesh'])]:
        hashes[str(p)] = sha(p)
    hashes.update(q['source_rgb_receipt']['source_rgb_hashes'])
    for pattern in ['dec5_forearm_layer_qualified*.log', 'dec5_forearm_stereo_ambiguity.log']:
        for p in Path('/mnt/data').glob(pattern):
            if 'freeze' not in p.name:
                hashes[str(p)] = sha(p)
    atomic_json(ROOT / 'artifact_manifest.json', dict(hashes=hashes, production_updated=False, full_video_approved=False))
    print('Frozen', len(hashes), 'hashes; original final veto observations', sum(c['original_trusted'] for c in last), flush=True)


def verify():
    hashes = read(ROOT / 'artifact_manifest.json')['hashes']
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Changed artifact: '+path)
    print('Rechecked', len(hashes), 'hashes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        verify()
    elif (ROOT / 'artifact_manifest.json').exists():
        raise ValueError('Already frozen; use --check')
    else:
        freeze()
