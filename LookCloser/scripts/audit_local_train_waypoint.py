"""Independent canary inventory, pose, render-hash and supervision checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy.spatial import ConvexHull
from joint_temporal_texture import read, sha, atomic_json, cameras, CALIBRATION
from study_local_train_waypoint import OUTPUT, PARENT, FRAMES, PHYSICAL


def check():
    ps = subprocess.check_output(['ps', '-eo', 'pid,etime,pcpu,rss,args'], text=True)
    processes = [s.strip() for s in ps.splitlines()
                 if 'python scripts/study_local_train_waypoint.py render' in s and '/bin/bash' not in s]
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader'], text=True).strip()
    worker_logs = sorted(Path('/mnt/data').glob('dec5_local_train_waypoint_worker*.log'))
    errors = {}
    for path in worker_logs:
        lines = path.read_text().splitlines()
        flagged = [s for s in lines if any(t in s for t in ['Traceback', 'CUDA error', 'OutOfMemoryError'])]
        if flagged: errors[str(path)] = flagged
    record = dict(utc=datetime.now(timezone.utc).isoformat(), processes=processes, gpu=gpu,
                  free_bytes=shutil.disk_usage(OUTPUT).free, errors=errors,
                  completed_frames=sorted(p.parent.name for p in (OUTPUT / 'frames').glob('*/complete.json')),
                  workers=[read(p) for p in sorted((OUTPUT / 'workers').glob('*.json'))])
    with (OUTPUT / 'checks.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
    print(record, flush=True)


def geometry_audit():
    from render_smooth_temporal_mesh_video import verify_request, calibration_pose
    q = verify_request(OUTPUT); p = verify_request(PARENT); cal = read(CALIBRATION)
    assert q['ordered_frame_ids'] == p['ordered_frame_ids']
    assert len(q['inventory']) == 150 and len(set(q['ordered_frame_ids'])) == 150
    assert q['recipe']['static_registration'] is False and q['recipe']['averages_rgb'] is False
    assert not q['uses_heldout_rgb']
    poses = []; oldposes = []
    for old, new in zip(p['inventory'], q['inventory']):
        for key in set(old) - {'camera'}: assert old[key] == new[key], key
        for key in ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'k1', 'k2', 'p1', 'p2']:
            assert old['camera'].get(key) == new['camera'].get(key)
        poses.append(np.array(calibration_pose(new['camera'], cal, read(new['metadata']))['transform_matrix']))
        oldposes.append(np.array(calibration_pose(old['camera'], cal, read(old['metadata']))['transform_matrix']))
    rows, _, meta = cameras('001193')
    centers = np.array([calibration_pose(r, cal, read(meta))['transform_matrix'] for r in rows])[:, :3, 3]
    target = next(r for r in rows if r['physical_camera'] == PHYSICAL)
    np.testing.assert_allclose(q['inventory'][147]['camera']['transform_matrix'], target['transform_matrix'], atol=2e-7)
    hull = ConvexHull(centers); equations = hull.equations
    residual = np.array(poses)[:, :3, 3] @ equations[:, :3].T + equations[:, 3]
    oldresidual = np.array(oldposes)[:, :3, 3] @ equations[:, :3].T + equations[:, 3]
    result = dict(request_sha256=sha(OUTPUT / 'request.json'), actual_times_unchanged=True,
        mesh_inventory_unchanged=True, intrinsics_unchanged=True, exact_waypoint_within_float32_tolerance=True,
        full_train_center_hull_check_not_two_row_clearance=True,
        original_max_outside_hull=float(oldresidual.max()), candidate_max_outside_hull=float(residual.max()),
        candidate_outside_hull_indices=np.flatnonzero(residual.max(1) > 1e-6).tolist(),
        hull_coordinate_system='fixed calibration', full_video_approved=False,
        script_sha256=sha(__file__))
    atomic_json(OUTPUT / 'geometry_audit.json', result); print(result, flush=True)


def artifacts():
    from review_jaw_repair_transfer import verified_image
    from render_smooth_temporal_mesh_video import verify_request
    request = verify_request(OUTPUT)
    assert sorted(p.parent.name for p in (OUTPUT / 'frames').glob('*/complete.json')) == sorted(FRAMES)
    for frame in FRAMES:
        image, result = verified_image(OUTPUT, frame)
        assert image.shape == (1920, 1080, 3) and np.isfinite(image).all()
        assert result['frame_id'] == frame and result['source_time_frame_count'] == 1
        assert sha(result['mesh_path']) == result['mesh_sha256']
        assert len(result['source_cameras']) == 62
    review = read(OUTPUT / 'visual_review.json')
    assert sorted(r['frame'] for r in review['records']) == sorted(FRAMES)
    assert all(r['status'] in ['pass', 'fail', 'uncertain'] for r in review['records'])
    for row in review['inspected_images']:
        assert sha(row['path']) == row['sha256']
    hashes = {str(p): sha(p) for p in sorted(OUTPUT.rglob('*'))
              if p.is_file() and p.name not in ['artifact_manifest.json', 'checks.jsonl'] and 'workers' not in p.parts}
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    for name in request['script_hashes']:
        path = Path(__file__).resolve().with_name(name); hashes[str(path)] = sha(path)
    for name in ['review_local_train_waypoint.py']:
        path = Path(__file__).resolve().with_name(name); hashes[str(path)] = sha(path)
    for row in request['inventory']:
        if row['frame_id'] not in FRAMES: continue
        for key in ['mesh', 'metadata']: hashes[row[key]] = sha(row[key])
    manifest = dict(hashes=hashes, frames=FRAMES,
        full_video_approved=False, geometry_improvement=False, full_frame_metrics=False)
    path = OUTPUT / 'artifact_manifest.json'
    if path.exists() and read(path) != manifest: raise ValueError('Frozen artifact inventory changed')
    atomic_json(path, manifest)
    print('Verified', len(hashes), 'retained hashes', flush=True)


def verify():
    manifest = read(OUTPUT / 'artifact_manifest.json')
    for path, digest in manifest['hashes'].items():
        if sha(path) != digest: raise ValueError('Changed frozen artifact: ' + path)
    print('Rechecked', len(manifest['hashes']), 'hashes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'geometry', 'artifacts', 'verify'])
    args = parser.parse_args()
    {'check': check, 'geometry': geometry_audit, 'artifacts': artifacts, 'verify': verify}[args.action]()
