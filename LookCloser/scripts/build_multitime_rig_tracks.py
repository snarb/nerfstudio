#!/usr/bin/env python3
"""Build time-separated train-only tracks for ONE shared physical camera rig.

No cross-time image matches, semantic masks, eval RGB, or pose changes. Temporary
features/edges are resumable only for the identical request and source hashes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256, validate_source_frame, calibration_by_physical_camera
from build_angular_camera_subset import camera_geometry
from audit_source_epipolar_residuals import projection


def unique_feature_indices(xy):
    return np.sort(np.unique(np.asarray(xy), axis=0, return_index=True)[1])


def camera_pairs(frames, neighbors):
    if not 1 <= neighbors < len(frames):
        raise ValueError('Invalid camera neighbor count')
    _, direction, _ = camera_geometry(frames)
    distance = 1 - direction @ direction.T
    np.fill_diagonal(distance, np.inf)
    return sorted({tuple(sorted((i, int(j)))) for i in range(len(frames))
                   for j in np.argsort(distance[i], kind='stable')[:neighbors]})


def merge_tracks(edges, minimum_views=3):
    """Edges are (camera, feature) pairs; reject unions with camera conflicts."""
    parent, members = {}, {}
    def find(node):
        if node not in parent:
            parent[node] = node
            members[node] = {node[0]: node}
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    conflicts = 0
    for left, right in edges:
        a, b = find(tuple(left)), find(tuple(right))
        if a == b:
            continue
        if members[a].keys() & members[b].keys():
            conflicts += 1
            continue
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parent[b] = a
        members[a].update(members.pop(b))
    tracks = [sorted(m.values()) for m in members.values() if len(m) >= minimum_views]
    return sorted(tracks), conflicts


def triangulate_track(observations, projections, *, max_error=8., min_angle=.5):
    """DLT seed in original world units; never fit a camera to seed points."""
    rows = []
    for camera, xy in observations:
        p = projections[camera]
        rows.extend([xy[0] * p[2] - p[0], xy[1] * p[2] - p[1]])
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-12:
        return None
    xyz = vt[-1, :3] / vt[-1, 3]
    if not np.isfinite(xyz).all():
        return None
    errors, rays = [], []
    for camera, xy in observations:
        p = projections[camera]
        q = p @ np.r_[xyz, 1.]
        if q[2] <= 0:
            return None
        errors.append(np.linalg.norm(q[:2] / q[2] - xy))
        center = -np.linalg.solve(p[:, :3], p[:, 3])
        ray = xyz - center
        rays.append(ray / np.linalg.norm(ray))
    angle = np.degrees(np.arccos(np.clip(np.min(np.asarray(rays) @ np.asarray(rays).T), -1, 1)))
    if max(errors) > max_error or angle < min_angle:
        return None
    return xyz, float(np.median(errors)), float(angle)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main():
    from convert_exr_nerfstudio_to_jpeg import tone_map
    from nerfstudio.data.utils.data_utils import load_exr_image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--frames', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--neighbors', type=int, default=10)
    a = parser.parse_args()
    if len(a.frames) != len(set(a.frames)) or len(a.frames) < 2:
        parser.error('Need unique multiple source times')
    calibration = calibration_by_physical_camera(a.calibration)
    input_hashes = {str(a.calibration): sha256(a.calibration), str(Path(__file__)): sha256(Path(__file__))}
    sources, camera_names = [], None
    for time in a.frames:
        source = a.parent / time
        validate_source_frame(source)
        transforms = source / 'transforms.json'
        input_hashes[str(transforms)] = sha256(transforms)
        payload = json.loads(transforms.read_text())
        train = sorted([f for f in payload['frames'] if Path(f['file_path']).name.startswith('frame_train_')], key=lambda f: f['physical_camera'])
        names = [f['physical_camera'] for f in train]
        if camera_names is not None and names != camera_names:
            raise ValueError('Temporal train camera inventory changed')
        camera_names = names
        for i, f in enumerate(train):
            path = source / f['file_path']
            input_hashes[str(path)] = sha256(path)
            sources.append((time, i, path))
    frames = [calibration[n] for n in camera_names]
    if len(frames) != 62 or any(n in {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'} for n in camera_names):
        raise ValueError('Exactly 62 train cameras required')
    if any(any(f.get(k, 0) != 0 for k in ['k1', 'k2', 'k3', 'k4', 'p1', 'p2']) for f in frames):
        raise ValueError('This opt-in control currently requires undistorted pinhole input')
    for helper in ['convert_exr_nerfstudio_to_jpeg.py', 'build_angular_camera_subset.py', 'audit_source_epipolar_residuals.py']:
        path = Path(__file__).with_name(helper)
        input_hashes[str(path)] = sha256(path)
    pairs = camera_pairs(frames, a.neighbors)
    request = dict(frames=a.frames, physical_cameras=camera_names, pairs=pairs, input_hashes=input_hashes,
        uses_eval_rgb=False, uses_semantic_masks=False, cross_time_matching=False, changes_calibration=False,
        thresholds=dict(features=6000, sift_contrast=.01, sift_edge=12, ratio=.7, fundamental_ransac_pixels=1.5,
                        min_pair_inliers=20, min_track_views=3, seed_max_error=8., seed_min_angle=.5),
        display='per-image percentile70 exposure -> Reinhard -> sRGB; JPEG98 4:4:4, optimize=True')
    a.output.mkdir(parents=True, exist_ok=True)
    request_path = a.output / 'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != json.loads(json.dumps(request)):
            raise ValueError('Changed request on resume')
    else:
        atomic_json(request_path, request)
    (a.output / 'features').mkdir(exist_ok=True)
    (a.output / 'pairs').mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    def extract(source):
        time, camera, path = source
        output = a.output / 'features' / f'{time}_{camera:02d}.npz'
        receipt = output.with_suffix('.json')
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if sha256(output) != row['feature_sha256']:
                raise ValueError('Changed feature cache')
            return row
        rgb = load_exr_image(path)
        if rgb.shape != (1080, 1920, 3) or not np.isfinite(rgb).all():
            raise ValueError('Invalid finite source EXR')
        sample = np.maximum(rgb[::8, ::8, :3], 0)
        lum = sample @ np.array([.2126, .7152, .0722], np.float32)
        gain = .18 / (max(float(np.percentile(lum, 70)), 1e-8) * .82)
        memory = BytesIO()
        Image.fromarray(tone_map(rgb, gain)).save(memory, format='JPEG', quality=98, subsampling=0, optimize=True)
        gray = cv2.imdecode(np.frombuffer(memory.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.01, edgeThreshold=12)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            raise ValueError('No source features')
        xy = np.array([k.pt for k in keypoints], np.float32)
        keep = unique_feature_indices(xy)
        atomic_npz(output, xy=xy[keep], descriptors=descriptors[keep])
        row = dict(frame_id=time, camera_index=camera, physical_camera=camera_names[camera], source=str(path),
                   source_sha256=input_hashes[str(path)], gain=gain, features=len(keep), feature_path=str(output), feature_sha256=sha256(output))
        atomic_json(receipt, row)
        return row
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        receipts = []
        for row in pool.map(extract, sources):
            receipts.append(row)
            print(json.dumps(dict(stage='features', completed=len(receipts), total=len(sources), frame=row['frame_id'], camera=row['physical_camera'])), flush=True)
    projections = []
    for frame in frames:
        k, w = projection(frame, .5)
        projections.append(k @ w[:3])
    all_tracks, time_stats = [], []
    for time in a.frames:
        features = [dict(np.load(a.output / 'features' / f'{time}_{i:02d}.npz', allow_pickle=False)) for i in range(62)]
        def match(pair):
            left, right = pair
            output = a.output / 'pairs' / f'{time}_{left:02d}_{right:02d}.npz'
            receipt = output.with_suffix('.json')
            if receipt.exists():
                row = json.loads(receipt.read_text())
                if sha256(output) != row['sha256']:
                    raise ValueError('Changed pair cache')
                return dict(np.load(output, allow_pickle=False)), row
            cv2.setRNGSeed(left * 100 + right)
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=96))
            def accepted(x, y):
                return {m.queryIdx: (m.trainIdx, m.distance / max(n.distance, 1e-12))
                        for row in matcher.knnMatch(x, y, k=2) if len(row) == 2 for m, n in [row] if m.distance < .7 * n.distance}
            ab = accepted(features[left]['descriptors'], features[right]['descriptors'])
            ba = accepted(features[right]['descriptors'], features[left]['descriptors'])
            matches = [(i, j, ratio) for i, (j, ratio) in ab.items() if j in ba and ba[j][0] == i]
            matches = np.asarray(matches, float).reshape(-1, 3)
            if len(matches) >= 20:
                matrix, inlier = cv2.findFundamentalMat(features[left]['xy'][matches[:, 0].astype(int)], features[right]['xy'][matches[:, 1].astype(int)],
                    cv2.USAC_MAGSAC, 1.5, .999, 10000)
                matches = matches[inlier.ravel() > 0] if matrix is not None else matches[:0]
            else:
                matches = matches[:0]
            if len(matches) < 20:
                matches = matches[:0]
            values = dict(matches=matches)
            atomic_npz(output, **values)
            row = dict(frame_id=time, cameras=pair, matches=len(matches), path=str(output), sha256=sha256(output))
            atomic_json(receipt, row)
            return values, row
        edge_rows, pair_rows = [], []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for pair, (values, row) in zip(pairs, pool.map(match, pairs)):
                pair_rows.append(row)
                for i, j, ratio in values['matches']:
                    edge_rows.append((float(ratio), (pair[0], int(i)), (pair[1], int(j))))
                if len(pair_rows) % 25 == 0:
                    print(json.dumps(dict(stage='matching', frame=time, completed=len(pair_rows), total=len(pairs))), flush=True)
        tracks, conflicts = merge_tracks([(l, r) for _, l, r in sorted(edge_rows)])
        accepted_tracks = []
        for track in tracks:
            observations = [(camera, features[camera]['xy'][feature]) for camera, feature in track]
            seed = triangulate_track(observations, projections)
            if seed is not None:
                xyz, error, angle = seed
                accepted_tracks.append(dict(frame_id=time, xyz=xyz.tolist(), observations=[dict(camera=c, feature=f, xy=features[c]['xy'][f].tolist()) for c, f in track],
                                            seed_error_median=error, seed_angle_degrees=angle))
        counts = np.bincount([o['camera'] for t in accepted_tracks for o in t['observations']], minlength=62)
        stats = dict(frame_id=time, pair_inliers=len(edge_rows), union_conflicts=conflicts, tracks_before_seed=len(tracks), tracks=len(accepted_tracks), observations_per_camera=counts.tolist())
        all_tracks.extend(accepted_tracks)
        time_stats.append(stats)
        atomic_json(a.output / f'tracks_{time}.json', dict(tracks=accepted_tracks, stats=stats, pairs=pair_rows))
        print(json.dumps(dict(stage='tracks', **stats)), flush=True)
    if min(sum(np.array(s['observations_per_camera']) for s in time_stats)) < 30:
        raise ValueError('Insufficient joint rig support: some train camera has fewer than 30 observations')
    atomic_json(a.output / 'tracks.json', dict(request_sha256=sha256(request_path), request=str(request_path),
        uses_eval_rgb=False, uses_semantic_masks=False, changes_calibration=False, cross_time_matching=False,
        physical_cameras=camera_names, calibration=str(a.calibration), calibration_sha256=sha256(a.calibration),
        frames=a.frames, tracks=all_tracks, time_stats=time_stats, ingest_receipts=receipts))
    print(json.dumps(dict(stage='complete', tracks=len(all_tracks))), flush=True)


if __name__ == '__main__':
    main()
