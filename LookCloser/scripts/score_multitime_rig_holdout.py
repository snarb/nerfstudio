#!/usr/bin/env python3
"""Compare candidate common rigs on unchanged held-time train correspondences.

No camera or point optimization. Pair matches were selected before candidates;
no candidate inlier mask or surface mask can remove a held observation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from audit_source_epipolar_residuals import fundamental
from audit_spatial_temporal_residuals import held_block_summary


def symmetric_epipolar_error(matrix, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ah, bh = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    la, lb = bh @ matrix, ah @ matrix.T
    da, db = np.linalg.norm(la[:, :2], axis=1), np.linalg.norm(lb[:, :2], axis=1)
    if np.minimum(da, db).min() <= 1e-15:
        raise ValueError('Degenerate epipolar line')
    numerator = np.sum(bh * lb, axis=1)
    error = abs(numerator) * np.sqrt((da ** -2 + db ** -2) / 2)
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite epipolar error')
    return error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--held-tracks', type=Path, required=True)
    p.add_argument('--calibrations', nargs='+', type=Path, required=True)
    p.add_argument('--fit-frames', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve existing held evaluation')
    dataset = json.loads(a.held_tracks.read_text())
    if set(dataset['frames']) & set(a.fit_frames):
        raise ValueError('Fit and held temporal frames overlap')
    if any(dataset[k] for k in ['uses_eval_rgb', 'uses_semantic_masks', 'cross_time_matching', 'changes_calibration']):
        raise ValueError('Only frozen train correspondences allowed')
    names = dataset['physical_cameras']
    hashes = {str(a.held_tracks): sha256(a.held_tracks), str(Path(__file__)): sha256(Path(__file__))}
    request = Path(dataset['request'])
    if sha256(request) != dataset['request_sha256']:
        raise ValueError('Held request changed')
    hashes[str(request)] = sha256(request)
    for path, digest in json.loads(request.read_text())['input_hashes'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed held source')
        hashes[path] = digest
    calibrations = []
    for path in a.calibrations:
        payload = json.loads(path.read_text())
        cameras = {f['physical_camera']: {**payload, **f} for f in payload['frames']}
        if len(cameras) != len(payload['frames']) or not set(names) <= set(cameras):
            raise ValueError('Incomplete calibration inventory')
        hashes[str(path)] = sha256(path)
        calibrations.append((str(path), cameras))
    baseline_cameras = calibrations[0][1]
    forbidden = {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'}
    if len(names) != 62 or set(names) & forbidden:
        raise ValueError('Held geometry comparison uses only train cameras')
    results = {path: [] for path, _ in calibrations}
    for time in dataset['frames']:
        features = []
        for i in range(62):
            path = a.held_tracks.parent / 'features' / f'{time}_{i:02d}.npz'
            receipt_path = path.with_suffix('.json')
            receipt = json.loads(receipt_path.read_text())
            if sha256(path) != receipt['feature_sha256']:
                raise ValueError('Changed held feature array')
            hashes[str(path)] = receipt['feature_sha256']
            hashes[str(receipt_path)] = sha256(receipt_path)
            features.append(np.load(path, allow_pickle=False)['xy'])
        time_path = a.held_tracks.parent / f'tracks_{time}.json'
        hashes[str(time_path)] = sha256(time_path)
        pairs = json.loads(time_path.read_text())['pairs']
        for pair in pairs:
            path = Path(pair['path'])
            if sha256(path) != pair['sha256']:
                raise ValueError('Changed held match array')
            hashes[str(path)] = pair['sha256']
            matches = np.load(path, allow_pickle=False)['matches']
            if not len(matches):
                continue
            left, right = pair['cameras']
            pa, pb = features[left][matches[:, 0].astype(int)], features[right][matches[:, 1].astype(int)]
            groups = np.floor(pa / 128).astype(int).tolist()
            for label, cameras in calibrations:
                matrix = fundamental(cameras[names[left]], cameras[names[right]], .5)
                error = symmetric_epipolar_error(matrix, pa, pb)
                result = held_block_summary(error, groups)
                result.pop('blocks')
                results[label].append(dict(frame_id=time, left_camera=names[left], right_camera=names[right], **result))
    summaries = []
    baseline = results[calibrations[0][0]]
    baseline_errors = np.asarray([r['block_median_absolute_error'] for r in baseline])
    for label, _ in calibrations:
        rows = results[label]
        errors = np.asarray([r['block_median_absolute_error'] for r in rows])
        if len(rows) != len(baseline):
            raise ValueError('Candidate changed held inventory')
        camera_rows = []
        for name in names:
            selected = np.array([name in [r['left_camera'], r['right_camera']] for r in rows])
            camera_rows.append(dict(physical_camera=name, pairs=int(selected.sum()),
                median=float(np.median(errors[selected])), baseline=float(np.median(baseline_errors[selected]))))
        row = dict(calibration=label, camera_time_pairs=len(rows), matched_observations=sum(r['points'] for r in rows),
            pair_block_median=float(np.median(errors)), pair_block_p90=float(np.quantile(errors, .9)),
            median_paired_change=float(np.median(errors - baseline_errors)), fraction_pairs_improved=float(np.mean(errors < baseline_errors)),
            camera_summary=camera_rows)
        summaries.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'camera_summary'}), flush=True)
    atomic_json(a.output, dict(method='fixed_held_train_pairs_symmetric_epipolar_block_error', uses_eval_rgb=False,
        uses_semantic_masks=False, changes_calibration=False, changes_prediction=False, fit_frames=a.fit_frames,
        held_frames=dataset['frames'], summaries=summaries, results=results, input_hashes=hashes,
        caveat='Calibration diagnostic, not PSNR/SSIM/LPIPS or a render visual pass. Matches are fixed across candidates; observations within a pair/block are correlated.'))


if __name__ == '__main__':
    main()
