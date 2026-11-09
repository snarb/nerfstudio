#!/usr/bin/env python3
"""Train-only RoMa correspondences, independent cycles and calibrated triangulation.

Cycles are qualified without poses before known-pose reprojection is measured.
No MVS depth is consulted to generate or qualify the new 3D points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from dec5_hair import VALIDATION, TEST, cv_pose, project
from audit_dec5_calibration import fundamental, distances
from run_dec5_hair_campaign import atomic_json, sha


def crop_to_camera(uv, frame, crop):
    return np.stack((frame['w']-uv[..., 1]-crop[1], uv[..., 0]+crop[0]), -1)


def sample_normalized(array, uv):
    """Continuous align_corners=False coordinates, zero outside the image."""
    h, w = array.shape[:2]
    x = ((uv[..., 0]+1)*w/2-.5).astype(np.float32)
    y = ((uv[..., 1]+1)*h/2-.5).astype(np.float32)
    if len(x) == 0:
        return np.empty((0,)+array.shape[2:], dtype=array.dtype)
    # OpenCV's remap destination axes are limited to signed 16-bit lengths.
    return np.concatenate([cv2.remap(array, x[i:i+16384, None], y[i:i+16384, None],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[:, 0]
        for i in range(0, len(x), 16384)])


def triangulate(uv_a, uv_b, a, b):
    """Closest calibrated rays, returning midpoint and parallax in degrees."""
    rays = []
    poses = [cv_pose(a), cv_pose(b)]
    for uv, f, p in zip((uv_a, uv_b), (a, b), poses):
        ray = np.column_stack(((uv[:, 0]-f['cx'])/f['fl_x'],
                               (uv[:, 1]-f['cy'])/f['fl_y'], np.ones(len(uv)))) @ p[:3, :3].T
        rays.append(ray/np.linalg.norm(ray, axis=-1, keepdims=True))
    u, v = rays
    delta = poses[1][:3, 3]-poses[0][:3, 3]
    cosine = (u*v).sum(-1)
    denominator = np.maximum(1-cosine*cosine, 1e-12)
    along_u = (u@delta-cosine*(v@delta))/denominator
    along_v = (cosine*(u@delta)-v@delta)/denominator
    points = .5*(poses[0][:3, 3]+along_u[:, None]*u + poses[1][:3, 3]+along_v[:, None]*v)
    return points, np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def match(root, output, names):
    import torch
    from romatch import roma_outdoor
    output.mkdir(parents=True, exist_ok=False)
    payload = json.loads((root/'dataset/transforms.json').read_text())
    by = {f['physical_camera']: f for f in payload['frames'] if f['file_path'] in payload['train_filenames']}
    if len(by) != 59 or any(n in VALIDATION+TEST or n not in by for n in names):
        raise ValueError('Correspondences require the frozen train59 split')
    rows = []
    for index, name in enumerate(names):
        frame = by[name]
        crop = np.load(root/'matting_v3'/(name+'.npz'))['portrait_crop'].astype(int)
        x0, y0, x1, y1 = crop
        image = np.rot90(cv2.imread(str(root/'dataset'/frame['file_path'])))[y0:y1, x0:x1].copy()
        path = output/('%d.png'%index)
        cv2.imwrite(str(path), image)
        rows.append(dict(frame=frame, crop=crop.tolist(), shape=list(image.shape[:2]), image_sha256=sha(path)))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(12*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
    torch.set_float32_matmul_precision('highest')
    model = roma_outdoor(device='cuda', use_custom_corr=False, upsample_res=864)
    receipts = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        start = time.monotonic()
        with torch.inference_mode():
            warp, certainty = model.match(str(output/('%d.png'%i)), str(output/('%d.png'%j)), device='cuda')
        dest = output/('%d_%d.npz'%(i, j))
        np.savez_compressed(dest, warp=warp.float().cpu().numpy().squeeze(0),
                            certainty=certainty.float().cpu().numpy().squeeze(0))
        receipts.append(dict(pair=[i, j], seconds=time.monotonic()-start, sha256=sha(dest)))
        print('dense pair %d-%d complete %.1fs'%(i, j, time.monotonic()-start), flush=True)
    atomic_json(output/'matches.json', dict(rows=rows, pairs=receipts,
        peak_vram_bytes=torch.cuda.max_memory_allocated(), uses_validation_rgb=False, uses_test_rgb=False,
        implementation_sha256=sha(__file__)))


def semantic(root, row, normalized):
    h, w = row['shape']
    crop_uv = (normalized+1)*np.array([w, h])/2
    uv = crop_to_camera(crop_uv, row['frame'], row['crop'])
    f = row['frame']
    sem = np.load(root/'semantics'/(f['physical_camera']+'.npz'))['probability'].astype(np.float32)/65535
    return uv, sample_normalized(sem, uv/np.array([f['w'], f['h']])*2-1)


def analyze(root, output, calibration=None, analysis_output=None):
    receipt = json.loads((output/'matches.json').read_text())
    rows = receipt['rows']
    destination = analysis_output or output
    if analysis_output is not None: destination.mkdir(parents=True, exist_ok=False)
    if calibration is not None:
        payload = json.loads(calibration.read_text())
        train = {f['physical_camera']:f for f in payload['frames'] if f['file_path'] in payload['train_filenames']}
        if len(train) != 59 or any(n in train for n in VALIDATION+TEST): raise ValueError('Calibration must retain train59')
        for row in rows: row['frame'] = train[row['frame']['physical_camera']]
    pairs = {key: np.load(output/('%d_%d.npz'%key)) for key in ((0, 1), (0, 2), (1, 2))}
    ab = pairs[0, 1]['warp']; width = ab.shape[1]//2
    # Dense A->B samples, not the model's confidence-biased random sample.
    ab_forward = ab[:, :width][::2, ::2].reshape(-1, 4)
    a, b = ab_forward[:, :2], ab_forward[:, 2:]
    # RoMa keeps coordinate order (A, B) in both halves: the second half is
    # indexed in B, with B->A stored in its FIRST two channels.
    ba = sample_normalized(ab[:, width:, :2], b)
    c_a = sample_normalized(pairs[0, 2]['warp'][:, :width, 2:], a)
    c_b = sample_normalized(pairs[1, 2]['warp'][:, :width, 2:], b)
    ba_error = np.linalg.norm((ba-a)*np.array(rows[0]['shape'][::-1])/2, axis=-1)
    cycle_error = np.linalg.norm((c_a-c_b)*np.array(rows[2]['shape'][::-1])/2, axis=-1)
    confidence = np.minimum.reduce([
        pairs[0, 1]['certainty'][:, :width][::2, ::2].ravel(),
        sample_normalized(pairs[0, 2]['certainty'][:, :width], a),
        sample_normalized(pairs[1, 2]['certainty'][:, :width], b)])
    normalized = [a, b, .5*(c_a+c_b)]
    uv, sem = zip(*(semantic(root, row, xy) for row, xy in zip(rows, normalized)))
    valid = np.isfinite(ab_forward).all(-1) & (confidence > .5)
    for xy in [a, b, ba, c_a, c_b]:
        valid &= (np.abs(xy) < 1).all(-1)
    independent = valid & (ba_error < 1.5) & (cycle_error < 1.5)
    epipolar = distances(uv[0], uv[1], fundamental(rows[0]['frame'], rows[1]['frame']))
    points, parallax = triangulate(uv[0], uv[1], rows[0]['frame'], rows[1]['frame'])
    errors, depths = [], []
    for f, measured in zip((row['frame'] for row in rows), uv):
        predicted, z = project(points, f)
        errors.append(np.linalg.norm(predicted-measured, axis=-1)); depths.append(z)
    reprojection = np.max(errors, axis=0)
    positive = (np.stack(depths) > 0).all(0)
    hair = np.logical_and.reduce([s[:, 0] > .5 for s in sem])
    face = np.logical_and.reduce([s[:, 1] > .5 for s in sem])
    bg = np.logical_and.reduce([s[:, 2] > .9 for s in sem])
    np.savez_compressed(destination/'cycle_correspondences.npz',
        uv_a=uv[0][independent].astype(np.float32), uv_b=uv[1][independent].astype(np.float32),
        uv_c=uv[2][independent].astype(np.float32), hair=hair[independent], face=face[independent],
        background=bg[independent], cycle_error_px=cycle_error[independent].astype(np.float32))
    def summary(values, mask):
        return dict(count=int(mask.sum()), quantiles_10_50_90_95=np.quantile(values[mask], [.1,.5,.9,.95]).tolist() if mask.any() else None)
    metrics = {}
    for name, region in [('all', np.ones(len(a), bool)), ('hair', hair), ('face', face), ('background', bg)]:
        selected = region & independent
        metrics[name] = dict(confident_matches=int((region & valid).sum()), independent_cycle_matches=int(selected.sum()),
            epipolar_px=summary(epipolar, selected), reprojection_max_px=summary(reprojection, selected),
            parallax_degrees=summary(parallax, selected))
    accepted = independent & hair & positive & (parallax > 1) & (reprojection < 2)
    np.savez_compressed(destination/'triangulated.npz', points=points[accepted].astype(np.float32),
        uv_a=uv[0][accepted].astype(np.float32), uv_b=uv[1][accepted].astype(np.float32), uv_c=uv[2][accepted].astype(np.float32),
        reprojection_max_px=reprojection[accepted].astype(np.float32),
        parallax_degrees=parallax[accepted].astype(np.float32))
    atomic_json(destination/'metrics.json', dict(regions=metrics, accepted_hair_points=int(accepted.sum()),
        gates=dict(confidence=.5, forward_backward_px=1.5, independent_three_view_cycle_px=1.5,
                   parallax_degrees=1, maximum_three_view_reprojection_px=2),
        uses_mvs_depth=False, camera_poses_changed=calibration is not None, uses_validation_rgb=False, uses_test_rgb=False,
        calibration_sha256=sha(calibration) if calibration is not None else None,
        matching_sha256=sha(output/'matches.json'), implementation_sha256=sha(__file__)))
    print(json.dumps(dict(regions=metrics, accepted_hair_points=int(accepted.sum()))))


def sift_crosscheck(root, output):
    """Independent descriptor/feature-ID cycles on the identical train crops."""
    rows = json.loads((output/'matches.json').read_text())['rows']
    detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=.01)
    keys, descriptors = [], []
    for i in range(3):
        key, descriptor = detector.detectAndCompute(cv2.imread(str(output/('%d.png'%i)), 0), None)
        keys.append(np.asarray([p.pt for p in key], np.float32)+.5)
        descriptors.append(descriptor)
    matcher = cv2.BFMatcher(); maps = {}
    for i,j in ((0,1),(0,2),(1,2)):
        forward = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(descriptors[i],descriptors[j],k=2)
                   if m.distance < .75*n.distance}
        reverse = {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(descriptors[j],descriptors[i],k=2)
                   if m.distance < .75*n.distance}
        maps[i,j] = {a:b for a,b in forward.items() if reverse.get(b) == a}
    tracks = [(a,b,maps[0,2][a]) for a,b in maps[0,1].items()
              if a in maps[0,2] and b in maps[1,2] and maps[0,2][a] == maps[1,2][b]]
    if not tracks: raise ValueError('No independent SIFT cycles')
    indices = np.asarray(tracks); uv, sem = [], []
    for i,row in enumerate(rows):
        h,w = row['shape']
        p,s = semantic(root,row,keys[i][indices[:,i]]/np.array([w,h])*2-1)
        uv.append(p); sem.append(s)
    error = distances(uv[0],uv[1],fundamental(rows[0]['frame'],rows[1]['frame']))
    result = dict(features=[len(k) for k in keys], cycle_matches=len(tracks), uses_validation_rgb=False, uses_test_rgb=False)
    for label,index in (('hair',0),('face',1)):
        mask = np.logical_and.reduce([s[:,index] > .5 for s in sem])
        result[label] = dict(count=int(mask.sum()), epipolar_quantiles=np.quantile(error[mask],[.1,.5,.9,.95]).tolist() if mask.any() else [])
    np.savez_compressed(output/'sift_cycles.npz',uv_a=uv[0],uv_b=uv[1],uv_c=uv[2],sem=np.stack(sem))
    atomic_json(output/'sift_crosscheck.json',result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('match', 'analyze', 'sift'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cameras', nargs=3, default=['F004_C005_121059', 'G004_C005_121037', 'E004_C005_1210YM'])
    p.add_argument('--calibration', type=Path)
    p.add_argument('--analysis-output', type=Path)
    args = p.parse_args()
    if args.calibration is not None and (args.stage != 'analyze' or args.analysis_output is None):
        p.error('A calibration screen requires analyze plus a new --analysis-output directory')
    if args.stage == 'match': match(args.root, args.output, args.cameras)
    elif args.stage == 'sift': sift_crosscheck(args.root,args.output)
    else: analyze(args.root, args.output, args.calibration, args.analysis_output)


if __name__ == '__main__': main()
