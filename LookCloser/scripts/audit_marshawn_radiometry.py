#!/usr/bin/env python3
"""Read-only train-track radiometry screen, not an automatic color calibration.

Fits robust point RGB + camera log-gain on disjoint train-only tracks. View-
dependent reflectance and reprojection/patch errors remain confounders.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def centered_residual(log_rgb, points, cameras, gains):
    adjusted = log_rgb - gains[cameras]
    count = np.bincount(points, minlength=int(points.max())+1)
    means = np.stack([np.bincount(points, weights=adjusted[:, k], minlength=len(count))
                      / np.maximum(count, 1) for k in range(3)], axis=1)
    return adjusted-means[points]


def fit(log_rgb, points, cameras, ncam, iterations=300):
    npoint = int(points.max())+1
    gains = np.zeros((ncam, 3), np.float64)
    weights = np.ones(len(points))
    point_rgb = np.zeros((npoint, 3))
    for iteration in range(iterations):
        previous = gains.copy()
        denom = np.maximum(np.bincount(points, weights=weights, minlength=npoint), 1e-20)
        for k in range(3):
            point_rgb[:, k] = np.bincount(points, weights=weights*(log_rgb[:, k]-gains[cameras, k]), minlength=npoint)/denom
        denom_cam = np.maximum(np.bincount(cameras, weights=weights, minlength=ncam), 1e-20)
        for k in range(3):
            gains[:, k] = np.bincount(cameras, weights=weights*(log_rgb[:, k]-point_rgb[points, k]), minlength=ncam)/denom_cam
        gains -= gains.mean(axis=0, keepdims=True)
        residual = log_rgb - point_rgb[points] - gains[cameras]
        # One RGB-vector weight per observation rejects specular/edge outliers.
        magnitude = np.sqrt(np.mean(residual**2, axis=1))
        weights = np.minimum(1., .15/np.maximum(magnitude, 1e-12))
        if iteration > 30 and np.max(abs(gains-previous)) < 1e-7:
            break
    return gains, dict(iterations=iteration+1, camera_weight=denom_cam.tolist())


def score(log_rgb, points, cameras, gains):
    residual = centered_residual(log_rgb, points, cameras, gains)
    luma = residual @ np.array([.2126, .7152, .0722])
    return dict(median_abs_log_rgb=float(np.median(abs(residual))),
                rms_log_rgb=float(np.sqrt(np.mean(residual**2))),
                median_abs_luminance_ev=float(np.median(abs(luma))/np.log(2)),
                p90_abs_luminance_ev=float(np.quantile(abs(luma), .9)/np.log(2)))


def display(x):
    x = np.maximum(x, 0)
    x = x/(1+x)
    return np.rint(np.clip(np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055), 0, 1)*255).astype(np.uint8)


def run(args):
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.data/'transforms.json').read_text())
    train = set(meta['train_filenames'])
    frames = {f['physical_camera']: f for f in meta['frames'] if f['file_path'] in train}
    with np.load(args.observations, allow_pickle=False) as obs:
        points, cameras, uv, names = [obs[k].copy() for k in ['point_indices', 'camera_indices', 'uv', 'camera_names']]
    assert set(names) == set(frames) and len(names) == 63
    colors = np.zeros((len(uv), 3), np.float64)
    contrast = np.zeros(len(uv), np.float64)
    bindings = {}
    for c, name in enumerate(names):
        path = args.data/frames[str(name)]['file_path']
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.float32 or image.shape != (1012, 1920, 3):
            raise ValueError(str(path))
        image = image[..., ::-1].copy()
        mean = cv2.blur(image, (5, 5))
        variance = np.maximum(cv2.blur(image*image, (5, 5))-mean*mean, 0)
        selected = np.flatnonzero(cameras == c)
        mx = (uv[selected, 0]-.5).astype(np.float32)[:, None]
        my = (uv[selected, 1]-.5).astype(np.float32)[:, None]
        colors[selected] = cv2.remap(mean, mx, my, cv2.INTER_LINEAR)[:, 0]
        std = cv2.remap(np.sqrt(variance), mx, my, cv2.INTER_LINEAR)[:, 0]
        contrast[selected] = std.mean(axis=1)/np.maximum(colors[selected].mean(axis=1), 1e-6)
        bindings[str(name)] = sha(path)
        print('sampled', str(name), len(selected), flush=True)
    np.savez_compressed(args.output/'samples.npz', rgb=colors, contrast=contrast,
                        point_indices=points, camera_indices=cameras, uv=uv, camera_names=names)
    finite = (np.isfinite(colors).all(axis=1) & (colors.min(axis=1) > .002)
              & (colors.max(axis=1) < 5) & (uv.min(axis=1) >= 4)
              & (uv[:, 0] <= 1916) & (uv[:, 1] <= 1008))
    results = {}
    gain_candidates = {}
    for mode, extra in [('all_valid', np.ones(len(uv), bool)), ('lower_patch_contrast', contrast < .5)]:
        keep = finite & extra
        counts = np.bincount(points[keep], minlength=int(points.max())+1)
        keep &= counts[points] >= 3
        ids = np.flatnonzero(keep)
        p, c, logs = points[ids], cameras[ids], np.log(colors[ids])
        if len(ids) < 1000 or np.bincount(c, minlength=len(names)).min() < 20:
            results[mode] = dict(status='insufficient_stable_tracks', observations=len(ids))
            continue
        unique = np.unique(p)
        rng = np.random.default_rng(20261105)
        group = np.zeros(int(p.max())+1, np.int64)
        group[rng.permutation(unique)] = np.arange(len(unique)) % 5
        fold_gains, folds = [], []
        for fold in range(5):
            heldout = group[p] == fold
            gains, diagnostics = fit(logs[~heldout], p[~heldout], c[~heldout], len(names))
            fold_gains.append(gains)
            folds.append(dict(fold=fold, train_tracks=len(np.unique(p[~heldout])),
                              heldout_tracks=len(np.unique(p[heldout])),
                              before=score(logs[heldout], p[heldout], c[heldout], np.zeros_like(gains)),
                              after=score(logs[heldout], p[heldout], c[heldout], gains), fit=diagnostics))
        gains, diag = fit(logs, p, c, len(names))
        exposure = -(gains @ [.2126, .7152, .0722])/np.log(2)
        fold_std = np.std(np.stack(fold_gains), axis=0)
        rows = []
        for i, name in enumerate(names):
            correction = np.exp(-gains[i])
            rows.append(dict(camera=str(name), observations=int((c == i).sum()),
                             exposure_correction_ev=float(exposure[i]),
                             correction_rgb=correction.tolist(), r_over_g=float(correction[0]/correction[1]),
                             b_over_g=float(correction[2]/correction[1]),
                             fold_log_gain_std=fold_std[i].tolist()))
        results[mode] = dict(status='diagnostic_fit_only_not_applied', observations=len(ids),
                             tracks=len(unique), camera_rows=rows, folds=folds,
                             full_before=score(logs, p, c, np.zeros_like(gains)),
                             full_after=score(logs, p, c, gains), fit=diag,
                             exposure_ev_range=[float(exposure.min()), float(exposure.max())])
        gain_candidates[mode] = gains
    report = dict(kind='read_only_train_track_radiometric_consistency_screen', uses_eval_rgb=False,
                  source_transforms_sha256=sha(args.data/'transforms.json'), observations_sha256=sha(args.observations),
                  source_images_sha256=bindings, patch='5x5 linear mean at projected COLMAP pixel center minus0.5',
                  model='logRGB=pointRGB+cameraRGB; Huber IRLS; zero-mean camera log-gain gauge',
                  validation='five disjoint TRACK folds from training cameras only', results=results,
                  limitations=['Reflectance directionality, reprojection and spatial patch footprints confound camera gains.',
                               'Relative radiometry only, no absolute white balance or exposure standard.',
                               'No production images, training configuration or calibration were changed.'])
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    # Three independent diffuse-looking track contacts, diagnostic correction only.
    if 'lower_patch_contrast' in gain_candidates:
        gains = gain_candidates['lower_patch_contrast']
        candidate = finite & (contrast < .3)
        counts = np.bincount(points[candidate], minlength=int(points.max())+1)
        selected_points = np.flatnonzero(counts >= 4)
        if len(selected_points):
            chosen = selected_points[np.linspace(0, len(selected_points)-1, min(3, len(selected_points))).astype(int)]
            canvas = Image.new('RGB', (768, len(chosen)*200), 'black')
            draw = ImageDraw.Draw(canvas)
            for row, point in enumerate(chosen):
                indices = np.flatnonzero(candidate & (points == point))
                indices = indices[np.linspace(0, len(indices)-1, 4).astype(int)]
                for column, i in enumerate(indices):
                    c = int(cameras[i]); image = cv2.imread(str(args.data/frames[str(names[c])]['file_path']), cv2.IMREAD_UNCHANGED)[..., ::-1]
                    patch = cv2.getRectSubPix(image, (33, 33), tuple((uv[i]-.5).astype(float)))
                    for j, value in enumerate([patch, patch*np.exp(-gains[c])]):
                        tile = Image.fromarray(display(value)).resize((96, 144))
                        canvas.paste(tile, (column*192+j*96, row*200+38))
                    draw.text((column*192, row*200+3), f'{names[c]} point{point}', fill='white')
                    draw.text((column*192, row*200+19), 'raw | diagnostic gain', fill='white')
            canvas.save(args.output/'three_train_track_contacts.jpg', quality=94)
    print(json.dumps({k: {x: v[x] for x in ['status', 'observations', 'exposure_ev_range', 'full_before', 'full_after'] if x in v}
                      for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
