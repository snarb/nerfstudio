#!/usr/bin/env python3
"""Create an experimental scalar-only exposure variant; never overwrites inputs.

Training gains come from train tracks. Held-out gains are predicted from camera
positions alone. Camera-correlated reflectance remains a calibration confounder.
"""
import argparse
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


def encode(linear):
    x = np.maximum(linear, 0)
    x = x/(1+x)
    return np.rint(np.clip(np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055), 0, 1)*255).astype(np.uint8)


def run(a):
    cv2.setNumThreads(2)
    a.output.mkdir(parents=True, exist_ok=False)
    review = json.loads(a.scalar_review.read_text())
    if review['uses_eval_RGB'] is not False or review['connected_camera_components'] != 1:
        raise ValueError('Expected connected, train-only scalar fit')
    meta = json.loads((a.source/'transforms.json').read_text())
    by = {f['physical_camera']: f for f in meta['frames']}
    train_names = {f['physical_camera'] for f in meta['frames'] if f['file_path'] in meta['train_filenames']}
    evs = {row['camera']: float(row['scalar_correction_EV']) for row in review['camera_rows']}
    if set(evs) != train_names or len(evs) != 63 or len(by) != 66:
        raise ValueError('Camera split mismatch')
    names = sorted(evs)
    origins = np.array([by[n]['transform_matrix'] for n in names])[:, :3, 3]
    targets = np.array([evs[n] for n in names])
    distances = np.linalg.norm(origins[:, None]-origins[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    candidates = []
    for k in [2, 3, 4, 6, 8]:
        for power in [1, 2, 3]:
            indices = np.argsort(distances, axis=1)[:, :k]
            weights = np.take_along_axis(distances, indices, axis=1)**(-power)
            weights /= weights.sum(axis=1, keepdims=True)
            prediction = (weights*targets[indices]).sum(axis=1)
            error = prediction-targets
            candidates.append(dict(k=k, power=power, camera_loo_rmse_ev=float(np.sqrt(np.mean(error**2))),
                                   median_abs_ev=float(np.median(abs(error))), p90_abs_ev=float(np.quantile(abs(error), .9))))
    selected = min(candidates, key=lambda x: x['camera_loo_rmse_ev'])
    if selected['camera_loo_rmse_ev'] > .2:
        raise ValueError('Pose-only exposure interpolation too uncertain')
    eval_predictions = {}
    for name in sorted(set(by)-train_names):
        position = np.array(by[name]['transform_matrix'])[:3, 3]
        dist = np.linalg.norm(origins-position, axis=1)
        ids = np.argsort(dist)[:selected['k']]
        weights = np.maximum(dist[ids], 1e-8)**(-selected['power'])
        weights /= weights.sum()
        evs[name] = float(weights@targets[ids])
        eval_predictions[name] = dict(ev=evs[name], neighbors=[names[i] for i in ids], weights=weights.tolist())
    rows = []
    sdr = a.output/'sdr_png'
    linear = a.output/'linear_rec709_exr'
    for out in [sdr, linear]:
        (out/'images').mkdir(parents=True)
        (out/'masks').symlink_to((a.source/'masks').resolve(), target_is_directory=True)
    for frame in meta['frames']:
        name = frame['physical_camera']
        source = a.source/frame['file_path']
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.float32 or image.shape != (1012, 1920, 3):
            raise ValueError(str(source))
        image = image*(2**evs[name])
        if not np.isfinite(image).all():
            raise ValueError('Nonfinite corrected image')
        relative = Path(frame['file_path'])
        exr_out = linear/relative
        png_relative = relative.with_suffix('.png')
        if not cv2.imwrite(str(exr_out), image):
            raise RuntimeError(str(exr_out))
        Image.fromarray(encode(image[..., ::-1])).save(sdr/png_relative)
        rows.append(dict(camera=name, exposure_correction_ev=evs[name], source=str(source),
                         source_sha256=sha(source), linear_file=str(relative), linear_sha256=sha(exr_out),
                         png_file=str(png_relative), png_sha256=sha(sdr/png_relative),
                         gain_source='train_track_fit' if name in train_names else 'pose_only_prediction_no_eval_RGB'))
        print('converted', name, evs[name], flush=True)
    (linear/'transforms.json').write_text((a.source/'transforms.json').read_text())
    png_meta = json.loads(json.dumps(meta))
    for f in png_meta['frames']:
        f['file_path'] = str(Path(f['file_path']).with_suffix('.png'))
    for key in ['train_filenames', 'val_filenames', 'test_filenames']:
        png_meta[key] = [str(Path(p).with_suffix('.png')) for p in png_meta[key]]
    (sdr/'transforms.json').write_text(json.dumps(png_meta, indent=2)+'\n')
    contract = dict(kind='EXPERIMENTAL_scalar_exposure_normalization', script_sha256=sha(__file__),
                    source_master=str(a.source), source_transforms_sha256=sha(a.source/'transforms.json'),
                    scalar_review=str(a.scalar_review), scalar_review_sha256=sha(a.scalar_review),
                    train=63, eval=3, white_balance='as-shot unchanged; scalar-only RGB multiplication',
                    global_base_gain=4, nits_per_scene_unit=203, exposure_baked=True,
                    eval_rgb_used_to_fit_gains=False, eval_gain_predictions=eval_predictions,
                    interpolation_selected=selected, interpolation_candidates=candidates,
                    display='linear -> Reinhard -> sRGB with no additional exposure',
                    caveat='Camera-correlated directional reflectance can confound exposure; this is a matched experiment, not ground-truth ISO calibration.',
                    rows=rows)
    for out, encoding in [(sdr, 'sRGB8display'), (linear, 'scene-linearfloat32')]:
        (out/'color_contract.json').write_text(json.dumps(dict(contract, encoding=encoding), indent=2)+'\n')
    (a.output/'complete.json').write_text(json.dumps(contract, indent=2)+'\n')
    sheet = Image.new('RGB', (960, 834));draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(['DC', 'HC', 'KA']):
        frame = by[name]
        raw = cv2.imread(str(a.source/frame['file_path']), cv2.IMREAD_UNCHANGED)[..., ::-1]
        for j, value in enumerate([raw, raw*(2**evs[name])]):
            tile = Image.fromarray(encode(value));tile.thumbnail((480, 253));sheet.paste(tile, (j*480, i*278+24))
        draw.text((3, i*278+3), f'{name} gain4 base | experimental scalar EV {evs[name]:+.3f}', fill='white')
    sheet.save(a.output/'source_comparison.jpg', quality=93)
    print(json.dumps(dict(output=str(a.output), images=len(rows), selected=selected)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--scalar-review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
