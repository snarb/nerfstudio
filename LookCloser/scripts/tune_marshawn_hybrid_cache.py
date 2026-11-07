#!/usr/bin/env python3
"""Replay qualified native-source warps; never train or load a NeRF.

The input manifest binds a single cold model, a target crop with context, and
ordered independent donor warps. Target RGB participates only in measurement.
Warp admission and RAW photometry remain the qualified renderer's responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def pick(base, donors, count, pixel_scale, ambiguity=False):
    """Exactly reproduce legacy single-donor selection, plus optional tie fading.

    The experimental confidence fades only already accepted pixels. It never
    creates support, changes a donor, blends different photographs, or erodes
    the actor mask. An exact tie has confidence .5 rather than deleting hair.
    """
    shape = base.shape[:2]
    best = np.full(shape, np.inf, np.float32)
    second = best.copy()
    labels = np.full(shape, -1, np.int16)
    texture = base.copy()
    for donor in donors[:count]:
        valid, score = donor['valid'].astype(bool), donor['score']
        if valid.shape != shape or score.shape != shape or donor['rgb'].shape != base.shape:
            raise ValueError('Warp raster differs from target crop')
        if np.any(valid & ~np.isfinite(score)):
            raise ValueError('Accepted warp has nonfinite score')
        score = np.where(valid, score, np.inf)
        take = score < best
        second = np.minimum(second, np.where(take, best, score))
        best[take] = score[take]
        labels[take] = donor['camera_id']
        texture[take] = donor['rgb'][take]
    for label in np.unique(labels[labels >= 0]):
        mask = (labels == label).astype(np.uint8)
        kernel = 2 * max(1, round(pixel_scale)) + 1
        labels[(mask > 0) & (cv2.medianBlur(mask, kernel) == 0)] = -1
    confidence = (labels >= 0).astype(np.float32)
    if pixel_scale != 1:
        for label in np.unique(labels[labels >= 0]):
            mask = (labels == label).astype(np.uint8)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            factor = np.minimum(distance / (3 * pixel_scale), 1) / np.maximum(np.minimum(distance / 3, 1), 1e-8)
            confidence[mask > 0] *= factor[mask > 0]
    if ambiguity:
        pair = (labels >= 0) & np.isfinite(second)
        gap = np.ones(shape, np.float32)
        gap[pair] = np.clip((second[pair] - best[pair]) / .15, 0, 1)
        confidence *= .5 + .5 * gap
    return texture, labels, confidence


def configurations():
    # Keep a small, interpretable sweep. A winner needs separate visual review.
    variants = [dict(count=16, sigma=8., gain=1., ambiguity=False)]
    variants += [dict(count=16, sigma=s, gain=g, ambiguity=False)
                 for s, g in [(4., 1.), (12., 1.), (8., .75), (8., 1.25), (4., .75), (12., .75)]]
    variants += [dict(count=n, sigma=8., gain=1., ambiguity=False) for n in [8, 24]]
    variants += [dict(count=16, sigma=8., gain=g, ambiguity=True) for g in [1., .75]]
    return variants


class Measure:
    def __init__(self, device):
        import torch
        import lpips
        self.torch = torch
        self.device = device
        self.lpips = lpips.LPIPS(net='alex', spatial=True, verbose=False).to(device).eval()

    def __call__(self, pred, truth, actor, region):
        from skimage.metrics import structural_similarity
        torch = self.torch
        x0, y0, x1, y1 = region
        p = pred[y0:y1, x0:x1].copy()
        t = truth[y0:y1, x0:x1].copy()
        weight = actor[y0:y1, x0:x1].astype(np.float32)
        if not (weight.sum() > 0 and min(weight.shape) >= 32):
            raise ValueError('Empty/too small actor metric region')
        mse = float(np.sum(np.mean((p - t) ** 2, -1) * weight) / weight.sum())
        # Identical neutral RGB outside the actor prevents background dominance.
        p[weight <= 0] = .5
        t[weight <= 0] = .5
        _, ssim = structural_similarity(p, t, channel_axis=-1, data_range=1., full=True)
        ssim = np.mean(ssim, -1)
        with torch.inference_mode():
            a = torch.from_numpy(p).permute(2, 0, 1)[None].to(self.device) * 2 - 1
            b = torch.from_numpy(t).permute(2, 0, 1)[None].to(self.device) * 2 - 1
            spatial = self.lpips(a, b)
            w = torch.from_numpy(weight)[None, None].to(self.device)
            if spatial.shape[-2:] != w.shape[-2:]:
                raise ValueError('LPIPS spatial map was unexpectedly rescaled')
            distance = float((spatial * w).sum().item() / w.sum().item())
        return dict(psnr=float(-10 * np.log10(max(mse, 1e-20))),
                    ssim=float((ssim * weight).sum() / weight.sum()), lpips=distance,
                    actor_pixels=int((weight > 0).sum()))


def run(manifest_path, output, device):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest['target_excluded_from_donors'] is not True:
        raise ValueError('A target photograph cannot supply its own hybrid prediction')
    if not 0 < int(manifest['completed_iterations']) <= 140000:
        raise ValueError('Use the current capped campaign model')
    sys.path.insert(0, manifest['legacy_reference'])
    from transfer_luster_hybrid import merge_details
    measure = Measure(device)
    results = []
    for target in manifest['targets']:
        directory = Path(target['directory'])
        if sha(directory / 'target.npz') != target['target_sha256']:
            raise ValueError('Bound neural target/GT/mask bundle changed')
        with np.load(directory / 'target.npz') as z:
            base, truth, actor = [z[k].copy() for k in ['rgb', 'gt', 'actor']]
        if base.shape != truth.shape or actor.shape != base.shape[:2] or base.shape[-1] != 3:
            raise ValueError('Target RGB, GT and actor mask rasters differ')
        if not all(np.isfinite(a).all() for a in [base, truth, actor]) or np.any((actor < 0) | (actor > 1)):
            raise ValueError('Invalid target values or actor weights')
        donors = []
        for item in target['donors']:
            if item['camera'] == target['camera']:
                raise ValueError('Target donor leak')
            path = directory / item['path']
            if sha(path) != item['sha256']:
                raise ValueError('Qualified warp changed')
            with np.load(path) as z:
                donors.append(dict(camera_id=item['camera_id'], **{k: z[k].copy() for k in ['rgb', 'valid', 'score']}))
        scale = float(target['pixel_scale'])
        if scale <= 0 or 'parent_target_wh' not in target:
            raise ValueError('Explicit parent pixel scale required; never infer from crop width')
        region = target['evaluation_xyxy']
        x0, y0, x1, y1 = region
        if not (0 <= x0 < x1 <= base.shape[1] and 0 <= y0 < y1 <= base.shape[0]):
            raise ValueError('Metric region is outside the unresized crop')
        destination = output / target['name']
        save(destination / 'lookcloser.png', base)
        save(destination / 'GT.png', truth)
        variants = [dict(name='lookcloser', metrics=measure(base, truth, actor, region))]
        for config in configurations():
            if config['count'] > len(donors):
                continue
            name = 'detail_s{sigma:g}_g{gain:g}_n{count}_a{a}'.format(**config, a=int(config['ambiguity']))
            texture, labels, confidence = pick(base, donors, config['count'], scale, config['ambiguity'])
            rgb, weight = merge_details(base, texture, labels, confidence, 'detail', config['sigma'] * scale, config['gain'])
            if not np.array_equal(rgb[weight == 0], base[weight == 0]):
                raise AssertionError('Neural fallback changed')
            save(destination / (name + '.png'), rgb)
            save(destination / (name + '_transfer.png'), np.repeat(weight[..., None], 3, -1))
            value = dict(name=name, configuration=config, metrics=measure(rgb, truth, actor, region),
                         transfer_actor_fraction=float(((weight > 0) & (actor > 0)).sum() / max(1, (actor > 0).sum())))
            variants.append(value)
            write(output / 'progress.json', dict(target=target['name'], variant=name, time=time.time()))
        results.append(dict(target=target['name'], variants=variants))
        write(destination / 'metrics.json', results[-1])
    names = sorted(set.intersection(*[{v['name'] for v in t['variants']} for t in results]))
    aggregate = []
    for name in names:
        views = [next(v for v in t['variants'] if v['name'] == name)['metrics'] for t in results]
        aggregate.append(dict(name=name, **{k: float(np.mean([v[k] for v in views])) for k in ['psnr', 'ssim', 'lpips']}))
    aggregate.sort(key=lambda x: (x['lpips'], -x['psnr']))
    write(output / 'selection.json', dict(manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
          checkpoint_sha256=manifest['checkpoint_sha256'], completed_iterations=manifest['completed_iterations'],
          results=results, aggregate=aggregate, numerical_candidate=aggregate[0]['name'],
          visual_acceptance=False, temporal_validation=False, target_RGB_used_for_prediction=False,
          training=False, source_math='qualified unchanged native RAW warps', code_sha256=sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run(args.manifest, args.output, args.device)
