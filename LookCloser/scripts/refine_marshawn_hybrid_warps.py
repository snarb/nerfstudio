#!/usr/bin/env python3
"""Experimental bounded image alignment of qualified same-instant donor warps.

Uses only the frozen neural prediction as its alignment reference, never GT.
This is a pixel-transfer experiment, not a geometry correction or production
release. Original and sampled donor eligibility must both remain valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tune_marshawn_hybrid_cache import sha, write


def refine(base, donor, scale, radius_hd):
    valid = donor['valid'].astype(bool)
    colour = donor['rgb'].astype(np.float32)
    if not np.any(valid):
        return {**donor}, dict(accepted_pixels=0, original_pixels=0)
    sigma = float(scale)
    # Fill invalid regions with the SAME neural reference, so mask boundaries
    # cannot masquerade as photograph texture or an optical-flow feature.
    filled = np.where(valid[..., None], colour, base)
    smooth_base = cv2.GaussianBlur(base, (0, 0), sigma)
    smooth_source = cv2.GaussianBlur(filled, (0, 0), sigma)
    luminance = np.array([.2126, .7152, .0722], np.float32)
    prev = np.rint(np.clip(smooth_base @ luminance, 0, 1) * 255).astype(np.uint8)
    next_ = np.rint(np.clip(smooth_source @ luminance, 0, 1) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev, next_, None, .5, 3,
                                      max(15, 2 * round(5 * scale) + 1), 4, 7, 1.5,
                                      cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    yy, xx = np.indices(valid.shape, dtype=np.float32)
    mapx, mapy = xx + flow[..., 0], yy + flow[..., 1]
    remap = lambda image: cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    displaced_valid = remap(valid.astype(np.float32)) >= 1 - 1e-6
    shifted = remap(filled)
    smooth_shifted = cv2.GaussianBlur(shifted, (0, 0), sigma)
    before = np.mean(np.abs(smooth_source - smooth_base), axis=-1)
    after = np.mean(np.abs(smooth_shifted - smooth_base), axis=-1)
    magnitude = np.linalg.norm(flow, axis=-1)
    # No new support: uncertain, excessive or unhelpful shifts retain their
    # original qualified colour and score. Filters/GT do not add admission.
    accept = valid & displaced_valid & np.isfinite(flow).all(-1)
    accept &= (magnitude <= radius_hd * scale) & (after <= .95 * before)
    output = colour.copy()
    output[accept] = shifted[accept]
    result = dict(rgb=output, valid=valid, score=donor['score'].copy())
    assert np.array_equal(result['rgb'][~accept], colour[~accept])
    assert np.array_equal(result['valid'], donor['valid'])
    assert np.array_equal(result['score'], donor['score'])
    return result, dict(accepted_pixels=int(accept.sum()), original_pixels=int(valid.sum()),
                       maximum_accepted_shift_native=float(magnitude[accept].max()) if accept.any() else 0.,
                       mean_accepted_shift_native=float(magnitude[accept].mean()) if accept.any() else 0.,
                       radius_HD=radius_hd, target_pixel_scale=scale,
                       before_mean=float(before[accept].mean()) if accept.any() else None,
                       after_mean=float(after[accept].mean()) if accept.any() else None)


def run(path, output, radius_hd):
    cv2.setNumThreads(4)
    path, output = Path(path), Path(output)
    manifest = json.loads(path.read_text())
    if manifest['target_excluded_from_donors'] is not True:
        raise ValueError('Independent donors required')
    if not 0 < manifest['completed_iterations'] <= 140000:
        raise ValueError('Capped same-frame model required')
    if radius_hd <= 0:
        raise ValueError('Positive bounded search radius required')
    refined_manifest = {**manifest, 'targets': [], 'experimental_alignment': dict(
        reference='frozen neural RGB only, never GT', radius_HD=radius_hd,
        original_admission_unchanged=True, geometry_refined=False,
        source_manifest=str(path), source_manifest_sha256=sha(path), code_sha256=sha(__file__))}
    for target in manifest['targets']:
        original = Path(target['directory'])
        destination = output / target['name']
        destination.mkdir(parents=True, exist_ok=True)
        if sha(original / 'target.npz') != target['target_sha256']:
            raise ValueError('Target bundle changed')
        with np.load(original / 'target.npz') as z:
            base = z['rgb'].copy()
        # A hard link preserves the exact immutable neural/GT/metric bundle.
        bundle = destination / 'target.npz'
        if not bundle.exists():
            bundle.hardlink_to(original / 'target.npz')
        elif sha(bundle) != target['target_sha256']:
            raise ValueError('Target bundle binding conflict')
        new_target = {**target, 'directory': str(destination), 'donors': []}
        for donor in target['donors']:
            source = original / donor['path']
            if sha(source) != donor['sha256']:
                raise ValueError('Qualified donor cache changed')
            with np.load(source) as z:
                arrays = {k: z[k].copy() for k in ['rgb', 'valid', 'score']}
            result, receipt = refine(base, arrays, target['pixel_scale'], radius_hd)
            dest = destination / donor['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, **result)
            write(dest.with_suffix('.json'), dict(source_sha256=donor['sha256'], **receipt))
            new_target['donors'].append({**donor, 'sha256': sha(dest)})
        refined_manifest['targets'].append(new_target)
        write(output / 'manifest.json', refined_manifest)
        write(output / 'progress.json', dict(target=target['name'], completed=len(refined_manifest['targets']), total=len(manifest['targets'])))
    write(output / 'complete.json', refined_manifest['experimental_alignment'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius-HD', type=float, default=1.)
    args = parser.parse_args()
    run(args.manifest, args.output, args.radius_HD)
