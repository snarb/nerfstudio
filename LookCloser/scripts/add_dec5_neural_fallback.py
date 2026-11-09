#!/usr/bin/env python3
"""Complete unsupported RGB with the SAME frozen NeRF, retaining hole evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_dec5_hair_campaign import atomic_json, sha
from study_dec5_hair_geometry import display


def inverse_display(rgb):
    x = np.asarray(rgb)
    x = np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
    return x/np.maximum(1-x, 1e-6)/10.570320292648677


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--fallback', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.prediction/'complete.json').read_text())
    fallback = json.loads((args.fallback/'complete.json').read_text())
    by_name = {r['camera']:r for r in fallback['records']}
    if fallback.get('uses_eval_rgb') is not False:
        raise ValueError('Fallback must be produced without target RGB')
    records = []
    for row in receipt['records']:
        name = row['camera']
        path = args.prediction/(name+'.npz')
        if sha(path) != row['npz_sha256'] or sha(args.fallback/(name+'.npz')) != by_name[name]['sha256']:
            raise ValueError('Changed prediction/fallback archive')
        with np.load(path) as source:
            arrays = {k:source[k] for k in source.files}
        rgb = np.load(args.fallback/(name+'.npz'))['rgb_display']
        if rgb.shape != arrays['rgb'].shape:
            if fallback.get('rgb_downscale', 1) <= 1:
                raise ValueError('Fallback shape mismatch')
            rgb = cv2.resize(rgb, (arrays['rgb'].shape[1], arrays['rgb'].shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = arrays['source_id'] < 0
        arrays['rgb'][mask] = inverse_display(rgb)[mask]
        arrays['neural_fallback'] = mask
        np.savez_compressed(args.output/(name+'.npz'), **arrays)
        image = np.rint(display(arrays['rgb'])*255).astype(np.uint8)
        cv2.imwrite(str(args.output/(name+'.png')), image[..., ::-1])
        preview = cv2.resize(np.rot90(image), (360, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.output/(name+'_review.jpg')), preview[..., ::-1])
        records.append(dict(row, npz_sha256=sha(args.output/(name+'.npz')), png_sha256=sha(args.output/(name+'.png')),
                            neural_fallback_pixels=int(mask.sum())))
    receipt.update(records=records, parent_sha256=sha(args.prediction/'complete.json'),
                   fallback_receipt_sha256=sha(args.fallback/'complete.json'),
                   geometry_unchanged=True, missing_source_evidence_retained=True)
    atomic_json(args.output/'complete.json', receipt)


if __name__ == '__main__':
    main()
