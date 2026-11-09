#!/usr/bin/env python3
"""Direct NeRF RGB ROI readout, separately from fixed GT-transfer experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from run_dec5_hair_campaign import atomic_json, sha
from score_dec5_hair import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Neural ROI metrics already exist')
    receipt = json.loads((args.prediction/'complete.json').read_text())
    payload = json.loads((args.root/'dataset/transforms.json').read_text())
    by_name = {f['physical_camera']:f for f in payload['frames'] if f['file_path'] in payload['val_filenames']}
    torch.set_num_threads(2)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).eval()
    rows = []
    for record in receipt['records']:
        name = record['camera']
        path = args.prediction/(name+'.npz')
        if name not in by_name or sha(path) != record['sha256']:
            raise ValueError('Changed or unexpected validation prediction')
        pred = np.load(path)['rgb_display'].clip(0,1)
        gt = cv2.imread(str(args.root/'dataset'/by_name[name]['file_path']))[..., ::-1].astype(np.float32)/255
        prob = np.load(args.root/'validation_reference'/(name+'.npz'))['probability']/65535
        domains = dict(hair=prob[...,0]>.5, face_neck=prob[...,1]>.5, full_frame=np.ones(gt.shape[:2],bool))
        rows.append(dict(camera=name, **{key:metrics(pred,gt,mask,lpips) for key,mask in domains.items()}))
    aggregate = {region:{key:float(np.mean([r[region][key] for r in rows])) for key in ('psnr','ssim','lpips')}
                 for region in ('hair','face_neck','full_frame')}
    atomic_json(args.output, dict(aggregate=aggregate, per_view=rows,
        prediction_receipt_sha256=sha(args.prediction/'complete.json'),
        semantic_reference_receipt_sha256=sha(args.root/'validation_reference/complete.json'),
        scope='Direct frozen NeRF RGB; not GT transfer or a depth-accuracy measurement',
        render_samples=receipt.get('sampling'),
        checkpoint_sha256=receipt['checkpoint_sha256']))
    print(json.dumps(aggregate),flush=True)


if __name__ == '__main__':
    main()
