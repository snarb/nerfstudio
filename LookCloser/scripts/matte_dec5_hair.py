#!/usr/bin/env python3
"""Train-only BiRefNet alpha + PyMatting foreground on native head crops.

Person alpha is retained as such; a separate Sapiens hair domain qualifies its
use. Foreground estimation is an explicit matting estimate, not original GT RGB.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dec5_hair import VALIDATION, TEST
from run_dec5_hair_campaign import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoModelForImageSegmentation
    from pymatting import estimate_foreground_ml
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(24*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
    model = AutoModelForImageSegmentation.from_pretrained(str(args.model), trust_remote_code=True,
                    local_files_only=True).eval().cuda().half()
    preprocess = transforms.Compose([transforms.Resize((2048, 2048)), transforms.ToTensor(),
                        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    payload = json.loads((args.root/'dataset/transforms.json').read_text())
    frames = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    if len(frames) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):
        raise ValueError('Only 59 train cameras may provide matting inputs')
    sources = {r['camera']:r for r in json.loads((args.root/'prepare.json').read_text())['records']}
    atomic_json(args.output/'request.json', dict(model_weights_sha256=sha(args.model/'model.safetensors'),
        data_sha256=sha(args.root/'dataset/transforms.json'), count=59, input_resize=[2048, 2048],
        person_alpha_not_hair_semantics=True, foreground='PyMatting estimate_foreground_ml on original linear EXR',
        source_domain='train-only Sapiens hair crop plus context; unknown outside crop'))
    rows = []
    for index, frame in enumerate(frames):
        started = time.time()
        name = frame['physical_camera']
        semantic = np.load(args.root/'semantics'/(name+'.npz'))['probability'].astype(np.float32)/65535
        portrait_hair = np.rot90(semantic[..., 0])
        yy, xx = np.nonzero(portrait_hair > .25)
        if len(xx) == 0:
            raise ValueError('Empty hair semantic domain: '+name)
        image = cv2.imread(str(args.root/'dataset'/frame['file_path']))[..., ::-1]
        portrait = np.rot90(image)
        x0, x1 = max(0, int(xx.min())-96), min(portrait.shape[1], int(xx.max())+97)
        y0, y1 = max(0, int(yy.min())-96), min(portrait.shape[0], int(yy.max())+257)
        crop = np.ascontiguousarray(portrait[y0:y1, x0:x1])
        inputs = preprocess(Image.fromarray(crop))[None].cuda().half()
        with torch.inference_mode():
            logits = model(inputs)[-1]
            alpha = F.interpolate(logits.float(), size=crop.shape[:2], mode='bilinear', align_corners=False).sigmoid()[0, 0].cpu().numpy()
        linear = cv2.imread(str(args.root/'source/images'/Path(sources[name]['original_path']).name), cv2.IMREAD_UNCHANGED)[..., ::-1]
        linear_crop = np.ascontiguousarray(np.rot90(linear)[y0:y1, x0:x1])
        # Matting operates on bounded linear RGB; record the reversible crop scale.
        scale = max(float(np.quantile(np.maximum(linear_crop, 0), .999)), 1e-6)
        bounded = np.clip(np.maximum(linear_crop, 0)/scale, 0, 1).astype(np.float64)
        foreground = estimate_foreground_ml(bounded, alpha.astype(np.float64), n_small_iterations=5,
                                            n_big_iterations=2).astype(np.float32)*scale
        hair_domain = portrait_hair[y0:y1, x0:x1]
        output = args.output/(name+'.npz')
        np.savez_compressed(output, alpha=alpha.astype(np.float16), foreground=foreground.astype(np.float16),
            portrait_crop=np.array([x0, y0, x1, y1]), hair_probability=hair_domain.astype(np.float16),
            original_shape=np.array(linear.shape), linear_scale=scale)
        rows.append(dict(camera=name, sha256=sha(output), seconds=time.time()-started,
                         clipped_rgb_fraction=float((linear_crop > scale).mean()),
                         peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        if index % 8 == 0:
            checker = ((np.indices(alpha.shape).sum(0)//16)%2)[..., None]*.25+.25
            from study_dec5_hair_geometry import display
            composite = display(foreground)*alpha[..., None]+checker*(1-alpha[..., None])
            panel = np.concatenate((crop.astype(np.float32)/255, np.repeat(alpha[..., None], 3, -1), composite), 1)
            factor = min(1., 1200/panel.shape[1])
            panel = cv2.resize(panel, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.output/(name+'_review.jpg')), np.rint(np.clip(panel, 0, 1)*255).astype(np.uint8)[..., ::-1])
        atomic_json(args.output/'progress.json', dict(status='running', completed=index+1, total=59))
    atomic_json(args.output/'complete.json', dict(records=rows, request_sha256=sha(args.output/'request.json')))
    atomic_json(args.output/'progress.json', dict(status='complete', completed=59, total=59))


if __name__ == '__main__':
    main()
