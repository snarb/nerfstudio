#!/usr/bin/env python3
"""Train-only Sapiens2 hair evidence, native coordinates, immutable receipts.

Uses the official model pipeline (portrait rotation, non-aspect-preserving
1024x768 resize) and reverses rotation after resizing probabilities. These are
semantic probabilities, never presented as opacity or ground-truth masks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dec5_hair import VALIDATION, TEST
from run_dec5_hair_campaign import atomic_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--partition', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--prediction-receipt', type=Path)
    parser.add_argument('--selection-receipt', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    from sapiens.dense.models import init_model
    config = args.repo/'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py'
    payload = json.loads((args.data/'transforms.json').read_text())
    split_key = {'train':'train_filenames', 'validation':'val_filenames', 'test':'test_filenames'}[args.partition]
    frames = [f for f in payload['frames'] if f['file_path'] in payload[split_key]]
    if args.partition == 'train':
        if len(frames) != 59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):
            raise ValueError('Expected the 59 train cameras only')
    elif (len(frames) != 3 or {f['physical_camera'] for f in frames} != set(TEST if args.partition == 'test' else VALIDATION)
          or args.prediction_receipt is None or not args.prediction_receipt.is_file()):
        raise ValueError('Validation reference masks require already sealed predictions')
    if args.partition == 'test' and (args.selection_receipt is None or not args.selection_receipt.is_file()):
        raise ValueError('Test masks require locked selection')
    torch.cuda.set_per_process_memory_fraction(24*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    model = init_model(config, args.checkpoint, device='cuda:0')
    atomic_json(args.output/'request.json', dict(checkpoint_sha256=sha(args.checkpoint),
        config_sha256=sha(config), data_sha256=sha(args.data/'transforms.json'),
        channels=['hair_probability', 'face_neck_probability', 'background_probability'],
        hair_class=4, face_class=3, background_class=0,
        probability_is_not_alpha=True, rotation='numpy.rot90; inverse after resampling', count=len(frames),
        partition=args.partition, reference_masks_only=args.partition != 'train',
        prediction_receipt_sha256=sha(args.prediction_receipt) if args.prediction_receipt else None))
    records = []
    for index, row in enumerate(frames):
        image = cv2.imread(str(args.data/row['file_path']))
        portrait = np.ascontiguousarray(np.rot90(image))
        data = model.data_preprocessor(model.pipeline(dict(img=portrait)))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(data['inputs'])
            logits = F.interpolate(logits.float(), size=portrait.shape[:2], mode='bilinear', align_corners=False)
            probability = logits.softmax(1)[0, [4, 3, 0]].permute(1, 2, 0).cpu().numpy()
        probability = np.ascontiguousarray(np.rot90(probability, -1))
        if probability.shape[:2] != image.shape[:2] or not np.isfinite(probability).all():
            raise ValueError('Invalid semantic evidence')
        output = args.output/(row['physical_camera']+'.npz')
        np.savez_compressed(output, probability=np.rint(probability*65535).astype(np.uint16))
        records.append(dict(camera=row['physical_camera'], source_sha256=sha(args.data/row['file_path']),
                            output_sha256=sha(output), hair_pixels=int((probability[..., 0] > .5).sum())))
        if index % 8 == 0:
            overlay = image.copy()
            mask = probability[..., 0] > .5
            overlay[mask] = (overlay[mask]*.35+np.array([40, 220, 40])*.65).astype(np.uint8)
            panel = np.rot90(np.concatenate([image, overlay], axis=0))
            panel = cv2.resize(panel, (720, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.output/(row['physical_camera']+'_review.jpg')), panel)
        atomic_json(args.output/'progress.json', dict(status='running', completed=index+1, total=len(frames)))
    atomic_json(args.output/'complete.json', dict(records=records, request_sha256=sha(args.output/'request.json')))
    atomic_json(args.output/'progress.json', dict(status='complete', completed=len(frames), total=len(frames)))
    print('semantic evidence complete cameras='+str(len(frames)), flush=True)


if __name__ == '__main__':
    main()
