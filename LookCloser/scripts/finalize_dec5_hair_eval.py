#!/usr/bin/env python3
"""Reserve test-camera poses, then expose test RGB only after selection/prediction."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np

from dec5_hair import TEST
from run_dec5_hair_campaign import atomic_json, sha
from study_dec5_hair_geometry import display


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('poses', 'prepare-test'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--selection', type=Path)
    p.add_argument('--prediction-receipt', type=Path)
    args = p.parse_args()
    records = json.loads((args.root/'reserved_test.json').read_text())['records']
    if {r['frame']['physical_camera'] for r in records} != set(TEST):
        raise ValueError('Unexpected test-camera inventory')
    if args.stage == 'poses':
        output = args.root/'test_camera_poses.json'
        if output.exists():
            raise FileExistsError(output)
        frames = [deepcopy(r['frame']) for r in records]
        for frame in frames:
            frame.pop('file_path', None)
        atomic_json(output, dict(frames=frames, rgb_read=False))
        return
    if args.selection is None or args.prediction_receipt is None:
        raise ValueError('Final test requires a locked selection and already-produced predictions')
    selection = json.loads(args.selection.read_text())
    chosen = Path(selection['chosen_metrics'])
    if sha(chosen) != selection['chosen_metrics_sha256'] or selection['independent_test_rgb_used']:
        raise ValueError('Selection was changed or already used test RGB')
    predictions = json.loads(args.prediction_receipt.read_text())
    if {r['camera'] for r in predictions['records']} != set(TEST):
        raise ValueError('Produce all three test predictions before reading test RGB')
    output = args.root/'test_dataset'
    (output/'images').mkdir(parents=True, exist_ok=False)
    frames, names, image_hashes = [], [], []
    for record in records:
        row = deepcopy(record['frame'])
        image = cv2.imread(record['source_path'], cv2.IMREAD_UNCHANGED)
        if image is None or not np.isfinite(image).all():
            raise ValueError('Unreadable test RGB')
        name = 'images/frame_eval_'+row['physical_camera']+'.png'
        rgb = np.rint(display(image[..., ::-1])*255).astype(np.uint8)
        cv2.imwrite(str(output/name), rgb[..., ::-1])
        row['file_path'] = name
        names.append(name)
        frames.append(row)
        image_hashes.append(dict(camera=row['physical_camera'], source_sha256=sha(record['source_path']), png_sha256=sha(output/name)))
    atomic_json(output/'transforms.json', dict(frames=frames, train_filenames=[], val_filenames=[], test_filenames=names))
    atomic_json(output/'unlock_receipt.json', dict(selection_sha256=sha(args.selection),
        prediction_receipt_sha256=sha(args.prediction_receipt), images=image_hashes,
        policy='locked test readout; do not tune or retrain from these results'))


if __name__ == '__main__':
    main()
