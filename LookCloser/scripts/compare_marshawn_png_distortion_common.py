#!/usr/bin/env python3
"""Score saved PNG-control predictions against an existing sealed common GT.

The candidate is already display RGB: never PQ-decode or re-expose it. ns-eval
images contain GT|prediction; match their GT to the actual scalar PNG targets
instead of inferring camera order from filenames. No field loading or training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compare_marshawn_common_domain import CAMERAS, metric_engine, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--renders', type=Path, required=True)
    p.add_argument('--scalar-data', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True,
                   help='Previous common-domain comparison containing GT_CAM and baseline_CAM PNGs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    meta = json.loads((a.scalar_data / 'transforms.json').read_text())
    targets = {row['physical_camera']: a.scalar_data / row['file_path']
               for row in meta['frames'] if row['physical_camera'] in CAMERAS}
    target_arrays = {cam: np.asarray(Image.open(path).convert('RGB'))
                     for cam, path in targets.items()}
    matches = {}
    for path in sorted(a.renders.glob('eval_img_*.png')):
        image = np.asarray(Image.open(path).convert('RGB'))
        if image.shape != (1012, 3840, 3):
            raise ValueError('Requires actual native GT|prediction raster: ' + str(path))
        gt, prediction = image[:, :1920], image[:, 1920:]
        errors = {cam: float(np.mean(np.abs(gt.astype(float) - value)))
                  for cam, value in target_arrays.items()}
        camera = min(errors, key=errors.get)
        maximum = int(np.max(np.abs(gt.astype(int) - target_arrays[camera].astype(int))))
        # ns-eval float->uint8 can truncate where the dataset round-trips exactly.
        if errors[camera] > 1.0 or maximum > 1 or camera in matches:
            raise ValueError('Missing/ambiguous source GT identity: ' + str(errors))
        if sum(value <= 1.0 for value in errors.values()) != 1:
            raise ValueError('More than one target matches ns-eval GT')
        matches[camera] = (path, prediction.copy(), errors[camera], maximum)
    if set(matches) != set(CAMERAS):
        raise ValueError('Exactly three unique DC/HC/KA predictions required')
    reference = json.loads((a.reference / 'comparison.json').read_text())
    for row in reference['rows']:
        camera = row['camera']
        if sha(a.reference / ('baseline_' + camera + '.png')) != row['baseline_png_sha256']:
            raise ValueError('Reference baseline PNG bytes changed')
    a.output.mkdir(parents=True, exist_ok=False)
    score = metric_engine(a.device)
    rows, panels = [], []
    for camera in CAMERAS:
        source, prediction, mean_error, max_error = matches[camera]
        gt_path = a.reference / ('GT_' + camera + '.png')
        baseline_path = a.reference / ('baseline_' + camera + '.png')
        gt = np.asarray(Image.open(gt_path).convert('RGB'))
        baseline = np.asarray(Image.open(baseline_path).convert('RGB'))
        if gt.shape != prediction.shape or baseline.shape != prediction.shape:
            raise ValueError('Common target/prediction raster mismatch')
        Image.fromarray(prediction).save(a.output / (camera + '.png'))
        rows.append(dict(camera=camera, source_render=str(source),
                         source_render_sha256=sha(source),
                         scalar_GT_sha256=sha(targets[camera]),
                         source_GT_mean_code_error=mean_error,
                         source_GT_max_code_error=max_error,
                         canonical_GT_sha256=sha(gt_path),
                         baseline_png_sha256=sha(baseline_path),
                         baseline_metrics=score(baseline, gt),
                         distortion_metrics=score(prediction, gt)))
        panels.append([gt, baseline, prediction])
    averages = {label: {key: float(np.mean([row[label][key] for row in rows]))
                        for key in ['psnr', 'ssim', 'lpips']}
                for label in ['baseline_metrics', 'distortion_metrics']}
    canvas = Image.new('RGB', (1920, 1083), '#202020')
    draw = ImageDraw.Draw(canvas)
    for r, (camera, images) in enumerate(zip(CAMERAS, panels)):
        for c, image in enumerate(images):
            canvas.paste(Image.fromarray(image).resize((640, 337), Image.Resampling.LANCZOS), (640*c, 361*r+24))
            draw.text((640*c+4, 361*r+5), camera+' '+['canonical common GT', 'accepted PNG8k guard', 'linear distortion8k core0'][c], fill='white')
    canvas.save(a.output / 'common_GT_predictions_contact.jpg', quality=95)
    report = dict(implementation_sha256=sha(__file__),
                  metric_dependency_sha256=sha(Path(__file__).with_name('compare_marshawn_common_domain.py')),
                  reference_comparison=str(a.reference / 'comparison.json'),
                  reference_comparison_sha256=sha(a.reference / 'comparison.json'),
                  native_raster=[1920, 1012], no_field_rendering=True, no_training=True,
                  no_exposure_fit=True, no_candidate_PQ_decode=True,
                  candidate_and_baseline_PNG_unchanged=True,
                  metric_protocol=reference['metric_protocol'],
                  common_GT_display=reference['display'], rows=rows, averages=averages,
                  limitations=['Saved PNG predictions are quantized8bit; native float metrics differ.',
                               'Accepted baseline uses its released inference guard; candidate uses trained radius0.',
                               'Metrics do not certify absence of visible floor artifacts or physical grid correctness.'])
    (a.output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(averages))


if __name__ == '__main__':
    main()
