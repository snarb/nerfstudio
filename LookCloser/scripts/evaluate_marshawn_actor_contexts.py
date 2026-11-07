#!/usr/bin/env python3
"""Score immutable same-checkpoint camera crops on actual actor pixels.

RGB outside each measured actor is identical neutral .5 for LPIPS/SSIM;
spatial metric maps are averaged only over that actor, not the whole canvas.
This evaluator never constructs a prediction or modifies a training artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from tune_marshawn_hybrid_cache import Measure, sha, write


def evaluate(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest['kind'] not in {'ALL66_CURRENT_ACTOR_CONTEXT_FIXED_GRID_VALIDATION_V1',
                                'ALL66_CURRENT_ACTOR_CONTEXT_RENDER_VALIDATION_V1'}:
        raise ValueError('Qualified same-checkpoint actor contexts required')
    if manifest['completed_iterations'] != 140000 or manifest['no_image_resize'] is not True:
        raise ValueError('Unexpected training budget or resized metric images')
    contexts = manifest['contexts']
    if len(contexts) != 66 or len({r['camera'] for r in contexts}) != 66:
        raise ValueError('Exactly all 66 distinct cameras required')
    binding = dict(manifest_sha256=sha(manifest_path), evaluator_sha256=sha(__file__),
                   checkpoint_sha256=manifest['checkpoint_sha256'],
                   metric='actor-weighted spatial AlexNet LPIPS; masked PSNR; actor-weighted SSIM',
                   background='identical .5 outside measured actor', context_resize=False)
    output.mkdir(parents=True, exist_ok=True)
    measure = Measure('cpu')
    results = []
    for row in contexts:
        receipt = output / (row['camera'] + '.json')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['binding'] != binding:
                raise ValueError('Use a new output for a different metric/source revision')
            results.append(old)
            continue
        if sha(row['source']) != row['source_sha256']:
            raise ValueError('Source RGB/mask context changed')
        with np.load(row['source']) as z:
            truth, black, grey, valid = [z[k].copy() for k in ['gt', 'black', 'grey', 'valid']]
        if truth.shape[:2] != valid.shape or black.shape != valid.shape or grey.shape != valid.shape:
            raise ValueError('Actor/source rasters differ')
        if np.any(black & grey):
            raise ValueError('Actor ownership overlaps in the fixed metric labels')
        h, w = valid.shape
        masks = dict(black=black & valid, grey=grey & valid, union=(black | grey) & valid)
        phases = {}
        for phase in manifest['phases']:
            source = row[phase]
            if sha(source['path']) != source['sha256']:
                raise ValueError('Prediction context changed')
            with np.load(source['path']) as z:
                prediction = z['rgb'].copy()
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise ValueError('Prediction raster/value mismatch')
            phases[phase] = {actor: measure(prediction, truth, mask, [0, 0, w, h])
                             if mask.any() else dict(reviewed_absent=True)
                             for actor, mask in masks.items()}
        result = dict(camera=row['camera'], binding=binding, source_sha256=row['source_sha256'],
                      phases=phases, actor_invalid_pixels=int(((black | grey) & ~valid).sum()))
        write(receipt, result)
        results.append(result)
        write(output / 'progress.json', dict(camera=row['camera'], completed=len(results), total=66, time=time.time()))
    aggregate = {}
    for phase in manifest['phases']:
        groups = {}
        for actor in ['black', 'grey', 'union', 'equal_actor_camera_mean']:
            actors = ['black', 'grey'] if actor == 'equal_actor_camera_mean' else [actor]
            values = [r['phases'][phase][a] for r in results for a in actors
                      if not r['phases'][phase][a].get('reviewed_absent')]
            groups[actor] = dict(observations=len(values), **{
                metric: float(np.mean([v[metric] for v in values])) for metric in ['psnr', 'ssim', 'lpips']})
        aggregate[phase] = groups
    base = manifest['phases'][0]
    changes = []
    for r in results:
        for phase in manifest['phases'][1:]:
            for actor in ['black', 'grey', 'union']:
                a, b = r['phases'][base][actor], r['phases'][phase][actor]
                if a.get('reviewed_absent') or b.get('reviewed_absent'):
                    continue
                changes.append(dict(camera=r['camera'], phase=phase, actor=actor, **{
                    metric + '_delta': b[metric] - a[metric] for metric in ['psnr', 'ssim', 'lpips']}))
    write(output / 'complete.json', dict(binding=binding, completed_cameras=66, aggregate=aggregate,
          per_camera_files=[str(output / (r['camera'] + '.json')) for r in results],
          largest_LPIPS_regressions=sorted(changes, key=lambda r: -r['lpips_delta'])[:12],
          largest_LPIPS_gains=sorted(changes, key=lambda r: r['lpips_delta'])[:12],
          training_cameras='all66; IN-SAMPLE', selected_grid_not_automatically_promoted=True,
          ground_truth_used_only_for_metrics=True, finished=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.output)
