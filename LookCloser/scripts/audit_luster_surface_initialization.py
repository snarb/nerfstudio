#!/usr/bin/env python3
"""Audit actual opaque first hits against a train-only learned geometry teacher."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from run_luster_opacity import sha, write


def audit(selection, manifest_path, output, samples=1024):
    if output.exists():
        raise FileExistsError('Preserve earlier initialization audits')
    if samples < 1:
        raise ValueError('Sample count must be positive')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['measured_depth']:
        raise ValueError('Require a train-only learned teacher')
    archive = manifest_path.parent / 'points.npz'
    if sha(archive) != manifest['file_sha256']:
        raise ValueError('Teacher point cache changed')
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model = trainer.pipeline.model
    if getattr(model.config, 'representation', 'volume') != 'opaque_surface':
        raise ValueError('This audit requires hard opaque first-hit rendering')
    if sha(Path(request['data']) / 'transforms.json') != manifest['transforms_sha256']:
        raise ValueError('Teacher and student camera calibration differ')
    dataset = trainer.pipeline.datamanager.train_dataset
    with np.load(archive) as data:
        candidates = np.flatnonzero(data['width'] <= .005)
        take = np.random.default_rng(42).choice(candidates, min(samples, len(candidates)), replace=False)
        if not len(take):
            raise ValueError('No narrow teacher points')
        cameras = data['camera'][take]
        pixels = data['pixel_yx'][take]
        target = data['distance'][take]
        origins = data['origins'][take]
        directions = data['directions'][take]
    rows = []
    maximum_ray_error = 0.
    with torch.no_grad():
        for index in np.unique(cameras):
            group = np.flatnonzero(cameras == index)
            camera = dataset.cameras[int(index):int(index)+1].to(model.device)
            rays = camera.generate_rays(0, coords=torch.tensor(pixels[group], device=model.device).float()+.5)
            maximum_ray_error = max(maximum_ray_error,
                float((rays.origins-torch.tensor(origins[group], device=model.device)).abs().max()),
                float((rays.directions-torch.tensor(directions[group], device=model.device)).abs().max()))
            if maximum_ray_error > 1e-5:
                raise ValueError('Stored teacher rays no longer match calibration')
            result = model(rays)
            distance = torch.tensor(target[group], device=model.device)[:, None]
            points = rays.origins+rays.directions*distance
            anchor = model.field.query_surface(points).flatten().abs().cpu().numpy()
            hit = result['surface_hit'].flatten().bool().cpu().numpy()
            error = result['depth'].flatten().cpu().numpy()-target[group]
            fractions = torch.tensor([.05, .25, .5, .75, .95], device=model.device)
            probes = rays.origins[:, None]+rays.directions[:, None]*(distance*fractions)[..., None]
            signs = model.field.query_surface(probes.reshape(-1, 3)).reshape(len(group), -1)
            negative = (signs < 0).any(-1).cpu().numpy()
            for i in range(len(group)):
                rows.append(dict(hit=bool(hit[i]), depth_error=float(error[i]) if hit[i] else None,
                                 anchor_abs_sdf=float(anchor[i]), negative_before_teacher=bool(negative[i])))
    errors = np.array([abs(r['depth_error']) for r in rows if r['hit']])
    summary = dict(samples=len(rows), hit_fraction=float(np.mean([r['hit'] for r in rows])),
        anchor_abs_sdf_median=float(np.median([r['anchor_abs_sdf'] for r in rows])),
        first_hit_abs_error_median=float(np.median(errors)) if len(errors) else None,
        first_hit_abs_error_p90=float(np.quantile(errors, .9)) if len(errors) else None,
        first_hit_within_0_005_fraction=float(np.mean([r['hit'] and abs(r['depth_error']) <= .005 for r in rows])),
        negative_before_teacher_fraction=float(np.mean([r['negative_before_teacher'] for r in rows])))
    write(output, dict(summary=summary, selected=selected,
        checkpoint_sha256=sha(selected['checkpoint']), teacher_manifest_sha256=sha(manifest_path),
        maximum_ray_error=maximum_ray_error, seed=42,
        distance_units='normalized scene units, not calibrated physical meters',first_hit_tolerance=.005,
        scope='Train-only learned teacher diagnostic, not measured depth or held-out validation', rows=rows))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1024)
    args = parser.parse_args()
    audit(args.selection, args.manifest, args.output, args.samples)
