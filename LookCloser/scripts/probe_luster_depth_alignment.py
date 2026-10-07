#!/usr/bin/env python3
"""Measure train-only depth agreement on fixed native pixel rays, without masking RGB."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from run_luster_opacity import sha, write


def probe(selection, manifest_path, output, grid):
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model = trainer.pipeline.model
    dataset = trainer.pipeline.datamanager.train_dataset
    manifest = json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['transforms_sha256'] != sha(Path(request['data']) / 'transforms.json'):
        raise ValueError('Require the same calibrated train-only depth reference')
    records = []
    with torch.no_grad():
        for cid in [11, 97, 151]:
            index = next(i for i, p in enumerate(dataset.image_filenames) if int(p.stem.split('_')[1]) == cid)
            name = dataset.image_filenames[index].name
            receipt = manifest['maps'][name]
            path = manifest_path.parent / receipt['path']
            if sha(path) != receipt['sha256']:
                raise ValueError('Depth map hash mismatch')
            with np.load(path) as archive:
                measured = {k: archive[k] for k in ['depth', 'uncertainty', 'valid']}
            camera = dataset.cameras[index:index + 1].to(model.device)
            for region in ['face', 'hand', 'clothing']:
                x0, y0, x1, y1 = request['roi_boxes'][name][region]
                x, y = np.meshgrid(np.linspace(x0, x1 - 1, grid).astype(int), np.linspace(y0, y1 - 1, grid).astype(int))
                x, y = x.ravel(), y.ravel()
                rays = camera.generate_rays(0, coords=torch.tensor(np.stack([y, x], -1), device=model.device) + .5)
                result = model(rays)
                norm = rays.metadata['directions_norm'].flatten().cpu().numpy() * manifest['normalization_scale']
                target = measured['depth'][y, x] * norm
                tolerance = np.maximum(2 * measured['uncertainty'][y, x] * norm, .01 * target)
                indices = result['packed_ray_indices'].cpu().numpy()
                weights = result['packed_weights'].flatten().cpu().numpy()
                starts = result['packed_spacing_starts'].flatten().cpu().numpy()
                ends = result['packed_spacing_ends'].flatten().cpu().numpy()
                near, far = rays.nears.flatten().cpu().numpy(), rays.fars.flatten().cpu().numpy()
                mid = near[indices] + (far - near)[indices] * (starts + ends) / 2
                interval = (far - near)[indices] * (ends - starts)
                rgb = result['rgb'].cpu().numpy()
                for i in range(len(x)):
                    take = indices == i
                    w, t, dt = weights[take], mid[take], interval[take]
                    if not len(w) or not measured['valid'][y[i], x[i]]:
                        continue
                    mode = float(t[np.argmax(w / dt.clip(1e-8))])
                    distance = np.maximum(0, abs(t - target[i]) - tolerance[i]) / max(tolerance[i], 1e-6)
                    penalty = np.where(distance < 1, .5 * distance ** 2, distance - .5)
                    records.append(dict(camera=cid, region=region, xy=[int(x[i]), int(y[i])],
                        target=float(target[i]), tolerance=float(tolerance[i]), mode=mode,
                        relative_mode_error=float((mode - target[i]) / target[i]),
                        front_mass=float(w[t < target[i] - tolerance[i]].sum()),
                        back_mass=float(w[t > target[i] + tolerance[i]].sum()),
                        weighted_distance_penalty=float((w * penalty).sum()), rgb=rgb[i].tolist()))
    summaries = []
    for cid in [11, 97, 151]:
        for region in ['face', 'hand', 'clothing']:
            rows = [r for r in records if r['camera'] == cid and r['region'] == region]
            summaries.append(dict(camera=cid, region=region, trusted=len(rows), sampled=grid ** 2,
                quantiles={key: np.quantile([r[key] for r in rows], [.1, .5, .9]).tolist() if rows else []
                           for key in ['relative_mode_error', 'front_mass', 'back_mass', 'weighted_distance_penalty']}))
    write(output, dict(selection=str(selection), sampling='Unchanged checkpoint renderer; integer pixel centres; train-only depth; no RGB masks',
                       summaries=summaries, rays=records))
    print(json.dumps(summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=20)
    args = parser.parse_args()
    probe(args.selection, args.manifest, args.output, args.grid)
