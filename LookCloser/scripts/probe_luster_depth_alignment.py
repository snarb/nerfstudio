#!/usr/bin/env python3
"""Measure train-only depth agreement on fixed native pixel rays, without masking RGB."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from run_luster_opacity import sha, write


def probe(selection, manifest_path, output, grid, band_strength=10., sampling_overrides=None):
    if band_strength <= 0:
        raise ValueError('Band reference strength must be positive')
    if (sampling_overrides or {}).get('adaptive_coarse_step_size', 1.) <= 0:
        raise ValueError('Coarse step must be positive')
    if (sampling_overrides or {}).get('max_steps_per_ray', 1) < 1:
        raise ValueError('Ray step cap must be positive')
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model = trainer.pipeline.model
    for key, value in (sampling_overrides or {}).items():
        setattr(model.config, key, value)
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
                native_sigma = np.empty(0)
                if len(indices):
                    ids = result['packed_ray_indices'].reshape(-1).long()
                    distance = torch.tensor(mid, device=model.device)
                    positions = rays.origins[ids] + rays.directions[ids] * distance[:, None]
                    native_sigma = model.field.density_fn(positions).flatten().float().cpu().numpy()
                rgb = result['rgb'].cpu().numpy()
                points = rays.origins.cpu().numpy() + rays.directions.cpu().numpy() * target[:, None]
                aabb = model.scene_box.aabb.cpu().numpy()
                effective_valid = measured['valid'][y, x] & np.isfinite(target) & (target > 0)
                effective_valid &= ((points >= aabb[0]) & (points <= aabb[1])).all(-1)
                # Fixed reference slab in both arms, independent of RGB nodes.
                band_tolerance = np.maximum(
                    2 * measured['uncertainty'][y, x] * norm, .01 * target / band_strength)
                band_valid = effective_valid & np.isfinite(band_tolerance) & (band_tolerance > 0)
                band_valid &= (target-band_tolerance >= near) & (target+band_tolerance <= far)
                band_opacity = np.full(len(x), np.nan)
                band_opacity_dense = np.full(len(x), np.nan)
                front_probe_opacity = np.full(len(x), np.nan)
                good = np.flatnonzero(band_valid)
                if len(good):
                    ids=torch.tensor(good,device=model.device)
                    center=torch.tensor(target[good],device=model.device)[:,None]
                    eps=torch.tensor(band_tolerance[good],device=model.device)[:,None]
                    offsets=torch.linspace(-1,1,9,device=model.device)
                    distances=center+eps*offsets[None,:]
                    p=rays.origins[ids,None]+rays.directions[ids,None]*distances[...,None]
                    sigma=model.field.density_fn(p.reshape(-1,3)).reshape(len(good),9).float()
                    tau=torch.trapezoid(sigma,dx=1/8,dim=-1)*2*eps.flatten()
                    band_opacity[good]=(-torch.expm1(-tau)).cpu().numpy()
                    # Independent denser quadrature detects nine-node aliasing.
                    offsets=torch.linspace(-1,1,65,device=model.device)
                    distances=center+eps*offsets[None,:]
                    p=rays.origins[ids,None]+rays.directions[ids,None]*distances[...,None]
                    sigma=model.field.density_fn(p.reshape(-1,3)).reshape(len(good),65).float()
                    tau=torch.trapezoid(sigma,dx=1/64,dim=-1)*2*eps.flatten()
                    band_opacity_dense[good]=(-torch.expm1(-tau)).cpu().numpy()
                    front_distance=torch.maximum(center-2*eps,rays.nears[ids])
                    p=rays.origins[ids]+rays.directions[ids]*front_distance
                    sigma=model.field.density_fn(p).flatten().float()
                    front_probe_opacity[good]=(-torch.expm1(-sigma*2*eps.flatten())).cpu().numpy()
                for i in range(len(x)):
                    take = indices == i
                    w, t, dt = weights[take], mid[take], interval[take]
                    if not len(w) or not effective_valid[i] or w.sum() <= 1e-8:
                        continue
                    mode = float(t[np.argmax(w / dt.clip(1e-8))])
                    order = np.argsort(t)
                    cdf = np.cumsum(w[order]) / w.sum()
                    lo, hi = np.searchsorted(cdf, [.05, .95]).clip(0, len(w) - 1)
                    width = float(t[order[hi]] - t[order[lo]])
                    distance = np.maximum(0, abs(t - target[i]) - tolerance[i]) / max(tolerance[i], 1e-6)
                    penalty = np.where(distance < 1, .5 * distance ** 2, distance - .5)
                    native_band_opacity, native_band_nodes = None, None
                    if band_valid[i]:
                        lo = near[i] + (far[i]-near[i]) * starts[take]
                        hi = near[i] + (far[i]-near[i]) * ends[take]
                        overlap = (np.minimum(hi, target[i]+band_tolerance[i])
                                   - np.maximum(lo, target[i]-band_tolerance[i])).clip(0)
                        native_band_opacity = float(-np.expm1(-(native_sigma[take]*overlap).sum()))
                        native_band_nodes = int(np.count_nonzero(abs(t-target[i]) <= band_tolerance[i]))
                    records.append(dict(camera=cid, region=region, xy=[int(x[i]), int(y[i])],
                        target=float(target[i]), tolerance=float(tolerance[i]), mode=mode,
                        relative_mode_error=float((mode - target[i]) / target[i]),
                        termination_90pct_width=width,
                        density_band_opacity=float(band_opacity[i]) if band_valid[i] else None,
                        density_band_opacity_dense=float(band_opacity_dense[i]) if band_valid[i] else None,
                        native_midpoint_band_opacity=native_band_opacity,
                        native_band_midpoint_nodes=native_band_nodes,
                        density_front_probe_opacity=float(front_probe_opacity[i]) if band_valid[i] else None,
                        band_valid=bool(band_valid[i]),
                        band_tolerance=float(band_tolerance[i]) if band_valid[i] else None,
                        front_mass_tight=float(w[t < target[i]-band_tolerance[i]].sum()) if band_valid[i] else None,
                        front_mass=float(w[t < target[i] - tolerance[i]].sum()),
                        back_mass=float(w[t > target[i] + tolerance[i]].sum()),
                        weighted_distance_penalty=float((w * penalty).sum()), rgb=rgb[i].tolist()))
    summaries = []
    for cid in [11, 97, 151]:
        for region in ['face', 'hand', 'clothing']:
            rows = [r for r in records if r['camera'] == cid and r['region'] == region]
            keys = ['relative_mode_error', 'front_mass', 'back_mass', 'termination_90pct_width',
                    'weighted_distance_penalty', 'density_band_opacity', 'density_band_opacity_dense',
                    'native_midpoint_band_opacity', 'native_band_midpoint_nodes',
                    'density_front_probe_opacity', 'front_mass_tight']
            quantiles = {}
            for key in keys:
                values = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
                quantiles[key] = np.quantile(values, [.1, .5, .9]).tolist() if values else []
            summaries.append(dict(camera=cid, region=region, trusted=len(rows), sampled=grid ** 2,
                                  band_qualified=sum(r['band_valid'] for r in rows), quantiles=quantiles))
    write(output, dict(selection=str(selection), sampling='Checkpoint renderer with declared sampling overrides; integer pixel centres; train-only depth; no RGB masks',
                       probe_sha256=sha(Path(__file__)), band_reference_strength=band_strength,
                       band_quadrature_nodes=[9,65], front_probe='Local 2*tolerance opacity proxy at max(near, depth-2*tolerance), not integrated front opacity',
                       native_band_probe='Density requery at native packed midpoints, integrated over interval overlap with slab; not a new RGB render',
                       sampling_overrides=sampling_overrides or {},
                       summaries=summaries, rays=records))
    print(json.dumps(summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=20)
    parser.add_argument('--band-strength', type=float, default=10.)
    parser.add_argument('--coarse-step', type=float)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--corrected-allocator', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    overrides = {key: value for key, value in [
        ('adaptive_coarse_step_size', args.coarse_step), ('max_steps_per_ray', args.max_steps),
        ('corrected_arm_allocator', args.corrected_allocator)] if value is not None}
    probe(args.selection, args.manifest, args.output, args.grid, args.band_strength, overrides)
