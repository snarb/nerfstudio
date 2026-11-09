#!/usr/bin/env python3
"""Separate face color error from trusted-depth placement on a frozen field."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from export_luster_opacity import load
from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import score, sha, write


def probe(selection, manifest_path, output, stride=2):
    if output.exists():
        raise FileExistsError('Preserve earlier face probes')
    if stride < 1:
        raise ValueError('Positive pixel stride required')
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model = trainer.pipeline.model
    dataset = trainer.pipeline.datamanager.train_dataset
    manifest = json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['transforms_sha256'] != sha(Path(request['data']) / 'transforms.json'):
        raise ValueError('Require matching calibrated training-only depth')
    before = field_fingerprint(model)
    output.mkdir(parents=True)
    rows, panels = [], []
    with torch.no_grad():
        for cid in [11, 97, 151]:
            index = next(i for i, p in enumerate(dataset.image_filenames) if int(p.stem.split('_')[1]) == cid)
            name = dataset.image_filenames[index].name
            receipt = manifest['maps'][name]
            archive_path = manifest_path.parent / receipt['path']
            if sha(archive_path) != receipt['sha256']:
                raise ValueError('Depth archive changed')
            with np.load(archive_path) as archive:
                measured = {k: archive[k].copy() for k in ['depth', 'uncertainty', 'valid']}
            box = request['roi_boxes'][name]['face']
            x0, y0, x1, y1 = box
            yy, xx = np.meshgrid(np.arange(y0, y1, stride), np.arange(x0, x1, stride), indexing='ij')
            shape = yy.shape
            y, x = yy.ravel(), xx.ravel()
            camera = dataset.cameras[index:index+1].to(model.device)
            rays = camera.generate_rays(0, coords=torch.tensor(np.stack([y, x], -1), device=model.device) + .5)
            native = torch.empty(len(x), 3, device=model.device)
            front, inside = np.zeros(len(x)), np.zeros(len(x))
            norm = rays.metadata['directions_norm'].flatten().cpu().numpy() * manifest['normalization_scale']
            target = measured['depth'][y, x] * norm
            eps = np.maximum(2 * measured['uncertainty'][y, x] * norm, .001 * target)
            valid = measured['valid'][y, x].copy() & np.isfinite(target) & np.isfinite(eps) & (eps > 0)
            for start in range(0, len(x), 1024):
                end = min(start + 1024, len(x))
                batch_rays = rays[start:end]
                rendered = model(batch_rays)
                native[start:end] = rendered['rgb'].float()
                indices = rendered['packed_ray_indices'].flatten().cpu().numpy()
                weights = rendered['packed_weights'].flatten().float().cpu().numpy()
                lo, hi = [rendered[k].flatten().cpu().numpy() for k in ['packed_spacing_starts', 'packed_spacing_ends']]
                near, far = [getattr(batch_rays, k).flatten().cpu().numpy() for k in ['nears', 'fars']]
                mid = near[indices] + (far-near)[indices] * (lo+hi) / 2
                local_target, local_eps = target[start:end], eps[start:end]
                valid[start:end] &= (local_target-local_eps >= near) & (local_target+local_eps <= far)
                np.add.at(front[start:end], indices, weights * (mid < local_target[indices]-local_eps[indices]))
                np.add.at(inside[start:end], indices, weights * (abs(mid-local_target[indices]) <= local_eps[indices]))
            positions = rays.origins + rays.directions * torch.tensor(target, device=model.device)[:, None]
            aabb = model.scene_box.aabb.to(model.device)
            valid &= ((positions >= aabb[0]) & (positions <= aabb[1])).all(-1).cpu().numpy()
            point, peak = native.clone(), native.clone()
            band_opacity = np.full(len(x), np.nan)
            for ids in np.array_split(np.flatnonzero(valid), max(1, (int(valid.sum())+511)//512)):
                if not len(ids):
                    continue
                take = torch.tensor(ids, device=model.device)
                center, width = [torch.tensor(a[ids], device=model.device) for a in [target, eps]]
                t = center[:, None] + width[:, None] * torch.linspace(-1, 1, 65, device=model.device)
                p = rays.origins[take, None] + rays.directions[take, None] * t[..., None]
                sigma = model.field.density_fn(p.reshape(-1, 3)).reshape(len(ids), 65).float()
                tau = torch.trapezoid(sigma, dx=1/64, dim=-1) * 2 * width
                band_opacity[ids] = (-torch.expm1(-tau)).cpu().numpy()
                best = sigma.argmax(-1)
                peak_positions = p[torch.arange(len(ids), device=model.device), best]
                point[take] = model.field.query_points(positions[take], rays.directions[take])[1].float()
                peak[take] = model.field.query_points(peak_positions, rays.directions[take])[1].float()
            gt = dataset[index]['image'][y, x].to(model.device).float()
            take = torch.tensor(np.flatnonzero(valid), device=model.device)
            errors = {}
            for label, rgb in [('native', native), ('measured_surface', point), ('band_density_peak', peak)]:
                image = rgb.reshape(*shape, 3)
                metrics = score(model, image, gt.reshape(*shape, 3))
                trusted_mse = (rgb[take].clamp(0, 1)-gt[take]).square().mean()
                errors[label] = dict(full_patch=metrics, trusted_psnr=float(-10*torch.log10(trusted_mse.clamp_min(1e-12))))
                pixels = np.rint(image.cpu().numpy().clip(0, 1)*255).astype('uint8')
                Image.fromarray(pixels).save(output/f'cam{cid:03d}_{label}.png')
            gt_pixels = np.rint(gt.reshape(*shape, 3).cpu().numpy().clip(0, 1)*255).astype('uint8')
            Image.fromarray(gt_pixels).save(output/f'cam{cid:03d}_gt.png')
            panel = Image.new('RGB', (shape[1]*4, shape[0]))
            for i, label in enumerate(['gt', 'native', 'measured_surface', 'band_density_peak']):
                panel.paste(Image.open(output/f'cam{cid:03d}_{label}.png'), (i*shape[1], 0))
            panels.append((cid, panel))
            rows.append(dict(camera=cid, box=box, stride=stride, sampled=len(x), trusted=int(valid.sum()),
                metrics=errors, median_front_termination_mass=float(np.median(front[valid])),
                median_native_band_termination_mass=float(np.median(inside[valid])),
                median_dense_band_opacity=float(np.nanmedian(band_opacity[valid]))))
            np.savez_compressed(output/f'cam{cid:03d}_geometry.npz', valid=valid.reshape(shape),
                target=target.reshape(shape), tolerance=eps.reshape(shape),
                front_mass=front.reshape(shape), band_mass=inside.reshape(shape),
                band_opacity=band_opacity.reshape(shape))
    if field_fingerprint(model) != before:
        raise ValueError('Diagnostic changed field parameters')
    width = max(p.width for _, p in panels)
    height = sum(p.height+30 for _, p in panels)
    sheet = Image.new('RGB', (width, height), '#222'); draw = ImageDraw.Draw(sheet); top = 0
    for cid, panel in panels:
        draw.text((4, top+4), f'cam{cid}: GT | native | measured depth color | band sigma peak color', fill='white')
        sheet.paste(panel, (0, top+25)); top += panel.height+30
    sheet.save(output/'comparison.png')
    write(output/'metrics.json', dict(selection=str(selection), field_sha256=before, depth_manifest=str(manifest_path),
        scope='Frozen-field counterfactual: trusted train pixels query opaque color at measured depth or local density peak; unknown pixels retain native RGB. Not a trained model or accepted renderer.',
        source_sha256=sha(Path(__file__)), rows=rows))
    print(json.dumps(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=2)
    args = parser.parse_args()
    probe(args.selection, args.manifest, args.output, args.stride)
