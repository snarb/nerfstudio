#!/usr/bin/env python3
"""Separate occupancy admission from step-size failures on train-only depth outliers."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from export_luster_opacity import load
from run_luster_opacity import density_geometry_fingerprint, sha, write


def cells(estimator, points):
    """Return valid voxel indices in every uncontracted nerfacc AABB level."""
    result = []
    for level, box in enumerate(estimator.aabbs):
        normalized = (points-box[:3])/(box[3:]-box[:3])
        inside = ((normalized >= 0) & (normalized < 1)).all(-1)
        ijk = torch.floor(normalized[inside]*estimator.resolution).long()
        result.append((level, ijk))
    return result


def audit(selection, reference, manifest_path, output):
    if output.exists():
        raise FileExistsError('Preserve previous diagnostics')
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model = trainer.pipeline.model
    dataset = trainer.pipeline.datamanager.train_dataset
    manifest = json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['transforms_sha256'] != sha(Path(request['data'])/'transforms.json'):
        raise ValueError('Require the same train-only calibration')
    failures = json.loads(reference.read_text())['outliers']
    estimator = model.occupancy_grid
    original = estimator.binaries.clone()
    field_hash = density_geometry_fingerprint(model)
    original_config = {k: getattr(model.config, k) for k in
                       ['adaptive_coarse_step_size', 'max_steps_per_ray']}
    rows = []
    try:
        with torch.no_grad():
            for item in failures:
                cid, xy = item['camera'], item['xy']
                index = next(i for i, p in enumerate(dataset.image_filenames) if int(p.stem.split('_')[1]) == cid)
                name = dataset.image_filenames[index].name
                receipt = manifest['maps'][name]
                path = manifest_path.parent/receipt['path']
                if sha(path) != receipt['sha256']:
                    raise ValueError('Depth receipt mismatch')
                with np.load(path) as z:
                    depth, unc, valid, witnesses = [z[k][xy[1], xy[0]] for k in
                                                    ['depth', 'uncertainty', 'valid', 'witnesses']]
                if not valid:
                    raise ValueError('Outlier must have qualified reference depth')
                camera = dataset.cameras[index:index+1].to(model.device)
                def ray():
                    return camera.generate_rays(0, coords=torch.tensor([[xy[1]+.5, xy[0]+.5]], device=model.device))
                rays = ray()
                norm = rays.metadata['directions_norm'].flatten()[0]*manifest['normalization_scale']
                target = float(depth*norm)
                eps = max(float(2*unc*norm), .001*target)
                distance = torch.linspace(target-eps, target+eps, 9, device=model.device)
                points = rays.origins + rays.directions*distance[:, None]
                indices = cells(estimator, points)
                status = []
                for level, ijk in indices:
                    for xyz in torch.unique(ijk, dim=0):
                        x, y, z = xyz.tolist()
                        status.append(dict(level=level, xyz=[x,y,z], occupied=bool(original[level,x,y,z])))
                records = []
                for admit, dense in [(False,False), (True,False), (False,True), (True,True)]:
                    estimator.binaries.copy_(original)
                    if admit:
                        for level, ijk in indices:
                            if len(ijk):
                                estimator.binaries[level, ijk[:,0], ijk[:,1], ijk[:,2]] = True
                    model.config.adaptive_coarse_step_size = .001 if dense else original_config['adaptive_coarse_step_size']
                    model.config.max_steps_per_ray = 4096 if dense else original_config['max_steps_per_ray']
                    rays = ray()
                    result = model(rays)
                    idx = result['packed_ray_indices'].reshape(-1)
                    start, end = [result[k].reshape(-1) for k in ['packed_spacing_starts','packed_spacing_ends']]
                    near, far = rays.nears.reshape(-1), rays.fars.reshape(-1)
                    lo, hi = near[idx]+(far-near)[idx]*start, near[idx]+(far-near)[idx]*end
                    mid = (lo+hi)*.5
                    pos = rays.origins[idx]+rays.directions[idx]*mid[:,None]
                    sigma = model.field.density_fn(pos).flatten().float()
                    overlap = (torch.minimum(hi, hi.new_tensor(target+eps))
                               -torch.maximum(lo, lo.new_tensor(target-eps))).clamp_min(0)
                    opacity = float(-torch.expm1(-(sigma*overlap).sum()))
                    records.append(dict(admit_band_cells=admit, dense_sampling=dense,
                         band_opacity=opacity, band_midpoint_nodes=int((abs(mid-target)<=eps).sum()),
                         total_nodes=len(mid), rgb=result['rgb'].flatten().cpu().tolist()))
                with Image.open(Path(request['data'])/'images'/name) as im:
                    gt = (np.asarray(im)[xy[1],xy[0],:3]/255).tolist()
                rows.append(dict(**item, ground_truth_rgb=gt, witnesses=int(witnesses),
                                 band_cells=status, support_variants=records))
    finally:
        estimator.binaries.copy_(original)
        for key, value in original_config.items():
            setattr(model.config, key, value)
    after = density_geometry_fingerprint(model)
    if after != field_hash or not torch.equal(estimator.binaries, original):
        raise ValueError('Diagnostic changed field or failed to restore occupancy')
    write(output, dict(scope='Train-only per-ray occupancy admission diagnostic, not deployable evaluation guidance',
                       selection=str(selection), reference=str(reference), source_sha256=sha(Path(__file__)),
                       density_before=field_hash, density_after=after, occupancy_restored=True, rays=rows))
    print(json.dumps(dict(rays=len(rows), initially_missing_cells=sum(not x['occupied'] for r in rows for x in r['band_cells']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.selection, args.reference, args.manifest, args.output)
