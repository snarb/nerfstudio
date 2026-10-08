#!/usr/bin/env python3
"""Rasterize qualified train-depth uncertainty bands into a spatial admission grid."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from run_luster_opacity import sha, write
from nerfstudio.model_components.lookcloser_occupancy import admit_occupancy_points_


def build(selection, manifest_path, output):
    if output.exists():
        raise FileExistsError('Use a new admission prior directory')
    selected = json.loads(selection.read_text())
    selected = selected.get('selected', selected)
    trainer, request = load(selected)
    model, dataset = trainer.pipeline.model, trainer.pipeline.datamanager.train_dataset
    manifest = json.loads(manifest_path.read_text())
    transforms_sha = sha(Path(request['data'])/'transforms.json')
    if not manifest['train_only'] or manifest['transforms_sha256'] != transforms_sha:
        raise ValueError('Require matching train-only calibrated depth')
    output.mkdir(parents=True)
    occupied = torch.zeros_like(model.occupancy_grid.binaries)
    offsets = torch.linspace(-1,1,9,device=model.device)
    measured_count, admitted_count = 0, 0
    rows = []
    with torch.no_grad():
        for index, image in enumerate(dataset.image_filenames):
            receipt = manifest['maps'].get(image.name)
            if receipt is None:
                raise ValueError(f'Missing training depth receipt: {image.name}')
            path = manifest_path.parent/receipt['path']
            if sha(path) != receipt['sha256']:
                raise ValueError('Depth map hash mismatch')
            with np.load(path) as archive:
                depth, uncertainty, valid = [archive[k] for k in ['depth','uncertainty','valid']]
            y, x = np.nonzero(valid & np.isfinite(depth) & (depth>0) & np.isfinite(uncertainty))
            measured_count += len(x)
            admitted = 0
            camera = dataset.cameras[index:index+1].to(model.device)
            for start in range(0,len(x),65536):
                yy, xx = y[start:start+65536], x[start:start+65536]
                coords = torch.tensor(np.stack([yy,xx],-1),device=model.device)+.5
                rays = model.collider(camera.generate_rays(0,coords=coords))
                norm = rays.metadata['directions_norm'].flatten()*manifest['normalization_scale']
                d = torch.tensor(depth[yy,xx],device=model.device)*norm
                unc = torch.tensor(uncertainty[yy,xx],device=model.device)*norm
                eps = torch.maximum(2*unc,.001*d)
                keep = (eps>0)&(d-eps>=rays.nears.flatten())&(d+eps<=rays.fars.flatten())
                distances = d[keep,None]+eps[keep,None]*offsets
                points = rays.origins[keep,None]+rays.directions[keep,None]*distances[...,None]
                admit_occupancy_points_(occupied,model.occupancy_grid.aabbs,points.reshape(-1,3))
                admitted += int(keep.sum())
            admitted_count += admitted
            rows.append(dict(image=image.name,qualified_pixels=len(x),admitted_bands=admitted))
    archive = output/'grid.npz'
    np.savez_compressed(archive,occupied=occupied.cpu().numpy(),aabbs=model.occupancy_grid.aabbs.cpu().numpy())
    result = dict(train_only=True,transforms_sha256=transforms_sha,
                  confidence_manifest=str(manifest_path),confidence_manifest_sha256=sha(manifest_path),
                  minimum_witnesses=manifest['minimum_witnesses'],source_sha256=sha(Path(__file__)),
                  grid_path=archive.name,grid_sha256=sha(archive),shape=list(occupied.shape),
                  band_strength=10.,band_points=9,qualified_pixels=measured_count,admitted_bands=admitted_count,
                  occupied_cells=int(occupied.sum()),occupied_fraction=float(occupied.float().mean()),maps=rows,
                  rule='Union-only occupancy admission; no RGB exclusion, density assignment or unknown-space veto')
    write(output/'manifest.json',result)
    print(json.dumps({k:result[k] for k in ['qualified_pixels','admitted_bands','occupied_cells','occupied_fraction']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    build(args.selection,args.manifest,args.output)
