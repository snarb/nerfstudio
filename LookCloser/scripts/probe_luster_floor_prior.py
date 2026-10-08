#!/usr/bin/env python3
"""Measure density along a fixed auxiliary floor-ray bank without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import sha, write


def probe(selection, manifest_path, output, free_samples=65):
    if free_samples < 2:
        raise ValueError('Require at least two quadrature points')
    if output.exists():
        raise FileExistsError('Preserve previous floor probes')
    manifest = json.loads(manifest_path.read_text())
    archive = manifest_path.parent / manifest['archive']
    if sha(archive) != manifest['archive_sha256']:
        raise ValueError('Floor bank hash mismatch')
    selected = json.loads(selection.read_text())
    trainer, request = load(selected.get('selected', selected))
    model = trainer.pipeline.model
    if sha(Path(request['data']) / 'transforms.json') != manifest['transforms_sha256']:
        raise ValueError('Floor bank calibration mismatch')
    with np.load(archive, allow_pickle=False) as arrays:
        cameras = arrays['cameras'].copy()
        bank = {k: torch.tensor(arrays[k], device=model.device)
                for k in ['origins', 'directions', 'depth', 'tolerance']}
    before = field_fingerprint(model)
    free, band = [], []
    with torch.no_grad():
        for start in range(0, len(cameras), 64):
            o, d, center, eps = [bank[k][start:start+64] for k in bank]
            length = center - eps - model.config.near_plane
            t = model.config.near_plane + length[:, None] * torch.linspace(0, 1, free_samples, device=model.device)
            sigma = model.field.density_fn((o[:, None] + d[:, None] * t[..., None]).reshape(-1, 3)).reshape(-1, free_samples)
            free.append((torch.trapezoid(sigma.float(), dx=1/(free_samples-1), dim=-1) * length).cpu().numpy())
            t = center[:, None] + eps[:, None] * torch.linspace(-1, 1, 9, device=model.device)
            sigma = model.field.density_fn((o[:, None] + d[:, None] * t[..., None]).reshape(-1, 3)).reshape(-1, 9)
            band.append((torch.trapezoid(sigma.float(), dx=1/8, dim=-1) * 2 * eps).cpu().numpy())
    if field_fingerprint(model) != before:
        raise ValueError('Probe changed field parameters')
    free, band = np.concatenate(free), np.concatenate(band)
    if not (np.isfinite(free).all() and np.isfinite(band).all()):
        raise ValueError('Nonfinite floor density integration')
    rows = []
    for cid in np.unique(cameras):
        take = cameras == cid
        rows.append(dict(camera=int(cid), rays=int(take.sum()),
            free_optical_thickness_mean=float(free[take].mean()),
            free_optical_thickness_p50_p90=np.quantile(free[take], [.5, .9]).tolist(),
            floor_band_opacity_mean=float((-np.expm1(-band[take])).mean())))
    write(output, dict(selection=str(selection), manifest=str(manifest_path),
        archive_sha256=manifest['archive_sha256'], field_sha256=before,
        quadrature=f'{free_samples} uniform free-segment and nine floor-band points; diagnostic approximation',
        per_camera=rows))
    print(json.dumps(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--free-samples', type=int, default=65)
    args = parser.parse_args()
    probe(args.selection, args.manifest, args.output, args.free_samples)
