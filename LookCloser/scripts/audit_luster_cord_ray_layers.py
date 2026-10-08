#!/usr/bin/env python3
"""Read-only decomposition of held-out cord rays on a pinned local renderer."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from nerfstudio.field_components.field_heads import FieldHeadNames

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, sha, write


def audit(request_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    request = deepcopy(json.loads(request_path.read_text()))
    if not request.get('resume') or not request['model'].get('cord_sampling_fit_path'):
        raise ValueError('Need a pinned checkpoint and train-only cord fit')
    request['output'] = str(output)
    configure_runtime(request)
    output.mkdir(parents=True)
    cfg = configuration(request)
    trainer = cfg.setup(local_rank=0, world_size=1)
    trainer.setup()
    trainer.pipeline.eval()
    model = trainer.pipeline.model
    model.current_train_step = int(Path(request['resume']).stem.split('-')[-1])
    before = field_fingerprint(model)
    ds = trainer.pipeline.datamanager.eval_dataset
    index = next(i for i, p in enumerate(ds.image_filenames)
                 if int(p.stem.split('_')[1]) == 95)
    camera = ds.cameras[index:index+1].to(model.device)
    # The line is approximate: a horizontal sweep also includes both flanks.
    points = [(x, y) for y in (1240, 1270, 1300, 1330, 1360, 1380)
              for x in range(874, 897)]
    coords = torch.tensor([[y, x] for x, y in points], device=model.device,
                          dtype=torch.float32) + .5
    ray_bundle = camera.generate_rays(0, coords=coords)
    start, end = model.cord_sampling_endpoints.to(model.device)
    rows = []
    with torch.no_grad():
        for left in range(0, len(points), 32):
            right = min(left+32, len(points))
            rays = ray_bundle[left:right]
            o, d = rays.origins, rays.directions
            dd = (d*d).sum(-1)
            segment = end-start
            projected = segment-d*((segment*d).sum(-1)/dd)[:, None]
            u = (-((start-o)*projected).sum(-1)/(projected*projected).sum(-1).clamp_min(1e-12)).clamp(0,1)
            closest = start+u[:,None]*segment
            t_line = ((closest-o)*d).sum(-1)/dd
            distance = (closest-o-t_line[:,None]*d).norm(dim=-1)
            captured = {}

            def capture(_module, args, values):
                samples = args[0]
                captured['t'] = ((samples.frustums.starts+samples.frustums.ends)/2).flatten().detach()
                captured['sigma'] = values[FieldHeadNames.DENSITY].flatten().detach()
                captured['color'] = values[FieldHeadNames.RGB].detach()

            hook = model.field.register_forward_hook(capture)
            try:
                prediction = model(rays)
            finally:
                hook.remove()
            ids = prediction['packed_ray_indices'].flatten().long()
            weights = prediction['packed_weights'].flatten().float()
            t, sigma, colors = captured['t'], captured['sigma'], captured['color']
            if len(ids) != len(t):
                raise RuntimeError('Field sample hook and packed renderer disagree')
            reconstructed = torch.zeros(right-left,3,device=model.device).index_add_(0,ids,weights[:,None]*colors)
            parity = float((reconstructed-prediction['rgb']).abs().max())
            if parity > 2e-4:
                raise RuntimeError(f'RGB reconstruction error {parity}')
            for i in range(right-left):
                x,y = points[left+i]
                take = ids == i
                wi, ti, ci, si = weights[take],t[take],colors[take],sigma[take]
                band = (ti-t_line[i]).abs() <= .003
                band_weight = wi[band].sum()
                winner = int(wi.argmax()) if len(wi) else None
                gt = ds[index]['image'][y,x].float().to(model.device)
                rows.append(dict(x=x,y=y,gt=gt.cpu().tolist(),rgb=prediction['rgb'][i].cpu().tolist(),
                                 error_l1=float((prediction['rgb'][i]-gt).abs().mean()),
                                 line_fraction=float(u[i]),line_t=float(t_line[i]),
                                 ray_line_distance=float(distance[i]),sample_count=len(wi),
                                 total_mass=float(wi.sum()),front_mass=float(wi[ti<t_line[i]-.003].sum()),
                                 line_band_mass=float(band_weight),
                                 line_band_rgb=((wi[band,None]*ci[band]).sum(0)/band_weight.clamp_min(1e-9)).cpu().tolist(),
                                 back_mass=float(wi[ti>t_line[i]+.003].sum()),
                                 winner_t=float(ti[winner]) if winner is not None else None,
                                 winner_weight=float(wi[winner]) if winner is not None else None,
                                 winner_rgb=ci[winner].cpu().tolist() if winner is not None else None,
                                 line_band_sigma_max=float(si[band].max()) if band.any() else None))
    if field_fingerprint(model) != before:
        raise RuntimeError('Forward audit changed field parameters')
    write(output/'result.json',dict(read_only=True,source_request=str(request_path),
          checkpoint=request['resume'],checkpoint_sha256=sha(Path(request['resume'])),
          fit_sha256=request['model']['cord_sampling_fit_sha256'],field_sha256=before,
          camera=95,masks_used=False,band_halfwidth=.003,rows=rows))
    for y in (1240,1270,1300,1330,1360,1380):
        local=[r for r in rows if r['y']==y and r['ray_line_distance']<.002]
        print(json.dumps(dict(y=y,closest=min(local,key=lambda r:r['ray_line_distance']) if local else None),
                         ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.request,args.output)
