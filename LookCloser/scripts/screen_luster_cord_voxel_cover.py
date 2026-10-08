#!/usr/bin/env python3
"""Conservatively occupy every voxel touching a train-fitted cord tube."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, evaluate, sha, write


@torch.no_grad()
def conservative_cover(grid: torch.Tensor, aabbs: torch.Tensor,
                       endpoints: torch.Tensor, radius: float) -> torch.Tensor:
    """A voxel is admitted if its bounding sphere intersects the line capsule.

    The sphere test can admit extra cells, but cannot miss a voxel intersected
    by the finite radius tube. Only the small segment bounding box is visited.
    """
    cover = torch.zeros_like(grid)
    resolution = torch.tensor(grid.shape[1:], device=aabbs.device)
    start, end = endpoints.to(aabbs)
    for level, box in enumerate(aabbs):
        cell = (box[3:]-box[:3])/resolution
        half_diagonal = cell.norm()/2
        pad = radius + cell
        lo = torch.floor((torch.minimum(start,end)-pad-box[:3])/cell).long().clamp_min(0)
        hi = torch.floor((torch.maximum(start,end)+pad-box[:3])/cell).long()
        hi = torch.minimum(hi,resolution-1)
        if (hi<lo).any():
            continue
        axes = [torch.arange(int(lo[d]),int(hi[d])+1,device=aabbs.device) for d in range(3)]
        xyz = torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        centers = box[:3]+(xyz+.5)*cell
        segment = end-start
        fraction = (((centers-start)*segment).sum(-1)/(segment*segment).sum()).clamp(0,1)
        distance = (centers-(start+fraction[:,None]*segment)).norm(dim=-1)
        included = xyz[distance<=radius+half_diagonal+1e-7]
        if len(included):
            cover[level,included[:,0],included[:,1],included[:,2]]=True
    return cover


def screen(request_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    request=deepcopy(json.loads(request_path.read_text()))
    if not request.get('resume') or request['model'].get('cord_occupancy_radius',0)<=0:
        raise ValueError('Need pinned checkpoint and cord occupancy admission')
    request['output']=str(output)
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    cfg=configuration(request)
    cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    model=trainer.pipeline.model
    step=int(Path(request['resume']).stem.split('-')[-1])
    model.current_train_step=step
    grid=model.occupancy_grid.binaries
    original=grid.clone()
    sparse=model.cord_occupancy_admission.clone()
    cover=conservative_cover(grid,model.occupancy_grid.aabbs,
                             model.cord_sampling_endpoints,
                             request['model']['cord_occupancy_radius'])
    if not torch.all(cover[sparse]):
        raise ValueError('Conservative voxel cover omitted a previously admitted cell')
    model.cord_occupancy_admission.logical_or_(cover)
    grid.logical_or_(cover)
    before=field_fingerprint(model)
    row=evaluate(trainer,request,step)
    after=field_fingerprint(model)
    if before!=after:
        raise RuntimeError('Frozen forward screen changed field weights')
    row['checkpoint']=request['resume']
    write(Path(row['renders'])/'metrics.json',row)
    write(output/'result.json',dict(scope='Frozen forward-only conservative cord voxel cover',
          checkpoint=request['resume'],checkpoint_sha256=sha(Path(request['resume'])),
          fit_sha256=request['model']['cord_sampling_fit_sha256'],
          radius=request['model']['cord_occupancy_radius'],
          sparse_cells=int(sparse.sum()),sparse_occupied_before=int((sparse&original).sum()),
          cover_cells=int(cover.sum()),extra_cover_cells=int((cover&~sparse).sum()),
          extra_cover_occupied_before=int((cover&~sparse&original).sum()),
          originally_occupied_in_cover=int((cover&original).sum()),
          newly_admitted=int((cover&~original).sum()),
          field_before=before,field_after=after,
          metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
          visual_review='pending'))
    print(json.dumps(dict(cover_cells=int(cover.sum()),newly_admitted=int((cover&~original).sum()),
                          metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    screen(args.request,args.output)
