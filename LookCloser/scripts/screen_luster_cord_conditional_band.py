#!/usr/bin/env python3
"""Diagnose front/back colour mixing with a frozen cord-depth counterfactual."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from nerfstudio.field_components.field_heads import FieldHeadNames

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, evaluate, sha, write


def screen(request_path: Path, output: Path, halfwidth: float,
           minimum_mass: float, strength: float, apply_radius: float | None,
           radial_falloff: bool) -> None:
    if output.exists():
        raise FileExistsError(output)
    if not (halfwidth > 0 and 0 < minimum_mass < 1 and 0 < strength <= 1
            and (apply_radius is None or apply_radius > 0)):
        raise ValueError('Invalid conditional band controls')
    request=deepcopy(json.loads(request_path.read_text()))
    if not request.get('resume') or not request['model'].get('cord_sampling_fit_path'):
        raise ValueError('Need a pinned checkpoint and train-only cord line')
    request['output']=str(output)
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    cfg=configuration(request)
    cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    model=trainer.pipeline.model
    radius=model.config.cord_sampling_radius if apply_radius is None else apply_radius
    step=int(Path(request['resume']).stem.split('-')[-1])
    model.current_train_step=step
    field_before=field_fingerprint(model)
    native_forward=model.forward
    changed_pixels=0
    eligible_pixels=0
    total_pixels=0

    def conditional_forward(ray_bundle):
        nonlocal changed_pixels,eligible_pixels,total_pixels
        captured={}

        def capture(_module,args,values):
            samples=args[0]
            captured['t']=((samples.frustums.starts+samples.frustums.ends)/2).flatten()
            captured['color']=values[FieldHeadNames.RGB].float()

        hook=model.field.register_forward_hook(capture)
        try:
            out=native_forward(ray_bundle)
        finally:
            hook.remove()
        ids=out['packed_ray_indices'].flatten().long()
        weights=out['packed_weights'].flatten().float()
        t,color=captured['t'],captured['color']
        if len(t)!=len(ids):
            raise RuntimeError('Packed renderer and field hook disagree')
        o,d=ray_bundle.origins,ray_bundle.directions
        start,end=model.cord_sampling_endpoints.to(o)
        segment=end-start
        dd=(d*d).sum(-1).clamp_min(1e-12)
        projected=segment-d*((segment*d).sum(-1)/dd)[:,None]
        fraction=(-((start-o)*projected).sum(-1)/(projected*projected).sum(-1).clamp_min(1e-12)).clamp(0,1)
        closest=start+fraction[:,None]*segment
        line_t=((closest-o)*d).sum(-1)/dd
        distance=(closest-o-line_t[:,None]*d).norm(dim=-1)
        within=(t-line_t[ids]).abs()<=halfwidth
        selected=within & (distance[ids]<=radius)
        band_mass=torch.zeros(len(o),device=o.device).index_add_(0,ids,weights*selected)
        band_rgb=torch.zeros(len(o),3,device=o.device).index_add_(0,ids,
                    weights[:,None]*color*selected[:,None])
        eligible=(distance<=radius)&(band_mass>=minimum_mass)
        conditional=band_rgb/band_mass[:,None].clamp_min(1e-8)
        mix=(1-distance/radius).clamp(0,1)*strength if radial_falloff else distance.new_full(distance.shape,strength)
        blended=out['rgb'].float()*(1-mix[:,None])+conditional*mix[:,None]
        rgb=torch.where(eligible[:,None],blended,out['rgb'].float())
        changed_pixels+=int(((rgb-out['rgb'].float()).abs().amax(-1)>1/255).sum())
        eligible_pixels+=int(eligible.sum())
        total_pixels+=len(o)
        out['rgb']=rgb
        return out

    model.forward=conditional_forward
    try:
        row=evaluate(trainer,request,step)
    finally:
        model.forward=native_forward
    field_after=field_fingerprint(model)
    if field_after!=field_before:
        raise RuntimeError('Frozen conditional screen changed field weights')
    row['checkpoint']=request['resume']
    write(Path(row['renders'])/'metrics.json',row)
    write(output/'result.json',dict(scope='Diagnostic frozen conditional cord-depth colour; not physical volume rendering',
          checkpoint=request['resume'],checkpoint_sha256=sha(Path(request['resume'])),
          fit_sha256=request['model']['cord_sampling_fit_sha256'],
          halfwidth=halfwidth,minimum_band_mass=minimum_mass,strength=strength,
          apply_radius=radius,radial_falloff=radial_falloff,
          eligible_pixels=eligible_pixels,changed_pixels=changed_pixels,total_pixels=total_pixels,
          field_before=field_before,field_after=field_after,
          metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']},
          visual_review='pending'))
    print(json.dumps(dict(changed_pixels=changed_pixels,eligible_pixels=eligible_pixels,
                          metrics={key:row[key] for key in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--halfwidth',type=float,default=.003)
    parser.add_argument('--minimum-mass',type=float,default=.5)
    parser.add_argument('--strength',type=float,default=1.)
    parser.add_argument('--apply-radius',type=float)
    parser.add_argument('--radial-falloff',action='store_true')
    args=parser.parse_args()
    screen(args.request,args.output,args.halfwidth,args.minimum_mass,args.strength,
           args.apply_radius,args.radial_falloff)
