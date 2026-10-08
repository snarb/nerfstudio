#!/usr/bin/env python3
"""Verify that a selected cord RGB anchor directly updates only the colour head."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from evaluate_luster_occupancy_prior import field_fingerprint
from run_luster_opacity import configuration, configure_runtime, sha, write


def audit(request_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    request=deepcopy(json.loads(request_path.read_text()))
    if (request['model'].get('cord_ray_color_regularization',0)<=0
            or not request.get('resume')):
        raise ValueError('Need a checkpoint continuation with selected cord RGB anchors')
    request['output']=str(output)
    configure_runtime(request)
    output.mkdir(parents=True)
    write(output/'request.json',request)
    trainer=configuration(request).setup(local_rank=0,world_size=1)
    trainer.setup()
    model=trainer.pipeline.model
    model.train()
    model.current_train_step=int(Path(request['resume']).stem.split('-')[-1])+1
    before=field_fingerprint(model)
    rays,_=trainer.pipeline.datamanager.next_train(model.current_train_step)
    model.zero_grad(set_to_none=True)
    outputs=model(rays)
    if 'cord_color_rgb' not in outputs or 'cord_color_target' not in outputs:
        raise RuntimeError('Selected cord RGB anchors were not evaluated')
    if outputs['cord_color_rgb'].shape!=outputs['cord_color_target'].shape:
        raise RuntimeError('Selected cord RGB shape mismatch')
    value=torch.sqrt((outputs['cord_color_rgb']-outputs['cord_color_target']).square()+1e-4).mean()
    value.backward()
    gradients={name:(0. if parameter.grad is None or parameter.grad.numel()==0
                     else float(parameter.grad.abs().max()))
               for name,parameter in model.field.named_parameters()}
    colour={name:value for name,value in gradients.items() if name.startswith('mlp_color.')}
    other={name:value for name,value in gradients.items() if not name.startswith('mlp_color.')}
    if not colour or max(colour.values())<=0 or (other and max(other.values())>0):
        raise RuntimeError('Cord RGB anchor gradient escaped the colour head')
    if field_fingerprint(model)!=before:
        raise RuntimeError('Read-only gradient audit changed field parameters')
    write(output/'result.json',dict(read_only=True,masks_used=False,
          source_request=str(request_path),checkpoint=request['resume'],
          checkpoint_sha256=sha(Path(request['resume'])),
          selected_rays=len(model.cord_ray_depths),
          sampled_anchor_rays=len(outputs['cord_color_rgb']),
          colour_head_gradient_max=max(colour.values()),
          noncolour_field_gradient_max=max(other.values()) if other else 0.,
          field_sha256=before))
    print(json.dumps(dict(selected_rays=len(model.cord_ray_depths),
                          sampled_anchor_rays=len(outputs['cord_color_rgb']),
                          colour_head_gradient_max=max(colour.values()),
                          noncolour_field_gradient_max=max(other.values()) if other else 0.)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.request,args.output)
