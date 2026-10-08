#!/usr/bin/env python3
"""Warm only an independent SDF head from train-only learned surface points."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from export_luster_opacity import load
from run_luster_opacity import configuration, configure_runtime, evaluate, sha, write


def bootstrap(manifest_path, output, steps=1000, batch_size=1024, eval_stride=4):
    if min(steps, batch_size, eval_stride) < 1:
        raise ValueError('Steps, batch size and evaluation stride must be positive')
    if output.exists():
        raise FileExistsError('Use a new bootstrap directory')
    manifest=json.loads(manifest_path.read_text())
    if not manifest['train_only'] or manifest['measured_depth']:
        raise ValueError('First opaque arm uses learned train-only geometry, not MVS targets')
    archive=manifest_path.parent/'points.npz'
    if sha(archive)!=manifest['file_sha256'] or sha(manifest['selected']['checkpoint'])!=manifest['checkpoint_sha256']:
        raise ValueError('Teacher data or checkpoint changed')
    teacher,source_request=load(manifest['selected'])
    if sha(Path(source_request['data'])/'transforms.json') != manifest['transforms_sha256']:
        raise ValueError('Teacher calibration or split changed')
    request=deepcopy(source_request)
    for key in ['baseline_preset','leader_preset','resume','warm_start','parent_history']:
        request.pop(key,None)
    request.update(output=str(output),incremental_standard=True,representation='opaque_surface',
                   opacity_regularization=0.,depth_prior_strength=0.,end_step=0,
                   eval_stride=eval_stride)
    request['model'].update(surface_sdf_features='continuous',surface_geometry_head='independent',
        surface_geometry_gradient_mode='implicit',surface_num_samples=256,surface_refine_steps=8,
        eikonal_loss_mult=.1,reconstruction_loss_type='charbonnier')
    configure_runtime(request)
    torch.manual_seed(42)
    config=configuration(request)
    config.save_config()
    student=config.setup(local_rank=0,world_size=1)
    student.setup()
    model=student.pipeline.model
    report=model.field.load_state_dict(teacher.pipeline.model.field.state_dict(),strict=False)
    allowed=('geometry_head.','opacity_head.','surface_field_version')
    if report.unexpected_keys or any(not k.startswith(allowed) for k in report.missing_keys):
        raise ValueError(f'Unexpected field transfer: {report}')
    for name,p in model.field.named_parameters():
        p.requires_grad_(name.startswith('geometry_head.'))
    with np.load(archive) as data:
        keep=data['width']<=.005
        origins=torch.tensor(data['origins'][keep],device=model.device)
        directions=torch.tensor(data['directions'][keep],device=model.device)
        distances=torch.tensor(data['distance'][keep,None],device=model.device)
    if not len(origins):
        raise ValueError('No narrow teacher surfaces')
    points=origins+directions*distances
    optimizer=torch.optim.Adam(model.field.geometry_head.parameters(),lr=.001)
    history=[]
    output.mkdir(parents=True,exist_ok=True)
    write(output/'request.json',request)
    for step in range(steps):
        take=torch.randint(len(points),(batch_size,),device=model.device)
        surface=points[take]
        direction=directions[take]
        # Sign margins, rather than ray-aligned distance targets, preserve
        # freedom for the actual surface normal and grazing observations.
        front=surface-direction*.005
        back=surface+direction*.0005
        sdf=model.field.query_surface(surface)
        before=model.field.query_surface(front)
        after=model.field.query_surface(back)
        anchor=sdf.square().mean()
        signs=(.00005-before).clamp_min(0).square().mean()+(.000005+after).clamp_min(0).square().mean()
        gradients=model.field.finite_gradients(surface[:128])
        eikonal=(gradients.norm(dim=-1)-1).square().mean()
        objective=anchor+signs+.001*eikonal
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite geometry bootstrap')
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if step%100==0 or step==steps-1:
            row=dict(step=step,teacher_surface_mae=float(sdf.detach().abs().mean()),
                     gradient_norm_mean=float(gradients.detach().norm(dim=-1).mean()))
            history.append(row);write(output/'progress.json',row);print(json.dumps(row),flush=True)
    for p in model.field.parameters():
        p.requires_grad_(True)
    model.current_train_step=0
    student.step=0
    student.save_checkpoint(0)
    write(output/'geometry_receipt.json',dict(teacher_manifest_sha256=sha(manifest_path),
        points=len(points),steps=steps,field_transfer_missing=report.missing_keys,
        geometry_only=True,no_mvs_targets=True,no_source_masks=True,history=history,
        eval_stride=eval_stride,scope='Initialization diagnostic; native quality requires eval_stride=1'))
    selected=dict(step=0,checkpoint=str(student.checkpoint_dir/'step-000000000.ckpt'))
    write(output/'selection.json',selected)
    row=evaluate(student,request,0)
    write(output/'initial_surface_metrics.json',row)
    print(json.dumps({k:row[k] for k in ['eval_all_psnr','eval_all_ssim','eval_all_lpips']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--eval-stride',type=int,default=4,
                   help='Debug preview stride; use 1 for native quality evaluation')
    a=p.parse_args()
    bootstrap(a.manifest,a.output,a.steps,a.batch_size,a.eval_stride)
