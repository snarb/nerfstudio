"""Offline opt-in foreground volume objective for the DEC5 controls, including native adaptive LookCloser.

Person head-crop mattes constrain opacity, not semantic hair labels or depth GT.
Optical thickness supplies a gradient even when accumulated opacity saturates.
Unknown pixels outside the matte crop receive no RGB or opacity supervision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_dec5_hair_campaign import sha, atomic_json


def crop_indices(indices, width, crop):
    """Global (image,y,x) integer sample indices -> upright cropped (row,col)."""
    return width-1-indices[:, 2]-crop[1], indices[:, 1]-crop[0]


def opacity_target_thickness(alpha):
    return -torch.log1p(-alpha.clamp(0, .999))


def integrate_packed_thickness(density,deltas,ray_indices,num_rays):
    values=F.relu(density.reshape(-1))*deltas.reshape(-1)
    if len(values)!=len(ray_indices):raise ValueError('Packed thickness/sample binding mismatch')
    return values.new_zeros(num_rays).scatter_add(0,ray_indices.long(),values)


def opacity_bernoulli_nll(thickness,alpha):
    """Exact soft-alpha Bernoulli likelihood for opacity 1-exp(-tau)."""
    opaque=-torch.log(-torch.expm1(-thickness.clamp_min(1e-6)))
    return alpha*opaque+(1-alpha)*thickness


def install(trainer, targets, output, opacity_weight=.1,head_crop_coverage=None,opacity_objective='thickness'):
    pipeline = trainer.pipeline; model = pipeline.model
    if model.config.ray_sampling_mode not in ('fixed','adaptive'):
        raise ValueError('Foreground thickness requires explicit fixed or native adaptive traversal')
    payload = json.loads((pipeline.datamanager.config.data/'transforms.json').read_text())
    by_file = {Path(f['file_path']).name:f for f in payload['frames'] if f['file_path'] in payload['train_filenames']}
    records = []; arrays = []
    receipt = json.loads((targets/'complete.json').read_text())
    if (receipt['dataset_sha256'] != sha(pipeline.datamanager.config.data/'transforms.json')
            or receipt.get('uses_validation_rgb') is not False or receipt.get('uses_test_rgb') is not False):
        raise ValueError('Foreground targets belong to another dataset or fitting split')
    by_name = {row['camera']:row for row in receipt['records']}
    if len(by_name) != 59: raise ValueError('Expected train59 foreground target receipt')
    if head_crop_coverage is not None:
        coverage=json.loads(head_crop_coverage.read_text())
        if (coverage['target_receipt_sha256']!=sha(targets/'complete.json')
                or {r['camera'] for r in coverage['records']}!=set(by_name)
                or any(r['outside_crop'] for r in coverage['records'])):
            raise ValueError('Head-only volume needs verified complete train hair crop coverage')
    for path in pipeline.datamanager.train_dataset.image_filenames:
        frame = by_file[Path(path).name]; name = frame['physical_camera']; target = targets/(name+'.npz')
        if sha(target) != by_name[name]['sha256']: raise ValueError('Changed foreground target '+name)
        data = np.load(target)
        foreground = data['foreground'].astype(np.float32)/255
        alpha = data['alpha'].astype(np.float32)/65535
        if foreground.shape[:2] != alpha.shape: raise ValueError('Target raster mismatch')
        rgba = np.concatenate((foreground,alpha[...,None]),-1)
        arrays.append((torch.as_tensor(rgba,device=pipeline.device,dtype=torch.float16),
                       data['portrait_crop'].astype(int),int(frame['w'])))
        records.append(dict(camera=name,sha256=sha(target)))
    if len(arrays) != 59: raise ValueError('Requires exact train59 target order')
    atomic_json(output/'foreground_objective.json',dict(records=records,opacity_weight=opacity_weight,
        opacity='person matting in known head crops; not hair alpha or depth GT',
        optical_thickness='sum relu(sigma)*delta, supervised against -log(1-alpha), alpha clipped at .999',
        target_rgb='alpha * display(decontaminated linear foreground), black background',
        unknown_crop_pixels='empty head volume outside verified padded crops' if head_crop_coverage else 'excluded from both RGB and opacity terms',
        head_crop_coverage_sha256=sha(head_crop_coverage) if head_crop_coverage else None,
        opacity_objective=opacity_objective,
        training_targets_receipt_sha256=sha(targets/'complete.json'),implementation_sha256=sha(__file__)))
    captured = {}; query = model.field.query_points
    original_outputs=model.get_outputs
    def get_outputs(rays):
        if model.training:
            captured.clear()
            captured['length']=(rays.fars-rays.nears).reshape(-1,1)
        return original_outputs(rays)
    model.get_outputs=get_outputs
    def query_points(*args, **kwargs):
        density,rgb = query(*args,**kwargs)
        if model.training: captured['density'] = density
        return density,rgb
    model.field.query_points = query_points
    def packed_field(module,inputs,outputs):
        if model.training:
            from nerfstudio.field_components.field_heads import FieldHeadNames
            samples=inputs[0]
            captured['packed_density_delta']=(outputs[FieldHeadNames.DENSITY],samples.frustums.ends-samples.frustums.starts)
    model.field.register_forward_hook(packed_field)
    original_loss=model.get_loss_dict
    def get_loss(outputs,batch,metrics=None):
        losses=original_loss(outputs,batch,metrics)
        indices = batch['indices'].to(pipeline.device)
        target = torch.zeros((len(indices),4),device=pipeline.device)
        known = torch.zeros(len(indices),device=pipeline.device,dtype=torch.bool)
        for index in torch.unique(indices[:,0]).tolist():
            select = torch.nonzero(indices[:,0] == index,as_tuple=True)[0]
            rgba,crop,width = arrays[index]
            y,x = crop_indices(indices[select],width,crop)
            valid = (x>=0)&(x<rgba.shape[1])&(y>=0)&(y<rgba.shape[0])
            chosen = select[valid]
            target[chosen] = rgba[y[valid],x[valid]].float(); known[chosen] = True
        if head_crop_coverage is not None:known[:]=True
        if not bool(known.any()): raise ValueError('No known matte rays in sampled batch')
        color = target[known,:3]*target[known,3:]
        difference = outputs['rgb'][known]-color
        kind = model.config.reconstruction_loss_type
        if kind == 'charbonnier': reconstruction = torch.sqrt(difference.square()+1e-4).mean()
        elif kind == 'mse': reconstruction = difference.square().mean()
        elif kind == 'huber': reconstruction = F.huber_loss(outputs['rgb'][known],color,delta=float(model.config.huber_delta))/5
        else: raise ValueError('Unsupported foreground reconstruction objective '+kind)
        if 'packed_ray_indices' in outputs:
            indices_packed=outputs['packed_ray_indices'].long()
            if len(indices_packed):
                density,deltas=captured.pop('packed_density_delta')
                thickness=integrate_packed_thickness(density,deltas,indices_packed,len(indices))
            else:thickness=outputs['accumulation'].reshape(-1)*0
        else:
            density = F.relu(captured.pop('density')).reshape(len(indices),-1)
            thickness = (density*captured['length']/density.shape[1]).sum(-1)
        desired = opacity_target_thickness(target[:,3])
        losses['rgb_loss'] = reconstruction
        if opacity_objective=='thickness':term=F.smooth_l1_loss(thickness[known],desired[known])
        elif opacity_objective=='bernoulli':term=opacity_bernoulli_nll(thickness[known],target[known,3]).mean()
        else:raise ValueError('Unknown foreground opacity objective')
        losses['foreground_opacity_loss'] = opacity_weight*term
        if metrics is not None:
            metrics['foreground_known_fraction'] = known.float().mean().detach()
            metrics['foreground_alpha_mae'] = (outputs['accumulation'].reshape(-1)[known]-target[known,3]).abs().mean().detach()
        captured.clear()
        return losses
    model.get_loss_dict=get_loss


def load_fallback(directory, names):
    receipt = json.loads((directory/'complete.json').read_text())
    if receipt.get('uses_eval_rgb') is not False: raise ValueError('Fallback must be independent of target RGB')
    by = {row['camera']:row for row in receipt['records']}; result = []
    for name in names:
        row=by[name];path=directory/(name+'.npz')
        if sha(path) != row['sha256']: raise ValueError('Fallback changed')
        result.append(np.load(path)['rgb_display'])
    return result
