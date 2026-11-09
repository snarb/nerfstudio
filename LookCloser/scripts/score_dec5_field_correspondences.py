#!/usr/bin/env python3
"""Independent-detector train-view depth proxy for a frozen learned field.

RoMa three-view cycles were not used for SfM initialization or field fitting.
They are noisy triangulated witnesses from training RGB, not independent 3D GT.
All methods use the same dense probe and camera-z/calibration units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dec5_hair import project,ray_to_z,volume_depths,packed_depth_quantile,packed_layer_depth_quantile
from run_dec5_hair_campaign import atomic_json,sha


def score(data,training,correspondences,matching_receipt,output,samples=4096,checkpoint_path=None):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    configs=list(training.rglob('config.yml'))
    if len(configs)!=1:raise ValueError('Expected one pinned model configuration')
    request=json.loads((training/'request.json').read_text())
    if sha(data/'transforms.json')!=request['data_sha256']:raise ValueError('Checkpoint/calibration dataset mismatch')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
    cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data
    pipeline=cfg.pipeline.setup(device='cuda',test_mode='val')
    checkpoint_path=checkpoint_path or training/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    pipeline.load_state_dict(checkpoint['pipeline'],strict=True);pipeline.eval();model=pipeline.model
    model.current_train_step=checkpoint['step'];native_mode=model.config.ray_sampling_mode
    native_adaptive=model.config.enable_adaptive_ray_marching;native_samples=model.config.fixed_num_samples_per_ray
    payload=json.loads((data/'transforms.json').read_text());by_file={Path(f['file_path']).name:f for f in payload['frames']}
    dataset=pipeline.datamanager.train_dataset
    frames=[by_file[Path(p).name] for p in dataset.image_filenames]
    index_by_name={f['physical_camera']:i for i,f in enumerate(frames)}
    normalized=dataset.cameras.camera_to_worlds.cpu().numpy();raw=np.asarray([f['transform_matrix'] for f in frames])
    rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3])
    translation=normalized[0,:3,3]-scale*rotation@raw[0,:3,3]
    if np.linalg.norm(scale*raw[:,:3,3]@rotation.T+translation-normalized[:,:3,3],axis=-1).max()>1e-4:raise ValueError('Bad field/camera similarity')
    witnesses=np.load(correspondences);receipt=json.loads(matching_receipt.read_text());records=[];pooled={};mass_pooled=[];traces=[]
    for key,row in zip(('uv_a','uv_b','uv_c'),receipt['rows']):
        name=row['frame']['physical_camera'];index=index_by_name[name];frame=frames[index]
        np.testing.assert_allclose(frame['transform_matrix'],row['frame']['transform_matrix'],atol=1e-10,rtol=0)
        np.testing.assert_allclose([frame[k] for k in ['fl_x','fl_y','cx','cy','w','h']],
                                  [row['frame'][k] for k in ['fl_x','fl_y','cx','cy','w','h']],atol=1e-10,rtol=0)
        uv=witnesses[key];camera=dataset.cameras[index:index+1].to('cuda')
        rays=camera.generate_rays(0,coords=torch.as_tensor(uv[:,::-1].copy(),device='cuda'),keep_shape=False)
        model.config.ray_sampling_mode=native_mode;model.config.enable_adaptive_ray_marching=native_adaptive
        model.config.fixed_num_samples_per_ray=native_samples
        with torch.no_grad():native=model(rays)
        native_z=ray_to_z(native['depth'].reshape(-1).float().cpu().numpy()/scale,uv,frame)
        native_valid=native['accumulation'].reshape(-1).float().cpu().numpy()>=.1
        native_median=None;native_head_median=None
        if 'packed_ray_indices' in native:
            values=[native[k].detach().float().cpu().numpy() for k in ('packed_spacing_starts','packed_spacing_ends','packed_weights')]
            distance=packed_depth_quantile(*values,native['packed_ray_indices'].cpu().numpy(),
                rays.nears.cpu().numpy(),rays.fars.cpu().numpy())
            native_median=ray_to_z(distance/scale,uv,frame)
            matrix=getattr(model.field,'density_geometry_roi_transform',None)
            if matrix is not None:
                ids=native['packed_ray_indices'].long();starts=native['packed_spacing_starts'].reshape(-1);ends=native['packed_spacing_ends'].reshape(-1)
                mid=rays.nears.reshape(-1)[ids]+(starts+ends)*.5*(rays.fars-rays.nears).reshape(-1)[ids]
                xyz=rays.origins[ids]+rays.directions[ids]*mid[:,None]
                unit=(xyz.float()[:,None,:]*matrix[None,:,:3]).sum(-1)+matrix[:,3];inside=((unit>=0)&(unit<=1)).all(-1)
                hd,hm=packed_layer_depth_quantile(*values,ids.cpu().numpy(),rays.nears.cpu().numpy(),rays.fars.cpu().numpy(),inside.cpu().numpy(),minimum_mass=.001)
                native_head_median=ray_to_z(hd/scale,uv,frame)
        model.config.ray_sampling_mode='fixed';model.config.enable_adaptive_ray_marching=False;model.config.fixed_num_samples_per_ray=samples
        with torch.no_grad():result=model(rays)
        weights=result['loss_weights'].float()[...,0].cpu().numpy()
        near=rays.nears[:,0].float().cpu().numpy();far=rays.fars[:,0].float().cpu().numpy()
        t=(near[:,None]+(far-near)[:,None]*(np.arange(samples)+.5)/samples)/scale
        depth_samples=ray_to_z(t,uv[:,None,:],frame);profiles=volume_depths(depth_samples,weights)
        _,reference=project(witnesses['points'],frame)
        native_relative=np.where(native_valid&np.isfinite(native_z)&(native_z>0),np.abs(native_z-reference)/reference,1.)
        metrics=dict(native_expected=dict(valid_fraction=float(native_valid.mean()),median_absolute_relative=float(np.median(native_relative)),
            p90_absolute_relative=float(np.quantile(native_relative,.9))))
        pooled.setdefault('native_expected',[]).append(native_relative)
        if native_median is not None:
            valid=native_valid&np.isfinite(native_median)&(native_median>0)
            relative=np.where(valid,np.abs(native_median-reference)/reference,1.)
            metrics['native_median']=dict(valid_fraction=float(valid.mean()),median_absolute_relative=float(np.median(relative)),p90_absolute_relative=float(np.quantile(relative,.9)))
            pooled.setdefault('native_median',[]).append(relative)
        if native_head_median is not None:
            valid=np.isfinite(native_head_median)&(native_head_median>0)&(hm>=.001)
            relative=np.where(valid,np.abs(native_head_median-reference)/reference,1.)
            metrics['native_head_median']=dict(valid_fraction=float(valid.mean()),median_absolute_relative=float(np.median(relative)),p90_absolute_relative=float(np.quantile(relative,.9)))
            pooled.setdefault('native_head_median',[]).append(relative)
        opacity=weights.sum(-1)
        mass=np.stack([(weights*(depth_samples<reference[:,None]*.99)).sum(-1),
                       (weights*((depth_samples>=reference[:,None]*.99)&(depth_samples<=reference[:,None]*1.01))).sum(-1),
                       (weights*(depth_samples>reference[:,None]*1.01)).sum(-1)],-1)/np.maximum(opacity[:,None],1e-8)
        mass_pooled.append(mass)
        selected=np.linspace(0,len(uv)-1,min(3,len(uv)),dtype=int)
        traces.append(dict(camera=name,witness_indices=selected,uv=uv[selected],reference=reference[selected],z=depth_samples[selected],weights=weights[selected]))
        for mode in ('expected','front','peak1','peak1_subpixel'):
            if mode not in profiles:continue
            depth=profiles[mode];valid=np.isfinite(depth)&(depth>0)
            relative=np.where(valid,np.abs(depth-reference)/reference,1.)
            signed=np.where(valid,(depth-reference)/reference,1.)
            metrics[mode]=dict(valid_fraction=float(valid.mean()),median_absolute_relative=float(np.median(relative)),
                p90_absolute_relative=float(np.quantile(relative,.9)),depth_over_5percent_behind_fraction=float((signed>.05).mean()))
            pooled.setdefault(mode,[]).append(relative)
        records.append(dict(camera=name,witnesses=len(uv),opacity_median=float(np.median(opacity)),metrics=metrics,
            median_weight_mass_before_near_behind_witness=np.median(mass,axis=0).tolist()))
    aggregate={mode:dict(median_absolute_relative=float(np.median(np.concatenate(rows))),
                       p90_absolute_relative=float(np.quantile(np.concatenate(rows),.9))) for mode,rows in pooled.items()}
    profile_path=output.with_suffix('.profiles.npz')
    np.savez_compressed(profile_path,**{str(i)+'_'+key:value for i,trace in enumerate(traces) for key,value in trace.items()})
    atomic_json(output,dict(checkpoint_sha256=sha(checkpoint_path),step=checkpoint['step'],samples=samples,
        native_ray_sampling_mode=native_mode,probe_ray_sampling_mode='fixed',records=records,aggregate=aggregate,
        witness_sha256=sha(correspondences),matching_receipt_sha256=sha(matching_receipt),
        proxy='independent-detector triangulated train-RGB witnesses, not independent 3D ground truth',
        witness_window_relative=.01,median_weight_mass_before_near_behind_witness=np.median(np.concatenate(mass_pooled),axis=0).tolist(),
        profiles_sha256=sha(profile_path),
        invalid_depth_relative_penalty=1.,depth_convention='camera-z in calibration units',uses_validation_rgb=False,
        uses_test_rgb=False,implementation_sha256=sha(__file__)))
    print(json.dumps(aggregate),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','training','correspondences','matching-receipt','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--samples',type=int,default=4096);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    score(a.data,a.training,a.correspondences,a.matching_receipt,a.output,a.samples,a.checkpoint)


if __name__=='__main__':main()
