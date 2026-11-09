#!/usr/bin/env python3
"""Export a frozen LookCloser field's actual native marcher depth in a fixed ROI.

This is separate from the common uniform density probe: occupancy and adaptive
intervals remain enabled exactly as configured by the selected checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import yaml

from dec5_hair import ray_to_z,packed_depth_quantile,packed_depth_quantile_vectorized
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def export(data,training,parent,output,manifest=None,checkpoint_path=None,source_semantics=None,include_median=False,include_rgb=False,uint8_cpu_cache=False,extra_quantiles=(),empty_space_grid=None,min_background_views=3,sigma_surface_calibration=None,rays_per_chunk=256,minimum_opacity=.1,source_roi_person=False,vectorized_quantiles=False):
    from nerfstudio.cameras.cameras import Cameras
    if not 1<=rays_per_chunk<=8192:raise ValueError('Bounded positive export batch required')
    if sigma_surface_calibration and empty_space_grid:raise ValueError('Sigma bisection and a hook-only empty-space gate cannot be combined')
    if not np.isfinite(minimum_opacity) or not 0<=minimum_opacity<=1:raise ValueError("Invalid minimum opacity")
    extra_quantiles=tuple(float(q) for q in extra_quantiles)
    quantile_function=packed_depth_quantile_vectorized if vectorized_quantiles else packed_depth_quantile
    if any(not 0<q<1 for q in extra_quantiles) or len(set(extra_quantiles))!=len(extra_quantiles):raise ValueError('Unique quantiles strictly inside (0,1) required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(24*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    request=json.loads((training/'request.json').read_text())
    if request['data_sha256']!=sha(data/'transforms.json'):raise ValueError('Model dataset/calibration mismatch')
    configs=list(training.rglob('config.yml'))
    if len(configs)!=1:raise ValueError('Expected one training configuration')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
    cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data
    if uint8_cpu_cache:cfg.pipeline.datamanager.cache_images_type='uint8'
    pipeline=cfg.pipeline.setup(device='cuda',test_mode='val')
    checkpoint_path=checkpoint_path or training/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    pipeline.load_state_dict(checkpoint['pipeline'],strict=True);pipeline.eval();model=pipeline.model
    from dec5_checkpoint_runtime import restore as restore_runtime
    restored_strength=restore_runtime(model,checkpoint,request)
    model.current_train_step=checkpoint['step'];before=fingerprint(model)
    if bool(getattr(model.field,'use_view_directions',True))!=bool(request.get('use_view_directions',True)):
        raise ValueError('Loaded field does not implement the requested color direction mode')
    background_checkpoint=None;background_provider=None
    if request.get('ray_background') or ('fr_switch_after' in request and (training/'background_dependency.json').is_file()):
        from dec5_background_separation import resolve_background,attach_inference
        background_checkpoint=resolve_background(request,data,training)
        if include_rgb:background_provider=attach_inference(pipeline,background_checkpoint,request.get('recipe',{}).get('background_volume'))
    payload=json.loads((data/'transforms.json').read_text());by_file={Path(f['file_path']).name:f for f in payload['frames']}
    train=pipeline.datamanager.train_dataset
    raw=np.asarray([by_file[Path(p).name]['transform_matrix'] for p in train.image_filenames])
    normalized=train.cameras.camera_to_worlds.cpu().numpy()
    rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3])
    translation=normalized[0,:3,3]-scale*rotation@raw[0,:3,3]
    error=np.linalg.norm(scale*raw[:,:3,3]@rotation.T+translation-normalized[:,:3,3],axis=-1).max()
    if error>1e-4:raise ValueError('Camera alignment is not one verified similarity')
    empty_gate=None
    if empty_space_grid:
        from dec5_empty_space_gate import EmptySpaceGate
        empty_gate=EmptySpaceGate(model,empty_space_grid,min_background_views,payload['frame_id']);empty_gate.check_normalization(rotation,translation,scale)
    surface_meta=None;capture={}
    if sigma_surface_calibration:
        from dec5_sigma_surface import first_sigma_surface
        from nerfstudio.field_components.field_heads import FieldHeadNames
        surface=json.loads(sigma_surface_calibration.read_text())
        if surface['checkpoint_sha256']!=sha(checkpoint_path) or surface['dataset_sha256']!=sha(data/'transforms.json') or surface['uses_validation_rgb'] or surface['uses_test_rgb']:raise ValueError('Sigma surface calibration mismatch')
        np.testing.assert_allclose(rotation,surface['rotation'],rtol=0,atol=1e-5);np.testing.assert_allclose(translation,surface['translation'],rtol=0,atol=1e-5);np.testing.assert_allclose(scale,surface['scale'],rtol=1e-5,atol=1e-7)
        surface_meta=dict(calibration_sha256=sha(sigma_surface_calibration),threshold_density_normalized=surface['threshold_density_normalized'],threshold_density_per_calibration_unit=surface['threshold_density_per_calibration_unit'],refine_steps=surface['refine_steps'],minimum_expected_opacity_not_applied=True,method='first detected absolute density crossing in native intervals, world-space bisection; not a certified first root in unsampled gaps')
        model.field.register_forward_hook(lambda module,args,out:capture.update(density=out[FieldHeadNames.DENSITY].detach()))
    frames=(json.loads(manifest.read_text())['frames'] if manifest else
            [f for f in payload['frames'] if f['file_path'] in payload['train_filenames' if source_semantics else 'val_filenames']])
    output.mkdir(parents=True,exist_ok=False);records=[]
    for frame in frames:
        name=frame['physical_camera']
        if source_semantics:
            if frame['file_path'] not in payload['train_filenames']:raise ValueError('Source visibility maps require train cameras')
            semantic=np.load(source_semantics/(name+'.npz'))['probability'].astype(np.float32)/65535
            domain=(semantic[...,0]>.05)|(semantic[...,1]>.05)
            if source_roi_person:domain|=semantic[...,2]<.5
            roi=cv2.dilate(domain.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)
        else:roi=np.load(parent/(name+'.npz'))['roi']
        y,x=np.nonzero(roi)
        uv=np.stack((x+.5,y+.5),-1);pose=np.asarray(frame['transform_matrix'])
        c2w=np.concatenate((rotation@pose[:3,:3],(scale*rotation@pose[:3,3]+translation)[:,None]),-1)
        camera=Cameras(camera_to_worlds=torch.as_tensor(c2w,dtype=torch.float32)[None],fx=frame['fl_x'],fy=frame['fl_y'],
            cx=frame['cx'],cy=frame['cy'],width=frame['w'],height=frame['h']).to('cuda')
        depth=np.zeros(roi.shape,np.float32);opacity=np.zeros(roi.shape,np.float32)
        median=np.zeros(roi.shape,np.float32) if include_median else None
        sigma_depth=np.zeros(roi.shape,np.float32) if surface_meta else None
        rgb=np.zeros(roi.shape+(3,),np.float32) if include_rgb else None
        quantiles={('q%04d'%round(q*10000)):np.zeros(roi.shape,np.float32) for q in extra_quantiles}
        weight_mass=np.zeros(roi.shape,np.float64) if include_median or extra_quantiles else None
        if len(quantiles)!=len(extra_quantiles):raise ValueError('Quantile channel names collide')
        for start in range(0,len(uv),rays_per_chunk):
            pixels=uv[start:start+rays_per_chunk]
            rays=camera.generate_rays(0,coords=torch.as_tensor(pixels[:,::-1].copy(),device='cuda',dtype=torch.float32),keep_shape=False)
            capture.clear()
            with torch.no_grad():result=model(rays)
            z=ray_to_z(result['depth'].reshape(-1).float().cpu().numpy()/scale,pixels,frame)
            a=result['accumulation'].reshape(-1).float().cpu().numpy()
            valid=np.isfinite(z)&(z>0)&np.isfinite(a)&(a>=minimum_opacity)
            yy,xx=y[start:start+len(pixels)],x[start:start+len(pixels)]
            depth[yy,xx]=np.where(valid,z,0);opacity[yy,xx]=a
            if include_rgb:rgb[yy,xx]=result['rgb'].reshape(-1,3).float().cpu().numpy()
            if surface_meta:
                ids=result['packed_ray_indices'];density=capture.get('density')
                if len(ids) and (density is None or density.numel()!=len(ids)):raise ValueError('Packed native density capture mismatch')
                with torch.no_grad():
                    distance=first_sigma_surface(model.field.density_fn,rays,result['packed_spacing_starts'],result['packed_spacing_ends'],density if density is not None else ids.new_empty(0),ids,surface_meta['threshold_density_normalized'],surface_meta['refine_steps'],assume_sorted=True)
                zs=ray_to_z(distance.cpu().numpy()/scale,pixels,frame);sigma_depth[yy,xx]=np.where(np.isfinite(zs)&(zs>0),zs,0)
            if include_median or extra_quantiles:
                if 'packed_ray_indices' not in result:raise ValueError('Native median requires packed adaptive intervals')
                values=[result[k].detach().float().cpu().numpy() for k in ('packed_spacing_starts','packed_spacing_ends','packed_weights')]
                ids=result['packed_ray_indices'].cpu().numpy();near=rays.nears.cpu().numpy();far=rays.fars.cpu().numpy()
                weight_mass[yy,xx]=np.bincount(ids,weights=np.maximum(values[2].reshape(-1).astype(np.float64),0),minlength=len(pixels))
                if include_median:
                    distance=quantile_function(*values,ids,near,far,minimum_mass=minimum_opacity)
                    zm=ray_to_z(distance/scale,pixels,frame)
                    median[yy,xx]=np.where(valid&np.isfinite(zm)&(zm>0),zm,0)
                for q in extra_quantiles:
                    distance=quantile_function(*values,ids,near,far,quantile=q,minimum_mass=minimum_opacity)
                    zq=ray_to_z(distance/scale,pixels,frame)
                    quantiles['q%04d'%round(q*10000)][yy,xx]=np.where(valid&np.isfinite(zq)&(zq>0),zq,0)
        arrays=dict(expected=depth,opacity=opacity,roi=roi,**quantiles)
        if weight_mass is not None:arrays['weight_mass']=weight_mass
        if include_median:arrays['median']=median
        if surface_meta:arrays['sigma_surface']=sigma_depth
        if include_rgb:arrays['rgb_display']=rgb
        path=output/(name+'.npz');np.savez_compressed(path,**arrays)
        records.append(dict(camera=name,sha256=sha(path),roi_pixels=len(uv),valid_depth_pixels=int((depth>0).sum()),sigma_surface_pixels=int((sigma_depth>0).sum()) if surface_meta else None))
        atomic_json(output/'progress.json',dict(completed=len(records),total=len(frames),camera=name))
    if background_provider:background_provider.verify()
    if fingerprint(model)!=before:raise ValueError('Native depth export mutated the frozen field')
    atomic_json(output/'complete.json',dict(records=records,checkpoint_sha256=sha(checkpoint_path),checkpoint_path=str(checkpoint_path),
        step=checkpoint['step'],frozen_state_unchanged=True,native_ray_sampling_mode=model.config.ray_sampling_mode,
        restored_feature_reweighting_strength=restored_strength,
        use_view_directions=bool(getattr(model.field,'use_view_directions',True)),
        native_adaptive_enabled=model.config.enable_adaptive_ray_marching,depth_convention='OpenCV camera-z',
        units='calibration units',normalized_per_calibration_unit=float(scale),pose_alignment_max_error=float(error),
        dataset_sha256=sha(data/'transforms.json'),camera_manifest_sha256=sha(manifest) if manifest else None,
        target_roi_parent_receipt_sha256=sha(parent/'complete.json') if parent else None,
        source_visibility_only=bool(source_semantics),source_roi=('train person union hair/face dilation12px' if source_roi_person else 'train semantic hair/face dilation12px') if source_semantics else None,
        quantile_evaluator='vectorized FP64 independent ray CDFs' if vectorized_quantiles else 'scalar reference',
        sigma_surface=surface_meta,rays_per_chunk=rays_per_chunk,empty_space_gate=empty_gate.meta if empty_gate else None,includes_native_rgb=include_rgb,extra_quantiles=list(extra_quantiles),ray_background_checkpoint_sha256=sha(background_checkpoint) if background_checkpoint else None,
        native_rgb_composition='foreground + frozen ray background' if background_provider else 'field RGB',cpu_image_cache='uint8' if uint8_cpu_cache else 'configured',
        uses_validation_rgb=False,uses_test_rgb=False,
        opacity='native volume opacity, not a semantic hair alpha',minimum_opacity=minimum_opacity,implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','training','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--source-semantics',type=Path)
    p.add_argument('--source-roi-person',action='store_true',help='Include all train-person semantics in source query domain; default head unchanged')
    p.add_argument('--vectorized-quantiles',action='store_true',help='Use the tested FP64 independent-CDF batch evaluator')
    p.add_argument('--include-median',action='store_true')
    p.add_argument('--sigma-surface-calibration',type=Path);p.add_argument('--rays-per-chunk',type=int,default=256)
    p.add_argument('--empty-space-grid',type=Path);p.add_argument('--min-background-views',type=int,choices=[3,8],default=3)
    p.add_argument('--extra-quantiles',type=float,nargs='*',default=[],help='Diagnostic conditional depths from the same native packed weights; no new field queries')
    p.add_argument('--minimum-opacity',type=float,default=.1)
    p.add_argument('--include-rgb',action='store_true')
    p.add_argument('--uint8-cpu-cache',action='store_true',help='Reduce unused source RGB cache during frozen geometry export')
    p.add_argument('--camera-manifest',type=Path)
    p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    if bool(a.parent)==bool(a.source_semantics):p.error('Choose exactly one fixed target ROI parent or train source semantics')
    export(a.data,a.training,a.parent,a.output,a.camera_manifest,a.checkpoint,a.source_semantics,a.include_median,a.include_rgb,a.uint8_cpu_cache,a.extra_quantiles,a.empty_space_grid,a.min_background_views,a.sigma_surface_calibration,a.rays_per_chunk,a.minimum_opacity,a.source_roi_person,a.vectorized_quantiles)


if __name__=='__main__':main()
