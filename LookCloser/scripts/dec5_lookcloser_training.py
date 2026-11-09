"""Paper-based LookCloser controls and train-only sparse initialization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nerfstudio.model_components.lookcloser_grid import normalized_frequency_focal
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng
from run_dec5_hair_campaign import atomic_json,sha


def configure(cfg,point_budget=1<<18):
    pipe=cfg.pipeline;model=pipe.model;sampler=pipe.datamanager.pixel_sampler
    pipe.enable_frequency_grid=True;pipe.frequency_projection_mode='normalized_image'
    pipe.grid_update_interval=1024;pipe.grid_update_batch_size=2048
    pipe.target_num_samples_per_batch=point_budget
    pipe.dynamic_rays_min=128;pipe.dynamic_rays_max=8192
    pipe.datamanager.cpu_fas_prefetch=False
    model.enable_frequency_grid=True;model.enable_feature_reweighting=True
    model.density_activation='trunc_exp'
    model.feature_reweighting_strength=1.;model.enable_adaptive_ray_marching=True
    model.ray_sampling_mode='adaptive';model.max_steps_per_ray=2048
    model.fixed_num_samples_per_ray=2048;model.adaptive_coarse_step_size=.00625
    model.adaptive_warmup_steps=4096;model.occupancy_warmup_steps=4096
    model.occupancy_binary_warmup_steps=4096;model.stable_occupancy_reduction=True
    model.depth_loss_mult=.001;model.depth_loss_steps=5000
    model.reconstruction_loss_type='charbonnier';model.distortion_loss_mult=.01
    sampler.enable_fas=True;sampler.sampling_ramp_start=1.;sampler.sampling_ramp_end=3.
    sampler.fas_strength=1.;sampler.fas_warmup_steps=1000;sampler.fas_ramp_steps=2000
    sampler.fas_level_count_alpha=0.;sampler.fas_patch_group_size=1
    cfg.optimizers['fields']['optimizer'].betas=(.9,.99)


def configure_sparse_scene_bounds(cfg,path):
    """The paper bounds scene content using SfM points, rather than camera extent."""
    data=cfg.pipeline.datamanager.data;receipt=json.loads(path.with_suffix('.json').read_text())
    if receipt['dataset_sha256']!=sha(data/'transforms.json') or receipt['npz_sha256']!=sha(path):
        raise ValueError('Sparse scene-bound input mismatch')
    parsed=cfg.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train')
    payload=json.loads((data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']}
    raw=np.asarray([by[Path(p).name]['transform_matrix'] for p in parsed.image_filenames])
    normalized=parsed.cameras.camera_to_worlds.cpu().numpy()
    rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3])
    translation=normalized[0,:3,3]-scale*rotation@raw[0,:3,3]
    points=scale*np.load(path)['xyz']@rotation.T+translation
    bounds=np.quantile(points,[.005,.995],axis=0)
    scene_scale=float(np.abs(bounds).max()*1.1+.005)
    coverage=float((np.abs(points)<=scene_scale).all(-1).mean())
    if coverage<.98:raise ValueError('Robust scene bound excludes too much reliable SfM content')
    old=cfg.pipeline.datamanager.dataparser.scene_scale
    cfg.pipeline.datamanager.dataparser.scene_scale=scene_scale
    span=2*scene_scale;model=cfg.pipeline.model
    model.adaptive_coarse_step_size=span/640
    model.adaptive_min_step_size=min(1e-4,span/(4*model.max_res))
    model.adaptive_max_step_size=min(.1,span/32)
    return dict(method='train59 SfM 0.5–99.5% bounds, origin-centered enclosing cube, 10% + .005 normalized margin',
        old_scene_scale=float(old),scene_scale=scene_scale,normalized_bounds=bounds.tolist(),sparse_point_coverage=coverage,
        adaptive_coarse_step_size=model.adaptive_coarse_step_size,adaptive_min_step_size=model.adaptive_min_step_size,
        adaptive_max_step_size=model.adaptive_max_step_size,sparse_sha256=sha(path),uses_validation_rgb=False,uses_test_rgb=False)


def initialize_sparse(trainer,path,output,initialize_grid=True):
    pipe=trainer.pipeline;dataset=pipe.datamanager.train_dataset;device=pipe.device
    receipt=json.loads(path.with_suffix('.json').read_text());source=np.load(path)
    data=pipe.datamanager.config.data
    if receipt['dataset_sha256']!=sha(data/'transforms.json') or sha(path)!=receipt['npz_sha256']:
        raise ValueError('Sparse initializer dataset or bytes mismatch')
    if receipt.get('uses_validation_rgb') is not False or receipt.get('uses_test_rgb') is not False:
        raise ValueError('Sparse initialization must be train-only')
    payload=json.loads((data/'transforms.json').read_text());by_file={Path(f['file_path']).name:f for f in payload['frames']}
    frames=[by_file[Path(p).name] for p in dataset.image_filenames]
    names={f['physical_camera']:i for i,f in enumerate(frames)}
    remap=np.asarray([names[name] for name in source['camera_names']])
    normalized=dataset.cameras.camera_to_worlds.cpu().numpy();raw=np.asarray([f['transform_matrix'] for f in frames])
    rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3])
    translation=normalized[0,:3,3]-scale*rotation@raw[0,:3,3]
    pose_error=np.linalg.norm(scale*raw[:,:3,3]@rotation.T+translation-normalized[:,:3,3],axis=-1).max()
    if pose_error>1e-4:raise ValueError('Sparse normalization is not a single verified camera similarity')
    points=torch.as_tensor(scale*source['xyz']@rotation.T+translation,device=device,dtype=torch.float32)
    op=torch.as_tensor(source['observation_point'].astype(np.int64),device=device)
    oc=torch.as_tensor(remap[source['observation_camera']],device=device)
    uv=torch.as_tensor(source['uv'],device=device)
    cameras=dataset.cameras.to(device);grid=pipe.model.freq_grid
    xyz=points[op];center=cameras.camera_to_worlds[oc,:3,3]
    z=-((xyz-center)*cameras.camera_to_worlds[oc,:3,2]).sum(-1)
    inside=((xyz>=grid.aabb_min_buf)&(xyz<=grid.aabb_max_buf)).all(-1)&(z>0)
    if not bool(inside.any()):raise ValueError('No reliable sparse observations inside the normalized scene')
    f2d=torch.zeros(len(op),device=device)
    for index in torch.unique(oc).tolist():
        mask=oc==index;fmap=pipe.cached_freq_maps[index].to(device)
        stride=pipe.cached_freq_strides[index]
        x=(uv[mask,0]/stride).long().clamp(0,fmap.shape[1]-1)
        y=(uv[mask,1]/stride).long().clamp(0,fmap.shape[0]-1)
        f2d[mask]=fmap[y,x]
    fx=cameras.fx[oc].reshape(-1);fy=cameras.fy[oc].reshape(-1)
    focal=normalized_frequency_focal(fx,fy,cameras.width[oc].reshape(-1),cameras.height[oc].reshape(-1),grid.aabb_size_buf.max())
    frequency=(f2d*focal/z.clamp_min(1e-6)).cpu().numpy()
    ids=op.cpu().numpy();valid=inside.cpu().numpy();order=np.argsort(ids[valid],kind='stable')
    sorted_ids=ids[valid][order];values=frequency[valid][order]
    unique,counts=np.unique(sorted_ids,return_counts=True)
    medians=np.asarray([np.median(x) for x in np.split(values,np.cumsum(counts)[:-1])],np.float32)
    levels=grid.freq_to_level(torch.as_tensor(medians,device=device))
    if initialize_grid:
        with torch.no_grad():grid.update_max(points[torch.as_tensor(unique,device=device)],levels)
    atomic_json(output/'sparse_frequency_initialization.json',dict(points=len(unique),observations=int(inside.sum()),
        pose_similarity_max_error=float(pose_error),normalized_per_calibration_unit=float(scale),
        frequency_projection='UV resolution * mean(fx/W,fy/H) * AABB span / camera-z',
        frequency_level_histogram=torch.bincount(levels.long(),minlength=grid.num_levels).cpu().tolist(),
        occupied_frequency_cells=int((grid.grid>0).sum()),sparse_npz_sha256=sha(path),
        initialized_grid=initialize_grid,implementation_sha256=sha(__file__),uses_validation_rgb=False,uses_test_rgb=False))
    # The same early sparse supervision is used by all sampling arms.
    original=pipe.get_train_loss_dict
    available=torch.nonzero(inside,as_tuple=True)[0]
    def train_loss(step):
        outputs,losses,metrics=original(step)
        if step<int(pipe.model.config.depth_loss_steps):
            with fork_seeded_rng(pipe.config.training_seed,'sparse_depth',step,device):
                selected=available[torch.randint(len(available),(64,),device=device)]
            index=oc[selected];rays=cameras.generate_rays(index[:,None],coords=uv[selected][:,[1,0]],keep_shape=False)
            result=pipe.model(rays)
            target=torch.linalg.norm(points[op[selected]]-cameras.camera_to_worlds[index,:3,3],dim=-1)
            predicted=result['depth'].reshape(-1)*result['accumulation'].reshape(-1)
            losses['sparse_depth_loss']=float(pipe.model.config.depth_loss_mult)*torch.sqrt((predicted-target).square()+1e-4).mean()
            extra=(result['packed_ray_indices'].numel() if 'packed_ray_indices' in result else result['loss_weights'].numel())
            pipe.cumulative_point_samples.add_(extra)
            metrics['cumulative_point_samples']=pipe.cumulative_point_samples.clone()
            metrics['sparse_prior_rays']=torch.tensor(64,device=device)
        return outputs,losses,metrics
    pipe.get_train_loss_dict=train_loss
