"""Quiet, checkpointed LookCloser actor training and masked-ROI diagnostic probes.

Explicitly separates diagnostic subsets from a final campaign. Uses the real
LookCloser pipeline/field, field/Adam continuation and measured evaluation.
Selection uses masked ROI metrics, not full-frame final validation.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import random
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mesh_distillation_background import check, read, write, sha, publish_hardlink
from nerfstudio.pipelines.mesh_distillation_pipeline import DistillationPipelineConfig, DistillationParserConfig
from nerfstudio.model_components.mesh_distillation import projected_frequency


def configuration(args):
    cfg=DistillationPipelineConfig(grid_update_interval=0 if args.freeze_grid else 1024)
    cfg.datamanager.dataparser=DistillationParserConfig(data=args.data,orientation_method='none',center_method='none',
        auto_scale_poses=False,scale_factor=1.,depth_unit_scale_factor=1.,downscale_factor=1,load_3D_points=False,eval_mode='filename')
    cfg.datamanager.train_num_rays_per_batch=args.rays;cfg.datamanager.eval_num_rays_per_batch=1024
    cfg.datamanager.images_on_gpu=True;cfg.datamanager.masks_on_gpu=True
    cfg.datamanager.native_training_manifest=getattr(args,'native_training_manifest',None)
    cfg.datamanager.observed_background_path=getattr(args,'observed_background',None)
    cfg.datamanager.observed_trimap_path=getattr(args,'observed_trimap',None)
    cfg.datamanager.native_observed_background=getattr(args,'native_observed_background',False)
    cfg.datamanager.stereo_depth_receipt=getattr(args,'stereo_depth',None)
    cfg.model.stereo_depth_weight=getattr(args,'stereo_depth_weight',0.)
    cfg.model.grid_resolution=128;cfg.model.log2_hashmap_size=args.hash_size
    cfg.model.max_res=args.max_resolution
    cfg.model.ray_sampling_mode=args.sampling;cfg.model.fixed_num_samples_per_ray=args.samples
    cfg.model.fixed_stratified_sampling=args.stratified
    cfg.model.fixed_importance_samples=args.importance_samples
    cfg.model.optimize_training_cameras=args.camera_lr>0
    cfg.model.freeze_decoders=args.freeze_decoders
    cfg.model.adaptive_warmup_steps=1024;cfg.model.occupancy_warmup_steps=1024;cfg.model.occupancy_binary_warmup_steps=1024
    cfg.model.adaptive_coarse_step_size=.001;cfg.model.adaptive_min_step_size=1e-5
    cfg.model.corrected_arm_allocator=True;cfg.model.adaptive_interval_level_mode='max3'
    cfg.model.correct_sh_directions=not args.legacy_sh
    cfg.model.view_independent_color=args.view_independent_color
    cfg.model.normalized_exponential_density=args.normalized_density
    cfg.model.foreground_support_path=args.foreground_support
    cfg.model.background_checkpoint=args.background_checkpoint
    cfg.model.equipment_slab=None if args.equipment_slab is None else tuple(args.equipment_slab)
    cfg.model.density_slab_plane=None if args.slab_plane is None else tuple(read(args.slab_plane)['plane'])
    cfg.datamanager.pixel_sampler.empty_fraction=args.empty_fraction
    cfg.datamanager.pixel_sampler.detail_fraction=args.detail_fraction
    cfg.datamanager.pixel_sampler.boundary_fraction=getattr(args,'boundary_fraction',0.)
    cfg.datamanager.pixel_sampler.boundary_radius=getattr(args,'boundary_radius',3)
    cfg.model.empty_opacity_weight=args.empty_weight
    cfg.model.matte_opacity_weight=args.matte_weight
    cfg.model.matte_boundary_only=not args.matte_all_pixels
    cfg.model.matte_soft_target_weight=getattr(args,'matte_soft_target_weight',1.)
    cfg.model.opacity_hard_fraction=getattr(args,'opacity_hard_fraction',1.)
    cfg.model.freeze_frequency_grid=args.freeze_grid
    cfg.model.depth_loss_mult=args.depth_weight;cfg.model.depth_loss_steps=args.depth_steps
    cfg.model.depth_distribution_weight=args.depth_distribution_weight
    cfg.model.depth_distribution_sigma=args.depth_distribution_sigma
    cfg.model.distortion_loss_mult=args.distortion_weight;cfg.model.eval_num_rays_per_chunk=2048
    cfg.model.opacity_neutral_distortion=getattr(args,'opacity_neutral_distortion',False)
    return cfg


@torch.no_grad()
def initialize_grid(pipe):
    """Dense trusted teacher point evidence; unknown voxels retain capacity."""
    ds=pipe.datamanager.train_dataset;grid=pipe.model.freq_grid;count=0
    for i in range(len(ds)):
        batch=ds[i];z=batch['depth_image'][::4,::4,0].cuda();h,w=z.shape
        yy,xx=torch.meshgrid(torch.arange(h,device='cuda')*4,torch.arange(w,device='cuda')*4,indexing='ij')
        good=(z>0)&torch.isfinite(z)&(batch['confidence'][::4,::4,0].cuda()>0)
        good &= pipe.cached_valid_maps[i][yy//8,xx//8]
        if not good.any():continue
        y,x=yy[good],xx[good];depth=z[good]
        camera=ds.cameras[i:i+1].to('cuda');rays=camera.generate_rays(0,coords=torch.stack([y,x],-1).float()+.5)
        freq=pipe.cached_freq_maps[i].cuda()[y//8,x//8]
        resolution=projected_frequency(freq,camera.fx,camera.fy,camera.width,camera.height,depth,grid.aabb_size_buf).flatten()
        point=rays.origins+rays.directions*(depth[:,None]*rays.metadata['directions_norm'])
        grid.update_max(point,grid.freq_to_level(resolution));count+=len(point)
    grid.initialized.copy_(grid.observed.any())
    return dict(projected_points=count,observed_voxels=int(grid.observed.sum()),frequency_digest=pipe.frequency_digest())


@torch.no_grad()
def evaluate(pipe,folder,step,scale=2):
    pipe.eval();folder.mkdir(parents=True,exist_ok=True);results=[]
    datasets=[('train',pipe.datamanager.train_dataset,[0]),('val',pipe.datamanager.eval_dataset,range(len(pipe.datamanager.eval_dataset)))]
    for split,ds,indices in datasets:
        for i in indices:
            data=ds[i];gt=data['image'][::scale,::scale].cuda();h,w=gt.shape[:2]
            camera=ds.cameras[i:i+1].to('cuda')
            if split=='train' and pipe.model.config.optimize_training_cameras:
                camera.metadata={'cam_idx':i}
                camera.camera_to_worlds=pipe.model.training_camera_optimizer.apply_to_camera(camera)
            yy,xx=torch.meshgrid(torch.arange(h,device='cuda')*scale,torch.arange(w,device='cuda')*scale,indexing='ij')
            rays=camera.generate_rays(0,coords=torch.stack([yy,xx],-1).float()+.5)
            output=pipe.model.get_outputs_for_camera_ray_bundle(rays)
            data['image']=gt;data['evaluation_mask']=data['evaluation_mask'][::scale,::scale].cuda()
            metrics,_=pipe.model.get_image_metrics_and_images(output,data)
            mask=data['evaluation_mask'][...,0]>0
            metrics.update(split=split,index=i,image=ds.image_filenames[i].name,
                           mean_actor_opacity=float(output['accumulation'][mask].mean()))
            results.append(metrics)
            panel=torch.cat([torch.where(mask[...,None],gt,0),output['rgb'],output['accumulation'].expand_as(gt)],1)
            Image.fromarray((panel.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{split}_{i:04d}.jpg')
            # Save face/hair/hand region at the actual evaluation sampling scale.
            for name,(x0,y0,x1,y1) in {'head':(640,256,1408,960),'hand':(384,256,896,704)}.items():
                crop=torch.cat([gt[y0//scale:y1//scale,x0//scale:x1//scale],output['rgb'][y0//scale:y1//scale,x0//scale:x1//scale]],1)
                Image.fromarray((crop.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{split}_{i:04d}_{name}.png')
    val=[r for r in results if r['split']=='val']
    metrics={f'eval_all_{key}':float(np.mean([r[key] for r in val])) for key in ['psnr','ssim','lpips']}
    receipt=dict(step=step,metrics=metrics,per_view=results,scale=scale,
                 protocol='Exact GT mask PSNR; tight zero-masked ROI SSIM/LPIPS; sampled native pixel centers')
    write(folder/'metrics.json',receipt);pipe.train();return receipt


def save(path,pipe,opt,step,args,cfg,camera_opt=None):
    tmp=path.with_suffix('.tmp')
    torch.save(dict(pipeline=pipe.state_dict(),optimizer=opt.state_dict(),step=step,args=vars(args),config=cfg,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),sampling=dict(cache_order=pipe.datamanager.train_pixel_sampler.cache_order.cpu(),
                        sample_count=pipe.datamanager.train_pixel_sampler.sample_count),
                    camera_optimizer=None if camera_opt is None else camera_opt.state_dict()),tmp)
    tmp.replace(path)


def main():
    # Same validated Blackwell/CUDA12.6 JIT environment as the existing quiet
    # runner; using a venv Python executable alone does not activate its PATH.
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    if Path('/usr/local/cuda-12.6').is_dir():
        os.environ.setdefault('CUDA_HOME','/usr/local/cuda-12.6')
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0+PTX')
        os.environ.setdefault('TORCH_EXTENSIONS_DIR','/home/brans/.cache/torch_extensions_lookcloser')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--rays',type=int,default=2048);p.add_argument('--samples',type=int,default=128)
    p.add_argument('--sampling',choices=['fixed','adaptive','occupancy'],default='fixed');p.add_argument('--hash-size',type=int,default=19)
    p.add_argument('--max-resolution',type=int,default=8192)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--lr-final',type=float,default=.001)
    p.add_argument('--depth-weight',type=float,default=.1);p.add_argument('--depth-steps',type=int,default=1500)
    p.add_argument('--depth-distribution-weight',type=float,default=0.)
    p.add_argument('--depth-distribution-sigma',type=float,default=.0015)
    p.add_argument('--distortion-weight',type=float,default=.01)
    p.add_argument('--opacity-neutral-distortion',action='store_true',help='Experimental distortion shape gradient without direct ray-opacity pressure')
    p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--eval-scale',type=int,default=2)
    p.add_argument('--load',type=Path);p.add_argument('--freeze-grid',action='store_true');p.add_argument('--legacy-sh',action='store_true')
    p.add_argument('--normalized-density',action='store_true')
    p.add_argument('--view-independent-color',action='store_true')
    p.add_argument('--foreground-support',type=Path)
    p.add_argument('--background-checkpoint',type=Path)
    p.add_argument('--equipment-slab',type=float,nargs=2,metavar=('WALL_GAP','FRONT_LIMIT'))
    p.add_argument('--slab-plane',type=Path,help='Explicit world plane for a bounded room field without a separate wall texture')
    p.add_argument('--empty-fraction',type=float,default=0.)
    p.add_argument('--detail-fraction',type=float,default=0.)
    p.add_argument('--boundary-fraction',type=float,default=0.,help='Train rays near matte gradients; no anatomical regions')
    p.add_argument('--boundary-radius',type=int,default=3)
    p.add_argument('--empty-weight',type=float,default=0.)
    p.add_argument('--matte-weight',type=float,default=0.)
    p.add_argument('--matte-all-pixels',action='store_true')
    p.add_argument('--matte-soft-target-weight',type=float,default=1.)
    p.add_argument('--opacity-hard-fraction',type=float,default=1.)
    p.add_argument('--native-patches',type=int,default=0,help='Auxiliary native opaque RGB patches per patch step')
    p.add_argument('--patch-size',type=int,default=16)
    p.add_argument('--patch-every',type=int,default=4)
    p.add_argument('--patch-ssim-weight',type=float,default=0.)
    p.add_argument('--stratified',action='store_true')
    p.add_argument('--importance-samples',type=int,default=0)
    p.add_argument('--camera-lr',type=float,default=0.)
    p.add_argument('--freeze-decoders',action='store_true')
    p.add_argument('--initial-frequency-grid',type=Path)
    p.add_argument('--replace-frequency-grid',type=Path,help='Explicit frequency-only transition after a full checkpoint load')
    p.add_argument('--initialize-occupancy',action='store_true')
    p.add_argument('--native-training-manifest',type=Path)
    p.add_argument('--observed-background',type=Path,help='Independent temporal train plates; replaces uncertain matte/empty supervision where observed')
    p.add_argument('--observed-trimap',type=Path,help='Audited train trimaps; observed unknown pixels use composition even if estimated alpha is saturated')
    p.add_argument('--native-observed-background',action='store_true',help='Native photo and subpixel measured-background targets on already-observed non-opaque rays')
    p.add_argument('--stereo-depth',type=Path,help='Verified disjoint-pair actor stereo receipt')
    p.add_argument('--stereo-depth-weight',type=float,default=0.)
    p.add_argument('--eval-only',action='store_true');args=p.parse_args()
    if args.depth_distribution_sigma<=0:raise ValueError('Depth-distribution sigma must be positive')
    if args.load and args.initial_frequency_grid:raise ValueError('Choose a full checkpoint or an initial Frequency Grid, not both')
    if args.replace_frequency_grid and (not args.load or args.initial_frequency_grid):raise ValueError('Frequency replacement requires only a full checkpoint source')
    if args.slab_plane and args.equipment_slab is None:raise ValueError('Explicit slab plane needs signed-distance bounds')
    if args.observed_background and (args.camera_lr>0 or args.matte_weight<=0):raise ValueError('Observed-background probe requires fixed calibration and a matte control')
    if args.observed_trimap and not args.observed_background:raise ValueError('Trimap override requires observed backgrounds')
    if args.native_observed_background and (not args.observed_background or not args.native_training_manifest):
        raise ValueError('Native composites require observed backgrounds and native photographs')
    if args.native_observed_background and args.stereo_depth_weight:
        raise ValueError('Native composites have no subpixel stereo-target adapter')
    if args.native_observed_background and args.depth_steps and (args.depth_weight or args.depth_distribution_weight):
        raise ValueError('Native composites have no subpixel teacher-depth adapter')
    if args.stereo_depth_weight<0 or (args.stereo_depth_weight and (not args.stereo_depth or args.camera_lr>0 or args.sampling!='fixed')):
        raise ValueError('Stereo depth requires a receipt, fixed cameras and fixed sampling')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists():raise ValueError('Output directory already has a run; preserve it and choose a fresh output')
    request={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    request.update(data_manifest_sha256=sha(args.data/'transforms.json'),pid=os.getpid(),script_sha256=sha(__file__))
    if args.slab_plane:request['slab_plane_sha256']=sha(args.slab_plane)
    write(out/'request.json',request)
    source_dir=out/'source';source_dir.mkdir()
    repo=Path(__file__).resolve().parents[2]
    source_paths=[Path(__file__),repo/'nerfstudio/pipelines/mesh_distillation_pipeline.py',
                  repo/'nerfstudio/fields/lookcloser_field.py',repo/'nerfstudio/models/lookcloser.py',
                  repo/'nerfstudio/model_components/mesh_distillation.py',
                  repo/'nerfstudio/model_components/native_training.py',
                  repo/'nerfstudio/model_components/native_patches.py']
    source_hashes={}
    if args.observed_background:source_paths.append(repo/'nerfstudio/model_components/observed_background.py')
    if args.stereo_depth:source_paths.append(repo/'nerfstudio/model_components/stereo_depth_evidence.py')
    for source in source_paths:
        shutil.copyfile(source,source_dir/source.name);source_hashes[str(source.relative_to(repo))]=sha(source)
    write(out/'source_hashes.json',source_hashes)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42);start=time.monotonic()
    cfg=configuration(args);pipe=cfg.setup(device='cuda');pipe.train()
    opt=torch.optim.Adam(pipe.model.field.parameters(),lr=args.lr,eps=1e-15,betas=(.9,.99))
    offset=0
    if args.load:
        state=torch.load(args.load,map_location='cuda',weights_only=False)
        if float(state['config'].model.max_res)!=float(cfg.model.max_res):
            raise ValueError('Full-state resume cannot reinterpret hash levels at a different maximum resolution')
        camera_key='_model.training_camera_optimizer.pose_adjustment'
        if camera_key in state['pipeline']:
            source_data=state['config'].datamanager.dataparser.data
            if read(source_data/'transforms.json')['train_filenames']!=read(args.data/'transforms.json')['train_filenames']:
                raise ValueError('Camera adjustment tables require identical ordered training identities')
        if args.camera_lr>0 and camera_key not in state['pipeline']:
            # Explicit identity initialization of the sole new parameter table;
            # every existing field/frequency/occupancy tensor loads strictly.
            state['pipeline'][camera_key]=torch.zeros_like(pipe.state_dict()[camera_key])
        support_transition=None
        slab_transition=None
        if args.equipment_slab is not None:
            key='_model.field.density_slab';requested=pipe.state_dict()[key].detach().clone()
            previous=state['pipeline'].get(key)
            slab_transition=dict(previous=None if previous is None else previous.cpu().tolist(),requested=requested.cpu().tolist())
            state['pipeline'][key]=requested
        if args.foreground_support:
            from nerfstudio.model_components.mesh_distillation import tensor_digest
            key='_model.field.foreground_support'
            requested=pipe.state_dict()[key].detach().clone()
            previous=state['pipeline'].get(key)
            support_transition=dict(source_digest=None if previous is None else tensor_digest(previous),
                requested_digest=tensor_digest(requested),requested_file_sha256=sha(args.foreground_support))
            # A caller-specified corrected hull must survive full-state loading.
            # Record this explicit geometry transition instead of silently
            # restoring the checkpoint's superseded support buffer.
            state['pipeline'][key]=requested
        pipe.load_pipeline(state['pipeline'],state['step']);opt.load_state_dict(state['optimizer']);offset=state['step']+1
        for group in opt.param_groups:group['lr']=args.lr
        write(out/'transition.json',dict(source=str(args.load),source_sha256=sha(args.load),offset=offset,
              frequency_digest=pipe.frequency_digest(),actual_lr=opt.param_groups[0]['lr'],support_transition=support_transition,slab_transition=slab_transition))
    elif args.initial_frequency_grid:
        identity=read(args.initial_frequency_grid.parent/'initialization.json')
        expected=(cfg.model.grid_resolution,cfg.model.min_res,cfg.model.max_res)
        actual=(identity['grid_resolution'],identity['min_res'],identity['max_res'])
        if actual!=expected or identity['frequency_grid_sha256']!=sha(args.initial_frequency_grid):
            raise ValueError('Initial Frequency Grid resolution/range/checksum does not match the model')
        pipe.model.freq_grid.load_state_dict(torch.load(args.initial_frequency_grid,map_location='cuda',weights_only=True))
        if not bool(pipe.model.freq_grid.initialized):raise ValueError('External Frequency Grid is uninitialized')
        if args.freeze_grid:pipe._frozen_grid_digest=pipe.frequency_digest()
        write(out/'frequency_initialization.json',dict(source=str(args.initial_frequency_grid),source_sha256=sha(args.initial_frequency_grid),frequency_digest=pipe.frequency_digest()))
    else:write(out/'frequency_initialization.json',initialize_grid(pipe))
    if args.replace_frequency_grid:
        identity=read(args.replace_frequency_grid.parent/'initialization.json')
        if identity['frequency_grid_sha256']!=sha(args.replace_frequency_grid) or identity.get('source_checkpoint_sha256')!=sha(args.load):
            raise ValueError('Replacement frequency grid does not match its source checkpoint')
        if identity.get('source_manifest_sha256')!=sha(args.data/'transforms.json'):
            raise ValueError('Replacement frequency dataset mismatch')
        if (identity['grid_resolution'],identity['min_res'],identity['max_res'])!=(cfg.model.grid_resolution,cfg.model.min_res,cfg.model.max_res):
            raise ValueError('Replacement frequency range mismatch')
        before=pipe.frequency_digest()
        pipe.model.freq_grid.load_state_dict(torch.load(args.replace_frequency_grid,map_location='cuda',weights_only=True))
        if not bool(pipe.model.freq_grid.initialized):raise ValueError('Replacement frequency grid is uninitialized')
        if args.freeze_grid:pipe._frozen_grid_digest=pipe.frequency_digest()
        write(out/'frequency_transition.json',dict(previous=before,replacement=pipe.frequency_digest(),
            source=str(args.replace_frequency_grid),source_sha256=sha(args.replace_frequency_grid),field_optimizer_preserved=True))
    callbacks=pipe.model.get_training_callbacks(None)
    if args.initialize_occupancy:
        before=pipe.frequency_digest()
        with torch.no_grad():
            for warmup_step in range(0,65,16):
                pipe.model._stable_update_occupancy_grid(warmup_step,lambda x:pipe.model.field.density_fn(x)*pipe.model.config.render_step_size)
        if before!=pipe.frequency_digest():raise RuntimeError('Occupancy initialization changed Frequency Grid')
        write(out/'occupancy_initialization.json',dict(frequency_unchanged=True,occupied_fraction=float(pipe.model.occupancy_grid.binaries.float().mean())))
    history=[];candidates=[];scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    camera_opt=torch.optim.Adam(pipe.model.training_camera_optimizer.parameters(),lr=args.camera_lr) if args.camera_lr>0 else None
    if camera_opt and args.load and state.get('camera_optimizer'):
        camera_opt.load_state_dict(state['camera_optimizer'])
        for group in camera_opt.param_groups:group['lr']=args.camera_lr
    check(out,'actor_train',offset,start)
    if args.eval_only:
        evaluate(pipe,out/'native_review',offset,1);write(out/'complete.json',dict(evaluation_only=True));return
    if args.native_patches and (not args.native_training_manifest or args.camera_lr>0 or args.patch_every<1 or args.sampling!='fixed'):
        raise ValueError('Native patch control requires native targets, fixed calibration/quadrature and a positive interval')
    best_psnr=-float('inf');no_improve=0;patch_sampler=None
    for local in range(args.steps):
        step=offset+local
        for callback in callbacks:callback.func(step)
        rate=args.lr*(args.lr_final/args.lr)**(local/max(args.steps-1,1))
        for group in opt.param_groups:group['lr']=rate
        opt.zero_grad(set_to_none=True)
        if camera_opt:camera_opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            output,losses,metrics=pipe.get_train_loss_dict(step)
            if args.observed_background and local==0:
                write(out/'observed_background.json',pipe.datamanager._observed_background.receipt)
            if args.native_observed_background and local==0:
                write(out/'native_observed.json',pipe.datamanager._native_targets.observed_sampling_summary)
            if args.stereo_depth and local==0:
                write(out/'stereo_depth.json',pipe.datamanager._stereo_depth.receipt)
            loss=sum(losses.values())
            if args.native_patches and local%args.patch_every==0:
                from nerfstudio.model_components.native_patches import NativeOpaquePatches, patch_objective
                if patch_sampler is None:
                    dataset=pipe.datamanager.train_dataset
                    rows=dataset.metadata['distillation_rows'];root=Path(dataset.metadata['distillation_root'])
                    if not all('alpha_file_path' in row and 'foreground_file_path' in row for row in rows):
                        raise ValueError('Native opaque patches require explicit training matte targets')
                    confidence=None
                    if any('confidence_file_path' in row for row in rows):
                        confidence=torch.stack([torch.from_numpy(np.array(Image.open(root/row['confidence_file_path']))).float()/255
                            if 'confidence_file_path' in row else torch.ones_like(pipe.datamanager._native_targets.core[i],device='cpu',dtype=torch.float32)
                            for i,row in enumerate(rows)])
                    patch_sampler=NativeOpaquePatches(pipe.datamanager._native_targets,args.patch_size,confidence=confidence)
                    write(out/'patch_protocol.json',dict(size=args.patch_size,count=args.native_patches,
                        every=args.patch_every,ssim_weight=args.patch_ssim_weight,seed=1447,
                        anchors_per_camera=[len(v) for v in patch_sampler.anchors],
                        target='Verified native train RGB inside fully opaque core',
                        depth_sampling='Deterministic patch quadrature; ordinary ray sampling unchanged'))
                patch_rays,patch_target,_=patch_sampler.sample(args.native_patches,pipe.datamanager.train_dataset.cameras)
                stratified=pipe.model.config.fixed_stratified_sampling
                try:
                    pipe.model.config.fixed_stratified_sampling=False
                    patch_output=pipe.model(patch_rays)
                finally:pipe.model.config.fixed_stratified_sampling=stratified
                loss=loss+patch_objective(patch_output['actor_rgb'],patch_target,args.patch_ssim_weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor objective')
        scaler.scale(loss).backward();scaler.step(opt)
        if camera_opt:
            scaler.step(camera_opt)
            with torch.no_grad():pipe.model.training_camera_optimizer.pose_adjustment.clamp_(-.0005,.0005)
        scaler.update()
        if (local+1)%250==0:
            check(out,'actor_train',step+1,start,train_psnr=float(metrics['psnr']),lr=rate,
                  opacity=float(output['accumulation'].mean()),observed_voxels=int(pipe.model.freq_grid.observed.sum()))
        stop=local+1==args.steps or time.monotonic()-start>=args.seconds
        if (local+1)%args.eval_every==0 or stop:
            # Preserve the actual state before evaluation, including failures on
            # unobserved rays that random train batches may not expose.
            save(out/'latest.pt',pipe,opt,step,args,cfg,camera_opt)
            if camera_opt:
                adjustment=pipe.model.training_camera_optimizer.pose_adjustment.detach().cpu()
                write(out/'camera_adjustments.json',dict(step=step+1,translation_units='frozen normalized world',rotation_units='radians',
                    train_filenames=[str(p.name) for p in pipe.datamanager.train_dataset.image_filenames],adjustments=adjustment.tolist(),
                    maximum_translation=float(adjustment[:,:3].norm(dim=-1).max()),maximum_rotation=float(adjustment[:,3:].norm(dim=-1).max()),
                    evaluation_camera_adjusted=False))
            result=evaluate(pipe,out/f'eval_{step+1:06d}',step+1,args.eval_scale)
            history.append(result);write(out/'history.json',history)
            score=result['metrics']['eval_all_psnr'];lpips=result['metrics']['eval_all_lpips']
            print('EVAL',step+1,json.dumps(result['metrics']),flush=True)
            no_improve=0 if score>best_psnr+.01 else no_improve+1
            best_psnr=max(best_psnr,score)
            if score>=best_psnr-.07:
                # The pre-evaluation snapshot already captures this exact
                # training step. Reuse its immutable inode instead of another
                # full optimizer copy. Future latest writes replace atomically.
                path=out/f'candidate_{step+1:06d}.pt';os.link(out/'latest.pt',path)
                candidates.append(dict(path=path,psnr=score,lpips=lpips))
            for candidate in candidates[:]:
                if candidate['psnr']<best_psnr-.07:
                    candidate['path'].unlink();candidates.remove(candidate)
            winner=min(candidates,key=lambda r:r['lpips'])
            # Candidates are immutable. An atomic hard link preserves the exact
            # selected bytes without another full optimizer-state copy.
            publish_hardlink(winner['path'],out/'best.pt')
            if local+1>=10000 and no_improve>=3:stop=True
        if stop:break
    check(out,'actor_train',step+1,start,finished=True)
    write(out/'complete.json',dict(step=step+1,seconds=time.monotonic()-start,best_checkpoint_sha256=sha(out/'best.pt'),
          visual_review_pending=True,stop_reason='budget_or_steps_or_plateau'))


if __name__=='__main__':main()
