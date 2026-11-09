#!/usr/bin/env python3
"""Bounded, quiet LookCloser-core control with the campaign's explicit split.

Uses existing nerfstudio training primitives, not a replacement trainer/model.
Frequency features are disabled for the initial geometry diagnosis. Native ray
weights remain available for the later frozen-field probe.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from dec5_hair import VALIDATION, TEST
from run_dec5_hair_campaign import atomic_json, sha


def capture_rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng_state(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


def configuration(data, output, steps, samples):
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    cfg = deepcopy(method_configs['lookcloser'])
    cfg.output_dir = output
    cfg.experiment_name = 'dec5_001083_hair_B0'
    cfg.timestamp = 'fixed'
    cfg.max_num_iterations = steps
    cfg.save_only_latest_checkpoint = True
    cfg.vis = 'tensorboard'
    cfg.logging.local_writer.enable = False
    cfg.logging.csv_writer.enable = False
    dm = cfg.pipeline.datamanager
    dm.data = data
    dm.dataparser = NerfstudioDataParserConfig(data=data, eval_mode='filename', downscale_factor=1,
        scene_scale=2., orientation_method='up', center_method='focus', auto_scale_poses=True,
        load_3D_points=False)
    dm.train_num_rays_per_batch = 1024
    dm.eval_num_rays_per_batch = 1024
    dm.pixel_sampler.enable_fas = False
    cfg.pipeline.enable_frequency_grid = False
    cfg.pipeline.independent_rng_streams = True
    model = cfg.pipeline.model
    model.independent_rng_streams = True
    model.enable_frequency_grid = False
    model.enable_feature_reweighting = False
    model.correct_sh_directions = True
    model.eval_use_training_near_plane = True
    model.corrected_arm_allocator = True
    model.enable_adaptive_ray_marching = False
    model.ray_sampling_mode = 'fixed'
    model.fixed_num_samples_per_ray = samples
    model.eval_num_rays_per_chunk = 256
    model.log2_hashmap_size = 19
    model.max_res = 8192.
    model.near_plane = .01
    model.far_plane = 1000.
    model.depth_loss_mult = 0.  # No inherited sparse geometry from test cameras.
    model.distortion_loss_mult = .01
    model.background_color = 'black'
    return cfg


def evaluate(trainer, destination, stride=2, foreground_fallback=None):
    """Only validation RGB, and only after each prediction is constructed."""
    from PIL import Image
    pipe = trainer.pipeline
    state = capture_rng_state()
    pipe.eval()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        with torch.no_grad():
            dataset = pipe.datamanager.eval_dataset
            for index in range(len(dataset)):
                camera = dataset.cameras[index:index+1].to(pipe.device)
                h, w = int(camera.height[0]), int(camera.width[0])
                yy, xx = torch.meshgrid(torch.arange(0, h, stride, device=pipe.device),
                                         torch.arange(0, w, stride, device=pipe.device), indexing='ij')
                rays = camera.generate_rays(0, coords=torch.stack([yy, xx], -1).float()+.5)
                result = pipe.model.get_outputs_for_camera_ray_bundle(rays)
                prediction = result['rgb'].float()
                if foreground_fallback is not None:
                    background = torch.as_tensor(foreground_fallback[index][::stride,::stride],device=pipe.device)
                    if background.shape != prediction.shape: raise ValueError('Foreground evaluation fallback raster mismatch')
                    prediction = prediction+(1-result['accumulation'].float()).clamp(0,1)*background
                gt = dataset[index]['image'][::stride, ::stride].to(pipe.device)
                p, g = [x.permute(2, 0, 1)[None].contiguous() for x in (prediction, gt)]
                row = dict(image=dataset.image_filenames[index].name,
                    psnr=float(-10*torch.log10((p-g).square().mean().clamp_min(1e-12))),
                    ssim=float(pipe.model.ssim(p, g, data_range=1.)), lpips=float(pipe.model.lpips(p, g)))
                if not all(np.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips')):
                    raise ValueError('Nonfinite validation metrics')
                panel = torch.cat([gt, prediction], 1).cpu().numpy()
                Image.fromarray(np.rint(np.clip(panel, 0, 1)*255).astype(np.uint8)).save(destination/(str(index)+'.jpg'))
                np.savez_compressed(destination/(str(index)+'.npz'), rgb=prediction.cpu().numpy(),
                    depth=result['depth'].float().cpu().numpy(),
                    accumulation=result['accumulation'].float().cpu().numpy())
                rows.append(row)
        return dict(per_view=rows, eval_stride=stride,
                    **{'eval_all_'+k:float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')})
    finally:
        restore_rng_state(state)
        pipe.train()


def select_best(history):
    top = max(r['eval_all_psnr'] for r in history)
    eligible = [r for r in history if r['eval_all_psnr'] >= top-.07]
    return min(eligible, key=lambda r:(r['eval_all_lpips'], -r['eval_all_psnr']))


def validation_is_progressing(row,previous):
    """Stopping and PSNR-band checkpoint selection are separate decisions."""
    if not previous:return True
    dense=[r['eval_dense_anchor_median_relative'] for r in previous if 'eval_dense_anchor_median_relative' in r]
    opacity=[r['eval_head_opacity_mae'] for r in previous if 'eval_head_opacity_mae' in r]
    return (('eval_head_opacity_mae' in row and opacity and row['eval_head_opacity_mae']<min(opacity)*.95)
            or ('eval_dense_anchor_median_relative' in row and dense and row['eval_dense_anchor_median_relative']<min(dense)*.95)
            or row['eval_all_psnr']>max(r['eval_all_psnr'] for r in previous)+.03
            or row['eval_all_lpips']<min(r['eval_all_lpips'] for r in previous)*.995
            or row['eval_all_ssim']>max(r['eval_all_ssim'] for r in previous)+.001)


def publish_best(history, destination):
    """A moving PSNR band may select an older gate, not the latest or prior best."""
    best = select_best(history)
    temporary = destination/'.best.pt.tmp'
    shutil.copyfile(destination/('gate_%06d.pt'%best['step']), temporary)
    temporary.replace(destination/'best.pt')
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=40000)
    parser.add_argument('--max-seconds', type=float, default=4*3600)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-steps', type=int, nargs='+')
    parser.add_argument('--evaluation-stride',type=int,default=2)
    parser.add_argument('--foreground-targets',type=Path)
    parser.add_argument('--foreground-fallback',type=Path)
    parser.add_argument('--foreground-volume-experiment',action='store_true')
    parser.add_argument('--head-volume-crop-coverage',type=Path)
    parser.add_argument('--foreground-opacity-objective',choices=('thickness','bernoulli'),default='thickness')
    parser.add_argument('--dense-hair-anchors',type=Path)
    parser.add_argument('--dense-hair-weight',type=float,default=.01)
    parser.add_argument('--dense-hair-lr',type=float,default=.0005)
    parser.add_argument('--dense-hair-density-only',action='store_true')
    parser.add_argument('--dense-hair-center-median',action='store_true')
    parser.add_argument('--head-opacity-targets',type=Path)
    parser.add_argument('--head-opacity-weight',type=float,default=.05)
    parser.add_argument('--head-opacity-joint',action='store_true',help='Joint RGB/geometry continuation with scoped head opacity; separate from frozen-appearance control')
    parser.add_argument('--appearance-only',action='store_true',help='Refit only the color head after a geometry change; freeze density, frequency and occupancy')
    parser.add_argument('--appearance-lr',type=float,default=.0005)
    parser.add_argument('--view-independent',action='store_true',help='Use constant color direction; re-evaluate resumed parent before native checkpoint selection')
    parser.add_argument('--initial-checkpoint',type=Path)
    parser.add_argument('--background-warmstart-experiment',action='store_true',help='Separate background trainer entrypoint only; not a native continuation')
    parser.add_argument('--full-lookcloser',action='store_true')
    parser.add_argument('--disable-feature-reweighting',action='store_true',
                        help='Separate fresh-run ablation; preserve grid, FAS, ARM and subject quotas.')
    parser.add_argument('--sparse-initialization',type=Path)
    parser.add_argument('--subject-semantics',type=Path)
    parser.add_argument('--hair-fraction',type=float,default=0.)
    parser.add_argument('--person-fraction',type=float,default=0.)
    parser.add_argument('--subject-boundary-width',type=int,default=11)
    parser.add_argument('--point-budget',type=int,default=1<<18)
    parser.add_argument('--resume-from',type=Path,
                        help='Continue a full LookCloser run with its native optimizer, scheduler, scaler and RNG state.')
    parser.add_argument('--resume-step',type=int)
    args = parser.parse_args()
    if bool(args.foreground_targets) != bool(args.foreground_fallback):
        parser.error('Foreground training requires both targets and frozen evaluation fallback')
    if args.full_lookcloser and (args.sparse_initialization is None or args.subject_semantics is None):
        parser.error('The full LookCloser campaign requires verified train-only sparse initialization and semantic audit maps')
    if args.full_lookcloser and args.foreground_targets is not None and not args.foreground_volume_experiment:
        parser.error('Full LookCloser foreground objective requires the explicit separate-experiment flag')
    if args.foreground_volume_experiment and (not args.full_lookcloser or args.foreground_targets is None or args.resume_from):
        parser.error('Foreground-volume experiment requires a fresh full LookCloser run and frozen foreground targets')
    if args.head_volume_crop_coverage is not None and not args.foreground_volume_experiment:
        parser.error('Head-only crop-empty supervision is a separate foreground-volume experiment')
    if args.dense_hair_anchors is not None and (not args.full_lookcloser or args.resume_from is None or args.foreground_targets is not None):
        parser.error('Dense hair guidance is a separate full native continuation experiment')
    if args.dense_hair_density_only and args.dense_hair_anchors is None:
        parser.error('Density-only refinement requires dense hair anchors')
    if args.dense_hair_center_median and not args.dense_hair_density_only:
        parser.error('Centered median is a separate density-only refinement experiment')
    if args.head_opacity_targets is not None and not (args.dense_hair_density_only or args.head_opacity_joint):
        parser.error('Bounded head opacity requires the separate density-only continuation')
    if args.head_opacity_joint and (args.head_opacity_targets is None or args.dense_hair_density_only or args.appearance_only):
        parser.error('Joint head opacity requires head targets and excludes frozen-density/appearance controls')
    if args.view_independent and (not args.full_lookcloser or args.dense_hair_anchors is not None or args.head_opacity_targets is not None):
        parser.error('View-independent appearance is a separate full-LookCloser control without added geometric objectives')
    if args.evaluation_stride<1:parser.error('Evaluation stride must be positive')
    if args.appearance_only and (args.resume_from is None or args.dense_hair_anchors is not None or args.head_opacity_targets is not None):
        parser.error('Appearance-only requires a native continuation without geometric objectives')
    if args.background_warmstart_experiment and (not globals().get('_background_warmstart_installer_active',False) or not args.full_lookcloser or args.initial_checkpoint is None or args.resume_from is not None):
        parser.error('Background warmstart is restricted to the explicit separated-background experiment entrypoint')
    if args.full_lookcloser and args.initial_checkpoint is not None and not args.background_warmstart_experiment:
        parser.error('Full LookCloser screens start from scratch; use a complete native trainer resume for continuation')
    if args.resume_from is not None and (not args.full_lookcloser or args.initial_checkpoint is not None):
        parser.error('--resume-from is the full LookCloser native continuation path')
    if args.disable_feature_reweighting and (not args.full_lookcloser or args.resume_from is not None):
        parser.error('Feature reweighting ablation requires a fresh full LookCloser run')
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.loads((args.data/'transforms.json').read_text())
    train = [f for f in payload['frames'] if f['file_path'] in payload['train_filenames']]
    validation = [f for f in payload['frames'] if f['file_path'] in payload['val_filenames']]
    if len(train) != 59 or {f['physical_camera'] for f in validation} != set(VALIDATION):
        raise ValueError('Unexpected campaign split')
    if any(f['physical_camera'] in VALIDATION+TEST for f in train):
        raise ValueError('Training-camera leak')
    if args.full_lookcloser:
        expected={Path(f['file_path']).stem for f in train}
        directory=args.data/'lookcloser_frequencies'
        if {p.stem for p in directory.glob('*.pt')} != expected:
            raise ValueError('Full LookCloser requires all 59 genuine train-only frequency maps before setup')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(24*(1 << 30)/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    parent_history=[]
    if args.resume_from is not None:
        import yaml
        source_request=json.loads((args.resume_from/'request.json').read_text())
        if (source_request['data_sha256']!=sha(args.data/'transforms.json') or not source_request.get('full_lookcloser')
                or source_request['hair_quota']!=args.hair_fraction or source_request['person_nonhair_quota']!=args.person_fraction
                or source_request['point_budget']!=args.point_budget or source_request['seed']!=args.seed
                or source_request['samples']!=args.samples or source_request.get('subject_boundary_width',11)!=args.subject_boundary_width):
            raise ValueError('Continuation data, architecture or sampling-policy mismatch')
        configs=list(args.resume_from.rglob('config.yml'))
        if len(configs)!=1:raise ValueError('Expected one parent training configuration')
        cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
        native_dirs={p.parent for p in args.resume_from.rglob('step-*.ckpt')}
        if len(native_dirs)!=1:raise ValueError('Expected one native checkpoint directory')
        cfg.output_dir=args.output;cfg.max_num_iterations=args.max_steps
        cfg.load_dir=next(iter(native_dirs));cfg.load_step=args.resume_step
        cfg.load_optimizers=True;cfg.load_scheduler=True
        cfg.pipeline.datamanager.data=args.data;cfg.pipeline.datamanager.dataparser.data=args.data
        parent_history=json.loads((args.resume_from/'history.json').read_text())
        shutil.copy2(args.resume_from/'scene_bounds.json',args.output/'scene_bounds.json')
    else:cfg = configuration(args.data, args.output, args.max_steps, args.samples)
    if args.full_lookcloser and args.resume_from is None:
        from dec5_lookcloser_training import configure,configure_sparse_scene_bounds
        configure(cfg,args.point_budget)
        atomic_json(args.output/'scene_bounds.json',configure_sparse_scene_bounds(cfg,args.sparse_initialization))
        cfg.save_only_latest_checkpoint=False
        cfg.experiment_name='dec5_'+str(payload.get('frame_id','001083'))+'_lookcloser'
    if args.initial_checkpoint is not None:
        for group in cfg.optimizers.values():
            group['optimizer'].lr *= .1
    cfg.machine.seed = args.seed
    if args.view_independent:cfg.pipeline.model.use_view_directions=False
    if args.disable_feature_reweighting:cfg.pipeline.model.enable_feature_reweighting=False
    if args.dense_hair_density_only:
        if not cfg.pipeline.model.separate_density_encoding:
            raise ValueError('Convert the parent checkpoint explicitly before density-only refinement')
        cfg.pipeline.grid_update_interval=1<<30
    if args.dense_hair_anchors is not None:cfg.resume_fields_lr_override=args.dense_hair_lr
    if args.head_opacity_joint:cfg.pipeline.grid_update_interval=1<<30
    if args.appearance_only:
        cfg.pipeline.grid_update_interval=1<<30
        cfg.pipeline.model.occupancy_update_interval=1<<30
        cfg.resume_fields_lr_override=args.appearance_lr
    cfg.pipeline.training_seed=args.seed
    cfg.pipeline.model.training_seed=args.seed
    atomic_json(args.output/'request.json', dict(data_sha256=sha(args.data/'transforms.json'),
        seed=args.seed, max_steps=args.max_steps, max_seconds=args.max_seconds, samples=args.samples,
        baseline=('LookCloser frequency grid + FAS + ARM; feature reweighting ablated' if args.disable_feature_reweighting else
                  'LookCloser frequency grid + FR + FAS + ARM' if args.full_lookcloser else
                  'LookCloser core, fixed dense traversal, no frequency/FAS/FR'),
        correct_sh_directions=True, sparse_depth_prior=args.full_lookcloser, evaluation_stride=args.evaluation_stride,
        full_lookcloser=args.full_lookcloser,point_budget=args.point_budget if args.full_lookcloser else None,
        feature_reweighting_enabled=cfg.pipeline.model.enable_feature_reweighting,
        appearance_only=args.appearance_only,background_warmstart_experiment=args.background_warmstart_experiment,
        head_opacity_joint=args.head_opacity_joint,
        use_view_directions=cfg.pipeline.model.use_view_directions,
        hair_quota=args.hair_fraction,person_nonhair_quota=args.person_fraction,
        subject_boundary_width=args.subject_boundary_width,
        eval_steps=args.eval_steps,foreground_volume_experiment=args.foreground_volume_experiment,
        foreground_targets_receipt_sha256=sha(args.foreground_targets/'complete.json') if args.foreground_targets else None,
        head_volume_crop_coverage_sha256=sha(args.head_volume_crop_coverage) if args.head_volume_crop_coverage else None,
        foreground_opacity_objective=args.foreground_opacity_objective,
        dense_hair_anchor_receipt_sha256=sha(args.dense_hair_anchors.with_suffix('.json')) if args.dense_hair_anchors else None,
        dense_hair_weight=args.dense_hair_weight,dense_hair_lr=args.dense_hair_lr,
        dense_hair_density_only=args.dense_hair_density_only,
        dense_hair_center_median=args.dense_hair_center_median,
        head_opacity_targets_receipt_sha256=sha(args.head_opacity_targets/'complete.json') if args.head_opacity_targets else None,
        head_opacity_weight=args.head_opacity_weight,
        implementation_sha256=sha(__file__)))
    cfg.save_config()
    trainer = cfg.setup(local_rank=0, world_size=1)
    trainer.setup()
    start_step=int(getattr(trainer,'_start_step',0))
    if args.resume_from is not None:
        audit=trainer.checkpoint_load_audit
        if not all(audit.get(key) for key in ('optimizer_loaded','scheduler_loaded','scaler_loaded','rng_loaded')):
            raise ValueError('Native continuation failed to restore complete training state')
        atomic_json(args.output/'continuation.json',dict(parent=str(args.resume_from),audit=audit))
        parent_history=[row for row in parent_history if row['step']<start_step]
        if args.view_independent:parent_history=[]
        for row in parent_history:
            filename='gate_%06d.pt'%row['step'];shutil.copy2(args.resume_from/filename,args.output/filename)
        if parent_history:publish_best(parent_history,args.output)
    import inspect
    atomic_json(args.output/'implementation_environment.json',dict(torch_version=torch.__version__,
        modules={name:dict(path=inspect.getfile(type(obj)),sha256=sha(inspect.getfile(type(obj))))
                 for name,obj in [('pipeline',trainer.pipeline),('model',trainer.pipeline.model),
                                  ('field',trainer.pipeline.model.field),('datamanager',trainer.pipeline.datamanager),
                                  ('sampler',trainer.pipeline.datamanager.train_pixel_sampler),('frequency_grid',trainer.pipeline.model.freq_grid)]}))
    subject_sampler=None
    if args.full_lookcloser:
        from dec5_lookcloser_training import initialize_sparse
        initialize_sparse(trainer,args.sparse_initialization,args.output,initialize_grid=args.resume_from is None)
        from dec5_subject_sampling import SubjectSampler
        subject_sampler=SubjectSampler(trainer.pipeline.datamanager.train_pixel_sampler,
            trainer.pipeline.datamanager.train_dataset,args.subject_semantics,args.hair_fraction,args.person_fraction,boundary_width=args.subject_boundary_width)
        subject_sampler.install(trainer.pipeline.model);subject_sampler.save(args.output/'sampling_audit.json')
    foreground_fallback = None
    if args.initial_checkpoint is not None:
        source_request=json.loads((args.initial_checkpoint.parent/'request.json').read_text())
        if source_request['data_sha256'] != sha(args.data/'transforms.json'):
            raise ValueError('Initialization checkpoint belongs to a different dataset/calibration')
        checkpoint=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
        trainer.pipeline.load_state_dict(checkpoint['pipeline'],strict=True)
        atomic_json(args.output/'initialization.json',dict(checkpoint_sha256=sha(args.initial_checkpoint),
            source_step=checkpoint['step'],optimizer='fresh; only model/pipeline weights restored',learning_rate_scale=.1))
    if args.foreground_targets is not None:
        from dec5_foreground_training import install,load_fallback
        install(trainer,args.foreground_targets,args.output,head_crop_coverage=args.head_volume_crop_coverage,
                opacity_objective=args.foreground_opacity_objective)
        by_file={Path(f['file_path']).name:f['physical_camera'] for f in payload['frames']}
        names=[by_file[Path(p).name] for p in trainer.pipeline.datamanager.eval_dataset.image_filenames]
        foreground_fallback=load_fallback(args.foreground_fallback,names)
    dense_guidance=None
    if args.dense_hair_anchors is not None:
        from dec5_dense_hair_training import DenseHairGuidance
        trainer.dec5_density_only=args.dense_hair_density_only
        trainer.dec5_joint_head=args.head_opacity_joint
        trainer.dec5_center_median=args.dense_hair_center_median
        dense_guidance=DenseHairGuidance(trainer,args.dense_hair_anchors,args.output,args.dense_hair_weight)
        dense_guidance.install();dense_baseline=dense_guidance.evaluate()
        atomic_json(args.output/'dense_hair_baseline.json',dense_baseline)
        if parent_history:parent_history[-1]=dict(parent_history[-1],**dense_baseline)
    head_opacity=None
    if args.head_opacity_targets is not None:
        from dec5_head_opacity_training import HeadOpacityGuidance
        head_opacity=HeadOpacityGuidance(trainer,args.head_opacity_targets,args.subject_semantics,args.output,args.head_opacity_weight)
        head_opacity.install();opacity_baseline=head_opacity.evaluate()
        atomic_json(args.output/'head_opacity_baseline.json',opacity_baseline)
        if parent_history:parent_history[-1]=dict(parent_history[-1],**opacity_baseline)
    appearance_frozen=None
    if args.appearance_only:
        trainer.pipeline.model.field.requires_grad_(False)
        trainer.pipeline.model.field.mlp_color.requires_grad_(True)
        appearance_frozen={k:v.detach().cpu().clone() for k,v in trainer.pipeline.state_dict().items()
                           if ('field.' in k and '.mlp_color.' not in k) or 'freq_grid.' in k or 'occupancy_grid.' in k}
    if args.view_independent and args.resume_from is not None:
        initial_step=start_step-1;trainer.pipeline.model.current_train_step=initial_step
        initial=dict(step=initial_step,**evaluate(trainer,args.output/('eval_%06d'%initial_step),stride=args.evaluation_stride))
        torch.save(dict(step=initial_step,pipeline=trainer.pipeline.state_dict()),args.output/('gate_%06d.pt'%initial_step))
        parent_history=[initial];publish_best(parent_history,args.output)
        atomic_json(args.output/'history.json',parent_history)
        atomic_json(args.output/'history_rebase.json',dict(parent_view_dependent_metrics_reused=False,
            reason='Direction-independent color changes the initial forward function; evaluate it before selecting checkpoints',initial=initial))
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    started = time.time()
    history = parent_history
    stale = 0
    # This runner calls train_iteration directly, bypassing Trainer.train's
    # last-moment RNG restore after callback/cache initialization.
    pending_rng=getattr(trainer,'_loaded_rng_state',None)
    if pending_rng is not None:
        restore_rng_state(pending_rng)
        trainer._loaded_rng_state=None
        continuation=json.loads((args.output/'continuation.json').read_text())
        continuation['rng_restored_before_iteration']=True
        atomic_json(args.output/'continuation.json',continuation)
    for step in range(start_step,args.max_steps):
        trainer.step = step
        trainer.pipeline.train()
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        objective, _, metrics = trainer.train_iteration(step)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite optimization objective')
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        elapsed = time.time()-started
        if step % 100 == 0:
            atomic_json(args.output/'progress.json', dict(status='training', step=step, elapsed_seconds=elapsed,
                        train_psnr=float(metrics['psnr']), memory_allocated=torch.cuda.memory_allocated()))
            if subject_sampler is not None:subject_sampler.save(args.output/'sampling_audit.json')
        limited = elapsed >= args.max_seconds
        scheduled = (step+1 in args.eval_steps) if args.eval_steps else ((step+1) % 2000 == 0)
        if scheduled or step == args.max_steps-1 or limited:
            atomic_json(args.output/'progress.json', dict(status='evaluating', step=step, elapsed_seconds=elapsed))
            trainer.save_checkpoint(step)
            row = dict(step=step, **evaluate(trainer, args.output/('eval_%06d'%step),stride=args.evaluation_stride,foreground_fallback=foreground_fallback))
            if dense_guidance is not None:row.update(dense_guidance.evaluate())
            if head_opacity is not None:row.update(head_opacity.evaluate())
            row['cumulative_training_field_queries'] = int(metrics['cumulative_point_samples']) if 'cumulative_point_samples' in metrics else (step+1)*1024*args.samples
            torch.save(dict(step=step, pipeline=trainer.pipeline.state_dict()),
                       args.output/('gate_%06d.pt'%step))
            progressing=validation_is_progressing(row,history)
            history.append(row)
            best = publish_best(history, args.output)
            if (progressing if args.full_lookcloser else best['step'] == step):
                stale = 0
            else:
                stale += 1
            atomic_json(args.output/'history.json', history)
            atomic_json(args.output/'selection.json', best)
            print(json.dumps({k:row[k] for k in ('step','eval_all_psnr','eval_all_ssim','eval_all_lpips')}), flush=True)
            if limited or stale >= 2:
                break
    if subject_sampler is not None:subject_sampler.save(args.output/'sampling_audit.json')
    if appearance_frozen is not None:
        state=trainer.pipeline.state_dict()
        changed=[k for k,v in appearance_frozen.items() if not torch.equal(v,state[k].detach().cpu())]
        atomic_json(args.output/'frozen_geometry_audit.json',dict(unchanged=not changed,changed=changed,checked_tensors=len(appearance_frozen),
            scope='all non-color-head field state, frequency-grid state and occupancy-grid state'))
        if changed:raise ValueError('Appearance-only continuation changed frozen geometry')
    atomic_json(args.output/'progress.json', dict(status='complete', step=step,
                elapsed_seconds=time.time()-started, selection=select_best(history)))


if __name__ == '__main__':
    main()
