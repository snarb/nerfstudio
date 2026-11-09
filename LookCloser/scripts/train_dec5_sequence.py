"""Full, scene-adapted LookCloser leader (75940 + FR.3 through 106316).

Each frame is independent. Resume restores the same-frame field, Adam, scheduler,
scaler and RNG. Frequency maps and RGB from the one held-out camera never fit
geometry, semantics, background, or frequency models.
"""
from copy import deepcopy
from pathlib import Path
import json
import random
import time
import cv2
import numpy as np
import torch
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def fit_background(directory,origin_levels=1,direction_levels=5,width=64,steps=5000,batch_size=4096,seed=None):
    from dec5_ray_background import RayBackgroundNetwork
    from dec5_hair import cv_pose
    output=directory/'ray_background';output.mkdir(exist_ok=True)
    options=dict(origin_levels=origin_levels,direction_levels=direction_levels,width=width,steps=steps,batch_size=batch_size,seed=seed)
    if (output/'complete.json').exists():
        if sha(output/'background.pt')!=read(output/'complete.json')['checkpoint_sha256']:
            raise ValueError('Frozen background checksum mismatch')
        fitted=read(output/'complete.json').get('fit_options')
        if fitted is not None and fitted!=options:raise ValueError('Frozen background recipe mismatch')
        return output/'background.pt'
    data=directory/'semantic_data';meta=read(data/'transforms.json')
    frames=[f for f in meta['frames'] if f['file_path'] in meta['train_filenames']]
    center=np.mean([cv_pose(f)[:3,3] for f in frames],axis=0).astype(np.float32)
    radius=max(float(np.linalg.norm(cv_pose(f)[:3,3]-center)) for f in frames)
    rng=np.random.RandomState(42);samples=[]
    for f in frames:
        value=np.load(directory/'semantics'/(f['physical_camera']+'.npz'))['probability'][...,2].astype(np.float32)/65535
        mask=cv2.erode((value>.98).astype(np.uint8),np.ones((33,33),np.uint8))>0
        y,x=np.nonzero(mask)
        if not len(x):continue
        take=rng.choice(len(x),4352,replace=len(x)<4352);x,y=x[take],y[take];pose=cv_pose(f).astype(np.float32)
        direction=np.stack(((x+.5-f['cx'])/f['fl_x'],(y+.5-f['cy'])/f['fl_y'],np.ones(len(x))),-1).astype(np.float32)@pose[:3,:3].T
        direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        origin=np.broadcast_to((pose[:3,3]-center)/radius,direction.shape)
        rgb=cv2.imread(str(data/f['file_path']))[y,x,::-1].astype(np.float32)/255
        samples.append(np.concatenate((origin,direction,rgb),-1))
    if len(samples)<32:raise ValueError('Insufficient train-only background coverage')
    if seed is not None:torch.manual_seed(seed)
    pool=torch.as_tensor(np.concatenate(samples),device='cuda');model=RayBackgroundNetwork(origin_levels,direction_levels,width).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6)
    for step in range(steps):
        batch=pool[torch.randint(len(pool),(batch_size,),device='cuda')]
        objective=(model(batch[:,:3],batch[:,3:6])-batch[:,6:]).square().mean()
        optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step()
    temp=output/'background.pt.tmp'
    torch.save(dict(state_dict=model.state_dict(),center=center.tolist(),radius=radius,
                    origin_levels=origin_levels,direction_levels=direction_levels,width=width,step=steps),temp);temp.replace(output/'background.pt')
    write(output/'request.json',dict(data_sha256=sha(data/'transforms.json'),
        allowed_serving_dataset_sha256=sha(directory/'data/transforms.json'),uses_validation_rgb=False,uses_test_rgb=False))
    write(output/'complete.json',dict(checkpoint_sha256=sha(output/'background.pt'),train_cameras=len(frames),fit_options=options))
    return output/'background.pt'


def train(root,frame,recipe,max_steps=106317,experimental_parent=None):
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from nerfstudio.engine.trainer import _capture_rng_state
    from dec5_subject_sampling import SubjectSampler
    from dec5_background_separation import install_training
    from train_dec5_hair_baseline import evaluate,select_best
    cold_probe=bool(recipe.get('experimental_cold_probe')) and experimental_parent is None
    if cold_probe and (recipe.get('synthetic_smoke_test') or max_steps not in {15189,30377}):
        raise ValueError('Cold probes stop only at explicit canonical review gates')
    if experimental_parent is not None:
        if not recipe.get('experimental_continuation') or recipe.get('synthetic_smoke_test'):
            raise ValueError('Real continuation trials require an explicit experimental recipe')
        parent_state=torch.load(experimental_parent,map_location='cpu',weights_only=False)
        parent_step=int(parent_state['step']);del parent_state
        if not 1<=max_steps-parent_step-1<=8192:
            raise ValueError('Continuation trial must perform 1..8192 matched updates')
    elif max_steps!=106317 and not cold_probe and not (recipe.get('synthetic_smoke_test') and 1<=max_steps<=4):
        raise ValueError('Production frames use the full106317-update recipe; short runs are structural tests only')
    directory=root/'frames'/frame;data=directory/'data';meta=read(data/'transforms.json')
    manifest=read(root/'manifest.json');heldout=manifest['heldout_camera']
    if cold_probe and manifest['frames']!=[frame]:raise ValueError('Cold probes are isolated single-frame experiments')
    train_names={f['physical_camera'] for f in meta['frames'] if f['file_path'] in meta['train_filenames']}
    eval_names={f['physical_camera'] for f in meta['frames'] if f['file_path'] in meta['val_filenames']}
    if len(train_names)!=64 or eval_names!={heldout} or heldout in train_names:raise ValueError('64+1 split mismatch')
    if not all((root/'frames'/f/'frequency_ready.json').exists() for f in manifest['frames']):
        raise ValueError('All 50 frames must finish frequency preparation first')
    ready=read(directory/'frequency_ready.json')
    if ready['data_sha256']!=sha(data/'transforms.json') or ready['maps']!=64:raise ValueError('Frequency preparation receipt mismatch')
    expected={Path(name).stem for name in meta['train_filenames']}
    if {p.stem for p in (data/'lookcloser_frequencies').glob('*.pt')}!=expected:raise ValueError('Frequency inventory differs from 64 train cameras')
    if recipe.get('background_volume_training'):
        from dec5_sequence_background import ensure
        recipe=deepcopy(recipe)
        recipe['background_volume']=ensure(directory,root,recipe['background_volume_training'])
    out=directory/('training_'+recipe['revision']);out.mkdir(exist_ok=True)
    request=dict(recipe=recipe,data_sha256=sha(data/'transforms.json'),seed=42,train=64,eval=1,
        heldout=heldout,steps=max_steps,fr_switch_after=75940,background_empty_space_weight=.05,
        sparse_depth_supervision=False,reason='No independently reconstructed current-frame SfM depth supplied')
    if experimental_parent is not None:
        request.update(experimental_parent_sha256=sha(experimental_parent),experimental_parent_step=parent_step,
                       production_frame_completion=False)
    existing=read(out/'request.json')
    if existing is not None and existing!=request:raise ValueError('Training recipe changed inside an immutable run')
    write(out/'request.json',request)
    torch.set_num_threads(2);background=fit_background(directory,seed=recipe.get('background_seed'))
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=deepcopy(method_configs['lookcloser']);cfg.output_dir=out;cfg.experiment_name='dec5_'+frame;cfg.timestamp='fixed'
    cfg.max_num_iterations=max_steps;cfg.vis='tensorboard';cfg.logging.local_writer.enable=False;cfg.logging.csv_writer.enable=False
    cfg.pipeline.independent_rng_streams=True;cfg.pipeline.training_seed=42
    # Persist the render-phase schedule for normal checkpoint/eval loaders.
    cfg.pipeline.feature_reweighting_switch_step=75941
    cfg.pipeline.feature_reweighting_after_switch=.3
    cfg.pipeline.frequency_projection_mode=recipe.get('frequency_projection','normalized_image')
    dm=cfg.pipeline.datamanager;dm.data=data;dm.cpu_fas_prefetch=False
    dm.dataparser=NerfstudioDataParserConfig(data=data,eval_mode='filename',downscale_factor=1,
        scene_scale=recipe['scene_scale'],orientation_method='up',center_method='focus',auto_scale_poses=True,load_3D_points=False)
    dm.train_num_rays_per_batch=4096;dm.eval_num_rays_per_batch=4096
    cfg.pipeline.target_num_samples_per_batch=0
    model=cfg.pipeline.model;model.correct_sh_directions=True;model.density_activation='trunc_exp'
    model.eval_use_training_near_plane=True
    model.independent_rng_streams=True;model.training_seed=42;model.log2_hashmap_size=23;model.max_res=8192
    model.corrected_arm_allocator=True;model.depth_loss_mult=0
    model.fixed_num_samples_per_ray=256;model.max_steps_per_ray=4096
    model.adaptive_coarse_step_size=2*recipe['scene_scale']/640
    model.adaptive_min_step_size=2*recipe['scene_scale']/(4*8192)
    model.adaptive_max_step_size=2*recipe['scene_scale']/32
    model.eval_num_rays_per_chunk=recipe.get('eval_rays_per_chunk',1024)
    model.occupancy_dilation_radius=recipe.get('occupancy_dilation',0)
    latest=read(out/'latest.json');history=read(out/'history.json',[])
    if latest:
        checkpoint=Path(latest['path'])
        if sha(checkpoint)!=latest['sha256']:raise ValueError('Checkpoint checksum mismatch')
        cfg.load_checkpoint=checkpoint;cfg.load_optimizers=True;cfg.load_scheduler=True
    elif experimental_parent is not None:
        cfg.load_checkpoint=experimental_parent;cfg.load_optimizers=True;cfg.load_scheduler=True
    cfg.save_config();trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup()
    pipe=trainer.pipeline
    if (latest or experimental_parent is not None) and not all(trainer.checkpoint_load_audit.get(k) for k in ['optimizer_loaded','scheduler_loaded','scaler_loaded','rng_loaded']):
        raise ValueError('Incomplete same-frame resume')
    if recipe.get('sampling_policy')=='luster_body80_fas50':
        from dec5_leader_person_sampling import LeaderPersonSampler
        if recipe.get('hair_boundary_fraction',0):
            from dec5_boundary_budget_sampling import BoundaryBudgetSampler
            LeaderPersonSampler=__import__('functools').partial(BoundaryBudgetSampler,
                boundary_fraction=recipe['hair_boundary_fraction'],boundary_frequency=recipe.get('hair_boundary_frequency',False))
        sampler=LeaderPersonSampler(pipe.datamanager.train_pixel_sampler,pipe.datamanager.train_dataset,directory/'semantics',
            hair_core_mix=recipe.get('hair_core_mix',0.),hair_mix_start=recipe.get('hair_mix_start',0))
        pipe.dec5_frequency_sampler=sampler.frequency_pixels
    else:
        sampler=SubjectSampler(pipe.datamanager.train_pixel_sampler,pipe.datamanager.train_dataset,directory/'semantics',
            recipe['hair_fraction'],recipe['person_fraction'],expected_train_count=64,boundary_width=11,
            boundary_share=recipe.get('boundary_share',.3))
    sampler.install(pipe.model)
    provider,counts=install_training(pipe,background,directory/'semantics',out,weight=.05,
        known_background_rgb=recipe.get('known_background_rgb',True),volume_dependency=recipe.get('background_volume'))
    frequency_probe=None
    if recipe.get('frequency_depth_quantile') is not None or recipe.get('frequency_update_minimum_opacity') is not None:
        if experimental_parent is None:raise ValueError('Frequency-depth probe is experimental only')
        from dec5_frequency_depth_probe import install
        frequency_probe=install(pipe,recipe.get('frequency_depth_quantile'),recipe.get('frequency_update_minimum_opacity'))
    write(out/'background_dependency.json',dict(path=str(background),sha256=sha(background)))
    start=int(getattr(trainer,'_start_step',0));began=time.time()
    def checkpoint(step):
        path=out/f'step-{step:09d}.ckpt';temp=path.with_suffix('.ckpt.tmp')
        torch.save(dict(step=step,pipeline=pipe.state_dict(),
            runtime=dict(feature_reweighting_strength=float(pipe.model.field.feature_reweighting_strength)),
            optimizers={k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},
            schedulers={k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},
            scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state()),temp)
        temp.replace(path);write(out/'latest.json',dict(path=str(path),sha256=sha(path),step=step))
        protected={step}|gates
        best=read(out/'selection.json')
        if best:protected.add(best['step'])
        for old in out.glob('step-*.ckpt'):
            if int(old.stem.split('-')[-1]) not in protected:old.unlink()
        return path
    pending=getattr(trainer,'_loaded_rng_state',None)
    if pending is not None:
        from nerfstudio.engine.trainer import _restore_rng_state
        _restore_rng_state(pending);trainer._loaded_rng_state=None
    iterations=range(start,max_steps)
    gates={15188,30376,45564,60752,75940,91128,106316}
    if experimental_parent is not None:
        gates={max_steps-1}
        write(out/'continuation_load_audit.json',dict(parent_sha256=sha(experimental_parent),
            start_step=start,checkpoint_load_audit=trainer.checkpoint_load_audit))
    elif cold_probe:
        gates={step for step in gates if step<max_steps}
    elif max_steps!=106317:
        steps=recipe.get('synthetic_iteration_steps',list(range(max_steps)))
        if len(steps)!=max_steps or steps!=sorted(set(steps)) or min(steps)<0 or max(steps)>8192:
            raise ValueError('Invalid isolated structural-test iteration list')
        iterations=[step for step in steps if step>=start];gates={steps[-1]}
    if latest and latest['step'] in gates and not any(r['step']==latest['step'] for r in history):
        step=latest['step'];strength=1. if step<=75940 else .3
        pipe.model.config.feature_reweighting_strength=strength;pipe.model.field.feature_reweighting_strength=strength
        destination=out/f'eval_{step:06d}_recovered_{int(time.time())}'
        row=dict(step=step,**evaluate(trainer,destination,stride=1),renders=str(destination))
        history.append(row);write(out/'history.json',history);best=select_best(history);write(out/'selection.json',best)
        chosen=out/f"step-{best['step']:09d}.ckpt"
        write(out/'best.json',dict(path=str(chosen),sha256=sha(chosen),metrics=best,renders=best['renders']))
    # Recover an interruption between metric publication and best-pointer update.
    if history:
        best=select_best(history);chosen=out/f"step-{best['step']:09d}.ckpt"
        write(out/'selection.json',best)
        write(out/'best.json',dict(path=str(chosen),sha256=sha(chosen),metrics=best,renders=best['renders']))
    if not iterations:
        provider.verify()
        write(out/'resume_verification.json',dict(checkpoint_load_audit=trainer.checkpoint_load_audit,
            start_step=start,no_additional_updates=True,time=time.time()))
        if not (out/'complete.json').exists():
            write(out/'complete.json',dict(selection=read(out/'best.json'),background_unchanged=True,
                recovered_after_final_evaluation=True,finished=time.time()))
        return
    for step in iterations:
        sampler.step=step
        strength=1. if step<=75940 else .3
        pipe.model.config.feature_reweighting_strength=strength;pipe.model.field.feature_reweighting_strength=strength
        target_distortion=recipe.get('final_distortion',.01)
        pipe.model.config.distortion_loss_mult=.01+min(1.,max(0.,(step-4096)/8192))*(target_distortion-.01)
        trainer.step=step;pipe.train()
        for callback in trainer.callbacks:callback.run_callback_at_location(step,location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        objective,_,metrics=trainer.train_iteration(step)
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training objective')
        for callback in trainer.callbacks:callback.run_callback_at_location(step,location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        if step%100==0:
            write(directory/'progress.json',dict(phase='train',time=time.time(),step=step,revision=recipe['revision'],
                elapsed_seconds=time.time()-began,gpu_memory_bytes=torch.cuda.memory_allocated()))
            sampler.save(out/'sampling_audit.json')
            if frequency_probe is not None:write(out/'frequency_depth_probe.json',frequency_probe)
        if step%2000==1999 or step in gates:checkpoint(step)
        if step in gates:
            destination=out/f'eval_{step:06d}'
            # A crashed evaluation is preserved; retry in a separate directory.
            if destination.exists():destination=out/f'eval_{step:06d}_retry_{int(time.time())}'
            row=dict(step=step,**evaluate(trainer,destination,stride=1),renders=str(destination))
            history=[r for r in history if r['step']!=step]+[row];write(out/'history.json',history)
            best=select_best(history);write(out/'selection.json',best)
            chosen=out/f"step-{best['step']:09d}.ckpt"
            # Native full-state checkpoint selected by all-eval PSNR, .07 dB/LPIPS tie break.
            write(out/'best.json',dict(path=str(chosen),sha256=sha(chosen),metrics=best,renders=best['renders']))
            print(json.dumps({k:row[k] for k in ['step','eval_all_psnr','eval_all_ssim','eval_all_lpips']}),flush=True)
    provider.verify();sampler.save(out/'sampling_audit.json')
    write(out/'complete.json',dict(selection=read(out/'best.json'),background_unchanged=True,
        known_background_counts=counts,finished=time.time()))
