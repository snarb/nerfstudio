#!/usr/bin/env python3
"""Separate 3D background NeRF, fitted only to confident train background.

The subject's central box is an exact empty region of this background field.
This diagnostic neither changes the foreground field nor uses Gaussian splats.
"""
from pathlib import Path
from copy import deepcopy
import argparse,fcntl,os,random,time
import cv2
import numpy as np
import torch
from run_dec5_sequence import read,write,identity
from dec5_sequence_worker import sha


def install_empty_subject_box(field,half_width=.18):
    original=field.get_density
    def density(samples):
        value,embedding=original(samples)
        outside=(samples.frustums.get_positions().abs()>half_width).any(-1)
        return value*outside[...,None],embedding
    field.get_density=density


def prepare(source,semantics,destination):
    destination.mkdir(parents=True,exist_ok=True);meta=read(source/'transforms.json')
    train=set(meta['train_filenames']);validation=set(meta['val_filenames'])
    if len(train)!=64 or len(validation)!=1 or train&validation:raise ValueError('64+1 split required')
    records=[];(destination/'masks').mkdir(exist_ok=True)
    for f in meta['frames']:
        original=source/f['file_path'];link=destination/f['file_path'];link.parent.mkdir(exist_ok=True,parents=True)
        if not link.exists():link.symlink_to(original.resolve())
        if f['file_path'] in train:
            path=semantics/(f['physical_camera']+'.npz')
            probability=np.load(path)['probability']
            mask=cv2.erode((probability[...,2]>round(.98*65535)).astype(np.uint8),np.ones((33,33),np.uint8))
            if mask.shape!=(f['h'],f['w']):raise ValueError('Use genuine HD train masks')
            records.append(dict(camera=f['physical_camera'],semantic_sha256=sha(path),known_pixels=int(mask.sum())))
        else:mask=np.ones((f['h'],f['w']),np.uint8)
        relative='masks/'+Path(f['file_path']).stem+'.png';cv2.imwrite(str(destination/relative),mask*255);f['mask_path']=relative
    write(destination/'transforms.json',meta)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame-root',type=Path,help='Explicit autonomous sequence frame directory')
    p.add_argument('--production-stage',action='store_true')
    p.add_argument('--smoke',action='store_true');a=p.parse_args();r=a.root;out=a.output
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);cv2.setNumThreads(1)
    source_frame=a.frame_root or r/'frame'
    lock=(out/'fit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    # A former parent can die while this child survives. A second child waits
    # on this lock and adopts the sealed result without rewriting its config.
    sealed=read(out/'complete.json')
    if sealed is not None:
        request=read(out/'request.json',{})
        if (sealed['steps']!=(2 if a.smoke else 20000) or
            request.get('source_data_sha256')!=sha(source_frame/'semantic_data/transforms.json') or
            sha(Path(sealed['checkpoint']['path']))!=sealed['checkpoint']['sha256']):
            raise ValueError('Sealed background component differs from requested source/steps/checksum')
        return
    torch.cuda.set_per_process_memory_fraction(min(.8,12*2**30/torch.cuda.get_device_properties(0).total_memory))
    write(out/'worker.json',dict(pid=os.getpid(),starttime=identity(os.getpid()),started=time.time()))
    records=prepare(source_frame/'semantic_data',source_frame/'semantics',out/'data')
    request=dict(source_data_sha256=sha(source_frame/'semantic_data/transforms.json'),records=records,
        steps=2 if a.smoke else 20000,scene_scale=1.5,empty_subject_half_width=.18,
        rays_per_batch=1024,max_res=1024,levels=12,hash_log2=19,
        uses_validation_rgb_for_fitting=False,uses_test_rgb=False,foreground_model_changed=False,
        purpose='Separate geometric background diagnostic; not native production')
    if a.production_stage:
        meta=read(source_frame/'semantic_data/transforms.json')
        request.update(purpose='Autonomous train-only per-frame background component; foreground recipe decides activation',
            source_rgb_sha256={f['physical_camera']:sha(source_frame/'semantic_data'/f['file_path'])
                for f in meta['frames'] if f['file_path'] in meta['train_filenames']},
            implementation_sha256=sha(Path(__file__)))
    existing=read(out/'request.json')
    if existing is not None and existing!=request:raise ValueError('Immutable background recipe changed')
    write(out/'request.json',request)
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from nerfstudio.engine.trainer import _capture_rng_state
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    cfg=deepcopy(method_configs['instant-ngp-bounded']);cfg.output_dir=out/'training'
    cfg.experiment_name='dec5_background_volume';cfg.timestamp='fixed';cfg.vis='tensorboard'
    cfg.logging.local_writer.enable=False;cfg.logging.csv_writer.enable=False
    dm=cfg.pipeline.datamanager;model=cfg.pipeline.model
    cfg.pipeline=VanillaPipelineConfig(datamanager=dm,model=model)
    dm.data=out/'data';dm.train_num_rays_per_batch=1024;dm.eval_num_rays_per_batch=1024
    dm.dataparser=NerfstudioDataParserConfig(data=out/'data',eval_mode='filename',downscale_factor=1,
        scene_scale=1.5,orientation_method='up',center_method='focus',auto_scale_poses=True,load_3D_points=False)
    model.max_res=1024;model.num_levels=12;model.log2_hashmap_size=19
    model.raw_no_appearance_embedding=True;model.use_appearance_embedding=False
    model.eval_num_rays_per_chunk=4096;cfg.max_num_iterations=request['steps']
    latest=read(out/'latest.json')
    if latest:
        if sha(Path(latest['path']))!=latest['sha256']:raise ValueError('Background checkpoint changed')
        cfg.load_checkpoint=Path(latest['path']);cfg.load_optimizers=True;cfg.load_scheduler=True
    cfg.save_config();trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup()
    install_empty_subject_box(trainer.pipeline.model.field)
    audit=trainer.checkpoint_load_audit
    if latest and not all(audit.get(k) for k in ['optimizer_loaded','scheduler_loaded','scaler_loaded','rng_loaded']):
        raise ValueError('Incomplete background checkpoint resume')
    with torch.no_grad():
        points=torch.tensor([[0.,0.,0.],[.1,-.1,.1],[.3,0.,0.]],device=trainer.pipeline.device)
        density=trainer.pipeline.model.field.density_fn(points).reshape(-1)
        if not torch.equal(density[:2],torch.zeros_like(density[:2])) or not torch.isfinite(density).all() or density[2]<=0:
            raise ValueError('Background field violates the empty-subject-box invariant')
    write(out/'structural_checks.json',dict(checkpoint_load_audit=audit,
        inner_density_exactly_zero=True,outside_density_finite_positive=True,
        density_fn_is_used_by_occupancy_and_rendering=True))
    started=time.time();start=int(getattr(trainer,'_start_step',0))
    for step in range(start,request['steps']):
        trainer.step=step;trainer.pipeline.train()
        for callback in trainer.callbacks:callback.run_callback_at_location(step,location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        objective,_,metrics=trainer.train_iteration(step)
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite background objective')
        for callback in trainer.callbacks:callback.run_callback_at_location(step,location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        if step%100==0 or step==request['steps']-1:
            write(out/'progress.json',dict(phase='train',step=step,elapsed_seconds=time.time()-started,
                gpu_memory_bytes=torch.cuda.memory_allocated(),samples=float(metrics.get('num_samples_per_batch',0))))
        if step%2000==1999 or step==request['steps']-1:
            path=out/f'step-{step:09d}.ckpt';temp=path.with_suffix('.tmp')
            torch.save(dict(step=step,pipeline=trainer.pipeline.state_dict(),
                runtime=dict(background_empty_subject_half_width=.18),
                optimizers={k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},
                schedulers={k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},
                scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state()),temp)
            temp.replace(path);write(out/'latest.json',dict(path=str(path),sha256=sha(path),step=step))
    write(out/'complete.json',dict(checkpoint=read(out/'latest.json'),steps=request['steps'],smoke=a.smoke,
        configuration=str(cfg.get_base_dir()/'config.yml'),background_only=True,
        loader_requires_install_empty_subject_box=True,production_promoted=False,finished=time.time()))


if __name__=='__main__':main()
