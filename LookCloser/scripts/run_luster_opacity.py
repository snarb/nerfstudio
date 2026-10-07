#!/usr/bin/env python3
"""Quiet, resumable and visually gated Luster opacity experiment worker."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

# Absolute Python invocation must carry the Conda CUDA layout into nerfacc's
# JIT build, even when the caller did not source Conda activation scripts.
_prefix=Path(sys.executable).resolve().parents[1]
if (_prefix/'bin/nvcc').exists():
    os.environ.setdefault('CUDA_HOME',str(_prefix))
    os.environ['PATH']=str(_prefix/'bin')+os.pathsep+os.environ.get('PATH','')
    os.environ['CPATH']=str(_prefix/'targets/x86_64-linux/include')+os.pathsep+os.environ.get('CPATH','')
    os.environ['LIBRARY_PATH']=str(_prefix/'targets/x86_64-linux/lib')+os.pathsep+os.environ.get('LIBRARY_PATH','')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0')
    os.environ.setdefault('MAX_JOBS','4')


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def select_best(history):
    maximum=max(row['eval_all_psnr'] for row in history)
    return min((r for r in history if maximum-r['eval_all_psnr']<=.07),
               key=lambda r:(r['eval_all_lpips'],-r['eval_all_psnr']))


def configuration(request):
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from nerfstudio.data.dataparsers.luster_opacity_dataparser import LusterOpacityDataParserConfig
    from nerfstudio.data.datamanagers.luster_opacity_datamanager import LusterOpacityDataManagerConfig
    from nerfstudio.models.lookcloser_opacity import LookCloserOpacityModelConfig
    if request.get('resume') and request.get('warm_start'):
        raise ValueError('Choose full resume or parameter-only warm start')
    cfg=deepcopy(method_configs['lookcloser'])
    data=Path(request['data'])
    baseline=request.get('baseline_preset',False)
    incremental=request.get('incremental_standard',False)
    if baseline and incremental:
        raise ValueError('Choose ordinary preset or its incremental prior extension')
    standard=baseline or incremental
    if baseline and (request.get('representation','volume')!='volume' or request.get('opacity_regularization',0.)
                     or request.get('depth_prior_strength',0.) or request.get('depth_prior_manifest')
                     or request.get('semantic_grid_path') or request.get('warm_start')):
        raise ValueError('The standard baseline cannot include experimental priors or field transfer')
    if not baseline:
        cfg.pipeline.datamanager=LusterOpacityDataManagerConfig(**{k:v for k,v in vars(cfg.pipeline.datamanager).items() if k!='_target'})
    parser=NerfstudioDataParserConfig if standard else LusterOpacityDataParserConfig
    cfg.pipeline.datamanager.dataparser=parser(data=data,eval_mode='filename',
        scene_scale=cfg.pipeline.datamanager.dataparser.scene_scale,
        orientation_method='none',center_method='none',auto_scale_poses=False,downscale_factor=1,load_3D_points=False)
    cfg.pipeline.datamanager.data=data
    if not baseline:
        cfg.pipeline.datamanager.depth_prior_manifest=request.get('depth_prior_manifest')
        cfg.pipeline.model=LookCloserOpacityModelConfig(**{k:v for k,v in vars(cfg.pipeline.model).items() if k!='_target'})
    model=cfg.pipeline.model
    # Keep pixel batches comparable when surface eikonal/probing consumes extra
    # random numbers. This is recorded as an experimental campaign default.
    cfg.pipeline.independent_rng_streams=request.get('independent_rng_streams',not standard)
    model.independent_rng_streams=cfg.pipeline.independent_rng_streams
    cfg.pipeline.training_seed=request.get('seed',42)
    model.training_seed=request.get('seed',42)
    if not baseline:
        model.representation=request.get('representation','volume')
        model.opacity_regularization=request.get('opacity_regularization',0.)
        model.depth_prior_strength=request.get('depth_prior_strength',0.)
        model.semantic_grid_path=request.get('semantic_grid_path')
    if incremental:
        model.skip_invalid_depth_updates=False
        model.allow_unmarked_volume_checkpoint=True
    if not standard:
        model.density_activation='trunc_exp'
        model.correct_sh_directions=True
        model.depth_loss_mult=0.
        model.adaptive_coarse_step_size=.001
    model.eval_num_rays_per_chunk=request.get('eval_chunk',model.eval_num_rays_per_chunk)
    if request.get('resume') and int(Path(request['resume']).stem.split('-')[-1])>=20000:
        model.feature_reweighting_strength=.3
    for key,value in request.get('model',{}).items():
        if not hasattr(model,key):
            raise ValueError(f'Unknown model control {key}')
        setattr(model,key,value)
    cfg.pipeline.feature_reweighting_switch_step=20000
    cfg.pipeline.feature_reweighting_after_switch=.3
    cfg.pipeline.datamanager.pixel_sampler.ignore_mask=True
    cfg.pipeline.datamanager.train_num_rays_per_batch=request.get('rays',4096)
    if request.get('smoke',False):
        cfg.pipeline.enable_frequency_grid=False
        model.enable_frequency_grid=False
        cfg.pipeline.datamanager.pixel_sampler.enable_fas=False
        cfg.pipeline.datamanager.train_num_images_to_sample_from=4
    cfg.output_dir=Path(request['output'])
    cfg.experiment_name='field'
    cfg.method_name='lookcloser-opacity'
    cfg.timestamp='run'
    cfg.max_num_iterations=request['end_step']+1
    cfg.save_only_latest_checkpoint=False
    cfg.machine.seed=request.get('seed',42)
    cfg.vis='tensorboard'
    cfg.logging.local_writer.enable=False
    cfg.logging.csv_writer.enable=False
    cfg.logging.profiler='none'
    if request.get('resume'):
        cfg.load_checkpoint=Path(request['resume'])
        cfg.load_optimizers=True
        cfg.load_scheduler=True
    elif request.get('warm_start'):
        # For paired surface/semantic forks, keep the reviewed permission grid
        # and representation marker from the new config. Copy field parameters
        # only; restart both paired arms with fresh optimization/sampling state.
        cfg.load_checkpoint=Path(request['warm_start'])
        cfg.checkpoint_load_mode='model_parameters_only'
        cfg.checkpoint_load_parameter_hash_audit=True
    return cfg


def score(model,pred,gt):
    import torch
    p=pred.clamp(0,1).permute(2,0,1)[None].float()
    g=gt.permute(2,0,1)[None].float()
    return dict(psnr=float(-10*torch.log10((p-g).square().mean().clamp_min(1e-12))),
                ssim=float(model.ssim(p,g,data_range=1.)),lpips=float(model.lpips(p,g)))


def evaluate(trainer,request,step,preview=False):
    import numpy as np
    from PIL import Image
    import torch
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    pipe=trainer.pipeline
    state=_capture_rng_state()
    pipe.eval()
    evaluation_model=request.get('evaluation_model',{})
    allowed={'adaptive_coarse_step_size','max_steps_per_ray','surface_num_samples',
             'surface_refine_steps','corrected_arm_allocator'}
    if set(evaluation_model)-allowed:
        raise ValueError('Evaluation overrides must change sampling controls only')
    original_sampling={key:getattr(pipe.model.config,key) for key in evaluation_model}
    for key,value in evaluation_model.items():
        setattr(pipe.model.config,key,value)
    destination=Path(request['output'])/f'eval_{step:06d}'
    destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    stride=request.get('eval_stride',1)
    meta=json.loads((Path(request['data'])/'transforms.json').read_text())
    roi_boxes=request.get('roi_boxes',{})
    try:
        with torch.no_grad():
            for split,dataset in [('eval',pipe.datamanager.eval_dataset),('train',pipe.datamanager.train_dataset)]:
                indices=list(range(len(dataset))) if split=='eval' else [i for i,p in enumerate(dataset.image_filenames) if int(p.stem.split('_')[1]) in [11,97,151]]
                if preview:
                    indices=indices[:1]
                for index in indices:
                    camera=dataset.cameras[index:index+1].to(pipe.device)
                    name=dataset.image_filenames[index].name
                    gt=dataset[index]['image'][::stride,::stride].to(pipe.device)
                    h,w=gt.shape[:2]
                    yy,xx=torch.meshgrid(torch.arange(h,device=pipe.device)*stride,
                                         torch.arange(w,device=pipe.device)*stride,indexing='ij')
                    rays=camera.generate_rays(0,coords=torch.stack([yy,xx],-1).float()+.5)
                    result=pipe.model.get_outputs_for_camera_ray_bundle(rays)
                    pred=result['rgb'].float()
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError('Nonfinite RGB')
                    values=score(pipe.model,pred,gt)
                    rois={}
                    for label,box in roi_boxes.get(name,{}).items():
                        x0,y0,x1,y1=[int(v)//stride for v in box]
                        a,b=pred[y0:y1,x0:x1],gt[y0:y1,x0:x1]
                        if min(a.shape[:2])<32:
                            continue
                        rois[label]=score(pipe.model,a,b)
                        panel=torch.cat([b,a],1).cpu().numpy()
                        Image.fromarray(np.rint(panel.clip(0,1)*255).astype('uint8')).save(destination/f'{split}_{index:03d}_{label}.png')
                    array=np.rint(pred.cpu().numpy().clip(0,1)*255).astype('uint8')
                    Image.fromarray(array).save(destination/f'{split}_img_{index:04d}.png')
                    acc=result['accumulation'].float()
                    Image.fromarray(np.rint(acc.cpu().numpy()[...,0].clip(0,1)*255).astype('uint8')).save(destination/f'{split}_alpha_{index:04d}.png')
                    np.savez_compressed(destination/f'{split}_geometry_{index:04d}.npz',
                                        depth=result['depth'].float().cpu().numpy()[...,0],
                                        accumulation=acc.cpu().numpy()[...,0])
                    if 'normals' in result:
                        normals=(result['normals'].float().cpu().numpy()+1)*.5
                        Image.fromarray(np.rint(normals.clip(0,1)*255).astype('uint8')).save(destination/f'{split}_normals_{index:04d}.png')
                    panel=torch.cat([gt,pred,acc.expand_as(pred)],1).cpu().numpy()
                    im=Image.fromarray(np.rint(panel.clip(0,1)*255).astype('uint8'))
                    im.thumbnail((1500,640))
                    im.save(destination/f'{split}_{index:03d}_panel.jpg')
                    rows.append(dict(split=split,index=index,image=name,**values,rois=rois,
                                     mean_opacity=float(acc.mean()),nonbinary_opacity_fraction=float(((acc>.001)&(acc<.999)).float().mean())))
        held=[r for r in rows if r['split']=='eval']
        summary=dict(step=step,eval_stride=stride,evaluation_model=evaluation_model,
                     checkpoint=str(trainer.checkpoint_dir/f'step-{step:09d}.ckpt'),
                     renders=str(destination),per_view=rows,
                     **{f'eval_all_{k}':float(np.mean([r[k] for r in held])) for k in ['psnr','ssim','lpips']})
        write(destination/'metrics.json',summary)
        return summary
    finally:
        for key,value in original_sampling.items():
            setattr(pipe.model.config,key,value)
        _restore_rng_state(state)
        pipe.train()


def train(request):
    deterministic = request.get('deterministic_torch', False)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from nerfstudio.engine.trainer import _restore_rng_state
    output=Path(request['output'])
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists():
        raise FileExistsError('Use a new output directory for each stage')
    if time.time()>=request.get('campaign_deadline',float('inf')):
        raise TimeoutError('Campaign budget has expired; do not launch another stage')
    if request.get('resume') and request['end_step']<=int(Path(request['resume']).stem.split('-')[-1]):
        raise ValueError('Full continuation must advance beyond its source checkpoint')
    if not request.get('smoke',False) and not (Path(request['data'])/'frequency_ready.json').exists():
        raise ValueError('Train-only frequency preprocessing is incomplete')
    seed=request.get('seed',42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic=deterministic
    torch.backends.cudnn.benchmark=False
    write(output/'request.json',request)
    source_root=Path(__file__).resolve().parents[2]
    source_files=[
        'LookCloser/scripts/run_luster_opacity.py','nerfstudio/models/lookcloser_opacity.py',
        'nerfstudio/models/lookcloser.py','nerfstudio/fields/lookcloser_surface_field.py',
        'nerfstudio/fields/lookcloser_field.py','nerfstudio/model_components/lookcloser_opacity.py',
        'nerfstudio/model_components/lookcloser_grid.py','nerfstudio/model_components/lookcloser_samplers.py',
        'nerfstudio/pipelines/lookcloser_pipeline.py','nerfstudio/lookcloser_pixel_sampler.py',
        'nerfstudio/data/datamanagers/luster_opacity_datamanager.py',
        'nerfstudio/data/dataparsers/luster_opacity_dataparser.py','nerfstudio/engine/trainer.py']
    write(output/'source_manifest.json',dict(files={p:sha(source_root/p) for p in source_files},
                                          python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                                          gpu=torch.cuda.get_device_name(),matmul_tf32=False,
                                          deterministic_torch=deterministic,
                                          custom_tcnn_gradients_not_covered=True))
    cfg=configuration(request)
    cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1)
    trainer.setup()
    if trainer._loaded_rng_state is not None:
        _restore_rng_state(trainer._loaded_rng_state)
        trainer._loaded_rng_state=None
    original_next_train=trainer.pipeline.datamanager.next_train
    first_batch_step=trainer._start_step
    def recorded_next_train(step):
        rays,batch=original_next_train(step)
        if step-first_batch_step<32:
            indices=batch['indices'].detach().cpu().contiguous().numpy()
            rgb=batch['image'].detach().cpu().contiguous().numpy()
            row=dict(step=step,indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                     rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                     cameras=sorted(set(int(v) for v in indices[:,0])),
                     maximum_y=int(indices[:,1].max()),maximum_x=int(indices[:,2].max()),
                     trusted_depth_rays=int(batch.get('depth_valid',torch.zeros(1)).sum()))
            with (output/'ray_batch_receipts.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        return rays,batch
    trainer.pipeline.datamanager.next_train=recorded_next_train
    history=json.loads(Path(request['parent_history']).read_text()) if request.get('parent_history') else []
    gates=set(request.get('eval_steps',[2000,8000,20000,26000,32000,40000]))
    start=time.time()
    last_step=None
    write(output/'progress.json',dict(status='training',worker=os.getpid(),start=start,start_step=trainer._start_step))
    for step in range(trainer._start_step,request['end_step']+1):
        trainer.step=step
        trainer.pipeline.train()
        if step==32000:
            optimizer=trainer.optimizers.optimizers['fields']
            for group in optimizer.param_groups:
                group['lr']*=.2
                group['initial_lr']=.002
            scheduler=trainer.optimizers.schedulers['fields']
            scheduler.base_lrs=[v*.2 for v in scheduler.base_lrs]
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step,location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        loss,_,metrics=trainer.train_iteration(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training objective at step {step}')
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step,location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        last_step=step
        elapsed=time.time()-start
        limited=elapsed>=request.get('max_seconds',6*3600) or time.time()>=request.get('campaign_deadline',float('inf'))
        if step%100==0:
            write(output/'progress.json',dict(status='training',worker=os.getpid(),step=step,
                  elapsed=elapsed,train_psnr=float(metrics['psnr']),memory_allocated=torch.cuda.memory_allocated(),
                  scaler=float(trainer.grad_scaler.get_scale())))
        if step in gates or step==request['end_step'] or limited:
            trainer.save_checkpoint(step)
            cfg.pipeline.model.feature_reweighting_strength=trainer.pipeline.model.field.feature_reweighting_strength
            cfg.save_config()
            row=evaluate(trainer,request,step,preview=request.get('smoke',False))
            target=Path(row['renders'])/'config.yml'
            shutil.copy2(cfg.get_base_dir()/'config.yml',target)
            row['config']=str(target)
            write(Path(row['renders'])/'metrics.json',row)
            history.append(row)
            write(output/'history.json',history)
            write(output/'selection.json',select_best(history))
            print(json.dumps({k:row[k] for k in ['step','eval_all_psnr','eval_all_ssim','eval_all_lpips']}),flush=True)
        elif step%2000==0:
            trainer.save_checkpoint(step)
        if limited:
            break
    chosen=select_best(history)
    write(output/'result.json',dict(status='review_required',last_step=last_step,elapsed=time.time()-start,
                                   selected=chosen,checkpoint_load_audit=trainer.checkpoint_load_audit,
                                   transforms_sha256=sha(Path(request['data'])/'transforms.json'),
                                   checkpoint_sha256=sha(chosen['checkpoint'])))
    write(output/'progress.json',dict(status='review_required',step=last_step,elapsed=time.time()-start))
    trainer.pipeline.datamanager.train_batch_prefetch_barrier()


def supervise(request_path):
    request=json.loads(Path(request_path).read_text())
    output=Path(request['output']); output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists():
        raise FileExistsError('Use a new output directory; preserve existing logs')
    log=output/'train_stdout.log'
    with log.open('w') as handle:
        child=subprocess.Popen([sys.executable,__file__,'train','--request',str(request_path)],
                               stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        def stop(*_):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
            raise SystemExit(130)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while True:
            progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
            processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
            tail=log.read_text(errors='replace')[-8192:]
            row=dict(time=time.time(),controller=os.getpid(),worker=child.pid,exit=child.poll(),progress=progress,
                     gpu=gpu.stdout.strip(),gpu_processes=processes.stdout.strip(),
                     oom=('out of memory' in tail.lower()),latest_checkpoint=sorted(str(p) for p in output.rglob('*.ckpt'))[-1:] )
            with (output/'supervision.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(f"status={progress.get('status','setup')} step={progress.get('step','-')} gpu={row['gpu']} worker={child.pid}",flush=True)
            if child.poll() is not None:
                if child.returncode:
                    print(tail[-3000:],flush=True)
                return child.returncode
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','supervise'])
    p.add_argument('--request',type=Path,required=True)
    args=p.parse_args()
    if args.action=='supervise':
        raise SystemExit(supervise(args.request))
    train(json.loads(args.request.read_text()))
