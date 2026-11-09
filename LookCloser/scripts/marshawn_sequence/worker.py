"""Native-crop cold training, own-state recovery and checkpoint-bound actor metrics."""
import argparse
import copy
import fcntl
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import atomic, read, sha
from runtime import bootstrap, config, install_sampling, install_core, GPUCacheGather, display_rgb


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def score(trainer,spec,step,cap):
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    from marshawn_tuning_metrics import masked_actor_metrics
    state=_capture_rng_state();pipe=trainer.pipeline;model=pipe.model;root=Path(spec['data'])
    folder=Path(spec['output'])/'renders'/('%09d_%s'%(step,'default' if cap is None else 'cap'+str(cap)))
    folder.mkdir(parents=True,exist_ok=True)
    original=model.config.adaptive_max_frequency_level;rows=[];views={};details=[]
    try:
        model.config.adaptive_max_frequency_level=cap;pipe.eval();dataset=pipe.datamanager.eval_dataset
        with torch.inference_mode():
            for i,path in enumerate(dataset.image_filenames):
                name=Path(path).stem;camera=dataset.cameras[i:i+1].to(model.device)
                target=dataset[i]['image'].to(model.device).float()
                parts=read(root/'parts'/(name+'.json'))
                pred=torch.zeros_like(target);alpha=torch.zeros((*target.shape[:2],1),device=model.device)
                rendered_rois=[]
                for person in parts['records']:
                    x0,y0,x1,y1=person['xyxy']
                    x0=max(0,min(x0,target.shape[1]-64));y0=max(0,min(y0,target.shape[0]-64))
                    x1=min(target.shape[1],max(x1,x0+64));y1=min(target.shape[0],max(y1,y0+64))
                    crop=copy.deepcopy(camera);crop.cx-=x0;crop.cy-=y0;crop.width.fill_(x1-x0);crop.height.fill_(y1-y0)
                    xy=torch.tensor([[.5,.5],[y1-y0-.5,x1-x0-.5],[(y1-y0)/2,(x1-x0)/2]],device=model.device)
                    source_rays=camera.generate_rays(0,coords=xy+torch.tensor([y0,x0],device=model.device))
                    crop_rays=crop.generate_rays(0,coords=xy)
                    torch.testing.assert_close(source_rays.origins,crop_rays.origins,rtol=0,atol=1e-7)
                    torch.testing.assert_close(source_rays.directions,crop_rays.directions,rtol=0,atol=2e-6)
                    torch.testing.assert_close(source_rays.pixel_area,crop_rays.pixel_area,rtol=2e-4,atol=1e-10)
                    patch=model.get_outputs_for_camera(crop)
                    pred[y0:y1,x0:x1]=patch['rgb'].float();alpha[y0:y1,x0:x1]=patch['accumulation'].float()
                    rendered_rois.append([x0,y0,x1,y1])
                    # The saved patch is the raw field RGB, with no GT masking/compositing.
                    raw=display_rgb(patch['rgb']) if spec.get('format')=='exr' else patch['rgb']
                    Image.fromarray(np.uint8(np.clip(raw.float().cpu().numpy()*255+.5,0,255))).save(folder/(name+'_person'+str(person['id'])+'_raw.png'))
                atomic(folder/(name+'_render_domain.json'),dict(native_sensor_crop_size=[target.shape[1],target.shape[0]],rendered_rois=rendered_rois,unqueried_exterior='zero-filled overview canvas only; never scored; raw native patches saved individually',intrinsics='integer origin shift only; focal lengths unchanged; no raster resize'))
                if spec.get('format')=='exr':
                    pred=display_rgb(pred)
                    reference=next(r for r in read(root/'ready.json')['records'] if r['camera']==name)
                    if sha(reference['evaluation_png_reference'])!=reference['evaluation_png_sha256']:raise ValueError('Common PNG evaluation reference changed')
                    target=torch.from_numpy(np.array(Image.open(reference['evaluation_png_reference']),dtype=np.float32)/255).to(model.device)
                if pred.shape!=target.shape or not torch.isfinite(pred).all():raise ValueError('Invalid eval RGB')
                Image.fromarray(np.uint8(np.clip(pred.cpu().numpy()*255+.5,0,255))).save(folder/(name+'_sampled_actor_canvas.png'))
                Image.fromarray(np.uint16(np.clip(alpha[...,0].cpu().numpy()*65535+.5,0,65535))).save(folder/(name+'_alpha.png'))
                with np.load(root/'parts'/(name+'.npz')) as masks:
                    for person in parts['records']:
                        mask=masks['person'+str(person['id'])]
                        if mask.sum()<100:continue
                        x0,y0,x1,y1=person['xyxy']
                        # AlexNet requires at least64 pixels; pad source ROI, never resize.
                        x0=max(0,min(x0,pred.shape[1]-64));y0=max(0,min(y0,pred.shape[0]-64))
                        x1=min(pred.shape[1],max(x1,x0+64));y1=min(pred.shape[0],max(y1,y0+64))
                        metrics=masked_actor_metrics(pred.permute(2,0,1)[None],target.permute(2,0,1)[None],torch.from_numpy(mask.copy()).to(model.device)[None,None],model.lpips.net.eval(),crop_xyxy=[x0,y0,x1,y1])
                        rows.append(dict(view=name,person=person['id'],**metrics))
                        head=masks['head'+str(person['id'])]&mask
                        if head.sum()>=100:
                            yy,xx=np.nonzero(head);hx0=max(0,min(int(xx.min())-16,pred.shape[1]-64));hy0=max(0,min(int(yy.min())-16,pred.shape[0]-64))
                            hx1=min(pred.shape[1],max(int(xx.max())+17,hx0+64));hy1=min(pred.shape[0],max(int(yy.max())+17,hy0+64))
                            detail=masked_actor_metrics(pred.permute(2,0,1)[None],target.permute(2,0,1)[None],torch.from_numpy(head.copy()).to(model.device)[None,None],model.lpips.net.eval(),crop_xyxy=[hx0,hy0,hx1,hy1])
                            details.append(dict(view=name,person=person['id'],region='SAM head/hair/face',**detail))
                            patch=torch.cat([target[hy0:hy1,hx0:hx1],pred[hy0:hy1,hx0:hx1]],dim=1).cpu().numpy()
                            Image.fromarray(np.uint8(np.clip(patch*255+.5,0,255))).save(folder/(name+'_person'+str(person['id'])+'_head.png'))
                records=[r for r in rows if r['view']==name]
                if not records:raise ValueError('No visible evaluation people '+name)
                views[name]={k:float(np.mean([r[k] for r in records])) for k in ('psnr','ssim','lpips','validation_loss')}
                # Lightweight review with raw RGB retained separately.
                gt=np.uint8(np.clip(target.cpu().numpy()*255+.5,0,255));pr=np.uint8(np.clip(pred.cpu().numpy()*255+.5,0,255))
                sheet=Image.fromarray(np.concatenate([gt,pr],axis=1));sheet.thumbnail((1920,1080));sheet.save(folder/(name+'_review.jpg'))
                print('EVAL',step,name,views[name]['lpips'],flush=True)
        result=dict(step=step,cap=cap,metrics={k:float(np.mean([r[k] for r in views.values()])) for k in ('psnr','ssim','lpips','validation_loss')},by_view=views,records=rows,details=details,head_lpips_by_view={v:float(np.mean([r['lpips'] for r in details if r['view']==v])) for v in views if any(r['view']==v for r in details)},raw_render_directory=str(folder),domain='native RGB; fixed source SAM person masks; RGB0.5 exterior; equal visible people per view then equal views; camera-local IDs',in_sample=read(root/'ready.json')['in_sample'])
        atomic(folder/'metrics.json',result);return result
    finally:
        model.config.adaptive_max_frequency_level=original;pipe.train();_restore_rng_state(state)


def plateau(evaluations):
    rows=[r for r in evaluations if r['step']>=75941 and r['cap']==10]
    if len(rows)<4:return False
    tail=rows[-4:]
    return all(min(r['metrics']['lpips'] for r in tail[i:])>tail[i-1]['metrics']['lpips']*.995 and min(r['metrics']['validation_loss'] for r in tail[i:])>tail[i-1]['metrics']['validation_loss']*.99 for i in range(1,4))


def train(spec_path,technical_steps=None):
    spec_path=Path(spec_path);spec=read(spec_path);root=Path(spec['data']);out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    lease=(out/'worker.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=Path('/tmp/lookcloser_sequence_gpu_%s_%s.lock'%(os.getuid(),os.environ.get('CUDA_VISIBLE_DEVICES','0'))).open('a+')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not 1<=spec['hard_max_iterations']<=150000:raise ValueError('Invalid hard limit')
    ready=read(root/'ready.json');tf=read(root/'transforms.json')
    if ready['train'] not in (69,72) or len(tf['frames'])!=72:raise ValueError('Expected verified72 cameras')
    if ready['train']==69 and set(tf['train_filenames'])&set(tf['val_filenames']):raise ValueError('Heldout leakage')
    for r in ready['records']:
        if sha(root/r['image'])!=r['rgb_sha256'] or sha(root/'semantics'/(r['camera']+'.npz'))!=r['semantic_sha256']:raise ValueError('Source artifact changed')
    bootstrap(spec)
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
    from nerfstudio.utils import writer
    binding=dict(spec_sha256=sha(spec_path),ready_sha256=sha(root/'ready.json'),code_sha256=spec['code_sha256'],technical=technical_steps is not None)
    for filename,digest in spec['code_sha256'].items():
        if sha(filename)!=digest:raise ValueError('Runtime code changed '+filename)
    proof=out/'cold_provenance.json'
    if proof.exists():
        if read(proof)!=binding:raise ValueError('Run binding changed')
    else:
        if (out/'latest.json').exists():raise ValueError('Unbound existing output')
        atomic(proof,binding)
    latest=read(out/'latest.json') if (out/'latest.json').exists() else None
    resume=None
    if latest:
        resume=out/latest['checkpoint']
        if resume.resolve().parent!=(out/'checkpoints').resolve() or sha(resume)!=latest['sha256']:raise ValueError('Own checkpoint hash/ownership mismatch')
    seed();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    cfg=config(spec,resume);cfg.save_config();trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup()
    pipe=trainer.pipeline;model=pipe.model
    expected=torch.tensor(ready['aabb'],dtype=torch.float32)
    for box in (model.scene_aabb.reshape(2,3),model.field.aabb,model.occupancy_grid.aabbs[0].reshape(2,3),pipe.datamanager.train_dataparser_outputs.scene_box.aabb):
        torch.testing.assert_close(box.detach().cpu(),expected,rtol=0,atol=1e-6)
    import nerfstudio.models.lookcloser as model_module
    import nerfstudio.fields.lookcloser_field as field_module
    for module in (model_module,field_module):
        try:Path(module.__file__).resolve().relative_to(Path(spec['source_code']).resolve())
        except ValueError:raise ValueError('Mixed scientific checkout imports')
    atomic(out/'source_imports.json',dict(model=model_module.__file__,field=field_module.__file__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),bounds_parity=True))
    sampler=install_sampling(pipe,spec);install_core(pipe,spec)
    if spec.get('gpu_cache'):sampler.packed_gather=GPUCacheGather()
    if resume:
        loaded=torch.load(resume,map_location='cpu',weights_only=False)
        if loaded['binding']!=binding:raise ValueError('Checkpoint source binding differs')
        sampler.load_state_dict(loaded['sampling_state'])
        _restore_rng_state(loaded['rng_state']);del loaded
    else:
        audit=dict(cold=True,optimizer_states={k:len(v.state) for k,v in trainer.optimizers.optimizers.items()},occupied=int(model.occupancy_grid.binaries.sum()),nonzero_ema=int(torch.count_nonzero(model.occupancy_grid.occs)),nonzero_frequency=int(torch.count_nonzero(model.freq_grid.grid)))
        if audit['occupied'] or audit['nonzero_ema'] or audit['nonzero_frequency'] or any(audit['optimizer_states'].values()):raise ValueError('Noncold initialization')
        atomic(out/'cold_setup_audit.json',audit)
    atomic(out/'runtime.json',dict(pid=os.getpid(),host=__import__('socket').gethostname(),time=time.time(),config=str(cfg.get_base_dir()/'config.yml'),train=ready['train'],gpu_cache=spec.get('gpu_cache',False),binding=binding))
    checkpoint_dir=out/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    evaluations=read(out/'metrics.json') if (out/'metrics.json').exists() else []
    stopping=[]
    signal.signal(signal.SIGTERM,lambda s,f:stopping.append(s));signal.signal(signal.SIGINT,lambda s,f:stopping.append(s))
    def save(step):
        p=checkpoint_dir/('step-%09d.ckpt'%step);tmp=p.with_suffix('.tmp')
        if __import__('shutil').disk_usage(out).free<12*2**30:raise OSError('Need12GiB free for atomic checkpoint')
        torch.save(dict(step=step,pipeline=pipe.state_dict(),optimizers={k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state(),sampling_state=sampler.state_dict(),binding=binding),tmp)
        with tmp.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(tmp,p);receipt=dict(step=step,checkpoint=str(p.relative_to(out)),sha256=sha(p));atomic(out/'latest.json',receipt)
        keep={step,75940}
        for cap in {r['cap'] for r in evaluations}:
            keep.add(min((r for r in evaluations if r['cap']==cap),key=lambda r:r['metrics']['lpips'])['step'])
        for old in checkpoint_dir.glob('step-*.ckpt'):
            if int(old.stem.split('-')[1]) not in keep:old.unlink()
        return receipt
    def evaluate(step):
        # Canonical saved state precedes evaluation; journal survives a crash in rendering.
        checkpoint=read(out/'latest.json');atomic(out/'pending_evaluation.json',checkpoint)
        batch=[]
        for cap in (None,10):
            marker=out/'evaluation_commits'/('%09d_%s.json'%(step,str(cap)))
            if marker.exists():r=read(marker)
            else:
                result=score(trainer,spec,step,cap);r=dict(result,checkpoint_sha256=checkpoint['sha256']);atomic(marker,r)
            if r['checkpoint_sha256']!=checkpoint['sha256']:raise ValueError('Evaluation checkpoint changed')
            batch.append(r)
        for r in batch:
            if not any(x['step']==r['step'] and x['cap']==r['cap'] for x in evaluations):evaluations.append(r)
        atomic(out/'metrics.json',evaluations);(out/'pending_evaluation.json').unlink()
    if (out/'pending_evaluation.json').exists():evaluate(latest['step'])
    target=technical_steps or read(spec['stage_plan'])['stop_after_iterations']
    max_attempts=spec['hard_max_iterations'];ledger=read(out/'attempt_budget.json') if (out/'attempt_budget.json').exists() else dict(reserved_attempts=0)
    available=0;start=time.monotonic();last=0;step=trainer._start_step
    while step<target:
        if not available:
            reserve=min(128,target-step,max_attempts-ledger['reserved_attempts'])
            if reserve<=0:break
            ledger['reserved_attempts']+=reserve;atomic(out/'attempt_budget.json',ledger);available=reserve
        available-=1
        trainer.step=step;pipe.train()
        if step==75941:seed()
        sampler.set_step(step);model.config.distortion_loss_mult=.01+.09*min(1.,max(0.,(step-4096)/8192))
        trainer._apply_live_optimizer_switch(step)
        for callback in trainer.callbacks:callback.run_callback_at_location(step,location=Location.BEFORE_TRAIN_ITERATION)
        loss,_,metrics=trainer.train_iteration(step)
        if not torch.isfinite(loss.detach()):raise FloatingPointError('Nonfinite objective')
        for callback in trainer.callbacks:callback.run_callback_at_location(step,location=Location.AFTER_TRAIN_ITERATION)
        writer.write_out_storage()
        count=step+1;gate=count in (8000,16000,32000,75941) or count>=32000 and count%16000==0
        if technical_steps:gate=count==technical_steps
        target=technical_steps or read(spec['stage_plan'])['stop_after_iterations']
        done=count>=target
        if gate or done or stopping or count%2000==0:save(step)
        if gate or done:evaluate(step)
        if time.monotonic()-last>30 or gate or done:
            p=dict(pid=os.getpid(),time=time.time(),step=step,completed_iterations=count,elapsed_seconds=time.monotonic()-start,gpu_memory_bytes=torch.cuda.memory_allocated(),gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(),target=target,reserved_attempts=ledger['reserved_attempts'],samples={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if k in ('num_samples_per_batch','samples_per_ray_mean','adaptive_saturation_rate','zero_sample_ray_rate')})
            atomic(out/'progress.json',p);print(json.dumps({k:p[k] for k in ('step','elapsed_seconds','gpu_memory_bytes')}),flush=True);last=time.monotonic()
        step+=1
        if stopping:trainer.shutdown();return
        if spec.get('allow_plateau',False) and plateau(evaluations):break
    if evaluations:
        best=min(evaluations,key=lambda r:(r['metrics']['lpips'],-r['metrics']['psnr']))
        atomic(out/'stage_complete.json',dict(completed_iterations=step,selected=best,plateau=plateau(evaluations),attempt_budget=ledger,technical=technical_steps is not None))
    trainer.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--technical-steps',type=int);a=p.parse_args();train(a.spec,a.technical_steps)
