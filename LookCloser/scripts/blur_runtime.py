"""Repository-owned paired LookCloser experiments using the standard Trainer update."""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Type

import numpy as np
from PIL import Image
import torch
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.utils import writer


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


class ProbeParser(Nerfstudio):
    def _generate_dataparser_outputs(self, split='train'):
        out = super()._generate_dataparser_outputs(split)
        meta = json.loads((self.config.data/'transforms.json').read_text())
        bounds = meta.get('blur_aabb')
        if bounds is not None:
            if self.config.auto_scale_poses or self.config.center_method != 'none' or self.config.orientation_method != 'none':
                raise ValueError('Explicit bounds require unchanged camera coordinates')
            out.scene_box.aabb = torch.tensor(bounds, dtype=torch.float32)
        return out


@dataclass
class ProbeParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: ProbeParser)


def configuration(request):
    cfg = deepcopy(method_configs['lookcloser'])
    data = Path(request['data']); meta = json.loads((data/'transforms.json').read_text())
    cfg.output_dir = Path(request['output']); cfg.experiment_name = 'trainer'
    cfg.timestamp = 'seed42'; cfg.vis = 'tensorboard'
    cfg.logging.local_writer.enable = False
    cfg.pipeline.datamanager.dataparser = ProbeParserConfig(data=data, downscale_factor=1,
        eval_mode='filename', scene_scale=1.5, center_method='focus', orientation_method='up')
    if 'blur_aabb' in meta or meta.get('coordinate_system'):
        parser = cfg.pipeline.datamanager.dataparser
        parser.auto_scale_poses=False; parser.center_method='none'; parser.orientation_method='none'
    dm = cfg.pipeline.datamanager
    dm.train_num_images_to_sample_from = -1; dm.train_num_times_to_repeat_images = -1
    dm.images_on_gpu = False; dm.masks_on_gpu = False
    # Explicit changes only; serialized configuration captures inherited defaults.
    for name, value in request.get('model', {}).items():
        if not hasattr(cfg.pipeline.model, name): raise ValueError(name)
        setattr(cfg.pipeline.model, name, value)
    for name, value in request.get('pipeline', {}).items():
        if not hasattr(cfg.pipeline, name): raise ValueError(name)
        setattr(cfg.pipeline, name, value)
    for name, value in request.get('sampler', {}).items():
        if not hasattr(dm.pixel_sampler, name): raise ValueError(name)
        setattr(dm.pixel_sampler, name, value)
    dm.train_num_rays_per_batch = request.get('rays', 4096)
    cfg.optimizers['fields']['optimizer'].lr = request.get('lr', .01)
    cfg.optimizers['fields']['scheduler'].lr_final = request.get('lr_final', .0001)
    cfg.optimizers['fields']['scheduler'].max_steps = request.get('lr_steps', 200000)
    cfg.grad_scaler_init_scale = request.get('grad_scale', cfg.grad_scaler_init_scale)
    return cfg


def state_digest(module):
    h = hashlib.sha256()
    for key, value in module.named_parameters():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics(model, prediction, target):
    prediction = prediction.float().clamp(0, 1); target = target.float().clamp(0, 1)
    a = prediction.permute(2,0,1)[None]; b = target.permute(2,0,1)[None]
    return dict(psnr=float(-10*torch.log10((prediction-target).square().mean().clamp_min(1e-12))),
                ssim=float(model.ssim(a,b)), lpips=float(model.lpips(a,b)))


@torch.no_grad()
def evaluate(trainer, out, step, request):
    pipe = trainer.pipeline; pipe.eval(); per_view=[]
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
    scale = request.get('eval_stride', 1)
    folder = out/f'eval_{step:06d}'; folder.mkdir(exist_ok=True)
    try:
        for split, ds in [('eval', pipe.datamanager.eval_dataset), ('train', pipe.datamanager.train_dataset)]:
            indices = range(len(ds)) if split=='eval' else request.get('train_review_indices', [0])
            for i in indices:
                item=ds[i]; gt=item['image'][::scale,::scale].to(pipe.device)
                camera=ds.cameras[i:i+1].to(pipe.device)
                h,w=gt.shape[:2]
                yy,xx=torch.meshgrid(torch.arange(h,device=pipe.device)*scale,torch.arange(w,device=pipe.device)*scale,indexing='ij')
                rays=camera.generate_rays(0,coords=torch.stack([yy,xx],-1).float()+.5)
                outputs=pipe.model.get_outputs_for_camera_ray_bundle(rays)
                pred=outputs['rgb']
                if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite evaluation RGB')
                values=metrics(pipe.model,pred,gt)
                rois={}
                regions=request.get('rois_by_image',{}).get(Path(ds.image_filenames[i]).name,request.get('rois',{}))
                for name,box in regions.items():
                    x0,y0,x1,y1=[int(v)//scale for v in box]
                    rois[name]=metrics(pipe.model,pred[y0:y1,x0:x1],gt[y0:y1,x0:x1])
                    panel=torch.cat([gt[y0:y1,x0:x1],pred[y0:y1,x0:x1]],1)
                    Image.fromarray((panel.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{split}_{i:03d}_{name}.png')
                Image.fromarray((pred.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{split}_{i:03d}.png')
                thumbnail=torch.cat([gt,pred,outputs['accumulation'].expand_as(pred)],1)
                img=Image.fromarray((thumbnail.cpu().numpy().clip(0,1)*255).astype('uint8'))
                img.thumbnail((1440,540));img.save(folder/f'{split}_{i:03d}_panel.jpg')
                per_view.append(dict(split=split,index=i,image=str(ds.image_filenames[i]),**values,rois=rois))
        vals=[v for v in per_view if v['split']=='eval']
        result=dict(step=step,eval_stride=scale,per_view=per_view,
            **{f'eval_all_{k}':float(np.mean([v[k] for v in vals])) for k in ['psnr','ssim','lpips']})
        write(folder/'metrics.json',result)
        return result
    finally:
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3]);pipe.train()


def train(request):
    out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists() or (out/'history.json').exists():
        raise RuntimeError('Refusing to overwrite an existing experiment')
    seed=request.get('seed',42)
    torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=configuration(request);cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup()
    pipe=trainer.pipeline
    sampler=pipe.datamanager.train_pixel_sampler
    original_sample=sampler.sample
    trace=[]
    def traced_sample(*args, **kwargs):
        batch=original_sample(*args, **kwargs)
        if len(trace)<32:
            trace.append(hashlib.sha256(batch['indices'].detach().cpu().numpy().tobytes()).hexdigest())
            write(out/'first_batches.json',trace)
        return batch
    sampler.sample=traced_sample
    write(out/'identity.json',dict(initial_weights=state_digest(pipe.model.field),
        train_count=len(pipe.datamanager.train_dataset),eval_count=len(pipe.datamanager.eval_dataset),
        transforms_sha256=sha(Path(request['data'])/'transforms.json'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),seed=seed,
        model_source=sha(Path(__file__).resolve().parents[2]/'nerfstudio/models/lookcloser.py'),
        field_source=sha(Path(__file__).resolve().parents[2]/'nerfstudio/fields/lookcloser_field.py'),
        runtime_source=sha(__file__)))
    history=[]; candidates=[]; start=time.monotonic();last_status=start
    pipe.train()
    for step in range(request['steps']):
        trainer.step=step
        for cb in trainer.callbacks:cb.run_callback_at_location(step,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        loss,_,batch_metrics=trainer.train_iteration(step)
        for cb in trainer.callbacks:cb.run_callback_at_location(step,TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        if step%100==0:
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            writer.write_out_storage()
        if time.monotonic()-last_status>=30 or step==0:
            status=dict(step=step+1,seconds=time.monotonic()-start,pid=os.getpid(),
                gpu_allocated_GiB=torch.cuda.memory_allocated()/2**30,scaler=trainer.grad_scaler.get_scale(),
                point_samples=int(pipe.cumulative_point_samples))
            write(out/'progress.json',status);print(json.dumps(status),flush=True);last_status=time.monotonic()
        if (step+1)%request['eval_every']==0 or step+1==request['steps']:
            result=evaluate(trainer,out,step+1,request);history.append(result);write(out/'history.json',history)
            path=out/f'candidate_{step+1:06d}.pt'
            snapshot=dict(pipeline=pipe.state_dict(),step=step+1,config=cfg,request=request)
            torch.save(snapshot,path)
            latest=out/'latest.tmp';latest.unlink(missing_ok=True);os.link(path,latest);latest.replace(out/'latest.pt')
            candidates.append(dict(path=path,psnr=result['eval_all_psnr'],lpips=result['eval_all_lpips']))
            max_psnr=max(c['psnr'] for c in candidates)
            for c in candidates[:]:
                if c['psnr']<max_psnr-.07:c['path'].unlink();candidates.remove(c)
            best=min(candidates,key=lambda c:(c['lpips'],-c['psnr']))
            tmp=out/'best.tmp';tmp.unlink(missing_ok=True);os.link(best['path'],tmp);tmp.replace(out/'best.pt')
            write(out/'selection.json',best)
            print(json.dumps({k:v for k,v in result.items() if k!='per_view'}),flush=True)
    write(out/'progress.json',dict(step=request['steps'],seconds=time.monotonic()-start,pid=os.getpid(),finished=True,
        gpu_allocated_GiB=torch.cuda.memory_allocated()/2**30,point_samples=int(pipe.cumulative_point_samples)))
    write(out/'complete.json',dict(seconds=time.monotonic()-start,steps=request['steps'],
        selected=best,best_sha256=sha(out/'best.pt'),point_samples=int(pipe.cumulative_point_samples)))
    trainer.shutdown()
