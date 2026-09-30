"""Resumable, explicitly gated Luster experiments with native RGB evaluation."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from blur_runtime import configuration, sha, state_digest, write, metrics
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.engine.trainer import _restore_rng_state
from nerfstudio.utils import writer


def select_best(history):
    maximum = max(row['eval_all_psnr'] for row in history)
    candidates = [row for row in history if maximum-row['eval_all_psnr'] <= .07]
    return min(candidates, key=lambda r: (r['eval_all_lpips'], -r['eval_all_psnr']))


def load_background_lookup(dataset, request):
    """Keep soft edges/near-boundary pixels unknown; never force foreground opaque."""
    import cv2
    cv2.setNumThreads(2)
    margin=int(request.get('background_mask_margin',3))
    if margin<0:raise ValueError('Background mask margin must be nonnegative')
    excluded=set(request.get('background_mask_exclude_cameras',[164]))
    arrays=[];offsets=[];widths=[];offset=0
    for filename in dataset.image_filenames:
        name=Path(filename).name;cid=int(name.split('_')[1])
        coverage=np.asarray(Image.open(Path(request['data'])/'masks'/name))
        possible_foreground=cv2.dilate((coverage>0).astype('uint8'),np.ones((2*margin+1,2*margin+1),'uint8'))
        background=possible_foreground==0
        if cid in excluded:background[:]=False
        offsets.append(offset);widths.append(coverage.shape[1]);offset+=background.size;arrays.append(background.ravel())
    flat=torch.from_numpy(np.concatenate(arrays));offsets=torch.tensor(offsets);widths=torch.tensor(widths)
    def lookup(indices):
        idx=indices.detach().cpu().long();camera,y,x=idx.unbind(-1)
        return flat[offsets[camera]+y*widths[camera]+x,None]
    return lookup


@torch.no_grad()
def evaluate(trainer, request, step):
    pipe = trainer.pipeline
    pipe.eval()
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
    folder = Path(request['output']) / f'eval_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    stride = request.get('eval_stride', 1)
    meta = json.loads((Path(request['data'])/'transforms.json').read_text())
    by_name = {Path(r['file_path']).name:r for r in meta['frames']}
    try:
        for split, ds in [('eval', pipe.datamanager.eval_dataset), ('train', pipe.datamanager.train_dataset)]:
            indices = range(len(ds)) if split == 'eval' else request.get('train_review_indices', [0])
            for index in indices:
                camera = ds.cameras[index:index+1].to(pipe.device)
                gt = ds[index]['image'][::stride, ::stride].to(pipe.device)
                name = Path(ds.image_filenames[index]).name
                h, w = gt.shape[:2]
                yy, xx = torch.meshgrid(torch.arange(h, device=pipe.device)*stride,
                                        torch.arange(w, device=pipe.device)*stride, indexing='ij')
                ray = camera.generate_rays(0, coords=torch.stack([yy, xx], -1).float()+.5)
                outputs = pipe.model.get_outputs_for_camera_ray_bundle(ray)
                pred = outputs['rgb'].float()
                if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite RGB')
                mask_np = np.array(Image.open(Path(request['data'])/'masks'/name))[::stride, ::stride] >= 128
                mask = torch.from_numpy(mask_np).to(pipe.device)
                foreground_pixels=int(mask.sum())
                foreground_mse = (pred.clamp(0,1)[mask]-gt[mask]).square().mean() if foreground_pixels else None
                fg_psnr = float(-10*torch.log10(foreground_mse.clamp_min(1e-12))) if foreground_pixels else None
                values = metrics(pipe.model, pred, gt)
                rois = {}
                for label, box in request.get('rois_by_image', {}).get(name, {}).items():
                    x0,y0,x1,y1 = [int(v)//stride for v in box]
                    a,b = pred[y0:y1,x0:x1],gt[y0:y1,x0:x1]
                    if min(a.shape[:2]) < 32: raise ValueError(f'ROI too small: {name}/{label}')
                    rois[label] = metrics(pipe.model, a, b)
                    rois[label]['foreground_fraction']=float(mask[y0:y1,x0:x1].float().mean())
                    panel = torch.cat([b,a],1)
                    Image.fromarray(np.rint(panel.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{split}_{index:03d}_{label}.png')
                Image.fromarray(np.rint(pred.cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'{split}_{index:03d}.png')
                opacity = outputs['accumulation'].float()
                panel = torch.cat([gt,pred,opacity.expand_as(pred)],1)
                im = Image.fromarray(np.rint(panel.cpu().numpy().clip(0,1)*255).astype('uint8'))
                im.thumbnail((1500,640));im.save(folder/f'{split}_{index:03d}_panel.jpg')
                rows.append(dict(split=split,index=index,image=name,physical_camera=by_name[name]['physical_camera'],
                                 **values,foreground_psnr=fg_psnr,foreground_pixels=foreground_pixels,
                                 foreground_opacity=float(opacity[mask].mean()) if foreground_pixels else None,
                                 background_opacity=float(opacity[~mask].mean()) if (~mask).any() else None,rois=rois))
        ev = [r for r in rows if r['split']=='eval']
        result = dict(step=step,eval_stride=stride,per_view=rows,
                      **{f'eval_all_{k}':float(np.mean([r[k] for r in ev])) for k in ['psnr','ssim','lpips']})
        write(folder/'metrics.json',result)
        return result
    finally:
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2]);random.setstate(rng[3]);pipe.train()


def train(request):
    out = Path(request['output']);out.mkdir(parents=True, exist_ok=True)
    if not request.get('diagnostic_only',False):
        data=Path(request['data'])
        audit=json.loads((data/'audit_ready.json').read_text())
        if audit['frequency_maps']!=162 or audit['transforms_sha256']!=sha(data/'transforms.json'):
            raise ValueError('Dataset has not passed the final preprocessing audit')
    if (out/'request.json').exists(): raise RuntimeError('Refusing to overwrite an existing run')
    write(out/'request.json', request)
    torch.set_num_threads(2)
    seed=request.get('seed',42)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg = configuration(request)
    cfg.max_num_iterations=request['end_step']+1
    cfg.save_only_latest_checkpoint=False
    cfg.pipeline.model.depth_loss_mult=0.
    cfg.pipeline.datamanager.pixel_sampler.ignore_mask=True
    if request.get('warm_start'):
        if request.get('resume') or request.get('parent_history'):
            raise ValueError('A new temporal frame must not inherit optimizer state or metric history')
        parent=Path(request['warm_start'])
        parent_request=json.loads(Path(request['warm_start_request']).read_text())
        old_meta=json.loads((Path(parent_request['data'])/'transforms.json').read_text())
        new_meta=json.loads((Path(request['data'])/'transforms.json').read_text())
        if old_meta['blur_aabb'] != new_meta['blur_aabb']:
            raise ValueError('Temporal field weights require identical normalized AABBs')
        old_bounds=json.loads((Path(parent_request['data'])/'bounds_audit.json').read_text())
        new_bounds=json.loads((Path(request['data'])/'bounds_audit.json').read_text())
        if old_bounds['normalization']!=new_bounds['normalization'] or old_bounds['scale']!=new_bounds['scale']:
            raise ValueError('Temporal field weights require the same world-coordinate normalization')
        cfg.load_checkpoint=parent
        cfg.checkpoint_load_mode='model_parameters_only'
        cfg.checkpoint_load_parameter_hash_audit=True
        cfg.load_optimizers=False;cfg.load_scheduler=False
    if request.get('resume'):
        cfg.load_checkpoint=Path(request['resume'])
        cfg.load_scheduler=True;cfg.load_optimizers=True
        cfg.resume_fields_lr_override=request.get('resume_fields_lr_override')
    cfg.save_config()
    trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup()
    pipe=trainer.pipeline
    sampler=pipe.datamanager.train_pixel_sampler
    background_lookup=load_background_lookup(pipe.datamanager.train_dataset,request) if cfg.pipeline.model.background_opacity_loss_mult>0 else None
    sample=sampler.sample;trace=[];traced_indices=[]
    def traced(*a, **kw):
        batch=sample(*a,**kw)
        if background_lookup is not None:batch['background_mask']=background_lookup(batch['indices'])
        if len(trace)<32:
            import hashlib
            trace.append(hashlib.sha256(batch['indices'].detach().cpu().numpy().tobytes()).hexdigest())
            write(out/'first_batches.json',trace)
            traced_indices.append(batch['indices'].detach().cpu())
            if len(trace)==32:
                indices=torch.cat(traced_indices);coverage=[]
                for index,path in enumerate(pipe.datamanager.train_dataset.image_filenames):
                    xy=indices[indices[:,0]==index,1:]
                    coverage.append(dict(image=Path(path).name,samples=len(xy),
                                         min_yx=xy.min(0).values.tolist() if len(xy) else None,
                                         max_yx=xy.max(0).values.tolist() if len(xy) else None))
                write(out/'ray_coverage_first32.json',coverage);traced_indices.clear()
        return batch
    sampler.sample=traced
    identity=dict(initial_weights=state_digest(pipe.model.field),seed=seed,
                  transforms_sha256=sha(Path(request['data'])/'transforms.json'),
                  train_count=len(pipe.datamanager.train_dataset),eval_count=len(pipe.datamanager.eval_dataset),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),start_step=trainer._start_step,
                  runtime_sha256=sha(Path(__file__)),
                  sampler_sha256=sha(Path(__file__).resolve().parents[2]/'nerfstudio/lookcloser_pixel_sampler.py'),
                  field_sha256=sha(Path(__file__).resolve().parents[2]/'nerfstudio/fields/lookcloser_field.py'),
                  model_sha256=sha(Path(__file__).resolve().parents[2]/'nerfstudio/models/lookcloser.py'),
                  derived_manifest_sha256=sha(Path(request['data'])/'derived_manifest.json') if not request.get('diagnostic_only',False) else None,
                  checkpoint_load_audit=trainer.checkpoint_load_audit)
    write(out/'identity.json',identity)
    history=[]
    if request.get('parent_history'):
        history=deepcopy(json.loads(Path(request['parent_history']).read_text()))
    gates=set(request.get('eval_steps',[])) | {request['end_step']}
    start=time.monotonic();last_status=start
    # Restore after setup/instrumentation, exactly as Trainer.train does.
    if trainer._loaded_rng_state is not None:
        _restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None
    pipe.train()
    for step in range(trainer._start_step,request['end_step']+1):
        trainer.step=step
        for cb in trainer.callbacks:cb.run_callback_at_location(step,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        objective,_,batch_metrics=trainer.train_iteration(step)
        for cb in trainer.callbacks:cb.run_callback_at_location(step,TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        if step%100==0:
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training objective')
            writer.write_out_storage()
        if time.monotonic()-last_status>=30 or step==trainer._start_step:
            status=dict(step=step,seconds=time.monotonic()-start,pid=os.getpid(),
                        gpu_allocated_GiB=torch.cuda.memory_allocated()/2**30,scaler=trainer.grad_scaler.get_scale(),
                        point_samples=int(pipe.cumulative_point_samples),phase='train',
                        fields_lr=trainer.optimizers.optimizers['fields'].param_groups[0]['lr'])
            if 'psnr' in batch_metrics:
                status['batch_psnr']=float(batch_metrics['psnr'])
            write(out/'progress.json',status);print(json.dumps(status),flush=True);last_status=time.monotonic()
        if step in gates:
            write(out/'progress.json',dict(step=step,pid=os.getpid(),seconds=time.monotonic()-start,phase='evaluation'))
            result=evaluate(trainer,request,step)
            trainer.save_checkpoint(step)
            checkpoint=trainer.checkpoint_dir/f'step-{step:09d}.ckpt'
            result['checkpoint']=str(checkpoint)
            result['config']=str(trainer.base_dir/'config.yml')
            result['render_dir']=str(out/f'eval_{step:06d}')
            history.append(result);write(out/'history.json',history)
            write(out/'selection.json',select_best(history))
            print(json.dumps({k:v for k,v in result.items() if k!='per_view'}),flush=True)
    write(out/'complete.json',dict(step=request['end_step'],seconds=time.monotonic()-start,
                                   latest_checkpoint=str(checkpoint),selected=select_best(history)))
    write(out/'progress.json',dict(step=request['end_step'],pid=os.getpid(),seconds=time.monotonic()-start,finished=True))
    trainer.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    args=parser.parse_args()
    train(json.loads(args.request.read_text()))
