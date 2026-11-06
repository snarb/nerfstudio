"""Durable native optimizer-state continuation with atomic 5k common-metric gates.

Production requires a separately reviewed release document binding config and
helpers. Original20k remains eligible and immutable; technical probes are not
production ancestry. All model/geometry objectives come from frozen native code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.%d.tmp'%os.getpid())
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def select_best(evaluations):
    minimum=min(row['metrics']['lpips'] for row in evaluations)
    return min((row for row in evaluations if row['metrics']['lpips']<=minimum+.001),key=lambda row:(-row['metrics']['psnr'],row['step']))


def advance_gate(state,evaluation):
    from marshawn_tuning_metrics import PlateauTracker
    if evaluation['step'] in [r['step'] for r in state['evaluations']]:
        prior=next(r for r in state['evaluations'] if r['step']==evaluation['step'])
        if prior!=evaluation:raise ValueError('Conflicting gate replay')
        return state
    tracker=PlateauTracker.from_dict(state['plateau'])
    tracker.observe(evaluation['step'],evaluation['metrics']['validation_loss'],evaluation['metrics']['lpips'])
    result=dict(state,evaluations=state['evaluations']+[evaluation],plateau=tracker.to_dict())
    result['selection']=select_best(result['evaluations']);return result


def validate_release(path,restore_only=False):
    release=json.loads(Path(path).read_text())
    if not(release.get('production_launch_authorized') is True or (restore_only and release.get('restoration_probe_authorized') is True)):raise ValueError('Missing reviewed production release')
    for name,digest in release['bound_files'].items():
        if sha(name)!=digest:raise ValueError('Release-bound file changed: '+name)
    return release


def run(a):
    release=validate_release(a.release,a.restore_only)
    for path in reversed(release['python_path']):sys.path.insert(0,path)
    import copy,random
    import numpy as np
    import torch,yaml
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from nerfstudio.scripts.train import _set_random_seed
    from nerfstudio.utils import writer
    from marshawn_tuning_metrics import PlateauTracker
    torch.set_num_threads(4)
    cfg=yaml.load(Path(release['config']).read_text(),Loader=yaml.Loader)
    out=Path(release['output']);out.mkdir(parents=True,exist_ok=True)
    # Atomic lock prevents a duplicate worker, including while a gate reader runs.
    import fcntl
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pointer_path=out/'resume_pointer.json';state_path=out/'state.json'
    if pointer_path.exists() and not a.resume:raise ValueError('Existing run requires explicit --resume')
    if a.resume and not pointer_path.exists():raise ValueError('No committed resume pointer')
    initial=not pointer_path.exists()
    if initial:
        source=Path(release['source_checkpoint']);runtime=json.loads(Path(release['initial_runtime']).read_text());grid_path=Path(release['initial_grid'])
        source_sha=release['source_checkpoint_sha256']
    else:
        pointer=json.loads(pointer_path.read_text());source=Path(pointer['checkpoint']);source_sha=pointer['checkpoint_sha256'];runtime_path=Path(pointer['runtime']);assert sha(runtime_path)==pointer['runtime_sha256'];runtime=json.loads(runtime_path.read_text());grid_path=Path(pointer['grid_archive']);assert sha(grid_path)==pointer['grid_sha256']
    assert sha(source)==source_sha
    cfg.load_checkpoint=source
    saved=torch.load(source,map_location='cpu',weights_only=False)
    _set_random_seed(cfg.machine.seed)
    trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();pipeline=trainer.pipeline
    # Bind the exact restored state before any optimizer work.
    def tensor_equal(left,right):
        if torch.is_tensor(left):return torch.equal(left.detach().cpu(),right.detach().cpu())
        if isinstance(left,np.ndarray):return np.array_equal(left,right)
        if isinstance(left,dict):return left.keys()==right.keys() and all(tensor_equal(left[k],right[k]) for k in left)
        if isinstance(left,(list,tuple)):return len(left)==len(right) and all(tensor_equal(x,y) for x,y in zip(left,right))
        return left==right
    domains={'pipeline':pipeline.state_dict(),'optimizers':{k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},'schedulers':{k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},'scalers':trainer.grad_scaler.state_dict()}
    restored={k:tensor_equal(v,saved[k]) for k,v in domains.items()};assert all(restored.values()),restored
    for split,key in [('train','cached_train_image_idx'),('eval','eval_cached_image_idx')]:
        cache=getattr(pipeline.datamanager,split+'_image_dataloader').cached_collated_batch
        actual=cache['image_idx'].detach().cpu().tolist();wanted=runtime[key];lookup={int(v):i for i,v in enumerate(actual)};index=torch.tensor([lookup[v] for v in wanted])
        for k,v in list(cache.items()):
            if torch.is_tensor(v) and v.ndim and v.shape[0]==len(actual):cache[k]=v[index.to(v.device)]
        assert cache['image_idx'].detach().cpu().tolist()==wanted
    pipeline.dynamic_num_rays_per_batch=runtime['dynamic_num_rays_per_batch'];pipeline._update_pixel_samplers()
    with np.load(grid_path) as grid:
        object.__setattr__(pipeline,'_actual_core',torch.from_numpy(grid['core']).to(pipeline.device));object.__setattr__(pipeline,'_actual_active',torch.from_numpy(grid['active']).to(pipeline.device))
    meta=json.loads(grid_path.with_suffix('.json').read_text());object.__setattr__(pipeline,'_actual_threshold',meta['gpu_native_threshold']);object.__setattr__(pipeline,'_last_core_step',meta['core_update_step']);assert torch.equal(pipeline.model.occupancy_grid.binaries,pipeline._actual_active)
    if state_path.exists():state=json.loads(state_path.read_text())
    else:
        baseline=json.loads(Path(release['baseline_metrics']).read_text());evaluation=dict(step=20000,metrics=baseline['metrics'],by_actor=baseline['by_actor'],by_view=baseline['by_view'],checkpoint=release['source_checkpoint'],checkpoint_sha256=release['source_checkpoint_sha256'],renders=release['baseline_renders'])
        tracker=PlateauTracker(anchor_step=20000,interval=5000,patience=4,loss_relative_delta=.001,lpips_absolute_delta=.001);tracker.observe(20000,evaluation['metrics']['validation_loss'],evaluation['metrics']['lpips'])
        state=dict(evaluations=[evaluation],plateau=tracker.to_dict(),selection=evaluation);atomic(state_path,state)
    atomic(out/'restoration.json',dict(time=time.time(),pid=os.getpid(),source=str(source),source_sha256=source_sha,start_step=trainer._start_step,domains_equal=restored,dynamic_rays=pipeline.dynamic_num_rays_per_batch,runtime=runtime,release_sha256=sha(a.release)))
    if a.restore_only:
        _restore_rng_state(saved['rng_state']);assert tensor_equal(_capture_rng_state(),saved['rng_state'])
        atomic(out/'restore_only_complete.json',dict(updates=0,domains_equal=restored));return

    def checkpoint(step):
        trainer._train_batch_prefetch_barrier();trainer.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        path=trainer.checkpoint_dir/('step-%09d.ckpt'%step);tmp=path.with_suffix('.ckpt.tmp')
        torch.save(dict(step=step,pipeline=pipeline.state_dict(),optimizers={k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict(),rng_state=_capture_rng_state(),checkpoint_load_audit=trainer.checkpoint_load_audit),tmp);os.replace(tmp,path)
        grid=pipeline.export_grid_for_checkpoint(path,step)
        runtime=dict(dynamic_num_rays_per_batch=pipeline.dynamic_num_rays_per_batch,cached_train_image_idx=pipeline.datamanager.train_image_dataloader.cached_collated_batch['image_idx'].detach().cpu().tolist(),eval_cached_image_idx=pipeline.datamanager.eval_image_dataloader.cached_collated_batch['image_idx'].detach().cpu().tolist(),native_core_update_step=pipeline._last_core_step)
        runtime_path=out/'runtime'/('step-%09d.json'%step);atomic(runtime_path,runtime)
        pointer=dict(step=step,checkpoint=str(path),checkpoint_sha256=sha(path),runtime=str(runtime_path),runtime_sha256=sha(runtime_path),grid_archive=grid['grid_archive'],grid_sha256=grid['grid_archive_sha256'])
        atomic(pointer_path,pointer);return pointer

    def gate(pointer):
        nonlocal state
        step=pointer['step']
        gate_rng_before=_capture_rng_state()
        if any(row['step']==step for row in state['evaluations']):return
        folder=out/'evaluations'/('step-%09d_attempt-%d'%(step,time.time_ns()))
        atomic(out/'status.json',dict(status='evaluating_saved_checkpoint',step=step,pid=os.getpid(),time=time.time(),checkpoint=pointer['checkpoint']))
        command=[sys.executable,release['evaluator'],'--config',release['config'],'--checkpoint',pointer['checkpoint'],'--checkpoint-sha256',pointer['checkpoint_sha256'],'--output',str(folder),'--ready',release['ready'],'--box',release['box'],'--v2-manifest',release['v2_manifest'],'--renderer',release['renderer']]
        for path in release['python_path']:command+=['--python-path',path]
        with (out/('evaluation_%09d.log'%step)).open('a') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/'evaluation.json').read_text());assert result['checkpoint_sha256']==pointer['checkpoint_sha256'] and result['step']==step
        row=dict(step=step,**result['v1'],checkpoint=pointer['checkpoint'],checkpoint_sha256=pointer['checkpoint_sha256'],renders=str(folder/'renders'),evaluation=str(folder/'evaluation.json'),evaluation_sha256=sha(folder/'evaluation.json'))
        assert tensor_equal(gate_rng_before,_capture_rng_state()), 'Read-only gate altered parent RNG'
        state=advance_gate(state,row);atomic(state_path,state)
        atomic(out/'gate_complete.json',dict(step=step,state_sha256=sha(state_path),evaluation=row['evaluation'],evaluation_sha256=row['evaluation_sha256'],time=time.time(),stopped=state['plateau']['stopped']))

    # Replay an incomplete read-only gate before allowing another optimizer update.
    if not initial:gate(pointer)
    _restore_rng_state(saved['rng_state']);assert tensor_equal(_capture_rng_state(),saved['rng_state']);trainer._loaded_rng_state=None;del domains,saved
    if state['plateau']['stopped']:
        atomic(out/'complete.json',dict(selection=state['selection'],plateau=state['plateau'],reason='both_signals_plateau',time=time.time()));return
    atomic(out/'status.json',dict(status='training',step=trainer._start_step-1,pid=os.getpid(),time=time.time()))
    for step in range(trainer._start_step,cfg.max_num_iterations):
        trainer.step=step;pipeline.train()
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        value,terms,metrics=trainer.train_iteration(step)
        if not bool(torch.isfinite(value)):raise RuntimeError('Nonfinite training scalar; no further updates')
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        # Original 2k internal eval cadence and its stochastic RNG exposure remain.
        trainer.eval_iteration(step);writer.write_out_storage()
        if step%100==0:
            health=dict(status='training',step=step,pid=os.getpid(),time=time.time(),adam_updates=max(int(s['step']) for s in trainer.optimizers.optimizers['fields'].state.values()),dynamic_next_rays=pipeline.dynamic_num_rays_per_batch,lr=trainer.optimizers.optimizers['fields'].param_groups[0]['lr'],scale=trainer.grad_scaler.get_scale(),gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved())
            atomic(out/'status.json',health)
            with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(health)+'\n')
        if step%5000==0:
            pointer=checkpoint(step);gate(pointer)
            if state['plateau']['stopped']:
                atomic(out/'complete.json',dict(selection=state['selection'],plateau=state['plateau'],reason='both_signals_plateau',time=time.time()));return
            atomic(out/'status.json',dict(status='training',step=step,pid=os.getpid(),time=time.time()))
    raise RuntimeError('Operational iteration ceiling reached before plateau; not successful completion')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--restore-only',action='store_true')
    run(parser.parse_args())
