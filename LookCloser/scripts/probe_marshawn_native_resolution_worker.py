"""One isolated native resolution technical phase; requires an executor job receipt."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def main(a):
    job=json.loads(a.job.read_text());release=Path(job['root_release']);approval=json.loads(release.read_text())
    if not approval.get('technical_GPU_gate_authorized') or approval.get('production_launch_authorized') is not False:raise ValueError('Technical-only root release required')
    if sha(release)!=job['root_release_sha256']:raise ValueError('Release changed')
    if job['phase'] not in approval['allowed_phases']:raise ValueError('Phase is not authorized')
    expected=16 if job['phase'] in ['TRAIN4K_updates16','TRAIN4K_resume16'] else 0
    if job['successful_updates']!=expected or job['maximum_attempts']!=128:raise ValueError('Bounded technical update policy changed')
    if Path(job['output']).parent.resolve()!=Path(approval['output']).resolve():raise ValueError('Phase output outside reviewed technical root')
    for path,digest in approval['bound_files'].items():
        if sha(path)!=digest:raise ValueError('Bound file changed: '+path)
    for path,digest in job['bound_inputs'].items():
        if sha(path)!=digest:raise ValueError('Job input changed: '+path)
    for path in reversed(job['python_path']):sys.path.insert(0,path)
    import numpy as np,torch,yaml
    if torch.__version__!=approval['torch_version'] or torch.version.cuda!=approval['cuda_version']:raise ValueError('Historical Torch/CUDA runtime changed')
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from nerfstudio.scripts.train import _set_random_seed
    torch.set_num_threads(4)
    def digest(value):
        h=hashlib.sha256()
        def feed(v):
            if torch.is_tensor(v):h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
            elif isinstance(v,dict):
                for k in sorted(v,key=str):h.update(str(k).encode());feed(v[k])
            elif isinstance(v,(tuple,list)):
                for x in v:feed(x)
            else:h.update(repr(v).encode())
        feed(value);return h.hexdigest()
    out=Path(job['output']);out.mkdir(exist_ok=False)
    source=Path(job['source_checkpoint']);saved=torch.load(source,map_location='cpu',weights_only=False);runtime=json.loads(Path(job['source_runtime']).read_text())
    cfg=yaml.load(Path(job['config']).read_text(),Loader=yaml.Loader);assert Path(cfg.load_checkpoint)==source
    cfg.output_dir=out/'run';cfg.pipeline.final_grid_artifact_dir=out/'grids';cfg.save_only_latest_checkpoint=False
    _set_random_seed(cfg.machine.seed);trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();p=trainer.pipeline
    def domains():return dict(pipeline=p.state_dict(),optimizers={k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},schedulers={k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},scalers=trainer.grad_scaler.state_dict())
    restored={key:digest(value)==digest(saved[key]) for key,value in domains().items()};assert all(restored.values()),restored
    for split,key in [('train','cached_train_image_idx'),('eval','eval_cached_image_idx')]:
        cache=getattr(p.datamanager,split+'_image_dataloader').cached_collated_batch;actual=cache['image_idx'].detach().cpu().tolist();wanted=runtime[key];lookup={int(v):i for i,v in enumerate(actual)};order=torch.tensor([lookup[v] for v in wanted])
        for k,value in list(cache.items()):
            if torch.is_tensor(value) and value.ndim and value.shape[0]==len(actual):cache[k]=value[order.to(value.device)]
        assert cache['image_idx'].detach().cpu().tolist()==wanted
    p.dynamic_num_rays_per_batch=runtime['dynamic_num_rays_per_batch'];p._update_pixel_samplers()
    grid=Path(job['source_grid'])
    with np.load(grid) as z:
        object.__setattr__(p,'_actual_core',torch.from_numpy(z['core']).to(p.device));object.__setattr__(p,'_actual_active',torch.from_numpy(z['active']).to(p.device))
    gridmeta=json.loads(grid.with_suffix('.json').read_text());object.__setattr__(p,'_actual_threshold',gridmeta['gpu_native_threshold']);object.__setattr__(p,'_last_core_step',gridmeta['core_update_step']);assert torch.equal(p.model.occupancy_grid.binaries,p._actual_active)
    cache=p.datamanager.train_image_dataloader.cached_collated_batch;assert tuple(cache['image'].shape)==(63,1012*job['train_scale'],1920*job['train_scale'],3)
    assert cache['image'].dtype==torch.float32 and not cache['image'].is_cuda
    _restore_rng_state(saved['rng_state']);trainer._loaded_rng_state=None;assert digest(_capture_rng_state())==digest(saved['rng_state'])
    initial_model=digest(p.state_dict());initial_rng=digest(saved['rng_state']);initial_domains={k:digest(v) for k,v in domains().items()}
    def adam():
        counts=[int(v['step']) for v in trainer.optimizers.optimizers['fields'].state.values()];assert counts and min(counts)==max(counts);return max(counts)
    initial_adam=adam();prior_meta=None
    if job['measure_priors']:
        p.train();step=int(saved['step'])+1;p.model.current_train_step=int(saved['step']);named=[(n,v) for n,v in p.model.field.named_parameters() if v.requires_grad];parameters=[v for _,v in named];arrays={};values={};scale=float(trainer.grad_scaler.get_scale())
        for name in ['free','band','MVS']:
            with torch.autocast(device_type='cuda',enabled=cfg.mixed_precision):
                if name=='free':value,_=p._final_free(p.model.field,step)
                elif name=='band':value,_=p._final_band(p.model.field,step,p._final_free.sample_batch(step)['indices'])
                else:value,_=p._MVS_depth(p.model,step)
            assert bool(torch.isfinite(value));gradients=torch.autograd.grad(value*scale,parameters,allow_unused=True)
            vector=np.concatenate([(torch.zeros_like(param) if grad is None else grad).detach().float().cpu().numpy().reshape(-1)/scale for param,grad in zip(parameters,gradients)])
            assert np.isfinite(vector).all(),name;arrays[name+'_gradient']=vector;values[name]=float(value.detach());del value,gradients,vector
        assert digest(p.state_dict())==initial_model and digest(_capture_rng_state())==initial_rng
        np.savez_compressed(out/'prior_gradients.npz',**arrays);prior_meta=dict(step=step,values=values,parameter_names=[name for name,_ in named],parameter_shapes=[list(v.shape) for _,v in named],AMP_scale=scale,gradients_divided_by_scale=True,archive_sha256=sha(out/'prior_gradients.npz'));atomic(out/'prior_probe.json',prior_meta);del arrays
    _restore_rng_state(saved['rng_state']);assert digest(_capture_rng_state())==initial_rng;del saved
    requested=job['successful_updates'];assert requested in [0,16];steps=[];start=trainer._start_step;last_step=start-1
    while adam()-initial_adam<requested:
        if len(steps)>=job['maximum_attempts']:raise RuntimeError('Bounded technical attempts exhausted before requested actual Adam updates')
        step=start+len(steps);trainer.step=step;p.train();before=adam()
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        value,terms,metrics=trainer.train_iteration(step);assert bool(torch.isfinite(value))
        for cb in trainer.callbacks:cb.run_callback_at_location(step,location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        trainer.eval_iteration(step);after=adam();assert after-before in [0,1]
        row=dict(step=step,adam_before=before,adam_after=after,actual_update=after-before,lr=trainer.optimizers.optimizers['fields'].param_groups[0]['lr'],scale=trainer.grad_scaler.get_scale(),dynamic_next_rays=p.dynamic_num_rays_per_batch);steps.append(row);last_step=step;atomic(out/'progress.json',dict(time=time.time(),attempts=len(steps),actual_updates=after-initial_adam,latest=row))
    if requested:assert digest(p.state_dict())!=initial_model and adam()-initial_adam==requested
    else:assert {k:digest(v) for k,v in domains().items()}==initial_domains and digest(_capture_rng_state())==initial_rng
    trainer._train_batch_prefetch_barrier();checkpoint=out/('step-%09d.ckpt'%last_step)
    state=dict(step=last_step,**domains(),rng_state=_capture_rng_state(),checkpoint_load_audit=trainer.checkpoint_load_audit);tmp=checkpoint.with_suffix('.ckpt.tmp');torch.save(state,tmp);os.replace(tmp,checkpoint)
    reloaded=torch.load(checkpoint,map_location='cpu',weights_only=False);roundtrip={k:digest(state[k])==digest(reloaded[k]) for k in ['pipeline','optimizers','schedulers','scalers','rng_state']};assert all(roundtrip.values())
    exported=p.export_grid_for_checkpoint(checkpoint,last_step)
    runtime_new=dict(dynamic_num_rays_per_batch=p.dynamic_num_rays_per_batch,cached_train_image_idx=p.datamanager.train_image_dataloader.cached_collated_batch['image_idx'].detach().cpu().tolist(),eval_cached_image_idx=p.datamanager.eval_image_dataloader.cached_collated_batch['image_idx'].detach().cpu().tolist(),native_core_update_step=p._last_core_step)
    atomic(out/'runtime.json',runtime_new)
    receipt=dict(kind='ISOLATED_NATIVE_RESOLUTION_TECHNICAL_PHASE',phase=job['phase'],source_checkpoint=str(source),source_checkpoint_sha256=sha(source),source_runtime_sha256=sha(job['source_runtime']),source_grid_sha256=sha(grid),output_checkpoint=str(checkpoint),output_checkpoint_sha256=sha(checkpoint),output_runtime=str(out/'runtime.json'),output_runtime_sha256=sha(out/'runtime.json'),output_grid=exported['grid_archive'],output_grid_sha256=exported['grid_archive_sha256'],restored_domain_equality=restored,serialization_roundtrip=roundtrip,initial_pipeline_hash=initial_model,final_pipeline_hash=digest(p.state_dict()),initial_rng_hash=initial_rng,initial_adam=initial_adam,final_adam=adam(),attempts=len(steps),actual_updates=adam()-initial_adam,AMP_skips=len(steps)-(adam()-initial_adam),steps=steps,source_step=start-1,final_step=last_step,train_scale=job['train_scale'],prior_probe=prior_meta,runtime_source=runtime,runtime_restored=True,torch_version=torch.__version__,cuda_version=torch.version.cuda,config_sha256=sha(job['config']),job_sha256=sha(a.job),no_production_ancestry=True)
    atomic(out/'receipt.json',receipt);print(json.dumps(dict(phase=job['phase'],attempts=len(steps),actual_updates=receipt['actual_updates'],checkpoint=str(checkpoint))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True);main(parser.parse_args())
