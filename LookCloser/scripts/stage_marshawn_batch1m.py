"""Stage ONE cold local batch/schedule quality trial; never launches training."""
from pathlib import Path
import copy,hashlib,json,os,shutil,sys

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
STAGE=ROOT/'native_png_batch1M_sched22k_LOCAL_STAGED'
ORIGINAL=ROOT/'native_png_linear_distortion_STAGED'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def normalize(x):
    if x is None or isinstance(x,(str,int,float,bool)):return x
    if isinstance(x,Path):return str(x)
    if isinstance(x,type):return x.__module__+'.'+x.__qualname__
    if isinstance(x,dict):return {str(k):normalize(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalize(v)for v in x]
    if hasattr(x,'__dict__'):return {k:normalize(v)for k,v in x.__dict__.items()}
    return repr(x)


def differences(a,b,prefix=''):
    if isinstance(a,dict)and isinstance(b,dict):
        out=[]
        for k in sorted(set(a)|set(b)):out+=differences(a.get(k),b.get(k),prefix+'.'+k if prefix else k)
        return out
    return []if a==b else[dict(field=prefix,original=a,candidate=b)]


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def main():
    STAGE.mkdir(exist_ok=False);code=STAGE/'code';code.mkdir()
    for name in ['run_marshawn_png_distortion_control.py','marshawn_packed_distortion_candidate.py']:
        shutil.copyfile(ORIGINAL/'code'/name,code/name)
        assert sha(code/name)==sha(ORIGINAL/'code'/name)
    # Minimal adaptation of the tested cold probe, not a new training driver.
    probe=(ORIGINAL/'code/probe_marshawn_distortion_cold_config.py').read_text()
    probe=probe.replace('pipe.dynamic_num_rays_per_batch==16 and pipe.datamanager.train_pixel_sampler.num_rays_per_batch==16','pipe.dynamic_num_rays_per_batch==64 and pipe.datamanager.train_pixel_sampler.num_rays_per_batch==64')
    probe=probe.replace('expected=int(16*(cfg.pipeline.target_num_samples/samples))','expected=int(64*(cfg.pipeline.target_num_samples/samples))')
    probe=probe.replace('initial_actual_rays=16','initial_actual_rays=64')
    probe=probe.replace("    callbacks=pipe.get_training_callbacks", "    captured=[]\n    native_next_train=pipe.datamanager.next_train\n    def capture_batch(step):\n        rays,batch=native_next_train(step);captured.append(batch['indices'].detach().cpu().clone());return rays,batch\n    pipe.datamanager.next_train=capture_batch\n    callbacks=pipe.get_training_callbacks")
    anchor="            assert digest_model(model)==after_callback\n            model.zero_grad(set_to_none=True)\n            for callback in callbacks:callback.run_callback_at_location(16,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)"
    budget="""            assert digest_model(model)==after_callback
            model.zero_grad(set_to_none=True)
            budget_rays=int(pipe.dynamic_num_rays_per_batch)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type='cuda',enabled=cfg.mixed_precision):
                budget_outputs,budget_losses,budget_metrics=pipe.get_train_loss_dict(1)
            budget_samples=int(budget_metrics['num_samples_per_batch'])
            budget_base=sum(v for k,v in budget_losses.items()if k!='packed_distortion')
            budget_combined=sum(budget_losses.values());budget_status=[];budget_scale=cfg.grad_scaler_init_scale
            for attempt in range(12):
                plain=gradients(model,budget_base,budget_scale);added=gradients(model,budget_combined,budget_scale)
                budget_status.append(dict(scale=budget_scale,baseline=plain,with_distortion=added))
                if plain['finite']and added['finite']:break
                if plain['finite']and not added['finite']:raise ValueError('Budget distortion-only AMP overflow')
                budget_scale*=.5
            else:raise ValueError('No finite actual budget-batch backward')
            budget=dict(actual_rays=budget_rays,packed_samples=budget_samples,
                unique_pixel_triplets=int(torch.unique(captured[-1],dim=0).shape[0]),
                physical_train_cameras=int(captured[-1][:,0].unique().numel()),
                actual_next_dynamic_rays=int(pipe.dynamic_num_rays_per_batch),full_AMP_backward=budget_status,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
            assert budget_rays==expected and len(captured)==2 and budget_samples>0
            assert digest_model(model)==after_callback
            model.zero_grad(set_to_none=True)
            for callback in callbacks:callback.run_callback_at_location(16,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)"""
    if anchor not in probe:raise ValueError('Frozen probe layout changed')
    probe=probe.replace(anchor,budget).replace('full_combined_AMP_backward=statuses,','full_combined_AMP_backward=statuses,actual_adapted_budget_batch=budget,')
    (code/'probe_batch1m_cold.py').write_text(probe)
    for p in [ROOT/'code_sfm/LookCloser/scripts',ROOT/'code_native/LookCloser/scripts',ROOT/'code_native',ROOT/'final_pair/code',code]:sys.path.insert(0,str(p))
    import yaml,torch
    base=yaml.load((ORIGINAL/'config.yml').read_text(),Loader=yaml.Loader);cfg=copy.deepcopy(base)
    cfg.pipeline.target_num_samples=1048576
    cfg.optimizers['fields']['scheduler'].max_steps=22000
    cfg.output_dir=STAGE/'run';cfg.experiment_name='marshawn_001660_png_batch1M_sched22k'
    cfg.pipeline.final_grid_artifact_dir=str(STAGE/'grids')
    assert cfg.max_num_iterations==22000 and cfg.load_checkpoint is None and cfg.load_dir is None and cfg.load_step is None and cfg.eval_checkpoint is None
    assert cfg.machine.seed==42 and cfg.pipeline.max_num_samples_per_ray==16384
    assert cfg.optimizers['fields']['optimizer'].lr==.01 and cfg.optimizers['fields']['scheduler'].lr_final==.0001
    delta=differences(normalize(base),normalize(cfg));assert {d['field']for d in delta}=={'pipeline.target_num_samples','optimizers.fields.scheduler.max_steps','output_dir','experiment_name','pipeline.final_grid_artifact_dir'}
    (STAGE/'config.yml').write_text(yaml.dump(cfg))
    write(STAGE/'configuration_diff.json',dict(baseline_config=str(ORIGINAL/'config.yml'),baseline_sha256=sha(ORIGINAL/'config.yml'),candidate_config_sha256=sha(STAGE/'config.yml'),differences=delta,
        interpretation='Combined optimization quality trial:4x target samples and decay horizon matched22k; NOTpurecausalbatchcomparison',NN_model_and_optimizer_initial_lr_unchanged=True,source_RGBA_data_AABB_and_geometry_losses_unchanged=True))
    # Actual official scheduler formula through a CPU dummy optimizer, zero steps.
    scheduler_rows={}
    for label,config in [('original',base),('candidate',cfg)]:
        dummy=torch.nn.Parameter(torch.ones(()));opt=torch.optim.Adam([dummy],lr=.01)
        sched=config.optimizers['fields']['scheduler'].setup().get_scheduler(opt,lr_init=.01)
        scheduler_rows[label]={str(step):float(.01*sched.lr_lambdas[0](step))for step in [0,2000,8000,22000]}
    write(STAGE/'scheduler_curve_cpu.json',dict(zero_optimizer_updates=True,dummy_CPU_parameter_only=True,official_scheduler_source_sha256=sha(ROOT/'code_native/nerfstudio/engine/schedulers.py'),curves=scheduler_rows))
    spec=json.loads((ORIGINAL/'run_spec.json').read_text());spec['config_sha256']=sha(STAGE/'config.yml');spec['baseline_config']=str(ORIGINAL/'config.yml');spec['baseline_config_sha256']=sha(ORIGINAL/'config.yml')
    for p in code.iterdir():
        if p.is_file():spec['sources_sha256'][str(p)]=sha(p)
    spec['sources_sha256'][str(ORIGINAL/'config.yml')]=sha(ORIGINAL/'config.yml')
    spec['release_checks']=dict(production_peer_review_passed=False,cold_probe_passed=False,root_read_required=True)
    spec['learning_change_only']='Combinedoptimizationtrial target_num_samples1048576 +schedulermax_steps22000; samefrozenNN/PNG/AABB/LOS/surface/distortion'
    spec['initial_actual_rays']=64;spec['target_num_samples']=1048576
    write(STAGE/'run_spec.json',spec)
    plan=json.loads((ORIGINAL/'launch_plan.json').read_text());cmd=plan['command'];cmd[0]='/home/pavlo/.conda/envs/nerfstudio/bin/python'
    cmd=[str(x).replace(str(ORIGINAL),str(STAGE)).replace('marshawn_001660_png_distortion_linear001',cfg.experiment_name)for x in cmd]
    plan['command']=cmd;plan['env']=dict(PYTHONPATH=':'.join(map(str,[ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code',code])),TORCH_EXTENSIONS_DIR=str(ROOT/'native_ngp4_corrected_sdr/local_torch_extensions_py311_cu128_sm120_ptx'),TORCH_CUDA_ARCH_LIST='12.0+PTX',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1',OPENCV_IO_ENABLE_OPENEXR='1')
    plan['expected_run_dir']=str(cfg.get_base_dir());plan['status']='STAGED_NOT_LAUNCHED_ROOT_REVIEW_REQUIRED';plan['no_Py38_compat']=True;plan['local_GPU']='RTX6000local98GB; combinedmemoryguard80GB';plan['quality_trial_not_pure_causal']=True
    plan['zero_optimizer_probe_command']=[cmd[0],str(code/'probe_batch1m_cold.py'),'--config',str(STAGE/'config.yml'),'--spec',str(STAGE/'run_spec.json'),'--output',str(STAGE/'cold_probe_local')]
    write(STAGE/'launch_plan.json',plan)
    print(json.dumps(dict(stage=str(STAGE),configuration_diff=delta,scheduler_curves=scheduler_rows,training_launched=False)))


if __name__=='__main__':main()
