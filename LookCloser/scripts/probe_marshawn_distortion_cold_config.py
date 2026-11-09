"""Exact production cold config: full AMP backward/callback/schema, zero updates."""
from __future__ import annotations
import argparse,copy,json,random,sys,time
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',
          ROOT/'final_pair/code',ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:sys.path.insert(0,str(d))
import numpy as np
import torch
import yaml
from nerfstudio.engine.callbacks import TrainingCallbackAttributes,TrainingCallbackLocation
from nerfstudio.scripts.train import _set_random_seed
from probe_marshawn_occupancy import digest_model
from run_marshawn_png_distortion_control import load,validate
from staged_native_hash import file_sha


def schema(model):
    return [(n,list(p.shape),str(p.dtype),p.requires_grad)for n,p in model.named_parameters()]


def gradients(model,term,scale):
    model.zero_grad(set_to_none=True)
    (term*scale).backward(retain_graph=True)
    values=[p.grad for p in model.parameters()if p.grad is not None]
    finite=all(bool(torch.isfinite(v).all())for v in values)
    norm=sum(float((v.detach().double()/scale).square().sum())for v in values)**.5 if finite else None
    return dict(finite=finite,unscaled_l2=norm)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    cfg,spec=load(a.config,a.spec)
    dumped=yaml.dump(cfg);again=yaml.load(dumped,Loader=yaml.Loader)
    assert again.pipeline._target is cfg.pipeline._target and '<locals>'not in dumped
    (a.output/'roundtrip_dev3.yml').write_text(dumped)
    base=yaml.load(Path(spec['baseline_config']).read_text(),Loader=yaml.Loader)
    _set_random_seed(42);reference=base.pipeline.setup(device='cuda',test_mode='val')
    baseline_hash=digest_model(reference.model);baseline_schema=schema(reference.model);baseline_keys=set(reference.state_dict());baseline_groups={k:sum(p.numel()for p in v)for k,v in reference.model.get_param_groups().items()}
    del reference;torch.cuda.empty_cache()
    _set_random_seed(42);pipe=again.pipeline.setup(device='cuda',test_mode='val');pipe.train();model=pipe.model
    bounds=validate(pipe);initial=digest_model(model)
    assert initial==baseline_hash and schema(model)==baseline_schema and set(pipe.state_dict())==baseline_keys
    assert {k:sum(p.numel()for p in v)for k,v in model.get_param_groups().items()}==baseline_groups
    assert pipe.dynamic_num_rays_per_batch==16 and pipe.datamanager.train_pixel_sampler.num_rays_per_batch==16
    callbacks=pipe.get_training_callbacks(TrainingCallbackAttributes(None,None,pipe,None))
    names=[c.func.__name__ for c in callbacks]
    assert names==['update_occupancy_grid','guard_after_native_core']
    grid={k:v.clone()for k,v in model.occupancy_grid.state_dict().items()}
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    statuses=[];hooks=(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks))
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            _set_random_seed(42)
            for callback in callbacks:callback.run_callback_at_location(0,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
            assert pipe._last_core_step==0 and torch.equal(model.occupancy_grid.binaries,pipe._actual_core)
            after_callback=digest_model(model)
            with torch.autocast(device_type='cuda',enabled=cfg.mixed_precision):
                outputs,losses,metrics=pipe.get_train_loss_dict(0)
            assert set(losses)=={'rgb_loss','observed_sfm_free_space','observed_sfm_surface_band','packed_distortion'}
            assert hooks==(len(model._forward_pre_hooks),len(model.renderer_depth._forward_pre_hooks))
            samples=int(metrics['num_samples_per_batch']);expected=int(16*(cfg.pipeline.target_num_samples/samples))
            assert pipe.dynamic_num_rays_per_batch==expected
            baseline=sum(v for k,v in losses.items()if k!='packed_distortion');combined=sum(losses.values())
            scale=cfg.grad_scaler_init_scale
            for attempt in range(12):
                plain=gradients(model,baseline,scale);added=gradients(model,combined,scale)
                statuses.append(dict(scale=scale,baseline=plain,with_distortion=added))
                if plain['finite']and added['finite']:break
                if plain['finite']and not added['finite']:raise ValueError('Distortion introduced an AMP overflow absent from matched baseline')
                scale*=.5  # Probe-only emulation of native scaler backoff, never a config edit.
            else:raise ValueError('No finite full cold AMP backward at native backoff scales')
            assert digest_model(model)==after_callback
            model.zero_grad(set_to_none=True)
            for callback in callbacks:callback.run_callback_at_location(16,TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
            assert pipe._last_core_step==16 and torch.equal(model.occupancy_grid.binaries,pipe._actual_core)
    finally:
        random.setstate(py);np.random.set_state(npstate)
        model.occupancy_grid.load_state_dict(grid)
        for n in ['_actual_core','_actual_active','_actual_threshold','_last_core_step']:object.__setattr__(pipe,n,None)
    assert digest_model(model)==initial
    assert random.getstate()==py
    state=np.random.get_state();assert state[0]==npstate[0]and np.array_equal(state[1],npstate[1])and state[2:]==npstate[2:]
    assert torch.equal(cpu,torch.get_rng_state())and all(torch.equal(x,y)for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    r=dict(passed=True,time=time.time(),config_sha256=file_sha(a.config),spec_sha256=file_sha(a.spec),source_sha256=file_sha(__file__),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),python=sys.version),
        cold=True,no_checkpoint_loaded=True,seed=42,optimizer_objects_created=0,optimizer_updates=0,
        initial_candidate_equals_matched_cold_baseline=True,baseline_initial_state_sha256=baseline_hash,
        exact_schema_param_groups_and_state_keys=True,actual_YAML_target_roundtrip=True,loss_key_present=True,
        callback_order=names,callback_steps_checked=[0,16],radius0_core_exact=True,
        initial_actual_rays=16,packed_samples=samples,next_native_dynamic_rays=expected,
        full_combined_AMP_backward=statuses,production_scaler_unchanged=cfg.grad_scaler_init_scale,
        hooks_removed=True,model_and_original_grid_restored=True,all_Python_NumPy_Torch_RNG_restored=True,
        fresh_training_must_reset_seed_in_separate_process=True,bounds=bounds)
    (a.output/'cold_probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)


if __name__=='__main__':main()
