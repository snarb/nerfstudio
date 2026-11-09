"""Exercise native Trainer.setup full resume and RNG restore, zero updates."""
from __future__ import annotations
import argparse,copy,json,sys,time,math,warnings
from types import SimpleNamespace
from pathlib import Path
ROOT=Path('/fsx/tmp/marshawn_001660_hd')
for directory in [ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',ROOT/'code_sfm/LookCloser/scripts',
                  ROOT/'final_pair/code',ROOT/'final_pair/release_tools_v3',Path(__file__).parent]:
    sys.path.insert(0,str(directory))
import numpy as np
import torch
from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
from nerfstudio.scripts.train import _set_random_seed
from probe_marshawn_occupancy import digest_model
from run_marshawn_patch_dssim_lowlr import load,validate,scale_loaded_lr,atomic_json
from staged_native_hash import file_sha


def assert_equal(actual,expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual.detach().cpu(),expected.detach().cpu());return
    if isinstance(expected,np.ndarray):assert np.array_equal(actual,expected);return
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for k in expected:assert_equal(actual[k],expected[k])
    elif isinstance(expected,(list,tuple)):
        assert type(actual)==type(expected)and len(actual)==len(expected)
        for a,b in zip(actual,expected):assert_equal(a,b)
    else:assert actual==expected


def cpu_lr_proof(cfg,saved):
    # Actual native LambdaLR, actual saved scheduler metadata; never optimizer.step.
    params=[torch.nn.Parameter(torch.zeros(1))for _ in range(4)]
    opt=torch.optim.Adam(params,lr=cfg.optimizers['fields']['optimizer'].lr)
    sched=cfg.optimizers['fields']['scheduler'].setup().get_scheduler(optimizer=opt,lr_init=cfg.optimizers['fields']['optimizer'].lr)
    sched.load_state_dict(saved['schedulers']['fields'])
    group=saved['optimizers']['fields']['param_groups'][0]
    opt.param_groups[0]['lr']=group['lr'];opt.param_groups[0]['initial_lr']=group['initial_lr']
    # Nonempty Adam witness state verifies the metadata helper never resets moments.
    for parameter in params:
        opt.state[parameter]=dict(step=torch.tensor(7.),exp_avg=torch.tensor([.125]),exp_avg_sq=torch.tensor([.25]))
    before=copy.deepcopy(opt.state_dict()['state']);epoch=sched.last_epoch;old_lambda=sched.lr_lambdas[0]
    target=.0005;audit=scale_loaded_lr(SimpleNamespace(optimizers={'fields':opt},schedulers={'fields':sched}),target)
    assert_equal(opt.state_dict()['state'],before)
    trace=[dict(successful_updates=0,lr=opt.param_groups[0]['lr'],scheduler_epoch=sched.last_epoch)]
    for j in range(1,65):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning);sched.step()
        expected=target*old_lambda(epoch+j)/old_lambda(epoch)
        assert math.isclose(opt.param_groups[0]['lr'],expected,rel_tol=1e-12)
        assert opt.param_groups[0]['lr']<=target
        trace.append(dict(successful_updates=j,lr=opt.param_groups[0]['lr'],scheduler_epoch=sched.last_epoch))
    assert_equal(opt.state_dict()['state'],before)
    return dict(passed=True,device='CPU',torch=torch.__version__,optimizer_updates=0,
        native_LambdaLR_used=True,parent_last_epoch=epoch,postload_lr_audit=audit,
        Adam_witness_moments_steps_unchanged=True,first64_successful_scheduler_updates=trace)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(exist_ok=False);torch.set_num_threads(2)
    cfg,spec=load(args.config,args.spec)
    original_rng=_capture_rng_state();saved=torch.load(cfg.load_checkpoint,map_location='cpu',weights_only=False)
    cpu_proof=cpu_lr_proof(cfg,saved)
    if args.cpu_only:
        atomic_json(args.output/'cpu_lr_proof.json',cpu_proof);print(json.dumps(cpu_proof));return
    assert saved['step']==8000 and all(k in saved for k in ['optimizers','schedulers','scalers','rng_state'])
    # Only output location changes for this setup-only proof; no production run
    # folder is touched and the original production config bytes are unchanged.
    cfg=copy.deepcopy(cfg);cfg.output_dir=args.output/'setup_outputs'
    try:
        _set_random_seed(42)
        trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();validate(trainer,spec)
        assert trainer._start_step==8001 and cfg.max_num_iterations-trainer._start_step==10999
        assert_equal(trainer.pipeline.state_dict(),saved['pipeline'])
        for name,opt in trainer.optimizers.optimizers.items():assert_equal(opt.state_dict(),saved['optimizers'][name])
        for name,scheduler in trainer.optimizers.schedulers.items():assert_equal(scheduler.state_dict(),saved['schedulers'][name])
        assert_equal(trainer.grad_scaler.state_dict(),saved['scalers'])
        assert float(trainer.grad_scaler.get_scale())==131072.
        assert_equal(trainer._loaded_rng_state,saved['rng_state'])
        before_policy_rng=_capture_rng_state()
        lr_audit=scale_loaded_lr(trainer.optimizers,spec['target_current_lr'])
        assert_equal(_capture_rng_state(),before_policy_rng)
        expected_opt=copy.copy(saved['optimizers']['fields'])
        expected_opt['param_groups']=copy.deepcopy(expected_opt['param_groups'])
        for group in expected_opt['param_groups']:
            group['lr']=spec['target_current_lr'];group['initial_lr']*=lr_audit['factor']
        assert_equal(trainer.optimizers.optimizers['fields'].state_dict(),expected_opt)
        expected_scheduler=copy.deepcopy(saved['schedulers']['fields'])
        expected_scheduler['base_lrs']=[v*lr_audit['factor']for v in expected_scheduler['base_lrs']]
        expected_scheduler['_last_lr']=[v*lr_audit['factor']for v in expected_scheduler['_last_lr']]
        assert_equal(trainer.optimizers.schedulers['fields'].state_dict(),expected_scheduler)
        assert_equal(trainer.grad_scaler.state_dict(),saved['scalers'])
        assert_equal(trainer._loaded_rng_state,saved['rng_state'])
        assert_equal(trainer.pipeline.state_dict(),saved['pipeline'])
        callback_names=[callback.func.__name__ for callback in trainer.callbacks]
        assert callback_names==['update_occupancy_grid','guard_after_native_core']
        before=digest_model(trainer.pipeline.model)
        # Exact native Trainer.train timing: setup/callbacks/writers complete,
        # restore just before the iteration loop; do not enter that loop here.
        _restore_rng_state(trainer._loaded_rng_state);trainer._loaded_rng_state=None
        assert_equal(_capture_rng_state(),saved['rng_state'])
        assert digest_model(trainer.pipeline.model)==before
        counts=[int(state['step'])for opt in trainer.optimizers.optimizers.values()for state in opt.state.values()if'step'in state]
        report=dict(passed=True,config_sha256=file_sha(args.config),spec_sha256=file_sha(args.spec),
            source_sha256=file_sha(__file__),parent_checkpoint_sha256=file_sha(cfg.load_checkpoint),
            runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),python=sys.version),
            actual_native_Trainer_setup=True,full_pipeline_values_and_keys_exact=True,
            Adam_moments_steps_exact=True,optimizer_groups_only_LR_metadata_scaled=True,scheduler_only_LR_metadata_scaled=True,scaler_state_exact=True,
            parent_scale=131072.,parent_RNG_all_components_exact=True,postload_lr_audit=lr_audit,CPU_LambdaLR_trace=cpu_proof,last_native_RNG_restore_timing_exercised=True,
            callbacks_registered_not_run=callback_names,optimizer_objects_created_and_loaded=len(trainer.optimizers.optimizers),
            optimizer_updates=0,training_iterations_called=0,model_unchanged_after_RNG_restore=True,
            loaded_adam_update_min=min(counts),loaded_adam_update_max=max(counts),phase_start=8001,
            global_end=19000,phase_attempt_cap=10999,total_attempt_upper_bound=21609,
            initial_dynamic_request=128,actual_initial_patch_rays=121,
            trajectory_note=spec['trajectory_note'],probe_output_override_only=str(cfg.output_dir))
    finally:_restore_rng_state(original_rng)
    assert_equal(_capture_rng_state(),original_rng);report['probe_process_original_RNG_restored']=True
    atomic_json(args.output/'fullstate_probe.json',report)
    print(json.dumps(dict(passed=True,optimizer_updates=0,output=str(args.output))),flush=True)


if __name__=='__main__':main()
