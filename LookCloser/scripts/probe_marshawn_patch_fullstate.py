"""Exercise native Trainer.setup full resume and RNG restore, zero updates."""
from __future__ import annotations
import argparse,copy,json,sys,time
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
from run_marshawn_patch_dssim_phase import load,validate
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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False);torch.set_num_threads(2)
    cfg,spec=load(args.config,args.spec)
    original_rng=_capture_rng_state();saved=torch.load(cfg.load_checkpoint,map_location='cpu',weights_only=False)
    assert saved['step']==8000 and all(k in saved for k in ['optimizers','schedulers','scalers','rng_state'])
    # Only output location changes for this setup-only proof; no production run
    # folder is touched and the original production config bytes are unchanged.
    cfg=copy.deepcopy(cfg);cfg.output_dir=args.output/'setup_outputs'
    try:
        _set_random_seed(42)
        trainer=cfg.setup(local_rank=0,world_size=1);trainer.setup();validate(trainer,spec)
        assert trainer._start_step==8001 and cfg.max_num_iterations-trainer._start_step==12999
        assert_equal(trainer.pipeline.state_dict(),saved['pipeline'])
        for name,opt in trainer.optimizers.optimizers.items():assert_equal(opt.state_dict(),saved['optimizers'][name])
        for name,scheduler in trainer.optimizers.schedulers.items():assert_equal(scheduler.state_dict(),saved['schedulers'][name])
        assert_equal(trainer.grad_scaler.state_dict(),saved['scalers'])
        assert float(trainer.grad_scaler.get_scale())==131072.
        assert_equal(trainer._loaded_rng_state,saved['rng_state'])
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
            Adam_states_and_param_groups_exact=True,scheduler_states_exact=True,scaler_state_exact=True,
            parent_scale=131072.,parent_RNG_all_components_exact=True,last_native_RNG_restore_timing_exercised=True,
            callbacks_registered_not_run=callback_names,optimizer_objects_created_and_loaded=len(trainer.optimizers.optimizers),
            optimizer_updates=0,training_iterations_called=0,model_unchanged_after_RNG_restore=True,
            loaded_adam_update_min=min(counts),loaded_adam_update_max=max(counts),phase_start=8001,
            global_end=21000,phase_attempt_cap=12999,total_attempt_upper_bound=21609,
            initial_dynamic_request=128,actual_initial_patch_rays=121,
            trajectory_note=spec['trajectory_note'],probe_output_override_only=str(cfg.output_dir))
    finally:_restore_rng_state(original_rng)
    assert_equal(_capture_rng_state(),original_rng);report['probe_process_original_RNG_restored']=True
    (args.output/'fullstate_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,optimizer_updates=0,output=str(args.output))),flush=True)


if __name__=='__main__':main()
