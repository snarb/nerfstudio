"""Native read-only64-step field replay; no optimizer or private-point queries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import types

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def run(path,execute):
    s=json.loads(path.read_text())
    if not execute:
        print(json.dumps(dict(readonly=True,cameras=['DC','HC','KA'],optimizer_created=False,private_queries=0,CUDA_started=False)));return
    release=json.loads(Path(s['root_release']).read_text())
    if release.get('reviewer')!='root'or release.get('readonly_replay_authorized')is not True or release['spec_sha256'].get(str(path))!=sha(path):raise ValueError('Exact root read-only replay release required')
    for p,h in release['bound_files'].items():
        if sha(p)!=h:raise ValueError('Bound replay input changed:'+p)
    if sha(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Selected trained checkpoint changed')
    sys.path[:0]=[s['source_code'],s['source_scripts']]
    from marshawn_gpu_isolation import require_controller_lease,verify_cuda
    lease=require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import torch
    import numpy as np
    from train_marshawn_semantic_late_distortion import tree_digest
    import train_marshawn_semantic_admission as base
    import nerfstudio.utils.eval_utils as eu
    from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
    verify_cuda(lease['expected_physical_uuid'],torch);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Read-only replay deadline')));signal.alarm(900)
    cp=Path(s['checkpoint']);payload=torch.load(cp,map_location='cpu',weights_only=False)
    if payload['step']!=126380:raise ValueError('Exact64 trained checkpoint required')
    expected=tree_digest(payload['pipeline'])
    def configure(cfg):cfg.eval_checkpoint=cp;return cfg
    cfg,pipe,actual,step=eu.eval_setup(Path(s['config']),test_mode='inference',update_config_callback=configure)
    m=pipe.model;pipe.eval();pipe.requires_grad_(False);m.current_train_step=step
    if Path(actual).resolve()!=cp.resolve()or step!=126380 or tree_digest(pipe.state_dict())!=expected:raise ValueError('Strict trained pipeline restore failed')
    if m.field.feature_reweighting_strength!=.3 or m.config.feature_reweighting_strength!=.3 or m.config.max_steps_per_ray!=1024 or m.config.corrected_arm_allocator:raise ValueError('Original FR.3/legacy1024 required')
    runtime=payload['below_floor_density_fork_runtime'];cache=runtime['model_cache']
    m.field._feature_weight_lut=cache['feature_weight_LUT'].to(m.device);m.field._feature_weight_lut_strength=cache['strength']
    pipe._feature_reweighting_switch_applied=cache['switch_applied']
    _restore_rng_state(payload['rng_state'])
    before=dict(pipeline=expected,RNG=tree_digest(_capture_rng_state()),LUT=tree_digest((m.field._feature_weight_lut,m.field._feature_weight_lut_strength)),private=tree_digest(runtime['below_floor_prior_state']))
    normalizer=m.field._normalize_positions
    def unchanged():
        now=dict(pipeline=tree_digest(pipe.state_dict()),RNG=tree_digest(_capture_rng_state()),LUT=tree_digest((m.field._feature_weight_lut,m.field._feature_weight_lut_strength)),private=tree_digest(runtime['below_floor_prior_state']))
        if now!=before or m.field._normalize_positions is not normalizer:raise ValueError('Read-only replay state changed')
        return dict(pipeline=True,RNG=True,LUT=True,private_cache=True,normalizer=True)
    ready=base.validate_data(Path(s['data']));dataset=pipe.datamanager.eval_dataset
    lookup={Path(r['image']).name:r['camera']for r in ready['records']}
    cameras=[lookup[Path(p).name]for p in dataset.image_filenames]
    if set(cameras)!={'DC','HC','KA'}or len(cameras)!=3:raise ValueError('Exact three heldout views required')
    # Capture the very same qualified native evaluation call, without a second
    # render or any field/sampler modification.
    native=m.get_outputs_for_camera;had='get_outputs_for_camera'in m.__dict__;binding=m.__dict__.get('get_outputs_for_camera');index=[0]
    def observed(*args,**kwargs):
        result=native(*args,**kwargs);camera=cameras[index[0]];index[0]+=1
        np.savez_compressed(out/(camera+'_float.npz'),rgb=result['rgb'].float().cpu().numpy(),alpha=result['accumulation'].float().cpu().numpy())
        return result
    m.get_outputs_for_camera=observed
    try:metrics=base.evaluate(types.SimpleNamespace(pipeline=pipe),ready,Path(s['data']),out,step)
    finally:
        if had:m.get_outputs_for_camera=binding
        else:delattr(m,'get_outputs_for_camera')
        pipe.eval()
    checks=unchanged()
    if sha(cp)!=s['checkpoint_sha256']:raise ValueError('Disk checkpoint changed')
    optimizer_metadata={name:[{k:g.get(k)for k in ['lr','eps','betas']}for g in opt['param_groups']]for name,opt in payload['optimizers'].items()}
    write(out/'complete.json',dict(passed=True,checkpoint_sha256=sha(cp),pipeline_digest=expected,state_checks=checks,
        metrics=metrics['metrics'],by_view=metrics['by_view'],Adam_metadata=optimizer_metadata,
        optimizer_created=False,optimizer_updates=0,private_queries=0,source_step=step,
        float_files={camera+'_float.npz':sha(out/(camera+'_float.npz'))for camera in cameras},spec_sha256=sha(path)))
    signal.alarm(0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();run(a.spec,a.execute)
