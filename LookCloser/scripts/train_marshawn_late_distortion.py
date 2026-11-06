#!/usr/bin/env python3
"""Explicit full-state fork for a reviewed 4000-attempt late-distortion pair.

Ordinary train_marshawn_tuning resume checks and all frozen scientific code stay
unchanged. This entrypoint owns separate checkpoints and explicit parent lineage.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def atomic(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(temp, path)


def tree_digest(value):
    """Device-independent tensor/scalar tree digest, including shape and dtype."""
    import torch
    import numpy as np
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str(('tensor', str(x.dtype), tuple(x.shape))).encode())
            h.update(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(('numpy', x.dtype.str, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for key in sorted(x, key=lambda v: (type(v).__name__, repr(v))): visit(key); visit(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        else: h.update((type(x).__name__+':'+repr(x)).encode())
    visit(value)
    return h.hexdigest()


def validate_spec(spec):
    expected = {'output', 'parent_checkpoint', 'parent_checkpoint_sha256', 'parent_spec', 'parent_spec_sha256',
                'parent_complete', 'parent_complete_sha256', 'mode', 'distortion_coefficient', 'release',
                'worker_sha256', 'frozen_source_hashes'}
    if set(spec) != expected: raise ValueError('Unexpected fork fields')
    if spec['mode'] not in ('zero', 'technical', 'diagnostic') or spec['distortion_coefficient'] not in (.01, 1.):
        raise ValueError('Only zero/technical or reviewed .01/1 diagnostic pair')
    if Path(spec['output']).resolve() in (Path(spec['parent_checkpoint']).resolve().parent, Path(spec['parent_spec']).resolve().parent):
        raise ValueError('Fork output must be separate from source')
    return spec


def adam_steps(optimizers):
    return {name: sorted({float(state['step']) for state in opt['state'].values() if 'step' in state})
            for name, opt in optimizers.items()}


def verify_state_equality(expected, actual):
    hashes = {}
    for key in ('pipeline', 'optimizers', 'schedulers', 'scalers', 'rng_state'):
        pair = [tree_digest(expected[key]), tree_digest(actual[key])]
        if pair[0] != pair[1]: raise ValueError('Restored state mismatch: '+key)
        hashes[key] = pair[0]
    if not all(values and min(values) > 0 for values in adam_steps(actual['optimizers']).values()):
        raise ValueError('Nonempty resumed Adam state required')
    return hashes


def verify_cache_ids(actual, seed, count=63):
    expected = random.Random(seed).sample(range(count), count)
    if list(actual) != expected: raise ValueError('Seed42 reconstructed complete-cache order differs')
    return hashlib.sha256(json.dumps(expected).encode()).hexdigest()


def run(spec_path, stop_after_attempts=None):
    spec_path = Path(spec_path).resolve(); spec = validate_spec(read(spec_path))
    release = read(spec['release'])
    authorized = 'diagnostic_authorized' if spec['mode'] == 'diagnostic' else 'technical_authorized'
    if release.get('reviewer') != 'root' or release.get(authorized) is not True or release.get('spec_sha256') != sha(spec_path):
        raise ValueError('Explicit source-bound fork release required')
    if sha(__file__) != spec['worker_sha256']: raise ValueError('Fork worker changed')
    for path, digest in spec['frozen_source_hashes'].items():
        if sha(path) != digest: raise ValueError('Frozen source changed: '+path)
    for key in ('parent_checkpoint', 'parent_spec', 'parent_complete'):
        if sha(spec[key]) != spec[key+'_sha256']: raise ValueError('Parent changed: '+key)
    import numpy as np
    import torch
    import train_marshawn_tuning as base
    # Match the original worker's model/pipeline import order before seed42.
    from nerfstudio.models.lookcloser import LookCloserModel
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    from marshawn_tuning_sampling import install_sampling
    from marshawn_tuning_occupancy import install_capture_hook
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from nerfstudio.utils import writer
    import inspect
    frozen_paths = {Path(p).resolve() for p in spec['frozen_source_hashes']}
    for obj in (base, LookCloserModel, LookCloserPipeline, install_sampling, install_capture_hook, _capture_rng_state):
        if Path(inspect.getfile(obj)).resolve() not in frozen_paths:
            raise ValueError('Imported runtime escaped the frozen fork snapshot')
    parent_spec = read(spec['parent_spec']); parent_complete = read(spec['parent_complete'])
    if parent_complete['selection']['step'] != 60752: raise ValueError('Selected60752 parent required')
    if parent_complete['config'] != parent_spec: raise ValueError('Parent completion and specification differ')
    if (parent_spec['objective'], parent_spec['sampling'], parent_spec['radius'], parent_spec['density_activation'], parent_spec['correct_sh_directions']) != ('full', 'body80', 0, 'trunc_exp', True):
        raise ValueError('Wrong selected structural parent recipe')
    if (parent_spec['width'], parent_spec['coarse_step'], parent_spec['distortion_final'], parent_spec['seed'], parent_spec['frequency_projection_mode']) != (64, .00625, .01, 42, 'legacy'):
        raise ValueError('Additional parent recipe factor differs')
    if stop_after_attempts is not None and (spec['mode'] != 'technical' or not 1 <= stop_after_attempts <= 32):
        raise ValueError('Partial invocation applies only to <=32 technical attempts')
    output = Path(spec['output']).resolve(); output.mkdir(parents=True, exist_ok=True)
    if output == Path(parent_spec['output']).resolve() or output in Path(spec['parent_checkpoint']).resolve().parents:
        raise ValueError('Source output cannot be reused')
    lock = (output/'worker.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = base.validate_data(Path(parent_spec['data']))
    checkpoint = Path(spec['parent_checkpoint']); latest = output/'latest.json'
    evaluations = []
    if latest.exists():
        receipt = read(latest); checkpoint = output/receipt['checkpoint']
        if sha(checkpoint) != receipt['sha256']: raise ValueError('Child checkpoint changed')
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if latest.exists():
        lineage = loaded['fork_runtime']
        if lineage['spec_sha256'] != sha(spec_path) or lineage['source_sha256'] != spec['parent_checkpoint_sha256']:
            raise ValueError('Strict child resume source/spec mismatch')
        evaluations = lineage['evaluations']
    elif loaded['step'] != 60752:
        raise ValueError('Wrong parent step')
    if loaded['tuning_runtime']['data_sha256'] != sha(Path(parent_spec['data'])/'ready.json'):
        raise ValueError('Source dataset changed')
    random.seed(parent_spec['seed']); np.random.seed(parent_spec['seed']); torch.manual_seed(parent_spec['seed']); torch.cuda.manual_seed_all(parent_spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True; torch.use_deterministic_algorithms(False)
    cfg = base.build_config(parent_spec, checkpoint)
    cfg.output_dir = output
    cfg.pipeline.model.distortion_loss_mult = spec['distortion_coefficient']
    if not cfg.load_optimizers or not cfg.load_scheduler: raise ValueError('Full-state optimizer/scheduler resume required')
    if cfg.checkpoint_load_mode != 'resume' or cfg.resume_fields_lr_override is not None or cfg.resume_reset_occupancy_grid or cfg.resume_reset_frequency_grid:
        raise ValueError('Reset or parameter-only fork is forbidden')
    cfg.save_config(); trainer = cfg.setup(local_rank=0, world_size=1); trainer.setup()
    pipe, model = trainer.pipeline, trainer.pipeline.model
    if len(pipe.datamanager.train_dataset) != 63 or len(pipe.datamanager.eval_dataset) != 3:
        raise ValueError('Original train/eval membership changed')
    if model.field.feature_reweighting_strength != 1. or pipe._feature_reweighting_switch_applied:
        raise ValueError('Selected60752 continuation must preserve pre-switch FR1')
    sampler = install_sampling(pipe, Path(parent_spec['data']), parent_spec['objective'], parent_spec['sampling'], parent_spec['width'])
    sampler.load_state_dict(loaded['tuning_runtime']['sampling_state'])
    install_capture_hook(model)
    capture = loaded.get('actual_occupancy_capture')
    if capture is not None: model._tuning_occupancy_capture = {k:v.to(model.device) if torch.is_tensor(v) else v for k,v in capture.items()}
    # Complete-cache ordering is rebuilt by the identical frozen seed/setup path.
    cache = pipe.datamanager.train_image_dataloader.cached_collated_batch
    cache_ids = cache['image_idx'].detach().cpu().tolist()
    cache_hash = verify_cache_ids(cache_ids, parent_spec['seed'])
    cache_sources = []
    for position, camera_index in enumerate(cache_ids):
        expected_row = pipe.datamanager.train_dataset[camera_index]
        if not torch.equal(cache['image'][position].cpu(), expected_row['image'].cpu()):
            raise ValueError('Cached camera RGB differs from original dataset association')
        if 'mask' in cache and not torch.equal(cache['mask'][position].cpu(), expected_row['mask'].cpu()):
            raise ValueError('Cached validity mask differs from original dataset association')
        cache_sources.append({'camera_index':camera_index,'rgb_sha256':tree_digest(cache['image'][position]),
                              'mask_sha256':tree_digest(cache['mask'][position]) if 'mask' in cache else None})
    pipe.datamanager.train_count = int(loaded['tuning_runtime']['sampling_state']['sample_count'])
    if trainer._loaded_rng_state is None: raise ValueError('Checkpoint RNG not restored')
    _restore_rng_state(trainer._loaded_rng_state); trainer._loaded_rng_state = None
    def actual_state():
        return {'pipeline': pipe.state_dict(), 'optimizers': {k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},
                'schedulers': {k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},
                'scalers': trainer.grad_scaler.state_dict(), 'rng_state': _capture_rng_state()}
    proof = verify_state_equality(loaded, actual_state())
    if sampler.state_dict() != loaded['tuning_runtime']['sampling_state']: raise ValueError('Private sampler state mismatch')
    if trainer._start_step != loaded['step']+1: raise ValueError('Absolute resume step changed')
    proof.update(cache_ids=cache_ids, cache_sha256=cache_hash, cache_sources=cache_sources,
                 historical_cache_order_observed=False,
                 cache_evidence='Seed42 current reconstruction and byte-exact dataset RGB/mask association; legacy historical cache IDs were not observed. Paired branches must match each other.',
                 sampler=copy.deepcopy(sampler.state_dict()), next_step=trainer._start_step, Adam=adam_steps(loaded['optimizers']), FR=model.field.feature_reweighting_strength, mode=spec['mode'])
    atomic(output/('resume_proof_'+str(loaded['step'])+'.json'), proof)
    if spec['mode'] == 'zero':
        atomic(output/'complete.json', {'status':'zero_update_state_equal','technical_smoke':True,'source_sha256':spec['parent_checkpoint_sha256'],'proof':proof})
        trainer.shutdown(); return
    target = 60752+(64 if spec['mode'] == 'technical' else 4000)
    if loaded['step'] >= target: raise ValueError('Completed child cannot silently rerun')
    runtime = copy.deepcopy(loaded['tuning_runtime']); del loaded
    stop = []; signal.signal(signal.SIGTERM, lambda signum, frame: stop.append(signum)); signal.signal(signal.SIGINT, lambda signum, frame: stop.append(signum))
    ckpts = output/'checkpoints'; ckpts.mkdir(exist_ok=True)
    started = time.monotonic(); last_report = 0.; attempts_this_call = 0
    ray_hash = hashlib.sha256(); first_ray = [None]; original_next = pipe.datamanager.next_train
    def observed_next(step):
        rays, batch = original_next(step); raw = batch['indices'].detach().cpu().contiguous().numpy().tobytes()
        if first_ray[0] is None: first_ray[0] = hashlib.sha256(raw).hexdigest()
        ray_hash.update(raw); return rays, batch
    pipe.datamanager.next_train = observed_next
    def sampling_segment(step):
        return {'first_step':trainer._start_step,'last_step':step,'attempts':attempts_this_call,'first_preupdate_ray_indices_sha256':first_ray[0],'ray_indices_sha256':ray_hash.hexdigest()}
    def save(step):
        runtime['sampling_state'] = sampler.state_dict()
        payload = dict(step=step, **actual_state(), tuning_runtime=copy.deepcopy(runtime), actual_occupancy_capture=getattr(model, '_tuning_occupancy_capture', None),
                       fork_runtime={'spec_sha256':sha(spec_path),'source_sha256':spec['parent_checkpoint_sha256'],'evaluations':evaluations,'worker_sha256':sha(__file__),'sampling_segment':sampling_segment(step)})
        path = ckpts/f'step-{step:09d}.ckpt'; temp = path.with_suffix('.tmp'); torch.save(payload,temp); os.replace(temp,path)
        atomic(latest, {'step':step,'checkpoint':str(path.relative_to(output)),'sha256':sha(path)})
    for step in range(trainer._start_step, target+1):
        trainer.step = step; pipe.train(); sampler.set_step(step)
        # Deliberately abrupt late intervention; no other schedule is changed.
        model.config.distortion_loss_mult = spec['distortion_coefficient']
        trainer._apply_live_optimizer_switch(step)
        for callback in trainer.callbacks: callback.run_callback_at_location(step, location=Location.BEFORE_TRAIN_ITERATION)
        loss, _, _ = trainer.train_iteration(step)
        if not torch.isfinite(loss.detach()): raise FloatingPointError('Nonfinite continuation objective')
        for callback in trainer.callbacks: callback.run_callback_at_location(step, location=Location.AFTER_TRAIN_ITERATION)
        writer.write_out_storage(); attempts_this_call += 1
        gate = spec['mode'] == 'diagnostic' and step-60752 in (500,1000,2000,4000)
        if gate:
            evaluations.append(base.evaluate(trainer, ready, Path(parent_spec['data']), output, step)); atomic(output/'metrics.json',evaluations)
        partial = stop_after_attempts is not None and attempts_this_call >= stop_after_attempts
        terminal = step == target
        if gate or partial or terminal or stop: save(step)
        if gate or partial or terminal or time.monotonic()-last_report >= 30:
            state = actual_state()
            atomic(output/'progress.json', {'step':step,'attempts':step-60752,'elapsed_seconds':time.monotonic()-started,'actual_adam_steps':adam_steps(state['optimizers']), 'ray_indices_this_invocation_sha256':ray_hash.hexdigest(),'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated(),'mode':spec['mode']})
            last_report = time.monotonic()
        if terminal:
            states = actual_state(); counts = adam_steps(states['optimizers'])
            if not all(min(v)>min(proof['Adam'][k]) for k,v in counts.items()): raise ValueError('No actual resumed Adam update')
            atomic(output/'complete.json', {'status':'diagnostic_visual_gate_pending' if spec['mode']=='diagnostic' else 'technical_resume_complete','technical_smoke':spec['mode']!='diagnostic','source_sha256':spec['parent_checkpoint_sha256'],'final_step':step,'attempts':step-60752,'actual_adam_steps':counts,'evaluations':evaluations,'sampling_segment':sampling_segment(step),'historical_cache_order_observed':False,'visual_review_complete':False,'geometry_approved':False,'quality_optimization_complete':False})
        if terminal or partial or stop: trainer.shutdown(); return


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--spec',type=Path,required=True); p.add_argument('--stop-after-attempts',type=int)
    a = p.parse_args(); run(a.spec,a.stop_after_attempts)
