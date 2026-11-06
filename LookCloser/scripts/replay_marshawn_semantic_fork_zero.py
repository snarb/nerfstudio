#!/usr/bin/env python3
"""Read-only semantic30376 original-factory versus fork-factory replay diagnostic.

Ordinary train_marshawn_semantic_admission resume checks and all frozen scientific code stay
unchanged. This entrypoint has no training iteration or optimizer update path.
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
                'anchor_audit', 'anchor_audit_sha256', 'mode', 'distortion_coefficient', 'release',
                'worker_sha256', 'frozen_source_hashes', 'source_render_sha256','construction'}
    if set(spec) != expected: raise ValueError('Unexpected fork fields')
    if spec['mode'] != 'zero' or spec['distortion_coefficient'] != .01 or spec['construction'] not in ('original','fork'):
        raise ValueError('Only zero/technical or reviewed .01/1 diagnostic pair')
    if Path(spec['output']).resolve() in (Path(spec['parent_checkpoint']).resolve().parent, Path(spec['parent_spec']).resolve().parent):
        raise ValueError('Fork output must be separate from source')
    if set(spec['source_render_sha256']) != {c+s for c in ('DC','HC','KA') for s in ('.png','_alpha.png')}:
        raise ValueError('Exact six source render hashes required')
    return spec


def adam_steps(optimizers):
    return {name: sorted({float(state['step']) for state in opt['state'].values() if 'step' in state})
            for name, opt in optimizers.items()}


def verify_state_equality(expected, actual):
    hashes = {}
    for key in ('pipeline', 'optimizers', 'schedulers', 'scalers', 'rng_state', 'tuning_runtime', 'actual_occupancy_capture'):
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


def verify_semantic_model(model, loaded, anchor):
    import numpy as np
    from marshawn_semantic_occupancy import validate_policy
    if model.__class__.__name__ != 'SemanticAdmissionModel': raise ValueError('Semantic model subclass required')
    capture = model._tuning_occupancy_capture
    npy = lambda v: v.detach().cpu().numpy()
    admission = npy(model.semantic_admission)[None]
    core, active = npy(capture['core']), npy(model.occupancy_grid.binaries)
    validate_policy(core, active, capture['radius'], capture['binary_warmup'], admission)
    if not np.array_equal(core,npy(model.semantic_learned_core)) or not np.array_equal(active,npy(capture['active'])):
        raise ValueError('Restored semantic learned core/actual capture differs')
    receipt = anchor['semantic_capture_metadata']['semantic_receipt']
    if model.semantic_receipt != receipt: raise ValueError('Sealed semantic support receipt differs')
    mapped_sha = hashlib.sha256(npy(model.semantic_admission).tobytes()).hexdigest()
    if mapped_sha != receipt['mapped_bool_sha256']: raise ValueError('Nonpersistent semantic admission reload differs')
    if not hasattr(model,'semantic_lookup'): raise ValueError('Shared semantic density selector missing')
    return dict(receipt=receipt,mapped_sha256=mapped_sha,schema=2,actual_core_and_active_equal=True,
                learned_core_preserved=True,semantic_selector_installed=hasattr(model,'semantic_lookup'))


def pixel_delta(expected,actual):
    import numpy as np
    if expected.shape!=actual.shape or expected.dtype!=actual.dtype:raise ValueError('Raster shape/dtype mismatch')
    diff=np.abs(expected.astype(np.float64)-actual.astype(np.float64))
    return dict(shape=list(expected.shape),dtype=str(expected.dtype),max_code_delta=float(diff.max()),mean_code_delta=float(diff.mean()),changed_values=int(np.count_nonzero(diff)),values=int(diff.size),changed_fraction=float(np.count_nonzero(diff)/diff.size))


def run(spec_path):
    spec_path = Path(spec_path).resolve(); spec = validate_spec(read(spec_path))
    release = read(spec['release'])
    authorized = 'read_only_authorized'
    if release.get('reviewer') != 'root' or release.get(authorized) is not True or release.get('spec_sha256') != sha(spec_path):
        raise ValueError('Explicit source-bound fork release required')
    if sha(__file__) != spec['worker_sha256']: raise ValueError('Fork worker changed')
    for path, digest in spec['frozen_source_hashes'].items():
        if sha(path) != digest: raise ValueError('Frozen source changed: '+path)
    for key in ('parent_checkpoint', 'parent_spec', 'anchor_audit'):
        if sha(spec[key]) != spec[key+'_sha256']: raise ValueError('Parent changed: '+key)
    import numpy as np
    import torch
    import train_marshawn_semantic_admission as base
    from marshawn_semantic_late_factory import build_config, validate_anchor
    from marshawn_semantic_admission import SemanticAdmissionModel
    # Match the original worker's model/pipeline import order before seed42.
    from nerfstudio.models.lookcloser import LookCloserModel
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    from marshawn_tuning_sampling import install_sampling
    from marshawn_semantic_occupancy import install_capture_hook, export_snapshot, load_snapshot
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from nerfstudio.utils import writer
    import inspect
    frozen_paths = {Path(p).resolve() for p in spec['frozen_source_hashes']}
    for obj in (base, LookCloserModel, SemanticAdmissionModel, LookCloserPipeline, install_sampling, install_capture_hook, build_config, _capture_rng_state):
        if Path(inspect.getfile(obj)).resolve() not in frozen_paths:
            raise ValueError('Imported runtime escaped the frozen fork snapshot')
    parent_spec = read(spec['parent_spec']); anchor = read(spec['anchor_audit'])
    validate_anchor(parent_spec, anchor, spec['distortion_coefficient'])
    for path, digest in anchor['source_hashes'].items():
        if sha(path) != digest: raise ValueError('Anchor evidence changed: '+path)
    if anchor['source_hashes'].get(spec['parent_checkpoint']) != spec['parent_checkpoint_sha256']:
        raise ValueError('Checkpoint is not the bound evaluated intermediate')
    output = Path(spec['output']).resolve(); output.mkdir(parents=True, exist_ok=True)
    if output == Path(parent_spec['output']).resolve() or output in Path(spec['parent_checkpoint']).resolve().parents:
        raise ValueError('Source output cannot be reused')
    lock = (output/'worker.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready = base.validate_data(Path(parent_spec['data']))
    checkpoint = Path(spec['parent_checkpoint']); latest = output/'latest.json'
    evaluations = []
    if latest.exists():raise ValueError('Read-only replay cannot resume or update')
    if latest.exists():
        receipt = read(latest); checkpoint = output/receipt['checkpoint']
        if sha(checkpoint) != receipt['sha256']: raise ValueError('Child checkpoint changed')
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if latest.exists():
        lineage = loaded['fork_runtime']
        if lineage['spec_sha256'] != sha(spec_path) or lineage['source_sha256'] != spec['parent_checkpoint_sha256']:
            raise ValueError('Strict child resume source/spec mismatch')
        evaluations = lineage['evaluations']
    elif loaded['step'] != 30376:
        raise ValueError('Wrong parent step')
    if loaded['tuning_runtime']['data_sha256'] != sha(Path(parent_spec['data'])/'ready.json'):
        raise ValueError('Source dataset changed')
    if not latest.exists():
        for key, digest in anchor['fullstate_hashes'].items():
            if tree_digest(loaded[key]) != digest: raise ValueError('Anchor serialized state differs: '+key)
    for name, digest in loaded['tuning_runtime']['input_sha256'].items():
        path = (Path(parent_spec['semantic_admission']['manifest']).parent/name[9:]
                if name.startswith('semantic/') else Path(parent_spec['semantic_admission']['manifest'])
                if name == 'semantic_manifest' else Path(parent_spec['data'])/name)
        if sha(path) != digest: raise ValueError('Parent input changed: '+name)
    for name,digest in loaded['tuning_runtime']['code_sha256'].items():
        matches = [Path(p) for p in spec['frozen_source_hashes'] if Path(p).name == name]
        if len(matches) != 1 or sha(matches[0]) != digest: raise ValueError('Parent runtime code differs: '+name)
    if spec['mode'] != 'diagnostic':
        torch.cuda.set_per_process_memory_fraction((10*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
    random.seed(parent_spec['seed']); np.random.seed(parent_spec['seed']); torch.manual_seed(parent_spec['seed']); torch.cuda.manual_seed_all(parent_spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True; torch.use_deterministic_algorithms(False)
    base.validate_spec(parent_spec)
    if spec['construction']=='original':
        cfg=base.build_config(parent_spec,checkpoint);cfg.output_dir=output
    else:
        cfg=build_config(parent_spec,anchor,checkpoint,output,.01)
    cfg.save_config(); trainer = cfg.setup(local_rank=0, world_size=1); trainer.setup()
    pipe, model = trainer.pipeline, trainer.pipeline.model
    if len(pipe.datamanager.train_dataset) != 63 or len(pipe.datamanager.eval_dataset) != 3:
        raise ValueError('Original train/eval membership changed')
    if model.field.feature_reweighting_strength != 1. or pipe._feature_reweighting_switch_applied:
        raise ValueError('Semantic30376 continuation must preserve pre-switch FR1')
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
    runtime = copy.deepcopy(loaded['tuning_runtime'])
    def actual_state():
        return {'pipeline': pipe.state_dict(), 'optimizers': {k:v.state_dict() for k,v in trainer.optimizers.optimizers.items()},
                'schedulers': {k:v.state_dict() for k,v in trainer.optimizers.schedulers.items()},
                'scalers': trainer.grad_scaler.state_dict(), 'rng_state': _capture_rng_state(),
                'tuning_runtime': copy.deepcopy(runtime), 'actual_occupancy_capture': getattr(model, '_tuning_occupancy_capture', None)}
    proof = verify_state_equality(loaded, actual_state())
    proof['semantic_support'] = verify_semantic_model(model, loaded, anchor)
    if sampler.state_dict() != loaded['tuning_runtime']['sampling_state']: raise ValueError('Private sampler state mismatch')
    if trainer._start_step != loaded['step']+1: raise ValueError('Absolute resume step changed')
    proof.update(cache_ids=cache_ids, cache_sha256=cache_hash, cache_sources=cache_sources,
                 historical_cache_order_observed=False,
                 cache_evidence='Seed42 current reconstruction and byte-exact dataset RGB/mask association; legacy historical cache IDs were not observed. Paired branches must match each other.',
                 sampler=copy.deepcopy(sampler.state_dict()), next_step=trainer._start_step, Adam=adam_steps(loaded['optimizers']), FR=model.field.feature_reweighting_strength, mode=spec['mode'])
    atomic(output/('resume_proof_'+str(loaded['step'])+'.json'), proof)
    if spec['mode'] == 'zero':
        runtime_attributes=dict(model_current_train_step=model.current_train_step,occupancy_warmup_start_step=model._occupancy_warmup_start_step,feature_reweighting_strength=model.field.feature_reweighting_strength,FR_switch_applied=pipe._feature_reweighting_switch_applied,eval_dilation=model.config.occupancy_eval_dilation_radius,model_training=model.training,shared_selector_installed=bool(getattr(model.field,'_semantic_admission_installed',False)),tcnn_geometry_jit=model.field.get_tcnn_network_jit(scope='geometry'),tcnn_color_jit=model.field.get_tcnn_network_jit(scope='color'))
        zero_metrics = base.evaluate(trainer, ready, Path(parent_spec['data']), output, loaded['step'])
        verify_state_equality(loaded, actual_state())
        source_folder = Path(spec['parent_checkpoint']).parent/'renders'
        render_hashes = {};pixel_deltas={}
        from PIL import Image
        for camera in ('DC','HC','KA'):
            for suffix in ('.png','_alpha.png'):
                name = camera+suffix
                expected, actual = sha(source_folder/name), sha(output/'renders'/f"step-{loaded['step']:09d}"/name)
                if expected != spec['source_render_sha256'][name]:raise ValueError('Original source render changed')
                pixel_deltas[name]=pixel_delta(np.asarray(Image.open(source_folder/name)),np.asarray(Image.open(output/'renders'/f"step-{loaded['step']:09d}"/name)))
                render_hashes[name] = actual
        export_snapshot(model, checkpoint, loaded['step'], output/'grid', sources=anchor['source_hashes'])
        atomic(output/'complete.json', {'status':'readonly_replay_complete','technical_smoke':True,'source_sha256':spec['parent_checkpoint_sha256'],'proof':proof,'zero_render_sha256':render_hashes,'zero_metrics':zero_metrics,'render_state_unchanged':True,'construction':spec['construction'],'runtime_attributes':runtime_attributes,'pixel_deltas_to_original':pixel_deltas,'optimizer_updates':0,'numeric_tolerance_approved':False,'cause_established':False})
        trainer.shutdown(); return

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
