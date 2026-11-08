"""Isolated native actor crop tuning worker with external review stages.

Run --spec SPEC.json; --smoke-steps N is an explicitly labelled technical run.
No training happens on import. The production loop has a140000-attempt hard ceiling and
preserves the native train_iteration and before/after callbacks.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import sys
import time

# Bootstrap exact sealed source before importing any nerfstudio/helper modules.
_DEFAULT_SOURCE = '/fsx/tmp/marshawn_video_1660_2015_24fps_140k_20261107/code'
_bootstrap_spec = {}
if '--spec' in sys.argv:
    _position = sys.argv.index('--spec') + 1
    if _position < len(sys.argv):
        _bootstrap_spec = json.loads(Path(sys.argv[_position]).read_text())
SOURCE_CODE = Path(_bootstrap_spec.get('source_code', os.environ.get('LOOKCLOSER_SEALED_CODE', _DEFAULT_SOURCE))).resolve()
PUBLIC_SCRIPT_DIR = Path(__file__).resolve().parent
# A public scripts directory also contains legacy NN helpers. The declared release
# owns those helpers; only the two crop sampling overlays come from this directory.
sys.path[:0] = [str(SOURCE_CODE), str(SOURCE_CODE/'LookCloser/scripts'), str(PUBLIC_SCRIPT_DIR)]


def _source_import(name):
    module = importlib.import_module(name)
    path = getattr(module, '__file__', None)
    if path is None or not Path(path).resolve().is_relative_to(SOURCE_CODE):
        raise ImportError('Source helper must come from declared sealed source: '+name+' '+str(path))
    # Catch cached transitive imports from another checkout as well.
    for module_name, loaded in tuple(sys.modules.items()):
        if module_name.startswith(('marshawn_', 'nerfstudio.')) or module_name == 'prepare_marshawn_tuning':
            loaded_path = getattr(loaded, '__file__', None)
            if loaded_path is not None and not Path(loaded_path).resolve().is_relative_to(SOURCE_CODE):
                raise ImportError('Mixed source helper checkout: '+module_name+' '+str(loaded_path))
    return module


def _public_sampling_import(name):
    path = PUBLIC_SCRIPT_DIR/(name+'.py')
    loaded = sys.modules.get(name)
    if loaded is not None:
        if Path(getattr(loaded, '__file__', '')).resolve() != path:
            raise ImportError('Crop sampling helper was loaded from another directory: '+name)
        return loaded
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_source_import('marshawn_tuning_sampling')
_public_sampling_import('native_crop_sampling')
_public_sampling_import('native_crop_sampling_v2')
EVAL_ITERATIONS = (8000, 16000, 32000, 50000, 80000, 140000)
HISTORICAL_EVAL_STEPS = tuple(x-1 for x in EVAL_ITERATIONS)
from types import SimpleNamespace

_source_import('marshawn_iteration_budget')
from marshawn_iteration_budget import validate_limit, check_step, completed
_source_import('marshawn_tuning_occupancy')
from marshawn_tuning_occupancy import atomic_json, sha

MODULE = 'train_native_crop'
if __name__ == '__main__':
    sys.modules[MODULE] = sys.modules[__name__]


def __getattr__(name):
    if name != 'TuningParser':
        raise AttributeError(name)
    import torch
    _source_import('nerfstudio.data.dataparsers.nerfstudio_dataparser')
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
    _source_import('nerfstudio.data.scene_box')
    from nerfstudio.data.scene_box import SceneBox

    class TuningParser(Nerfstudio):
        def _generate_dataparser_outputs(self, split='train'):
            c = self.config
            if c.auto_scale_poses or c.center_method != 'none' or c.orientation_method != 'none' or c.scale_factor != 1:
                raise ValueError('Prepared floor-frame poses require identity parser normalization')
            outputs = super()._generate_dataparser_outputs(split)
            box = json.loads((Path(c.data) / 'box.json').read_text())['aabb']
            outputs.scene_box = SceneBox(aabb=torch.tensor(box, dtype=torch.float32))
            return outputs
    TuningParser.__module__ = MODULE
    TuningParser.__qualname__ = name
    globals()[name] = TuningParser
    return TuningParser


DEFAULTS = dict(objective='actor', sampling='body80', width=64, radius=0,
    coarse_step=.0015625, distortion_final=.1, seed=42, frequency_threshold=.95,
    correct_sh_directions=True, density_activation='trunc_exp', frequency_projection_mode='legacy',
    lr_decay_steps=200000, fr_switch_step=75941, fr_after_switch=.3,
    corrected_arm_allocator=False, min_step_size=1e-4, max_steps_per_ray=1024,
    target_num_samples_per_batch=0, hard_max_iterations=140000, train_count=66, train_rays_per_batch=4096, exposure_scale=1, adam_beta2=.999, initial_lr=.01, final_lr=.0001, extra_eval_interval=0)


def read_stage_plan(spec):
    plan = json.loads(Path(spec['stage_plan']).read_text())
    target = plan.get('stop_after_iterations')
    if plan.get('schema') != 1 or type(target) is not int or target not in (32000, 50000, 80000, 140000):
        raise ValueError('Stage plan needs schema1 and stop_after_iterations32k/50k/80k/140k')
    return target


def validate_spec(spec):
    import math
    required = {'data','output','source_code','source_seal','source_seal_sha256','stage_plan','code_sha256','semantic_admission'}
    optional = set(DEFAULTS) | {'evaluation_data','technical_memory_cap_gib','technical_start_step','technical_pause_after'}
    if not required <= set(spec) or set(spec)-required-optional:
        raise ValueError('Missing or unsupported native crop spec keys: '+str(sorted((required-set(spec)) | (set(spec)-required-optional))))
    spec = dict(DEFAULTS, **spec)
    if type(spec['radius']) is not int or spec['radius'] not in (0,1):raise ValueError('Occupancy radius must be integer0or1')
    if isinstance(spec['distortion_final'], bool) or not isinstance(spec['distortion_final'], (int,float)) or spec['distortion_final'] not in (0,.01,.03,.1):raise ValueError('Final distortion must be0,0.01,0.03or0.1')
    for key in ('data','output','source_code','source_seal','stage_plan','evaluation_data'):
        if key in spec and (not isinstance(spec[key],str) or not Path(spec[key]).is_absolute()):
            raise ValueError('Absolute path required: '+key)
    if Path(spec['source_code']).resolve() != SOURCE_CODE:
        raise ValueError('Import source differs from specified sealed source')
    if sha(spec['source_seal']) != spec['source_seal_sha256']:
        raise ValueError('Source seal changed')
    seal = json.loads(Path(spec['source_seal']).read_text())
    for name, digest in seal['files_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(SOURCE_CODE/relative) != digest:
            raise ValueError('Sealed source changed: '+name)
    overlay = Path(__file__).resolve().parent
    if set(spec['code_sha256']) != {'train_native_crop.py','native_crop_sampling.py','native_crop_sampling_v2.py'}:
        raise ValueError('Both isolated worker/sampling code hashes required')
    for name, digest in spec['code_sha256'].items():
        if sha(overlay/name) != digest: raise ValueError('Isolated overlay code changed: '+name)
    for key, expected in dict(objective='actor',sampling='body80',width=64,
            frequency_threshold=.95,correct_sh_directions=True,density_activation='trunc_exp').items():
        if type(spec[key]) is not type(expected) or spec[key] != expected:
            raise ValueError('Fixed native crop recipe differs: '+key)
    if type(spec['exposure_scale']) is not int or spec['exposure_scale'] not in (1,2,4):raise ValueError('Exposure scale must be1,2or4')
    if not isinstance(spec['adam_beta2'], (int,float)) or not math.isfinite(spec['adam_beta2']) or not 0 < spec['adam_beta2'] < 1:raise ValueError('Adam beta2 must lie strictly in(0,1)')
    for key in ('initial_lr','final_lr'):
        if not isinstance(spec[key], (int,float)) or not math.isfinite(spec[key]) or not 0 < spec[key] <= .04:raise ValueError('LR must be positive finite and at most0.04: '+key)
    if type(spec['extra_eval_interval']) is not int or spec['extra_eval_interval'] < 0:raise ValueError('Extra eval interval must be a nonnegative integer')
    if spec['frequency_projection_mode'] not in ('legacy','normalized_jacobian'):
        raise ValueError('Unsupported frequency projection')
    if type(spec['corrected_arm_allocator']) is not bool:raise ValueError('Boolean corrected allocator required')
    for key in ('coarse_step','min_step_size'):
        if not isinstance(spec[key],(int,float)) or not math.isfinite(spec[key]) or not 0 < spec[key] <= .1:
            raise ValueError('Positive finite step required: '+key)
    for key in ('max_steps_per_ray','lr_decay_steps','train_rays_per_batch'):
        if type(spec[key]) is not int or spec[key] < 1:raise ValueError('Positive integer required: '+key)
    if type(spec['fr_switch_step']) is not int or not 0 <= spec['fr_switch_step'] < 140000:raise ValueError('Invalid FR switch')
    if not isinstance(spec['fr_after_switch'],(int,float)) or not 0 <= spec['fr_after_switch'] <= 1:raise ValueError('Invalid FR strength')
    if type(spec['target_num_samples_per_batch']) is not int or spec['target_num_samples_per_batch'] < 0:raise ValueError('Invalid point budget')
    if spec['train_count'] not in (63,66) or type(spec['seed']) is not int:raise ValueError('Invalid TRAIN count/seed')
    gate = spec['semantic_admission']
    if set(gate) != {'manifest','manifest_sha256','data'} or not Path(gate['data']).is_absolute():
        raise ValueError('Original semantic data/manifest binding required')
    if sha(gate['manifest']) != gate['manifest_sha256']:raise ValueError('Semantic manifest changed')
    if Path(gate['data']).resolve() == Path(spec['data']).resolve() and 'native' in Path(spec['data']).name:
        raise ValueError('Native RGB must not falsely rebind original semantic support')
    if json.loads((Path(gate['data'])/'box.json').read_text())['aabb'] != json.loads((Path(spec['data'])/'box.json').read_text())['aabb']:
        raise ValueError('Original semantic and training AABBs differ')
    validate_limit(spec)
    read_stage_plan(spec)
    return spec


def build_config(spec, resume=None):
    _source_import('nerfstudio.configs.method_configs')
    from nerfstudio.configs.method_configs import method_configs
    cfg = copy.deepcopy(method_configs['lookcloser'])
    cfg.output_dir = Path(spec['output'])
    cfg.experiment_name = 'worker'
    cfg.timestamp = 'frozen'
    cfg.vis = 'tensorboard'
    # The manual-loop guard is authoritative; framework display agrees.
    cfg.max_num_iterations = validate_limit(spec)
    cfg.machine.seed = spec['seed']
    cfg.machine.num_devices = 1
    cfg.logging.local_writer.enable = False
    cfg.logging.profiler = 'none'
    cfg.logging.csv_writer.enable = False
    cfg.save_only_latest_checkpoint = False
    cfg.viewer.quit_on_train_completion = True
    cfg.load_checkpoint = Path(resume) if resume else None
    cfg.load_dir = cfg.load_config = cfg.eval_checkpoint = None
    cfg.load_step = None
    cfg.pipeline.training_seed = spec['seed']
    cfg.pipeline.model.training_seed = spec['seed']
    _source_import('marshawn_semantic_admission')
    from marshawn_semantic_admission import SemanticAdmissionModelConfig
    cfg.pipeline.model = SemanticAdmissionModelConfig(**{k: v for k, v in vars(cfg.pipeline.model).items() if k != '_target'},
        semantic_manifest=spec['semantic_admission']['manifest'], semantic_manifest_sha256=spec['semantic_admission']['manifest_sha256'],
        semantic_data=spec['semantic_admission']['data'], semantic_identity_control=False)
    cfg.pipeline.model.correct_sh_directions = bool(spec.get('correct_sh_directions', False))
    cfg.pipeline.model.density_activation = spec.get('density_activation', 'softplus')
    cfg.pipeline.frequency_projection_mode = spec.get('frequency_projection_mode', 'legacy')
    cfg.optimizers['fields']['scheduler'].max_steps = spec.get('lr_decay_steps', 200000)
    # OptimizerConfig.setup forwards dynamic attributes through vars(self).
    cfg.optimizers['fields']['optimizer'].betas = (.9, spec.get('adam_beta2', .999))
    cfg.optimizers['fields']['optimizer'].lr = spec.get('initial_lr', .01)
    cfg.optimizers['fields']['scheduler'].lr_final = spec.get('final_lr', .0001)
    cfg.pipeline.feature_reweighting_switch_step = spec['fr_switch_step']
    cfg.pipeline.feature_reweighting_after_switch = spec['fr_after_switch']
    cfg.pipeline.model.occupancy_dilation_radius = spec['radius']
    cfg.pipeline.model.adaptive_coarse_step_size = spec['coarse_step']
    exposure_scale = spec.get('exposure_scale', 1)
    cfg.pipeline.model.adaptive_warmup_steps = 4096//exposure_scale
    cfg.pipeline.model.occupancy_warmup_steps = 4096//exposure_scale
    cfg.pipeline.model.occupancy_binary_warmup_steps = 4096//exposure_scale
    cfg.pipeline.model.occupancy_update_interval = 16//exposure_scale
    cfg.pipeline.grid_update_interval = 1024//exposure_scale
    cfg.pipeline.model.corrected_arm_allocator = spec['corrected_arm_allocator']
    cfg.pipeline.model.adaptive_min_step_size = spec['min_step_size']
    cfg.pipeline.model.max_steps_per_ray = spec['max_steps_per_ray']
    cfg.pipeline.target_num_samples_per_batch = spec['target_num_samples_per_batch']
    cfg.pipeline.dynamic_rays_start_step = 4096
    cfg.pipeline.dynamic_rays_max = 32768
    if spec['train_rays_per_batch'] > 32768:
        raise ValueError('Train ray batch exceeds reviewed32768 limit')
    cfg.pipeline.datamanager.train_num_rays_per_batch = spec['train_rays_per_batch']
    cfg.pipeline.datamanager.cpu_fas_prefetch = False
    cfg.pipeline.datamanager.cache_train_rays = False
    cfg.pipeline.model.eval_num_rays_per_chunk = 4096
    dm = cfg.pipeline.datamanager
    dm.data = Path(spec['data'])
    dp = dm.dataparser
    dp._target = __getattr__('TuningParser')
    dp.data = dm.data
    dp.orientation_method = dp.center_method = 'none'
    dp.auto_scale_poses = False
    dp.scale_factor = dp.scene_scale = 1.
    dp.downscale_factor = 1
    dp.eval_mode = 'filename'
    return cfg



def exposure_phase(actual_step, exposure_scale=1):
    """Return body80 phase and distortion ramp in4096-ray-equivalent steps.

    Trainer/model/optimizer/checkpoint step counters remain actual updates.
    This pure mapping consumes no RNG; LR and FR use explicit spec schedules.
    """
    virtual_step = actual_step*exposure_scale
    fraction = min(1., max(0., (virtual_step-4096)/8192))
    return virtual_step, fraction


def is_eval_iteration(completed_iterations, extra_eval_interval=0):
    """Evaluation cadence uses actual completed updates, including extra late gates."""
    return (completed_iterations in EVAL_ITERATIONS or
            (extra_eval_interval > 0 and completed_iterations >= 32000 and completed_iterations % extra_eval_interval == 0))

def choose_checkpoint(evaluations):
    return min(evaluations, key=lambda row:(row['metrics']['lpips'], -row['metrics']['psnr'], row['step'])) if evaluations else None


def protected_checkpoints(evaluations, latest_step, fr_boundary=75940):
    """Keep the moving LPIPS-tie Pareto frontier, plus resume and FR boundary.

    Keeping only today's tie winner can discard tomorrow's: as minimum LPIPS
    decreases, the high-PSNR winner can leave the .001 window. Candidates outside
    today's window can never reenter as the minimum only decreases.
    """
    protected = {int(latest_step), int(fr_boundary)}
    if not evaluations:
        return protected
    best = min(r['metrics']['lpips'] for r in evaluations)
    eligible = [r for r in evaluations if r['metrics']['lpips'] <= best + .001]
    for row in eligible:
        dominated = any(other['step'] != row['step'] and
            other['metrics']['lpips'] <= row['metrics']['lpips'] and
            (other['metrics']['psnr'] > row['metrics']['psnr'] or
             (other['metrics']['psnr'] == row['metrics']['psnr'] and other['step'] < row['step']))
            for other in eligible)
        if not dominated:
            protected.add(row['step'])
    return protected


def finish(output, spec, state, step, smoke_steps, sampling_audit):
    """Idempotent finalization, including a crash after terminal checkpoint commit."""
    limit = validate_limit(spec)
    if step + 1 > limit or (smoke_steps is None and step + 1 != limit):
        raise ValueError('Invalid terminal iteration count')
    chosen = choose_checkpoint(state['evaluations'])
    if chosen is None:
        raise ValueError('Cannot finalize without complete validation')
    atomic_json(output / 'result.json', dict(status='technical_smoke_complete' if smoke_steps is not None else 'fixed_iteration_budget_complete',
                technical_smoke=smoke_steps is not None, final_step=step, completed_iterations=step+1, hard_max_iterations=limit, plateau=state['plateau'],
                selected_step=chosen['step'], metrics={k:chosen['metrics'][k] for k in ('psnr','ssim','lpips')}, by_actor=chosen['by_actor'],
                evaluation_policy=state['evaluation_policy'], selection='raw minimum actor LPIPS; PSNR tie-break',
                checkpoint=str(output / 'checkpoints' / f"step-{chosen['step']:09d}.ckpt"),
                render_directory=str(output / 'renders' / f"step-{chosen['step']:09d}"),
                visual_review_complete=False, spec_sha256=state['spec_sha256'], data_sha256=state['data_sha256']))
    atomic_json(output / 'complete.json', dict(config=spec, selection=chosen,
                geometry_trigger=any(r['trigger'] for r in chosen['opacity_diagnostics']),
                technical_smoke=smoke_steps is not None, final_step=step, completed_iterations=step+1, hard_max_iterations=limit,
                plateau=state['plateau'], sampling_audit=sampling_audit,
                visual_review_complete=False))


def inventory_and_prune(output, checkpoint, step, evaluations, fr_boundary=75940):
    """Commit hash/size ledger and prune only dominated files owned by this run."""
    output, checkpoint = Path(output), Path(checkpoint)
    checkpoint_dir = output / 'checkpoints'
    if checkpoint.parent != checkpoint_dir:
        raise ValueError('Checkpoint outside owned directory')
    inventory_path = output / 'checkpoint_inventory.json'
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    inventory[str(step)] = dict(path=str(checkpoint.relative_to(output)), sha256=sha(checkpoint),
                                bytes=checkpoint.stat().st_size, retained=True)
    protected = protected_checkpoints(evaluations, step, fr_boundary)
    if evaluations:
        _source_import('marshawn_video_train')
        from marshawn_video_train import selectors
        protected.update(r['step'] for r in selectors(evaluations).values())
    for candidate in checkpoint_dir.glob('step-?????????.ckpt'):
        candidate_step = int(candidate.stem.split('-')[1])
        if candidate_step not in protected:
            if str(candidate_step) not in inventory:
                inventory[str(candidate_step)] = dict(path=str(candidate.relative_to(output)),
                    sha256=sha(candidate), bytes=candidate.stat().st_size)
            inventory[str(candidate_step)].update(retained=False,
                reason='outside future selector frontier; metrics/renders retained')
            candidate.unlink()
    atomic_json(inventory_path, inventory)
    return inventory


def validate_data(data, train_count=None):
    import numpy as np
    _source_import('marshawn_video_metrics')
    from marshawn_video_metrics import actor_visibility
    _source_import('prepare_marshawn_tuning')
    from prepare_marshawn_tuning import verify_inventory
    ready = verify_inventory(data)
    if ready['train'] not in (63,66) or (train_count is not None and ready['train'] != train_count) or len(ready['records']) != 66 or len(ready['validation']) != 3:
        raise ValueError('Expected declared TRAIN split and three eval views among66 camera records')
    if set(ready['validation']) != {'DC','HC','KA'}:
        review=ready.get('monitor_review',{})
        if review.get('reviewed') is not True or not review.get('evidence'):
            raise ValueError('Alternative TRAIN monitors require explicit reviewed evidence')
    transforms = json.loads((data / 'transforms.json').read_text())
    frames = {r['file_path']:r for r in transforms['frames']}
    train = transforms['train_filenames']
    monitors = transforms['val_filenames']
    if len(train) != ready['train'] or len(set(train)) != len(train) or not set(train) <= set(frames):
        raise ValueError('Actual TRAIN filenames differ')
    if ready['train'] == 66 and (set(train) != set(frames) or not set(monitors) <= set(train)):
        raise ValueError('ALL66 monitors must explicitly overlap TRAIN')
    if ready['train'] == 63 and (set(monitors)&set(train) or set(monitors)|set(train) != set(frames)):
        raise ValueError('63/3 split must be disjoint and complete')
    if set(frames[p]['physical_camera'] for p in monitors) != set(ready['validation']):
        raise ValueError('Wrong in-sample monitors')
    if sha(data / 'transforms.json') != ready['transforms_sha256']:raise ValueError('Transforms changed')
    for row in ready['records']:
        if sha(data / row['image']) != row['rgb_sha256']:raise ValueError('RGB changed')
        semantic = data / 'semantics' / (row['camera']+'.npz')
        if sha(semantic) != row['semantic_sha256']:raise ValueError('Semantic mask changed')
        with np.load(semantic,allow_pickle=False) as src:
            for actor in ('black','grey'):
                if src[actor].dtype != np.bool_:raise ValueError('Invalid actor mask dtype')
                actor_visibility(row,actor,src[actor])
    return ready


def evaluate(trainer, ready, data, output, step):
    import numpy as np
    import torch
    from PIL import Image
    from scipy.ndimage import binary_erosion
    _source_import('nerfstudio.engine.trainer')
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    _source_import('marshawn_tuning_metrics')
    from marshawn_tuning_metrics import masked_actor_metrics, aggregate_actor_metrics
    state = _capture_rng_state()
    pipeline = trainer.pipeline
    dataset = getattr(pipeline, 'native_evaluation_dataset', pipeline.datamanager.eval_dataset)
    records = {Path(row['image']).name: row for row in ready['records']}
    folder = output / 'renders' / f'step-{step:09d}'
    folder.mkdir(parents=True, exist_ok=True)
    rows, holes = [], []
    try:
        pipeline.eval()
        model = pipeline.model
        for index, filename in enumerate(dataset.image_filenames):
            row = records[filename.name]
            camera = dataset.cameras[index:index+1].to(model.device)
            with torch.inference_mode():
                rendered = model.get_outputs_for_camera(camera)
            pred = rendered['rgb'].float()
            alpha = rendered['accumulation'].float()
            target = dataset[index]['image'].to(pred.device).float()
            if pred.shape != target.shape:
                raise ValueError('Eval camera/source raster mismatch')
            if not torch.isfinite(pred).all() or not torch.isfinite(alpha).all():
                raise FloatingPointError('Nonfinite validation render')
            Image.fromarray((pred.clamp(0, 1).cpu().numpy()*255+.5).astype(np.uint8)).save(folder / f"{row['camera']}.png")
            Image.fromarray((alpha[..., 0].clamp(0, 1).cpu().numpy()*65535+.5).astype(np.uint16)).save(folder / f"{row['camera']}_alpha.png")
            with np.load(data / 'semantics' / f"{row['camera']}.npz", allow_pickle=False) as source:
                for actor in ('black', 'grey'):
                    mask = source[actor]
                    _source_import('marshawn_video_metrics')
                    from marshawn_video_metrics import actor_visibility
                    if actor_visibility(row, actor, mask) == 'absent':
                        continue
                    scores = masked_actor_metrics(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None],
                               torch.from_numpy(mask.copy()).to(pred.device)[None, None], model.lpips.net.eval(),
                               crop_xyxy=row['crops'][actor])
                    rows.append(dict(view=row['camera'], actor=actor, **scores))
                    inner = binary_erosion(mask, iterations=3)
                    opacity = alpha[..., 0].cpu().numpy()
                    fraction = float((opacity[inner] < .9).mean()) if inner.any() else None
                    holes.append(dict(view=row['camera'], actor=actor, eroded_pixels=int(inner.sum()),
                                      below09_fraction=fraction, trigger=bool(fraction is not None and fraction > .01)))
        _source_import('marshawn_video_metrics')
        from marshawn_video_metrics import aggregate_visible
        result = dict(step=step, **aggregate_visible(rows, ready), opacity_diagnostics=holes)
        result.update(masked_lpips=result['metrics']['lpips'], psnr=result['metrics']['psnr'],
                      ssim=result['metrics']['ssim'], validation_loss=result['metrics']['validation_loss'])
        atomic_json(folder / 'metrics.json', result)
        return result
    finally:
        pipeline.train()
        _restore_rng_state(state)



def bind_cold_output(output, spec_hash, data_hash, code_hashes):
    """Permit cold initialization only in a fresh, source-bound owned output."""
    output = Path(output).resolve()
    claim = dict(schema=1, output=str(output), spec_sha256=spec_hash,
        data_sha256=data_hash, code_sha256=code_hashes,
        initialization='cold field/Adam/occupancy; same-output resume only', learned_state_transfer=False)
    path = output/'cold_provenance.json'
    if path.exists():
        if json.loads(path.read_text()) != claim:raise ValueError('Cold output provenance differs')
    else:
        if (output/'latest.json').exists() or list((output/'checkpoints').glob('*.ckpt')):
            raise ValueError('Existing checkpoint without native cold provenance')
        atomic_json(path, claim)
    return claim

def train(spec_path, smoke_steps=None):
    import random
    import numpy as np
    import torch
    _source_import('nerfstudio.engine.callbacks')
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
    _source_import('nerfstudio.engine.trainer')
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    _source_import('nerfstudio.utils.writer')
    from nerfstudio.utils import writer
    from native_crop_sampling_v2 import install_sampling
    _source_import('marshawn_semantic_occupancy')
    from marshawn_semantic_occupancy import install_capture_hook, export_snapshot

    spec_path = Path(spec_path).resolve()
    spec = validate_spec(json.loads(spec_path.read_text()))
    hard_limit = validate_limit(spec)
    if smoke_steps is not None:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        cap = float(spec.get('technical_memory_cap_gib', .9*total_memory/2**30))
        if not 0 < cap*2**30 <= .9*total_memory:
            raise ValueError('Technical allocator cap must be positive and at most90% of this GPU')
        torch.cuda.set_per_process_memory_fraction(cap*2**30/total_memory)
    if smoke_steps is None and any(k.startswith('technical_') for k in spec):
        raise ValueError('Technical controls require a separate smoke output')
    output, data = Path(spec['output']).resolve(), Path(spec['data']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mode_path = output / 'mode.json'
    mode = dict(technical_smoke=smoke_steps is not None, smoke_steps=smoke_steps)
    if smoke_steps is not None and smoke_steps < 1:
        raise ValueError('Positive smoke length required')
    if mode_path.exists() and json.loads(mode_path.read_text()) != mode:
        raise ValueError('Technical smoke and production outputs must be separate')
    atomic_json(mode_path, mode)
    ready = validate_data(data, spec['train_count'])
    evaluation_data = Path(spec.get('evaluation_data', str(data))).resolve()
    evaluation_ready = ready if evaluation_data == data else validate_data(evaluation_data)
    stage_target = read_stage_plan(spec)
    spec_hash, data_hash = sha(spec_path), sha(data / 'ready.json')
    code_hashes = dict(spec['code_sha256'], source_seal=spec['source_seal_sha256'])
    input_hashes = {'train_ready': data_hash, 'evaluation_ready': sha(evaluation_data/'ready.json'),
                    'semantic_manifest': sha(spec['semantic_admission']['manifest']),
                    'semantic_ready': sha(Path(spec['semantic_admission']['data'])/'ready.json')}
    semantic_root = Path(spec['semantic_admission']['manifest']).parent
    for name, digest in json.loads((semantic_root / 'manifest.json').read_text())['files_sha256'].items():
        if sha(semantic_root/name) != digest:raise ValueError('Semantic source changed: '+name)
        input_hashes['semantic/'+name] = digest
    evaluation_policy = 'DC/HC/KA in-sample; all66 TRAIN' if spec['train_count'] == 66 else 'DC/HC/KA held-out;63 TRAIN'
    bind_cold_output(output, spec_hash, data_hash, code_hashes)
    latest_path = output / 'latest.json'
    resume = None
    resumed_capture = None
    state = dict(schema=1, spec_sha256=spec_hash, data_sha256=data_hash,
                 plateau=dict(policy='external stage gates; historical plateau disabled'), evaluations=[], boundary_rng_reset=False,
                 evaluation_policy=evaluation_policy,
                 code_sha256=code_hashes, input_sha256=input_hashes)
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        resume = (output / latest['checkpoint']).resolve()
        if resume.parent != (output/'checkpoints').resolve():raise ValueError('Resume checkpoint outside this owned run')
        if sha(resume) != latest['checkpoint_sha256']:
            raise ValueError('Last committed checkpoint hash mismatch')
        loaded = torch.load(resume, map_location='cpu', weights_only=False)
        state = loaded['tuning_runtime']
        if state['spec_sha256'] != spec_hash or state['data_sha256'] != data_hash:
            raise ValueError('Resume source/config mismatch')
        if state.get('code_sha256') != code_hashes or state.get('input_sha256') != input_hashes:
            raise ValueError('Runtime code, bounds or frequency maps changed since checkpoint')
        resumed_capture = loaded.get('actual_occupancy_capture')
        del loaded
    tracker = SimpleNamespace(to_dict=lambda:dict(policy='external stage gates; historical plateau disabled'), stopped=False)
    if resume is not None and latest['step'] + 1 > hard_limit:
        raise ValueError('Cannot resume an over-budget checkpoint')
    if resume is not None and (completed(latest['step'], hard_limit) or (smoke_steps is not None and latest['step'] + 1 >= smoke_steps)):
        finish(output, spec, state, latest['step'], smoke_steps, state.get('sampling_state', {}))
        return
    if resume is not None and latest['step'] + 1 >= stage_target and smoke_steps is None:
        atomic_json(output/'stage_paused.json', dict(completed_iterations=latest['step']+1, stage_target=stage_target, checkpoint=str(resume), policy='review then promote stage_plan and restart same output'))
        return
    (output/'stage_paused.json').unlink(missing_ok=True)
    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    seed_all(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)
    cfg = build_config(spec, resume)
    cfg.save_config()
    trainer = cfg.setup(local_rank=0, world_size=1)
    trainer.setup()
    _source_import('marshawn_video_train')
    from marshawn_video_train import cold_setup_audit
    cold_setup_audit(trainer, output, resume)
    pipe, model = trainer.pipeline, trainer.pipeline.model
    if len(pipe.datamanager.train_dataset) != spec['train_count'] or len(pipe.datamanager.eval_dataset) != len(ready['validation']):
        raise ValueError('Live train/validation split differs')
    expected = torch.tensor(ready['aabb'], dtype=torch.float32)
    for box in (model.scene_aabb.reshape(2, 3), model.field.aabb,
                model.occupancy_grid.aabbs[0].reshape(2, 3), pipe.datamanager.train_dataparser_outputs.scene_box.aabb):
        torch.testing.assert_close(box.cpu(), expected, rtol=0, atol=1e-6)
    if evaluation_data != data:
        evaluation_parser_config = copy.deepcopy(cfg.pipeline.datamanager.dataparser)
        evaluation_parser_config.data = evaluation_data
        evaluation_parser = evaluation_parser_config.setup()
        evaluation_outputs = evaluation_parser.get_dataparser_outputs(split='val')
        evaluation_dataset = pipe.datamanager.dataset_type(dataparser_outputs=evaluation_outputs, scale_factor=1.)
        object.__setattr__(pipe, 'native_evaluation_dataset', evaluation_dataset)
    sampler = install_sampling(pipe, data, spec['objective'],
                               'default' if spec['sampling'] == 'baseline' else spec['sampling'], spec['width'])
    if 'sampling_state' in state:
        sampler.load_state_dict(state['sampling_state'])
    install_capture_hook(model)
    if resumed_capture is not None:
        model._tuning_occupancy_capture = {k: v.to(model.device) if torch.is_tensor(v) else v
                                          for k, v in resumed_capture.items()}
    if trainer._loaded_rng_state is not None:
        _source_import('marshawn_semantic_admission')
        from marshawn_semantic_admission import rng_signature
        expected_rng = rng_signature(trainer._loaded_rng_state)
        _restore_rng_state(trainer._loaded_rng_state)
        actual_rng = rng_signature(_capture_rng_state())
        if expected_rng != actual_rng: raise ValueError('RNG restoration differs')
        atomic_json(output / 'resume_rng_verification.json', dict(passed=True, expected=expected_rng, actual=actual_rng, restored_checkpoint_step=latest['step']))
        trainer._loaded_rng_state = None
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    atomic_json(output / 'runtime.json', dict(pid=os.getpid(), host=socket.gethostname(), spec_sha256=spec_hash,
                data_sha256=data_hash, config=str(cfg.get_base_dir() / 'config.yml'), technical_smoke=smoke_steps,
                framework_max_num_iterations_display_only=cfg.max_num_iterations,
                actual_iteration_limit=hard_limit, optimizer_hyperparameters={k:[dict(betas=list(group['betas']),lr=group['lr'],eps=group['eps']) for group in opt.param_groups] for k,opt in trainer.optimizers.optimizers.items()}, extra_eval_interval=spec['extra_eval_interval'], exposure_scale=spec['exposure_scale'], phase_policy='body80/distortion use actual_step*exposure_scale; warmup/update cadence compressed; LR/FR explicit; trainer steps actual', source_sha256={Path(__file__).name: sha(__file__)},
                baseline=dict(density_activation=model.config.density_activation,
                              correct_sh_directions=model.config.correct_sh_directions,
                              frequency_projection=pipe.config.frequency_projection_mode,
                              sparse_depth='configured0.001; active only if dataset supplies depth_image')))
    requested_stop = []
    def terminate(signum, frame):
        requested_stop.append(signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def save(step, terminal=False):
        state['plateau'] = tracker.to_dict()
        state['sampling_state'] = sampler.state_dict()
        state['exposure_schedule'] = dict(exposure_scale=spec['exposure_scale'], actual_step=step, virtual_sampler_step=step*spec['exposure_scale'])
        path = checkpoint_dir / f'step-{step:09d}.ckpt'
        temporary = path.with_suffix('.ckpt.tmp')
        previous_size = 0
        if latest_path.exists():
            previous_file = output / json.loads(latest_path.read_text())['checkpoint']
            previous_size = previous_file.stat().st_size
        estimate = max(previous_size, sum(p.numel()*p.element_size() for p in model.parameters())*3)
        if shutil.disk_usage(output).free < estimate + (2 << 30):
            raise OSError('Insufficient disk for atomic checkpoint plus2GiB reserve; recipe unchanged')
        payload = dict(step=step, pipeline=pipe.state_dict(),
                       optimizers={k: v.state_dict() for k, v in trainer.optimizers.optimizers.items()},
                       schedulers={k: v.state_dict() for k, v in trainer.optimizers.schedulers.items()},
                       scalers=trainer.grad_scaler.state_dict(), rng_state=_capture_rng_state(),
                       tuning_runtime=copy.deepcopy(state),
                       actual_occupancy_capture=getattr(model, '_tuning_occupancy_capture', None))
        torch.save(payload, temporary)
        os.replace(temporary, path)
        digest = sha(path)
        atomic_json(latest_path, dict(checkpoint=str(path.relative_to(output)), checkpoint_sha256=digest,
                                      step=step, terminal=terminal))
        receipt = output / 'occupancy' / f'step-{step:09d}_grid.json'
        if not receipt.exists():
            receipt = export_snapshot(model, path, step, output / 'occupancy', sources={'spec': spec_hash, 'data': data_hash})
        # Actual core/active occupancy snapshot is retained; no expensive replay audit at every stage.
        inventory_and_prune(output, path, step, state['evaluations'], spec['fr_switch_step']-1)
        return path

    start_time, last_progress = time.monotonic(), 0.
    step = trainer._start_step
    if resume is None and smoke_steps is not None:
        step = int(spec.get('technical_start_step', step))
        if step not in (0, 4096): raise ValueError('Declared cold technical phase only')
    while True:
        check_step(step, hard_limit)
        trainer.step = step
        pipe.train()
        if step == spec['fr_switch_step'] and not state['boundary_rng_reset']:
            seed_all(spec['seed'])
            state['boundary_rng_reset'] = True
        sampler_step, fraction = exposure_phase(step, spec['exposure_scale'])
        sampler.set_step(sampler_step)
        model.config.distortion_loss_mult = .01 + (spec['distortion_final'] - .01) * fraction
        trainer._apply_live_optimizer_switch(step)
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, location=Location.BEFORE_TRAIN_ITERATION)
        loss, _, train_metrics = trainer.train_iteration(step)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError('Nonfinite training objective')
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, location=Location.AFTER_TRAIN_ITERATION)
        writer.write_out_storage()
        gate = is_eval_iteration(step+1, spec['extra_eval_interval'])
        smoke_done = smoke_steps is not None and step + 1 >= smoke_steps
        hard_done = completed(step, hard_limit)
        if step + 1 >= stage_target:stage_target = read_stage_plan(spec)
        stage_done = smoke_steps is None and step + 1 >= stage_target
        if gate or smoke_done or hard_done or stage_done:
            result = evaluate(trainer, evaluation_ready, evaluation_data, output, step)
            result['evaluation_policy'] = evaluation_policy
            state['evaluations'].append(result)
            atomic_json(output / 'metrics.json', state['evaluations'])
        terminal = hard_done or smoke_done
        if smoke_steps is not None and step == spec.get('technical_pause_after'):
            requested_stop.append(signal.SIGTERM)
        if gate or terminal or stage_done or requested_stop or (step+1)%2000 == 0 or step == spec['fr_switch_step']-1:
            save(step, terminal=terminal)
        now = time.monotonic()
        if now-last_progress >= 30 or gate or terminal or stage_done:
            atomic_json(output / 'progress.json', dict(step=step, pid=os.getpid(), time=time.time(),
                        elapsed_seconds=now-start_time, gpu_memory_bytes=torch.cuda.memory_allocated(),
                        gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                        cumulative_point_samples=int(pipe.cumulative_point_samples.item()),
                        completed_iterations=step+1, stage_target=stage_target, evaluation_policy=evaluation_policy,
                        exposure_scale=spec['exposure_scale'], virtual_sampler_step=sampler_step,
                        current_train_rays=int(pipe.datamanager.train_pixel_sampler.num_rays_per_batch),
                        sampling_metrics={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in train_metrics.items() if k in ('num_samples_per_batch','samples_per_ray_mean','adaptive_samples_max','adaptive_saturation_rate','zero_sample_ray_rate')},
                        plateau=tracker.to_dict(), technical_smoke=smoke_steps is not None))
            print(json.dumps(dict(step=step, elapsed_seconds=round(now-start_time), plateau=tracker.stopped)), flush=True)
            last_progress = now
        if terminal:
            finish(output, spec, state, step, smoke_steps, sampler.audit_state())
            trainer.shutdown()
            return
        if stage_done:
            atomic_json(output/'stage_paused.json', dict(completed_iterations=step+1, stage_target=stage_target, checkpoint=json.loads(latest_path.read_text()), policy='review then promote stage_plan and restart same output'))
            trainer.shutdown()
            return
        if requested_stop:
            trainer.shutdown()
            raise SystemExit(128 + requested_stop[-1])
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        spec = validate_spec(json.loads(args.spec.read_text()))
        cfg = build_config(spec)
        print(json.dumps(dict(spec=spec, config_base=str(cfg.get_base_dir()),
                             stage_target=read_stage_plan(spec), iteration_limit=140000, eval_completed_iterations=EVAL_ITERATIONS, CUDA_initialized=__import__('torch').cuda.is_initialized()), indent=2))
    else:
        train(args.spec, args.smoke_steps)
