"""Durable single-GPU LookCloser tuning worker, full leader ancestry then plateau.

Run --spec SPEC.json; --smoke-steps N is an explicitly labelled technical run.
No training happens on import. The manual loop has no iteration ceiling and
preserves the native train_iteration and before/after callbacks.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import sys
import time

from marshawn_tuning_metrics import HISTORICAL_EVAL_STEPS, PlateauTracker
from marshawn_tuning_occupancy import atomic_json, sha

MODULE = 'train_marshawn_tuning'
if __name__ == '__main__':
    sys.modules[MODULE] = sys.modules[__name__]


def __getattr__(name):
    if name != 'TuningParser':
        raise AttributeError(name)
    import torch
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
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


def validate_spec(spec):
    for key in ['data', 'output', 'objective', 'sampling', 'width', 'radius', 'coarse_step', 'distortion_final', 'seed']:
        if key not in spec:
            raise ValueError('Missing spec field ' + key)
    if spec['objective'] not in ('full', 'actor') or spec['sampling'] not in ('baseline', 'body80'):
        raise ValueError('Unknown objective or sampling')
    if spec['width'] not in (32, 64) or spec['radius'] not in (0, 1):
        raise ValueError('Unsupported width or occupancy policy')
    if spec['coarse_step'] not in (.003125, .00625, .0125) or spec['distortion_final'] not in (.01, 1.):
        raise ValueError('Unplanned marching or distortion parameter')
    if spec['seed'] not in (42, 43):
        raise ValueError('Only planned paired seeds42/43')
    return spec


def build_config(spec, resume=None):
    from nerfstudio.configs.method_configs import method_configs
    cfg = copy.deepcopy(method_configs['lookcloser'])
    cfg.output_dir = Path(spec['output'])
    cfg.experiment_name = 'worker'
    cfg.timestamp = 'frozen'
    cfg.vis = 'tensorboard'
    # Used by framework display/config only; our while-loop has NO step ceiling.
    cfg.max_num_iterations = sys.maxsize
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
    cfg.pipeline.feature_reweighting_switch_step = 75941
    cfg.pipeline.feature_reweighting_after_switch = .3
    cfg.pipeline.model.occupancy_dilation_radius = spec['radius']
    cfg.pipeline.model.adaptive_coarse_step_size = spec['coarse_step']
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


def choose_checkpoint(evaluations):
    if not evaluations:
        return None
    minimum = min(r['metrics']['lpips'] for r in evaluations)
    near = [r for r in evaluations if r['metrics']['lpips'] <= minimum + .001]
    return max(near, key=lambda r: (r['metrics']['psnr'], -r['step']))


def protected_checkpoints(evaluations, latest_step):
    """Keep the moving LPIPS-tie Pareto frontier, plus resume and FR boundary.

    Keeping only today's tie winner can discard tomorrow's: as minimum LPIPS
    decreases, the high-PSNR winner can leave the .001 window. Candidates outside
    today's window can never reenter as the minimum only decreases.
    """
    protected = {int(latest_step), 75940}
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
    chosen = choose_checkpoint(state['evaluations'])
    if chosen is None:
        raise ValueError('Cannot finalize without complete validation')
    atomic_json(output / 'result.json', dict(status='technical_smoke_complete' if smoke_steps is not None else 'plateau',
                technical_smoke=smoke_steps is not None, final_step=step, plateau=state['plateau'],
                selected_step=chosen['step'], metrics=chosen['metrics'], by_actor=chosen['by_actor'],
                checkpoint=str(output / 'checkpoints' / f"step-{chosen['step']:09d}.ckpt"),
                render_directory=str(output / 'renders' / f"step-{chosen['step']:09d}"),
                visual_review_complete=False, spec_sha256=state['spec_sha256'], data_sha256=state['data_sha256']))
    atomic_json(output / 'complete.json', dict(config=spec, selection=chosen,
                geometry_trigger=any(r['trigger'] for r in chosen['opacity_diagnostics']),
                technical_smoke=smoke_steps is not None, final_step=step,
                plateau=state['plateau'], sampling_audit=sampling_audit,
                visual_review_complete=False))


def inventory_and_prune(output, checkpoint, step, evaluations):
    """Commit hash/size ledger and prune only dominated files owned by this run."""
    output, checkpoint = Path(output), Path(checkpoint)
    checkpoint_dir = output / 'checkpoints'
    if checkpoint.parent != checkpoint_dir:
        raise ValueError('Checkpoint outside owned directory')
    inventory_path = output / 'checkpoint_inventory.json'
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    inventory[str(step)] = dict(path=str(checkpoint.relative_to(output)), sha256=sha(checkpoint),
                                bytes=checkpoint.stat().st_size, retained=True)
    protected = protected_checkpoints(evaluations, step)
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


def validate_data(data):
    import numpy as np
    from prepare_marshawn_tuning import verify_inventory
    ready = verify_inventory(data)
    if ready['train'] != 63 or set(ready['validation']) != {'DC', 'HC', 'KA'} or len(ready['records']) != 66:
        raise ValueError('Expected frozen63/3 split')
    if sha(data / 'transforms.json') != ready['transforms_sha256']:
        raise ValueError('Transforms changed')
    for row in ready['records']:
        if sha(data / row['image']) != row['rgb_sha256']:
            raise ValueError('RGB changed: ' + row['camera'])
        # ready.json can have preparation-host paths; use portable local layout.
        semantic = data / 'semantics' / (row['camera'] + '.npz')
        if sha(semantic) != row['semantic_sha256']:
            raise ValueError('Semantic mask changed: ' + row['camera'])
        with np.load(semantic, allow_pickle=False) as src:
            for actor in ('black', 'grey'):
                if src[actor].dtype != np.bool_ or not src[actor].any():
                    raise ValueError('Invalid actor mask')
    return ready


def evaluate(trainer, ready, data, output, step):
    import numpy as np
    import torch
    from PIL import Image
    from scipy.ndimage import binary_erosion
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from marshawn_tuning_metrics import masked_actor_metrics, aggregate_actor_metrics
    state = _capture_rng_state()
    pipeline = trainer.pipeline
    dataset = pipeline.datamanager.eval_dataset
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
                    scores = masked_actor_metrics(pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None],
                               torch.from_numpy(mask.copy()).to(pred.device)[None, None], model.lpips.net.eval(),
                               crop_xyxy=row['crops'][actor])
                    rows.append(dict(view=row['camera'], actor=actor, **scores))
                    inner = binary_erosion(mask, iterations=3)
                    opacity = alpha[..., 0].cpu().numpy()
                    fraction = float((opacity[inner] < .9).mean()) if inner.any() else None
                    holes.append(dict(view=row['camera'], actor=actor, eroded_pixels=int(inner.sum()),
                                      below09_fraction=fraction, trigger=bool(fraction is not None and fraction > .01)))
        result = dict(step=step, **aggregate_actor_metrics(rows), opacity_diagnostics=holes)
        result.update(masked_lpips=result['metrics']['lpips'], psnr=result['metrics']['psnr'],
                      ssim=result['metrics']['ssim'], validation_loss=result['metrics']['validation_loss'])
        atomic_json(folder / 'metrics.json', result)
        return result
    finally:
        pipeline.train()
        _restore_rng_state(state)


def train(spec_path, smoke_steps=None):
    import random
    import numpy as np
    import torch
    from nerfstudio.engine.callbacks import TrainingCallbackLocation as Location
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    from nerfstudio.utils import writer
    from marshawn_tuning_sampling import install_sampling
    from marshawn_tuning_occupancy import install_capture_hook, export_snapshot, audit_snapshot

    spec_path = Path(spec_path).resolve()
    spec = validate_spec(json.loads(spec_path.read_text()))
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
    ready = validate_data(data)
    spec_hash, data_hash = sha(spec_path), sha(data / 'ready.json')
    source_dir = Path(__file__).resolve().parent
    code_paths = [source_dir / name for name in ('train_marshawn_tuning.py', 'marshawn_tuning_sampling.py',
                  'marshawn_tuning_metrics.py', 'marshawn_tuning_occupancy.py')]
    import inspect
    from nerfstudio.models.lookcloser import LookCloserModel
    from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
    code_paths += [Path(inspect.getfile(LookCloserModel)), Path(inspect.getfile(LookCloserPipeline))]
    code_hashes = {p.name: sha(p) for p in code_paths}
    input_hashes = {'box.json': sha(data / 'box.json')}
    for path in sorted((data / 'lookcloser_frequencies').iterdir()):
        if path.is_file() and path.suffix in ('.pt', '.json'):
            input_hashes['lookcloser_frequencies/' + path.name] = sha(path)
    latest_path = output / 'latest.json'
    resume = None
    resumed_capture = None
    state = dict(schema=1, spec_sha256=spec_hash, data_sha256=data_hash,
                 plateau=PlateauTracker().to_dict(), evaluations=[], boundary_rng_reset=False,
                 code_sha256=code_hashes, input_sha256=input_hashes)
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        resume = output / latest['checkpoint']
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
    tracker = PlateauTracker.from_dict(state['plateau'])
    if tracker.stopped or (resume is not None and smoke_steps is not None and latest['step'] + 1 >= smoke_steps):
        finish(output, spec, state, latest['step'], smoke_steps, state.get('sampling_state', {}))
        return
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
    pipe, model = trainer.pipeline, trainer.pipeline.model
    if len(pipe.datamanager.train_dataset) != 63 or len(pipe.datamanager.eval_dataset) != 3:
        raise ValueError('Live train/validation split differs')
    expected = torch.tensor(ready['aabb'], dtype=torch.float32)
    for box in (model.scene_aabb.reshape(2, 3), model.field.aabb,
                model.occupancy_grid.aabbs[0].reshape(2, 3), pipe.datamanager.train_dataparser_outputs.scene_box.aabb):
        torch.testing.assert_close(box.cpu(), expected, rtol=0, atol=1e-6)
    sampler = install_sampling(pipe, data, spec['objective'],
                               'default' if spec['sampling'] == 'baseline' else spec['sampling'], spec['width'])
    if 'sampling_state' in state:
        sampler.load_state_dict(state['sampling_state'])
    install_capture_hook(model)
    if resumed_capture is not None:
        model._tuning_occupancy_capture = {k: v.to(model.device) if torch.is_tensor(v) else v
                                          for k, v in resumed_capture.items()}
    if trainer._loaded_rng_state is not None:
        _restore_rng_state(trainer._loaded_rng_state)
        trainer._loaded_rng_state = None
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    atomic_json(output / 'runtime.json', dict(pid=os.getpid(), host=socket.gethostname(), spec_sha256=spec_hash,
                data_sha256=data_hash, config=str(cfg.get_base_dir() / 'config.yml'), technical_smoke=smoke_steps,
                framework_max_num_iterations_display_only=cfg.max_num_iterations,
                actual_iteration_limit=None, source_sha256={Path(__file__).name: sha(__file__)},
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
        if step in HISTORICAL_EVAL_STEPS or step >= tracker.anchor_step or terminal:
            audit_snapshot(receipt, output / 'occupancy' / f'review-{step:09d}', checkpoint=path)
        inventory_and_prune(output, path, step, state['evaluations'])
        return path

    start_time, last_progress = time.monotonic(), 0.
    step = trainer._start_step
    while True:
        trainer.step = step
        pipe.train()
        if step == 75941 and not state['boundary_rng_reset']:
            seed_all(spec['seed'])
            state['boundary_rng_reset'] = True
        sampler.set_step(step)
        fraction = min(1., max(0., (step - 4096) / 8192))
        model.config.distortion_loss_mult = .01 + (spec['distortion_final'] - .01) * fraction
        trainer._apply_live_optimizer_switch(step)
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, location=Location.BEFORE_TRAIN_ITERATION)
        loss, _, _ = trainer.train_iteration(step)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError('Nonfinite training objective')
        for callback in trainer.callbacks:
            callback.run_callback_at_location(step, location=Location.AFTER_TRAIN_ITERATION)
        writer.write_out_storage()
        gate = step in HISTORICAL_EVAL_STEPS or (step > tracker.anchor_step and (step-tracker.anchor_step) % tracker.interval == 0)
        smoke_done = smoke_steps is not None and step + 1 >= smoke_steps
        if gate or smoke_done:
            result = evaluate(trainer, ready, data, output, step)
            state['evaluations'].append(result)
            if not smoke_done:
                tracker.observe(step, result['metrics']['validation_loss'], result['metrics']['lpips'])
            atomic_json(output / 'metrics.json', state['evaluations'])
        terminal = tracker.stopped or smoke_done
        if gate or terminal or requested_stop or (step > 0 and step % 2000 == 0):
            save(step, terminal=terminal)
        now = time.monotonic()
        if now-last_progress >= 30 or gate or terminal:
            atomic_json(output / 'progress.json', dict(step=step, pid=os.getpid(), time=time.time(),
                        elapsed_seconds=now-start_time, gpu_memory_bytes=torch.cuda.memory_allocated(),
                        gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                        cumulative_point_samples=int(pipe.cumulative_point_samples.item()),
                        plateau=tracker.to_dict(), technical_smoke=smoke_steps is not None))
            print(json.dumps(dict(step=step, elapsed_seconds=round(now-start_time), plateau=tracker.stopped)), flush=True)
            last_progress = now
        if terminal:
            finish(output, spec, state, step, smoke_steps, sampler.audit_state())
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
                             mandatory_end=106316, iteration_limit=None, plateau=PlateauTracker().to_dict()), indent=2))
    else:
        train(args.spec, args.smoke_steps)
