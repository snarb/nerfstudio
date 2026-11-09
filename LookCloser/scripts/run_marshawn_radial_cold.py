"""Gated cold radial PNG control. Preparation/dry-run never constructs a trainer."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path('/fsx/tmp/marshawn_001660_hd')
for d in [ROOT/'code_sfm/LookCloser/scripts', ROOT/'code_native/LookCloser/scripts',
          ROOT/'code_native', ROOT/'final_pair/code', Path(__file__).parent]:
    sys.path.insert(0, str(d))
from staged_native_hash import file_sha


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)


def load(path, spec_path, production=False):
    import yaml
    spec = json.loads(Path(spec_path).read_text())
    if file_sha(path) != spec['config_sha256']:
        raise ValueError('Config changed')
    for p, digest in spec['sources_sha256'].items():
        if file_sha(p) != digest:
            raise ValueError('Bound source changed '+p)
    cfg = yaml.load(Path(path).read_text(), Loader=yaml.Loader)
    if any([cfg.load_checkpoint, cfg.load_dir, cfg.load_config, cfg.eval_checkpoint,
            cfg.load_step is not None]):
        raise ValueError('Cold control must not load any checkpoint')
    if cfg.max_num_iterations != 22000 or cfg.machine.seed != 42:
        raise ValueError('Cold seed42 and absolute cap22000 required')
    p = cfg.pipeline
    if p._target.__name__ != 'NativeRadialColdPipeline' or p.final_grid_radius != 0:
        raise ValueError('Native radial cold/core0 required')
    if p.target_num_samples != 1048576 or p.max_num_samples_per_ray != 16384:
        raise ValueError('Original independent dynamic ray policy required')
    if p.datamanager.pixel_sampler._target.__name__ != 'PixelSampler':
        raise ValueError('No patch sampler')
    if p.model.near_plane != .052 or p.model.background_color != 'random':
        raise ValueError('Native random-background loss and near.052 required')
    if (p.packed_distortion_mode, p.packed_distortion_weight, p.MVS_depth_weight) != ('linear', .01, .01):
        raise ValueError('Only reviewed explicit auxiliary weights')
    if any(hasattr(p, k) for k in ['floor_band_prior_manifest', 'floor_band_prior_weight']):
        raise ValueError('Assumed floor prior must be absent')
    if cfg.optimizers['fields']['scheduler'].max_steps != 22000:
        raise ValueError('Cold native22k scheduler required')
    if [cfg.steps_per_save, cfg.steps_per_eval_batch, cfg.steps_per_eval_image,
        cfg.steps_per_eval_all_images] != [2000]*4:
        raise ValueError('All3 eval/save2k required')
    if production and not all(spec['release_checks'].values()):
        raise ValueError('Data, actual zeroopt, peer and root training gates required')
    return cfg, spec


def exact_eval_config(config, checkpoint, eval_num_rays_per_chunk=None):
    import yaml
    cfg = yaml.load(Path(config).read_text(), Loader=yaml.Loader)
    cfg.load_checkpoint = cfg.load_dir = cfg.load_step = cfg.load_config = None
    cfg.eval_checkpoint = Path(checkpoint).resolve()
    if eval_num_rays_per_chunk is not None:
        cfg.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk
    out = Path(config).with_name('eval_exact_'+Path(checkpoint).stem+'.yml')
    out.write_text(yaml.dump(cfg))
    return out


def choose(metrics, directory, spec):
    rows = []
    if Path(metrics).exists():
        for row in csv.DictReader(Path(metrics).open()):
            try:
                step = int(row['step'])
                values = [float(row[k]) for k in ['eval_all_psnr', 'eval_all_ssim', 'eval_all_lpips']]
            except (KeyError, ValueError, TypeError):
                continue
            checkpoint = Path(directory)/('step-%09d.ckpt' % step)
            if 0 <= step < 22000 and checkpoint.exists() and all(map(math.isfinite, values)):
                rows.append(dict(step=step, checkpoint=str(checkpoint), psnr=values[0], ssim=values[1], lpips=values[2]))
    terminal = Path(directory)/'step-000021999.ckpt'
    if terminal.exists() and not any(r['step'] == 21999 for r in rows):
        import run_bounded_ngp_quiet as quiet
        result = quiet.run_final_eval(Path(directory).parent, terminal, 'terminal_rank',
                                      eval_num_rays_per_chunk=512)
        if Path(result['checkpoint']).resolve() != terminal.resolve():
            raise ValueError('Wrong terminal checkpoint')
        values = result['results']
        rows.append(dict(step=21999, checkpoint=str(terminal),
                         **{k: float(values[k]) for k in ['psnr', 'ssim', 'lpips']}))
    if not rows:
        raise ValueError('No complete all3 checkpoint evaluation')
    best = max(r['psnr'] for r in rows)
    winner = min((r for r in rows if r['psnr'] >= best-.07-1e-12),
                 key=lambda r: (r['lpips'], -r['psnr']))
    atomic(spec['selection_receipt'], dict(candidates=rows, winner=winner,
        rule='all3PSNR; LPIPS tie within.07dB', GT_domain='NEW radial fixedk4+joint scalar PNG'))
    return Path(winner['checkpoint']), 'all3_psnr_lpips_tie_0.07'


def train(path, spec_path):
    import torch
    import nerfstudio.scripts.train as native
    cfg, spec = load(path, spec_path, production=True)
    def loop(local_rank, world_size, config, global_rank=0):
        native._set_random_seed(config.machine.seed+global_rank)
        trainer = config.setup(local_rank=local_rank, world_size=world_size)
        trainer.setup()
        # Native geometry/architecture guards; original validator expects the
        # parent class name, so check bounds explicitly without class deception.
        p, m = trainer.pipeline, trainer.pipeline.model
        assert type(p).__name__ == 'NativeRadialColdPipeline' and type(m).__name__ == 'NGPModel'
        assert m.field.appearance_embedding_dim == 0
        assert (m.config.render_step_size, m.config.max_res, m.config.log2_hashmap_size) == (.003, 8192, 21)
        assert m.renderer_rgb.background_color == 'random' and m.config.near_plane == .052
        assert trainer._start_step == 0 and not any(o.state for o in trainer.optimizers.optimizers.values())
        assert len(p.datamanager.train_dataset) == 63 and len(p.datamanager.eval_dataset) == 3
        parsed = p.datamanager.train_dataparser_outputs
        torch.testing.assert_close(parsed.dataparser_transform, torch.eye(4)[:3], rtol=0, atol=1e-7)
        assert parsed.dataparser_scale == 1
        expected = torch.tensor(json.loads((ROOT/'geometry/boxes_approved.json').read_text())['scopes']['scene']['loose'])
        for box in [m.scene_aabb.reshape(2, 3), m.field.aabb, m.occupancy_grid.aabbs[0].reshape(2, 3),
                    p.datamanager.train_dataparser_outputs.scene_box.aabb,
                    p.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb]:
            torch.testing.assert_close(box.cpu(), expected, rtol=0, atol=1e-6)
        folder = Path(config.output_dir)
        atomic(folder/'worker.json', dict(pid=os.getpid(), time=time.time(), cold_seed=42,
            start_step=0, end_exclusive=22000, max_attempts=22000, no_checkpoint_loaded=True))
        original_save = trainer.save_checkpoint
        def save(step):
            original_save(step)
            p.export_grid_for_checkpoint(trainer.checkpoint_dir/('step-%09d.ckpt' % step), step)
        trainer.save_checkpoint = save
        original_eval = trainer.eval_iteration
        def protected_eval(step):
            if step > 0 and step % 2000 == 0:
                trainer.save_checkpoint(step)
            return original_eval(step)
        trainer.eval_iteration = protected_eval
        from marshawn_crash_capture import TrainingCrashCapture
        capture = TrainingCrashCapture(trainer, folder/'failure_diagnostics', phase_start=0).install()
        original = trainer.train_iteration
        paused = [False]
        def observed(step):
            if step >= 22000:
                raise ValueError('Hard attempt guard: no update at step22000 or later')
            if step == 2001 and not paused[0]:
                checkpoint = trainer.checkpoint_dir/'step-000002000.ckpt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                assert saved['step'] == 2000
                assert all(bool(torch.isfinite(v).all()) for v in saved['pipeline'].values() if torch.is_tensor(v))
                atomic(folder/'gate_2000.json', dict(step=2000, next_step=2001, attempts=2001,
                    worker_pid=os.getpid(), checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint),
                    policy='SIGSTOP same worker BEFORE2001; root review before CONT'))
                del saved
                paused[0] = True
                os.kill(os.getpid(), signal.SIGSTOP)
            result = original(step)
            if step < 64 or step % 500 == 0:
                count = max([int(s['step']) for o in trainer.optimizers.optimizers.values()
                             for s in o.state.values() if 'step' in s] or [0])
                row = dict(step=step, attempts=step+1, adam_updates=count, skipped_attempts=step+1-count,
                           scale=float(trainer.grad_scaler.get_scale()),
                           lr=float(trainer.optimizers.optimizers['fields'].param_groups[0]['lr']),
                           scalar_finite=bool(torch.isfinite(result[0].detach())))
                if step < 64 or step % 2000 == 0:
                    row['weights_finite'] = all(bool(torch.isfinite(v).all()) for v in p.model.parameters())
                with (folder/'amp_optimizer_trace.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
            return result
        trainer.train_iteration = observed
        try:
            trainer.train()
        finally:
            capture.close()
    native.train_loop = loop
    native.main(cfg)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['controller', 'train'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    args, rest = p.parse_known_args()
    if args.mode == 'train':
        train(args.config, args.spec)
    else:
        cfg, spec = load(args.config, args.spec, production='--dry-run' not in rest)
        import run_bounded_ngp_quiet as quiet
        from run_marshawn_png_distortion_control import health_loop
        from marshawn_all_view_early_stop import configure_quiet_all_view_stop
        quiet.train_command = lambda _: [sys.executable, str(Path(__file__)), 'train',
            '--config', str(args.config), '--spec', str(args.spec)]
        quiet.run_dir = lambda _: cfg.get_base_dir()
        configure_quiet_all_view_stop(quiet, lambda m, d: choose(m, d, spec))
        quiet.eval_config_for_step = exact_eval_config
        old_eval = quiet.run_final_eval
        def checked(*a, **kw):
            result = old_eval(*a, **kw)
            if Path(result['checkpoint']).resolve() != Path(a[1]).resolve():
                raise ValueError('Final evaluator loaded a different checkpoint')
            return result
        quiet.run_final_eval = checked
        sys.argv = [sys.argv[0]]+rest
        os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', '')
        if '--dry-run' in rest:
            raise SystemExit(quiet.main())
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        stop = threading.Event()
        thread = threading.Thread(target=health_loop, args=(cfg, stop), daemon=True)
        thread.start()
        try:
            result = quiet.main()
        finally:
            stop.set()
            thread.join(timeout=2)
        raise SystemExit(result)
