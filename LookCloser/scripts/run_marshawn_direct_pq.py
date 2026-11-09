"""Actual CUDA zero-update preflight for a separately sealed PQ-code control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import threading
import time
import subprocess

ROOT = Path('/fsx/tmp/marshawn_001660_hd')
for p in [ROOT/'code_sfm/LookCloser/scripts', ROOT/'code_native/LookCloser/scripts',
          ROOT/'code_native', ROOT/'final_pair/code', Path(__file__).parent]:
    sys.path.insert(0, str(p))
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def validate(trainer):
    import torch
    p = trainer.pipeline; m = p.model
    assert type(p).__name__ == 'DirectPQNativePipeline' and type(m).__name__ == 'NGPModel'
    assert p.config.final_geometry_variant == 'scene_loose' and p.config.final_grid_radius == 0
    assert m.field.appearance_embedding_dim == 0 and m.config.max_res == 8192
    assert m.config.log2_hashmap_size == 21 and m.config.render_step_size == .003
    assert p.config.datamanager.cache_images_type == 'float32'
    assert len(p.datamanager.train_dataset) == 63 and len(p.datamanager.eval_dataset) == 3
    bounds = torch.tensor(json.loads((ROOT/'geometry/boxes_approved.json').read_text())['scopes']['scene']['loose'])
    for actual in [m.scene_aabb.reshape(2,3), m.field.aabb, m.occupancy_grid.aabbs[0].reshape(2,3),
                   p.datamanager.train_dataparser_outputs.scene_box.aabb,
                   p.datamanager.eval_dataset._dataparser_outputs.scene_box.aabb]:
        torch.testing.assert_close(actual.cpu(), bounds, rtol=0, atol=1e-6)
    out = p.datamanager.train_dataparser_outputs
    torch.testing.assert_close(out.dataparser_transform, torch.eye(4)[:3], rtol=0, atol=1e-7)
    assert out.dataparser_scale == 1.
    torch.testing.assert_close(m.renderer_rgb.background_color.cpu(), torch.full((3,), 988/65535.), rtol=0, atol=0)


def preflight(path):
    import cv2
    import numpy as np
    import torch
    import yaml
    from nerfstudio.scripts.train import _set_random_seed
    from nerfstudio.engine.callbacks import TrainingCallbackLocation
    from staged_native_hash import file_sha
    torch.set_num_threads(2)
    cfg = yaml.load(path.read_text(), Loader=yaml.Loader)
    assert cfg.max_num_iterations == 22000
    assert cfg.load_checkpoint is None and cfg.load_dir is None and cfg.load_step is None
    cfg.output_dir = path.parent/'noopt_probe'
    cfg.output_dir.mkdir(exist_ok=True)
    _set_random_seed(42)
    trainer = cfg.setup(local_rank=0, world_size=1); trainer.setup(); validate(trainer)
    pipe = trainer.pipeline; pipe.train()
    before = parameter_sha(pipe.model)
    ref = ROOT/'final_pair/radius0/marshawn_001660_final_radius0/instant-ngp-bounded/cold_seed42/nerfstudio_models/step-000008000.ckpt'
    refstate = torch.load(ref, map_location='cpu', weights_only=False)['pipeline']
    assert {k:tuple(v.shape) for k,v in pipe.state_dict().items()} == {k:tuple(v.shape) for k,v in refstate.items()}
    del refstate
    callback_names = [getattr(c.func, '__name__', '') for c in trainer.callbacks]
    assert callback_names[-1] == 'guard_after_native_core'
    for c in trainer.callbacks:
        c.run_callback_at_location(0, TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
    captured = {}
    next_train = pipe.datamanager.next_train
    def capture(step):
        rays, batch = next_train(step); captured.update(rays=rays, batch=batch)
        return rays, batch
    pipe.datamanager.next_train = capture
    with torch.autocast(device_type='cuda', enabled=cfg.mixed_precision):
        outputs, losses, _ = pipe.get_train_loss_dict(0)
        total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    gradients = [p.grad for p in pipe.model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert parameter_sha(pipe.model) == before
    assert all(not o.state for o in trainer.optimizers.optimizers.values())
    batch = captured['batch']; inds = batch['indices'][:128].cpu().numpy()
    target = batch['image'][:128].detach().cpu().numpy()
    filenames = pipe.datamanager.train_dataset.image_filenames
    physical = np.empty_like(target)
    for cam in np.unique(inds[:,0]):
        mask = inds[:,0] == cam; img = cv2.imread(str(filenames[cam]), cv2.IMREAD_UNCHANGED)[...,::-1]
        physical[mask] = img[inds[mask,1], inds[mask,2]]
    assert np.array_equal(physical, target), 'EXR code loader/pixel sampler mismatch'
    # Independent calibrated pinhole rays from actual parser K/c2w, y-down pixels.
    cameras = pipe.datamanager.train_dataset.cameras
    c2w = cameras.camera_to_worlds[inds[:,0]].cpu().numpy()
    ci = torch.as_tensor(inds[:,0])
    fx = cameras.fx[ci].cpu().numpy().reshape(-1)
    fy = cameras.fy[ci].cpu().numpy().reshape(-1)
    cx = cameras.cx[ci].cpu().numpy().reshape(-1)
    cy = cameras.cy[ci].cpu().numpy().reshape(-1)
    local = np.stack([(inds[:,2]+.5-cx)/fx, -(inds[:,1]+.5-cy)/fy, -np.ones(len(inds))],-1)
    directions = np.einsum('nij,nj->ni', c2w[:,:,:3], local)
    directions /= np.linalg.norm(directions,axis=-1,keepdims=True)
    actual_rays = captured['rays']
    origin_error = float(abs(actual_rays.origins[:128].detach().cpu().numpy()-c2w[:,:,3]).max())
    direction_error = float(abs(actual_rays.directions[:128].detach().cpu().numpy()-directions).max())
    assert origin_error < 1e-6 and direction_error < 1e-6
    zero = pipe.model.renderer_rgb(torch.full((2,1,3), .7, device='cuda'), torch.zeros((2,1,1),device='cuda'))
    torch.testing.assert_close(zero, torch.full_like(zero, 988/65535.),rtol=0,atol=0)
    np.savez_compressed(path.parent/'sampled_codes128.npz', indices=inds, target=target, physical_EXR=physical,
                        ray_origins=actual_rays.origins[:128].detach().cpu().numpy(),
                        ray_directions=actual_rays.directions[:128].detach().cpu().numpy())
    result = dict(passed=True, optimizer_updates=0, config_sha256=file_sha(path),
        pipeline_source_sha256=cfg.pipeline.direct_pq_source_sha256,
        actual_target_samples=128, numerical_EXR_target_exact=True,
        calibration_origin_max_error=origin_error, calibration_direction_max_error=direction_error,
        finite_combined_native_RGB_LOS_surface_backward=True, native_NN_schema_unchanged=True,
        native_NN_parameters_unchanged=True, renderer_zero_opacity_exact_floor=988/65535.,
        callback_order=callback_names, source_geometry_archive_sha256=pipe._final_free.receipt['archive_sha256'],
        maximum_iterations=22000, first_visual_gate=2000,
        encoded_compositing_not_physical=True, production_training_started=False)
    (path.parent/'actual_cuda_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def train(path):
    """One cold optimizer/RNG stream; pause in place after the first saved gate."""
    import run_final_pair as frozen
    check = validate
    def validate_and_gate(trainer):
        check(trainer)
        original = trainer.pipeline.export_grid_for_checkpoint
        def export_then_gate(checkpoint, step):
            receipt = original(checkpoint, step)
            if int(step) == 2000:
                pending = dict(status='WAITING_VISUAL_REVIEW', worker_pid=os.getpid(),
                    checkpoint=str(checkpoint), grid_receipt=receipt,
                    required='Decode final PQ codes offline; AP1 D60→709; fixed matched display; inspect all3')
                (path.parent/'visual_gate_2000_pending.json').write_text(json.dumps(pending,indent=2)+'\n')
                os.kill(os.getpid(), signal.SIGSTOP)
            return receipt
        trainer.pipeline.export_grid_for_checkpoint = export_then_gate
    frozen.validate = validate_and_gate
    frozen.train(path)


def controller(path, rest):
    """Quiet default plateau stopping, with a durable thirty-second health log."""
    import yaml
    from staged_native_hash import file_sha
    probe = json.loads((path.parent/'actual_cuda_preflight.json').read_text())
    if not probe['passed'] or probe['optimizer_updates'] != 0 or probe['config_sha256'] != file_sha(path):
        raise ValueError('Actual zero-update preflight must match this config')
    peer = json.loads((path.parent/'independent_review.json').read_text())
    if peer.get('passed') is not True:
        raise ValueError('Independent prelaunch review required')
    cfg = yaml.load(path.read_text(), Loader=yaml.Loader)
    import run_bounded_ngp_quiet as quiet
    import run_marshawn_native_ngp as native
    def health():
        while True:
            row = dict(time=time.time(), controller_pid=os.getpid())
            try:
                row['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
                row['children'] = subprocess.check_output(['ps','--ppid',str(os.getpid()),'-o','pid=,stat=,args='],text=True).strip()
                row['oom_in_log'] = any(x in (cfg.output_dir/cfg.experiment_name/cfg.method_name/cfg.timestamp/'train_stdout.log').read_text(errors='replace').lower() for x in ['cuda out of memory','outofmemoryerror']) if (cfg.output_dir/cfg.experiment_name/cfg.method_name/cfg.timestamp/'train_stdout.log').exists() else False
                metrics = cfg.output_dir/cfg.experiment_name/cfg.method_name/cfg.timestamp/'metrics_compact.csv'
                row['latest_compact'] = metrics.read_text().splitlines()[-1] if metrics.exists() else None
            except Exception as exc:
                row['health_error'] = str(exc)
            with (path.parent/'health.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            time.sleep(30)
    threading.Thread(target=health,daemon=True).start()
    quiet.train_command = lambda options:[sys.executable,str(Path(__file__)),'train','--config',str(path)]
    quiet.best_eval_checkpoint = native.select
    # Numerical-code metrics select checkpoints. Correct display is a separate,
    # mandatory decoded-EOTF review; generic ns-eval PNG display is disabled.
    sys.argv = [sys.argv[0], '--output-dir',str(cfg.output_dir),'--experiment-name',cfg.experiment_name,
                '--timestamp',cfg.timestamp,'--max-num-iterations','22000','--step-interval','2000',
                '--no-render-final','--no-update-summary'] + rest
    raise SystemExit(quiet.main())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight', 'controller', 'train'])
    p.add_argument('--config', type=Path, required=True)
    a, rest = p.parse_known_args()
    if a.mode == 'preflight': preflight(a.config)
    elif a.mode == 'controller': controller(a.config, rest)
    else: train(a.config)
