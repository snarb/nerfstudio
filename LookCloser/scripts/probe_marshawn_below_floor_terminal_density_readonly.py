"""Bound, zero-update native density queries on saved fields; no head or renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def verify_inputs(path, release):
    s = json.loads(path.read_text())
    if release['spec_sha256'].get(str(path)) != sha(path):
        raise ValueError('Exact bound density spec required')
    for p, digest in release['bound_files'].items():
        if sha(p) != digest:
            raise ValueError('Bound density input changed: ' + p)
    cp = Path(s['checkpoint'])
    if sha(cp) != s['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    return s


def run(path, execute):
    s = json.loads(path.read_text())
    release_path = Path(s['root_release']) if execute else Path(s['root_release']).with_name('root_review_TEMPLATE.json')
    release = json.loads(release_path.read_text())
    s = verify_inputs(path, release)
    if not execute:
        print(json.dumps(dict(CPU_dry_run=True, all_actual_input_hashes_verified=True, point_count=116,
            native_density_only=True, optimizer_created=False, head_created=False, CUDA_started=False)))
        return
    if release.get('reviewer') != 'root' or release.get('readonly_density_authorized') is not True:
        raise ValueError('Exact root read-only density release required')
    cp = Path(s['checkpoint'])
    sys.path[:0] = [s['source_code'], s['source_scripts']]
    from marshawn_gpu_isolation import require_controller_lease, verify_cuda
    lease = require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import torch
    import numpy as np
    from train_marshawn_semantic_late_distortion import tree_digest
    import nerfstudio.utils.eval_utils as eu
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    verify_cuda(lease['expected_physical_uuid'], torch)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(10 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    out = Path(s['output'])
    out.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Density probe deadline')))
    signal.alarm(300)
    payload = torch.load(cp, map_location='cpu', weights_only=False)
    if payload['step'] != s['step']:
        raise ValueError('Exact selected checkpoint step required')
    expected = tree_digest(payload['pipeline'])
    if expected != s['pipeline_digest']:
        raise ValueError('CPU-bound pipeline digest differs')
    def configure(cfg):
        cfg.eval_checkpoint = cp
        return cfg
    cfg, pipe, actual, step = eu.eval_setup(Path(s['config']), test_mode='inference', update_config_callback=configure)
    m = pipe.model
    pipe.eval()
    pipe.requires_grad_(False)
    m.current_train_step = step
    if Path(actual).resolve() != cp.resolve() or step != s['step'] or tree_digest(pipe.state_dict()) != expected:
        raise ValueError('Strict complete pipeline restore failed')
    c = m.config
    if (m.field.feature_reweighting_strength != .3 or c.feature_reweighting_strength != .3
            or c.max_steps_per_ray != 1024 or c.corrected_arm_allocator
            or c.occupancy_dilation_radius != 0 or c.occupancy_eval_dilation_radius != 0
            or c.adaptive_coarse_step_size != .0015625 or c.distortion_loss_mult != .1
            or m.field.density_activation != 'trunc_exp' or not m.field.correct_sh_directions
            or not getattr(m.field, '_semantic_admission_installed', False)):
        raise ValueError('Original FR.3/legacy1024/native density recipe required')
    lineage = payload.get('below_floor_density_fork_runtime')
    if lineage is not None:
        cache = lineage['model_cache']
        if cache['FR'] != .3:
            raise ValueError('Trained nonpersistent cache FR differs')
        m.field._feature_weight_lut = cache['feature_weight_LUT'].to(m.device)
        m.field._feature_weight_lut_strength = cache['strength']
        pipe._feature_reweighting_switch_applied = cache['switch_applied']
    else:
        if step != 126316 or s['checkpoint_sha256'] != '340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641':
            raise ValueError('Only the immutable source may prime its original LUT')
        with torch.inference_mode():
            m.field._get_discrete_feature_weights(torch.arange(m.field.num_levels,
                device=m.device, dtype=m.freq_grid.grid.dtype).unsqueeze(-1))
    capture = payload['actual_occupancy_capture']
    for old, now in [(capture['core'], m.semantic_learned_core),
                     (capture['active'], m.occupancy_grid.binaries),
                     (payload['pipeline']['_model.occupancy_grid.occs'], m.occupancy_grid.occs)]:
        if not torch.equal(old.reshape(-1).cpu(), now.reshape(-1).cpu()):
            raise ValueError('Actual saved core/active/EMA restore differs')
    _restore_rng_state(payload['rng_state'])
    normalizer = m.field._normalize_positions
    def state():
        return dict(pipeline=tree_digest(pipe.state_dict()), RNG=tree_digest(_capture_rng_state()),
            LUT=tree_digest((m.field._feature_weight_lut, m.field._feature_weight_lut_strength)),
            admission=tree_digest((m.semantic_admission, m.semantic_lookup.state_dict())),
            original_private_state=tree_digest(lineage['below_floor_prior_state']) if lineage else None,
            FR=m.field.feature_reweighting_strength)
    before = state()
    with np.load(s['points'], allow_pickle=False) as z:
        xyz = z['xyz_float32'].copy()
        world_to_box = z['world_to_box'].copy()
    if xyz.shape != (116, 3) or xyz.dtype != np.float32 or not np.array_equal(world_to_box, np.eye(4)):
        raise ValueError('Exact 7+109 floor-I points required')
    # Direct density_fn preserves the original source admission; no field selector,
    # support gate, ray sampler, floor carve, optimizer or semantic head is installed.
    with torch.inference_mode():
        sigma = m.field.density_fn(torch.from_numpy(xyz).to(m.device)).reshape(-1)
    if sigma.numel() != 116 or not bool(torch.isfinite(sigma).all()) or bool((sigma < 0).any()):
        raise ValueError('Finite nonnegative native density required')
    actual_sigma = sigma.float().cpu().numpy()
    after = state()
    if after != before or normalizer is not m.field._normalize_positions:
        raise ValueError('Read-only density query changed model/RNG/cache/admission')
    if sha(cp) != s['checkpoint_sha256']:
        raise ValueError('Disk checkpoint changed')
    np.savez_compressed(out / 'native_density.npz', xyz_float32=xyz, sigma=actual_sigma, world_to_box=world_to_box)
    report = dict(passed=True, spec_sha256=sha(path), checkpoint_sha256=sha(cp), source_step=step,
        pipeline_digest=expected, before=before, after=after, state_unchanged=True,
        actual_native_density_fn=True, field_selector_installed=False, native_admission_unchanged=True,
        floor_transform='IDENTITY; input points already source floor frame',
        points_sha256=sha(s['points']), results_sha256=sha(out / 'native_density.npz'),
        optimizer_created=False, optimizer_updates=0, private_random_queries=0, semantic_head_created=False,
        no_optical_contribution_or_free_space_claim=True, quality_promotion=False)
    (out / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
    signal.alarm(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    run(args.spec, args.execute)
