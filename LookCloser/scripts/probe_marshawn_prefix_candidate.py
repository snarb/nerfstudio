"""Memory-capped real-field AMP gradient probe; zero optimizer updates.

Shares a GPU only when separately authorized. Timing is not a benchmark.
Technical field setup and a private pixel draw never touch another worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random


def digest_state(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(args):
    import numpy as np
    import torch
    from train_marshawn_prefix import build_config, validate_spec
    from marshawn_tuning_sampling import install_sampling
    from marshawn_softplus_prefix_candidate import install_prefix, sha256
    from nerfstudio.engine.trainer import _capture_rng_state, _restore_rng_state
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if args.memory_cap_gib <= 0 or args.memory_cap_gib*2**30 > .95*total_memory:
        raise ValueError('Explicit probe memory cap must leave at least5% device reserve')
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*2**30/total_memory)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = validate_spec(json.loads(args.spec.read_text()))
    output = args.output; output.mkdir(parents=True, exist_ok=False)
    # This private probe has no resume optimizer and no training output directory.
    spec = dict(spec, output=str(output/'setup'))
    cfg = build_config(spec)
    trainer = cfg.setup(local_rank=0, world_size=1); trainer.setup()
    pipe, model = trainer.pipeline, trainer.pipeline.model
    checkpoint_sha = None
    step = 0
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        pipe.load_pipeline(payload['pipeline'], payload['step'])
        step = int(payload['step']); checkpoint_sha = sha256(args.checkpoint)
        del payload
        model.field.feature_reweighting_strength = 1. if step <= 75940 else .3
    else:
        # Fresh model's exact configured binary warmup policy, no density fit.
        model._postprocess_occupancy_grid(step=0)
    model.train()
    ready = json.loads((Path(spec['data'])/'ready.json').read_text())
    sampler = install_sampling(pipe, Path(spec['data']), 'full', 'body80', spec['width'])
    combined = install_prefix(pipe, sampler, spec, ready)
    sampler.set_step(step)
    pipe.datamanager.train_pixel_sampler.set_num_rays_per_batch(args.rays)
    model.current_train_step = step
    before = digest_state(model)
    parameters = list(model.field.encoding.parameters())+list(model.field.mlp_geo.parameters())
    rng = _capture_rng_state(); sampler_state = sampler.state_dict()
    rows = []
    try:
        # Direct real model forward avoids unrelated pipeline frequency/optimizer updates.
        ray_bundle, batch = pipe.datamanager.next_train(step)
        with torch.autocast(device_type='cuda', enabled=trainer.mixed_precision):
            outputs = model(ray_bundle)
            rgb = model.get_loss_dict(outputs, batch)['rgb_loss']
            combined.objective.weight = 1.
            prefix, diagnostic = combined.objective(model.field, step)
        scale = float(trainer.grad_scaler.get_scale())
        initial_scale = scale
        attempts = []
        while True:
            rgb_grads = torch.autograd.grad(rgb*scale, parameters, retain_graph=True, allow_unused=True)
            aux_grads = torch.autograd.grad(prefix*scale, parameters, retain_graph=True, allow_unused=True)
            finite = all(g is None or bool(torch.isfinite(g).all()) for g in (*rgb_grads, *aux_grads))
            attempts.append(dict(scale=scale, finite=finite))
            if finite or scale <= 1024:
                break
            del rgb_grads, aux_grads
            scale /= 2
        if not finite:
            raise FloatingPointError('No finite actual AMP geometry gradients down to scale1024')
        def norm(grads):
            return float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt()/scale)
        rgb_norm, prefix_norm = norm(rgb_grads), norm(aux_grads)
        for weight in (.001, .003, .01):
            finite_sum = all(bool(torch.isfinite((a if a is not None else 0)+weight*(b if b is not None else 0)).all())
                             for a, b in zip(rgb_grads, aux_grads) if a is not None or b is not None)
            rows.append(dict(weight=weight, auxiliary_to_rgb_geometry_norm=weight*prefix_norm/max(rgb_norm, 1e-30),
                             combined_gradients_finite=finite_sum))
        after = digest_state(model)
        if before != after or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Zero-update probe altered model state or accumulated parameter gradients')
        fitting = [r for r in rows if .05 <= r['auxiliary_to_rgb_geometry_norm'] <= .10]
        report = dict(kind='NATIVE_SOFTPLUS_PREFIX_AMP_ZERO_UPDATE', passed=True,
            spec_sha256=sha256(args.spec), manifest_sha256=spec['prefix_candidate']['manifest_sha256'],
            probe_script_sha256=sha256(__file__),
            candidate_helper_sha256=sha256(Path(__file__).parent/'marshawn_softplus_prefix_candidate.py'),
            checkpoint_sha256=checkpoint_sha, checkpoint_step=step, model_before_after_sha256=before,
            density_activation=model.field.density_activation, FR=model.field.feature_reweighting_strength,
            actual_rgb_rays=len(ray_bundle.origins), prior_rays=128, prior_samples_per_ray=64,
            gradient_parameters='hash encoding + geometry MLP; color head excluded',
            rgb_geometry_norm=rgb_norm, unit_prefix_geometry_norm=prefix_norm,
            scale_attempts=attempts, production_initial_scale=initial_scale,
            scanned=rows, weights_in_5_to_10_percent_band=[r['weight'] for r in fitting],
            optical_thickness_mean=float(diagnostic['prefix_tau_mean']),
            optimizer_updates=0, production_modified=False, parameter_grads_unset=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            GPU_overlap_authorized=args.overlap, walltime_is_benchmark=False,
            explicit_memory_cap_gib=args.memory_cap_gib,
            limitations=['Technical gradient diagnostic; quality requires separate root-reviewed training.',
                'Small batches require a full production batch repeat before release.' if args.rays < 4096 else 'Full4096ray RGB packet used.',
                'Gradient ratio is diagnostic, not a demonstrated quality improvement.',
                'A finite reduced AMP scale is a probe only; production scaler behavior remains unchanged.'])
        (output/'probe.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        sampler.load_state_dict(sampler_state); _restore_rng_state(rng)
        trainer.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--rays', type=int, default=128)
    p.add_argument('--memory-cap-gib', type=float, default=10.)
    p.add_argument('--overlap', action='store_true')
    a = p.parse_args(); main(a)
