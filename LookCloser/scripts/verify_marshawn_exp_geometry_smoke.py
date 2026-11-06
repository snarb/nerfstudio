"""Read-only real checkpoint/Adam and CUDA private-sampling resume verification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from marshawn_softplus_prefix_candidate import sha256
from marshawn_exp_geometry import ExpGeometry


def inspect_checkpoint(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    state = data['optimizers']['fields']['state']
    if not state:
        raise ValueError('Empty Adam state does not validate optimizer resume')
    steps = sorted({int(v['step']) for v in state.values()})
    for value in state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if not torch.isfinite(value[key]).all():
                raise FloatingPointError('Nonfinite saved Adam moment')
    sampler = data['tuning_runtime']['sampling_state']
    report = dict(checkpoint=str(path), sha256=sha256(path), attempted_step=data['step'],
        adam_entries=len(state), adam_steps=steps, scaler=data['scalers'],
        scheduler=data['schedulers'], prefix_state=sampler,
        model_finite=all(bool(torch.isfinite(v).all()) for v in data['pipeline'].values()
                         if torch.is_tensor(v) and v.is_floating_point()))
    return report


def main(args):
    before, after = inspect_checkpoint(args.before), inspect_checkpoint(args.after)
    if min(after['adam_steps']) <= max(before['adam_steps']) or not before['model_finite'] or not after['model_finite']:
        raise ValueError('Require successful finite optimizer updates on both sides of resume')
    a, b = before['prefix_state'], after['prefix_state']
    for key in ('schema', 'receipt', 'seed', 'weight', 'rays', 'samples', 'rng_policy', 'band_weight', 'band_samples', 'band_seed_offset', 'actual_SH_required', 'native_density'):
        if a[key] != b[key]:
            raise ValueError('Prefix state changed at resume')
    if a['native_density'] != 'trunc_exp_hplus1':
        raise ValueError('Native-exp checkpoint required')
    manifest = json.loads(args.manifest.read_text())
    with np.load(args.manifest.parent/manifest['archive']) as src:
        arrays = {k: src[k].copy() for k in src.files}
    objects = [ExpGeometry(arrays, manifest['aabb'], arrays['camera_names'].tolist(),
        weight=s['weight'], band_weight=s['band_weight'], rays=s['rays'], samples=s['samples'], seed=s['seed'], device='cuda') for s in (a, b)]
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    next_step = before['attempted_step']+1
    x, y = [obj.sample(next_step) for obj in objects]
    if any(not torch.equal(x[key], y[key]) for key in x):
        raise ValueError('Saved private policy does not reproduce next-step CUDA prefix samples')
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise ValueError('Private prior sampler changed global RNG')
    band_x, band_y = [obj.sample_band(next_step) for obj in objects]
    if any(not torch.equal(band_x[key], band_y[key]) for key in band_x):
        raise ValueError('Band private stream differs after resume')
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise ValueError('Band stream changed global RNG')
    sample_hash = hashlib.sha256(x['points'].cpu().numpy().tobytes()).hexdigest()
    renders = []
    for view in ('DC', 'HC', 'KA'):
        p = args.renders/f'{view}.png'; z = np.array(Image.open(p))
        alpha = np.array(Image.open(args.renders/f'{view}_alpha.png'))
        if z.shape != (1012, 1920, 3) or alpha.shape != (1012, 1920):
            raise ValueError('Incomplete technical eval rasters')
        renders.append(dict(view=view, rgb_sha256=sha256(p), rgb_min=int(z.min()), rgb_max=int(z.max()),
                            opacity_min=int(alpha.min()), opacity_max=int(alpha.max())))
    result = dict(kind='REAL_SH_NATIVE_EXP_GEOMETRY_ADAM_RESUME_TECHNICAL_VERIFICATION', passed=True,
        before=before, after=after, cuda_private_next_step=next_step,
        cuda_private_next_samples_sha256=sample_hash, band_private_stream_reproduced=True, global_cpu_cuda_rng_preserved=True,
        renders=renders, quality_claim=False, production_training_started=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, before_adam=before['adam_steps'], after_adam=after['adam_steps'],
                          private_next_step=next_step, output=str(args.output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('before', 'after', 'manifest', 'renders', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); main(a)
