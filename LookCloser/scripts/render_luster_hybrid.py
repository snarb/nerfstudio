"""Read-only native ARM inference with explicit camera-z termination quantiles."""
from __future__ import annotations
import argparse
import hashlib
import inspect
from pathlib import Path
import time

import numpy as np
import torch

from luster_hybrid_common import (CHECKPOINT, DIGEST, ORIGIN, TARGETS, bootstrap,
                                  image, log, native_rows, read, sha, write)


def packed_quantiles(outputs, rays, quantiles=(.1, .5, .9)):
    """Invert each ray's conditional termination CDF inside physical sample bins.

    Global float64 cumulative sums avoid cross-ray cancellation in long chunks.
    The model's RGB/weights are never modified. Empty rays return NaN.
    """
    indices = outputs['packed_ray_indices'].long()
    weights = outputs['packed_weights'].flatten().double()
    n = len(rays)
    result = torch.full((n, len(quantiles)), torch.nan, device=weights.device)
    if len(weights) == 0:
        return result
    counts = torch.bincount(indices, minlength=n)
    offsets = counts.cumsum(0) - counts
    cum = weights.cumsum(0)
    before_ray = torch.cat([cum.new_zeros(1), cum])[offsets]
    end = (offsets + counts - 1).clamp_min(0)
    total = torch.where(counts > 0, cum[end] - before_ray, 0)
    valid = (counts > 0) & (total >= .95)
    ids = torch.where(valid)[0]
    if not len(ids):
        return result
    q = torch.tensor(quantiles, device=weights.device, dtype=torch.float64)
    desired = before_ray[ids, None] + total[ids, None] * q
    pos = torch.searchsorted(cum, desired.flatten()).reshape(-1, len(q))
    pos = torch.minimum(torch.maximum(pos, offsets[ids, None]), end[ids, None])
    before = torch.cat([cum.new_zeros(1), cum])[pos] - before_ray[ids, None]
    mass = weights[pos]
    transmission = (1 - before).clamp_min(1e-12)
    optical = -torch.log1p(-(mass / transmission).clamp(0, 1-1e-12))
    partial = ((total[ids, None]*q-before)/transmission).clamp(0, 1-1e-12)
    fraction = torch.where(optical > 1e-12, -torch.log1p(-partial)/optical.clamp_min(1e-12), .5).clamp(0,1)
    starts = outputs['packed_spacing_starts'].flatten()[pos]
    ends = outputs['packed_spacing_ends'].flatten()[pos]
    fraction_t = starts + fraction * (ends-starts)
    near = rays.nears.flatten()[ids, None]
    span = rays.fars.flatten()[ids, None] - near
    result[ids] = (near + fraction_t * span).float()
    return result


def state_hashes(model):
    # Includes parameters and buffers, including occupancy/frequency grids.
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load(root, chunk):
    runtime = bootstrap(root)
    from nerfstudio.utils.eval_utils import eval_setup
    def update(cfg):
        cfg.eval_checkpoint = CHECKPOINT
        return cfg
    cfg, pipeline, loaded, step = eval_setup(CHECKPOINT.parent.parent/'config.yml',
            eval_num_rays_per_chunk=chunk, test_mode='inference', update_config_callback=update)
    assert loaded.resolve() == CHECKPOINT.resolve() and step == 106316
    m = pipeline.model.eval()
    assert Path(inspect.getfile(type(m))).is_relative_to(runtime)
    assert m._resolved_ray_sampling_mode() == 'adaptive'
    assert read(cfg.pipeline.datamanager.dataparser.data/'transforms.json') == read(ORIGIN/'data/unmasked/transforms.json')
    box = read(ORIGIN/'preflight/sequence_aabb.json')
    expected = torch.tensor([box['min'],box['max']], device=m.device)
    bounds = [m.scene_box.aabb, m.field.aabb, m.collider.scene_box.aabb,
              torch.stack([m.freq_grid.aabb_min_buf,m.freq_grid.aabb_max_buf]),
              m.occupancy_grid.aabbs[0].reshape(2,3),m.scene_aabb.reshape(2,3)]
    assert all(torch.allclose(v.to(expected),expected,atol=1e-6,rtol=0) for v in bounds)
    return cfg, m


def run(args):
    root = args.root
    cfg, model = load(root, args.chunk)
    from render_luster_quality import camera_for, measured_plate
    before = state_hashes(model)
    rows = native_rows()
    for cid in args.cameras:
        row = rows[cid]
        out = root/'geometry'/f'cam{cid:03d}'
        if (out/'receipt.json').exists():
            receipt = read(out/'receipt.json')
            assert receipt['checkpoint_sha256'] == DIGEST and receipt['state_hash'] == before
            assert receipt['code_sha256'] == sha(Path(__file__))
            assert all(sha(out/f'{k}.npy') == receipt['hashes'][k] for k in receipt['arrays'])
            log(root, 'render_cached', camera=cid)
            continue
        out.mkdir(parents=True, exist_ok=True)
        h,w = row['h'],row['w']
        arrays = {k: np.lib.format.open_memmap(out/f'{k}.npy', mode='w+', dtype=np.float32,
                  shape=(h,w,3) if k=='rgb' else (h,w)) for k in
                  ['rgb','alpha','ray_distance_mean','depth_z_q10','depth_z_q50','depth_z_q90']}
        camera = camera_for(row,model.device)
        plate, plate_receipt = measured_plate(cfg,row)
        started = last = time.monotonic()
        with torch.inference_mode():
            for start in range(0,h*w,args.chunk):
                ids = torch.arange(start,min(start+args.chunk,h*w),device=model.device)
                coords = torch.stack([ids//w,ids%w],-1).float()+.5
                rays = model.collider(camera.generate_rays(camera_indices=0,coords=coords))
                outputs = model.get_outputs(rays)
                q = packed_quantiles(outputs,rays)
                z_scale = -(rays.directions @ camera.camera_to_worlds[0,:3,:3])[:,2]
                q *= z_scale[:,None]
                a = outputs['accumulation'].float().cpu().numpy().reshape(-1)
                rgb = outputs['rgb'].float().cpu().numpy()
                if plate is not None:
                    rgb += (1-a[:,None])*plate.reshape(-1,3)[start:start+len(ids)]
                values = dict(rgb=rgb, alpha=a, ray_distance_mean=outputs['depth'].float().cpu().numpy().reshape(-1),
                              **{f'depth_z_q{k}':q[:,i].cpu().numpy() for i,k in enumerate((10,50,90))})
                for key, value in values.items():
                    arrays[key].reshape((-1,3) if key=='rgb' else (-1,))[start:start+len(ids)] = value
                if time.monotonic()-last>45:
                    log(root,'render_progress',camera=cid,pixels=start+len(ids),total=h*w,
                        elapsed=time.monotonic()-started,gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9)
                    last=time.monotonic()
        for v in arrays.values():v.flush()
        after=state_hashes(model)
        assert before==after, 'Inference changed model state'
        if cid in TARGETS:
            image(root/'baseline'/f'cam{cid:03d}.png',arrays['rgb'])
        receipt=dict(state='complete',camera=cid,row=row,arrays=list(arrays),
                     checkpoint_sha256=DIGEST,state_hash=before,code_sha256=sha(Path(__file__)),
                     hashes={k:sha(out/f'{k}.npy') for k in arrays},background=plate_receipt,
                     depth='conditional termination quantiles in camera-z; separate mean ray distance',
                     chunk=args.chunk,seconds=time.monotonic()-started)
        write(out/'receipt.json',receipt)
        log(root,'render_complete',camera=cid,seconds=receipt['seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cameras',type=int,nargs='+',default=list(TARGETS))
    p.add_argument('--chunk',type=int,default=4096)
    run(p.parse_args())
