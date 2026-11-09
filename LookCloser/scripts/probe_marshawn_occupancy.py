#!/usr/bin/env python3
"""Frozen calibrated-pixel density trace: native / halo1 / all cells / uniform.

Run in the nerfstudio GPU environment with --config, --checkpoint, --queries JSON,
--output. Query: {"split":"eval","pixels":[{"camera":0,"x":500,"y":300}]}.
x/y are integer raster pixel indices; rays pass through pixel centers. Native,
halo1 and all_cells preserve ARM and cap settings. Uniform references bypass
occupancy and sample caps; agreement between N and 2N must be reviewed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def integrate(starts, ends, sigma):
    starts, ends, sigma = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (starts, ends, sigma)]
    if len(starts) != len(ends) or len(starts) != len(sigma) or not np.isfinite([starts, ends, sigma]).all() or (ends < starts).any() or (sigma < 0).any():
        raise ValueError('Finite equal-length intervals and nonnegative sigma required')
    if len(starts) > 1 and ((starts[1:] < starts[:-1]).any() or (starts[1:] < ends[:-1]-1e-6).any()):
        raise ValueError('Samples must be ordered and nonoverlapping')
    optical = sigma*(ends-starts)
    alpha = -np.expm1(-optical)
    trans = np.exp(-np.concatenate(([0.], np.cumsum(optical)[:-1]))) if len(optical) else np.empty(0)
    return alpha, trans, alpha*trans




def occupancy_profiles(native, occs, source_radius, step, binary_warmup, cap=.01, mean_mult=1.):
    from marshawn_occupancy_audit import dilate_one
    native, occs = np.asarray(native), np.asarray(occs)
    if source_radius not in (0, 1):
        raise ValueError('Only source occupancy dilation 0/1 is supported')
    if native.dtype != np.bool_ or occs.size != native.size or not np.isfinite(occs).all():
        raise ValueError('Invalid native occupancy/EMA')
    threshold = min(float(occs.mean())*mean_mult, cap)
    core = (occs > threshold).reshape(native.shape)
    warmup = step < binary_warmup
    if warmup:
        if not native.all():
            raise ValueError('Binary warmup expects every cell admitted')
        return {'native': native, 'all_cells': np.ones_like(native)}, dict(source_radius=source_radius, binary_warmup_active=True, halo1_available=False, reason='All-cell warmup mask has no reconstructible one-cell halo comparison', ema_threshold=threshold)
    halo1 = dilate_one(core)
    expected = halo1 if source_radius else core
    if not np.array_equal(native, expected):
        raise ValueError('Native mask disagrees with EMA core and declared dilation; unsupported support/admission or runtime state')
    return {'native': native, 'base': core, 'halo1': halo1, 'all_cells': np.ones_like(native)}, dict(source_radius=source_radius, binary_warmup_active=False, halo1_available=True, native_matches_declared_policy=True, ema_threshold=threshold, base_occupied=int(core.sum()), halo1_occupied=int(halo1.sum()), halo1_definition='Exactly one non-cumulative cube dilation from undilated EMA core')


def assert_consistent_bounds(model, atol=1e-6):
    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float64)
        if result.size != 6 or not np.isfinite(result).all():
            raise ValueError('Invalid live AABB')
        result = result.reshape(2, 3)
        if (result[1] <= result[0]).any():
            raise ValueError('Live AABB must have positive extent')
        return result
    bounds = {'model.scene_aabb': array(model.scene_aabb),
              'model.scene_box.aabb': array(model.scene_box.aabb),
              'occupancy.aabbs[0]': array(model.occupancy_grid.aabbs[0])}
    if getattr(model.field, 'aabb', None) is not None:
        bounds['field.aabb'] = array(model.field.aabb)
    collider = getattr(model, 'collider', None)
    if collider is None or getattr(collider, 'scene_box', None) is None:
        raise ValueError('AABB collider required for bounded audit')
    bounds['collider.scene_box.aabb'] = array(collider.scene_box.aabb)
    reference = bounds['model.scene_aabb']
    mismatches = [key for key, value in bounds.items() if not np.allclose(reference, value, rtol=0, atol=atol)]
    if mismatches:
        raise ValueError('Live AABB mismatch: '+', '.join(mismatches)+'; check config/checkpoint agreement')
    return {key: value.tolist() for key, value in bounds.items()}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_model(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'checkpoint', 'queries', 'output']:
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--uniform-samples', type=int, default=8192)
    a = p.parse_args()
    if not 16 <= a.uniform_samples <= 131072:
        p.error('--uniform-samples must be between 16 and 131072')
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.field_components.field_heads import FieldHeadNames
    from marshawn_occupancy_audit import dilate_one, lookup, ray_intervals
    def configure(cfg):
        cfg.eval_checkpoint = a.checkpoint.resolve()
        cfg.pipeline.model.occupancy_eval_dilation_radius = 0
        if a.data:
            cfg.pipeline.datamanager.data = a.data
            cfg.pipeline.datamanager.dataparser.data = a.data
        return cfg
    cfg, pipe, checkpoint, step = eval_setup(a.config, test_mode='val', update_config_callback=configure)
    model = pipe.model
    model.current_train_step = step
    live_bounds = assert_consistent_bounds(model)
    if model._resolved_ray_sampling_mode() != 'adaptive':
        raise ValueError('This comparison requires adaptive ray marching')
    if getattr(model.config, 'occupancy_prior_grid_path', None) is not None:
        raise ValueError('Disable occupancy admission priors for an isolated mask comparison')
    if getattr(model.config, 'geometry_support_enabled', False):
        raise ValueError('Disable geometry support for an isolated EMA mask comparison')
    if getattr(model, '_eval_occupancy_backup', None) is not None:
        raise ValueError('Expected undilated loaded model')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    runtime = state.get('runtime', {})
    if 'feature_reweighting_strength' in runtime:
        model.config.feature_reweighting_strength = float(runtime['feature_reweighting_strength'])
        model.field.feature_reweighting_strength = model.config.feature_reweighting_strength
    request = json.loads(a.queries.read_text())
    split = request.get('split', 'eval')
    if split not in ['train', 'eval']:
        raise ValueError('split must be train or eval')
    ds = getattr(pipe.datamanager, split+'_dataset')
    cameras = ds.cameras.to(model.device)
    mask = model.occupancy_grid.binaries.clone()
    boxes = model.occupancy_grid.aabbs.detach().cpu().numpy()
    native = mask.cpu().numpy()
    numpy_profiles, occupancy_policy = occupancy_profiles(native, model.occupancy_grid.occs.detach().cpu().numpy(), int(model.config.occupancy_dilation_radius), step, int(model.config.occupancy_binary_warmup_steps), float(model.config.occupancy_occ_thre), float(model.config.occupancy_thre_clamp_mult))
    profiles = {name: torch.from_numpy(value).to(mask.device) for name,value in numpy_profiles.items()}
    before = digest_model(model)
    import yaml
    parser_config = yaml.dump(cfg.pipeline.datamanager.dataparser, sort_keys=True)
    parser_data = Path(cfg.pipeline.datamanager.dataparser.data)
    transforms_path = parser_data if parser_data.is_file() else parser_data/'transforms.json'
    transforms_sha = file_sha(transforms_path)
    parser_outputs = pipe.datamanager.train_dataparser_outputs
    normalization = dict(transform=parser_outputs.dataparser_transform.tolist(), scale=float(parser_outputs.dataparser_scale))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'dataparser_config.yml').write_text(parser_config)
    rows = []
    try:
        for i, pixel in enumerate(request['pixels']):
            camera, x, y = [pixel[k] for k in ['camera', 'x', 'y']]
            if not all(isinstance(v, int) for v in [camera, x, y]) or not 0 <= camera < len(cameras):
                raise ValueError('camera/x/y must be valid integer indices')
            if not (0 <= x < int(cameras.width[camera]) and 0 <= y < int(cameras.height[camera])):
                raise ValueError('Pixel outside calibrated raster')
            rays = cameras[camera:camera+1].generate_rays(0, coords=torch.tensor([[y+.5, x+.5]], device=model.device), keep_shape=False)
            rays = model.collider(rays) if model.collider else rays
            origin, direction = [v[0].detach().cpu().numpy() for v in [rays.origins, rays.directions]]
            near, far = float(rays.nears.item()), float(rays.fars.item())
            if not np.isfinite([near, far]).all() or far <= near:
                raise ValueError('Pixel ray has no finite positive AABB interval')
            row = dict(pixel=pixel, filename=str(ds.image_filenames[camera]), near=near, far=far, origin=origin.tolist(), direction=direction.tolist(), profiles={})
            for name in list(profiles)+['uniform', 'uniform2x']:
                captured = []
                def capture(module, inputs, outputs):
                    samples = inputs[0]
                    captured.append((samples.frustums.starts, samples.frustums.ends, outputs[FieldHeadNames.DENSITY], outputs[FieldHeadNames.RGB]))
                if name in profiles:
                    model.occupancy_grid.binaries.copy_(profiles[name])
                    hook = model.field.register_forward_hook(capture)
                    try:
                        with torch.no_grad():
                            result = model(rays)
                    finally:
                        hook.remove()
                    if len(captured) > 1:
                        raise ValueError('Unsupported multiple field calls; trace would be ambiguous')
                    if captured:
                        starts, ends, sigma, rgb = [v.detach().float().cpu().numpy().reshape(-1, 3 if j == 3 else 1) for j, v in enumerate(captured[0])]
                    else:
                        starts = ends = sigma = np.empty(0); rgb = np.empty((0, 3))
                    actual_weights = result['packed_weights'].detach().float().cpu().numpy().reshape(-1)
                    saturation = float(result['adaptive_saturation_rate'])
                    rendered = result['rgb'].detach().float().cpu().numpy().reshape(-1).tolist()
                else:
                    count = a.uniform_samples*(2 if name == 'uniform2x' else 1)
                    edges = torch.linspace(near, far, count+1, device=model.device)
                    starts_t, ends_t = edges[:-1], edges[1:]
                    positions = rays.origins+((starts_t+ends_t)/2)[:, None]*rays.directions
                    with torch.no_grad():
                        sigma_t, rgb_t = model.field.query_points(positions, rays.directions.expand(count, -1), camera_indices=None if rays.camera_indices is None else rays.camera_indices.expand(count, -1))
                    starts, ends, sigma, rgb = [v.detach().float().cpu().numpy() for v in [starts_t, ends_t, sigma_t, rgb_t]]
                    actual_weights = None; saturation = None; rendered = None
                starts, ends, sigma = [v.reshape(-1) for v in [starts, ends, sigma]]
                alpha, trans, weights = integrate(starts, ends, sigma)
                mid = (starts+ends)/2
                xyz = origin[None]+mid[:,None]*direction[None]
                cell_indices = np.full((len(mid), len(boxes), 3), -1, np.int32)
                native_occupied = np.zeros((len(mid), len(boxes)), bool)
                for j, point in enumerate(xyz):
                    for cell in lookup(native, boxes, point):
                        if cell['inside']:
                            cell_indices[j, cell['level']] = cell['index_xyz']
                            native_occupied[j, cell['level']] = cell['occupied']
                file = f'pixel{i:04d}_{name}.npz'
                np.savez_compressed(a.output/file, t_start=starts, t_end=ends, sigma=sigma, alpha=alpha, transmittance=trans, weights=weights, rgb=rgb, xyz=xyz, native_cell_indices=cell_indices, native_occupied=native_occupied, actual_weights=weights if actual_weights is None else actual_weights)
                opacity = float(weights.sum())
                row['profiles'][name] = dict(trace=file, samples=len(mid), opacity=opacity, expected_depth=float(np.sum(weights*mid)), conditional_mean_depth=float(np.sum(weights*mid)/opacity) if opacity > 0 else None, integrated_foreground_rgb=np.sum(weights[:,None]*rgb.reshape(-1,3),axis=0).tolist(), model_rgb=rendered, saturation=saturation, weight_max_abs_error=None if actual_weights is None else float(np.max(np.abs(weights-actual_weights), initial=0)), native_culled_weight_per_cascade=[float(weights[(cell_indices[:, level, 0] >= 0) & ~native_occupied[:, level]].sum()) for level in range(len(boxes))])
            first, second = [row['profiles'][name] for name in ['uniform', 'uniform2x']]
            row['uniform_convergence'] = dict(opacity_abs_difference=abs(first['opacity']-second['opacity']), expected_depth_abs_difference=abs(first['expected_depth']-second['expected_depth']), foreground_rgb_max_abs_difference=float(np.max(np.abs(np.array(first['integrated_foreground_rgb'])-second['integrated_foreground_rgb']))))
            row['native_cell_ray_intervals'] = ray_intervals(native, boxes, origin, direction, near, far)
            rows.append(row)
            (a.output/'progress.json').write_text(json.dumps({'completed': len(rows), 'total': len(request['pixels'])})+'\n')
    finally:
        model.occupancy_grid.binaries.copy_(mask)
    if digest_model(model) != before:
        raise ValueError('Frozen model changed after restoring occupancy')
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint), config_sha256=file_sha(a.config), dataparser_config_sha256=hashlib.sha256(parser_config.encode()).hexdigest(), transforms_path=str(transforms_path), transforms_sha256=transforms_sha, live_bounds=live_bounds, bounds_tolerance=1e-6, normalization=normalization, occupancy_policy=occupancy_policy, step=step, config=str(a.config), restored_state_sha256=before, frozen_state_unchanged=True, runtime=runtime, split=split, rows=rows, limitations=['Reference probes the same learned field, not independent geometry.', 'all_cells retains ARM caps/alpha policy; uniform bypasses both.', 'Uniform query_points and adaptive field paths must agree; compare N and 2N convergence.', 'No full-frame certification; only supplied calibrated pixels.', 'Native occupancy evidence is per cascade, not combined traversal selection.', 'Runtime absent from old checkpoints remains the configured feature strength.'])
    (a.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(a.output), 'pixels': len(rows), 'frozen_state_unchanged': True}))


if __name__ == '__main__':
    main()
