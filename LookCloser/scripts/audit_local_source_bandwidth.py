#!/usr/bin/env python3
"""Densely audit train-only bandwidth variation; never change a prediction.

Depth bins are derived from fit observations, not anatomical ROIs or eval RGB.
Only patches contained in one spatial block enter the disjoint fit/held audit.
Many overlapping patches in a block are not independent validation evidence.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from render_mesh_image_blend import load_depth
from source_bandwidth_prior import bandwidth_observations


def spatial_partition(x, y, *, half_width=32, block_size=128):
    """Return a disjoint block ID and split, or None for a boundary patch."""
    if half_width < 1 or block_size < 2 * half_width:
        raise ValueError('Invalid patch/block dimensions')
    bx, by = x // block_size, y // block_size
    if ((x-half_width)//block_size != bx or (x+half_width-1)//block_size != bx
            or (y-half_width)//block_size != by or (y+half_width-1)//block_size != by):
        return None
    held = ((bx*73856093) ^ (by*19349663)) % 5 == 0
    return {'block': [int(bx), int(by)], 'held': bool(held)}


def source_pairs(count, all_pairs=False):
    if count < 2:
        raise ValueError('Need at least two source cameras')
    return list(combinations(range(count), 2)) if all_pairs else [(0, i) for i in range(1, count)]


def summarize_observations(rows):
    """Block medians prevent dense samples from fabricating independent support."""
    result = {}
    signs = []
    for held, name in [(False, 'fit'), (True, 'held')]:
        selected = [r for r in rows if r['held'] == held]
        values = [r['relative_blur_variance'] for r in selected]
        blocks = {}
        for r in selected:
            blocks.setdefault(tuple(r['block']), []).append(r['relative_blur_variance'])
        medians = [float(np.median(v)) for v in blocks.values()]
        if values and not np.isfinite(values).all():
            raise ValueError('Nonfinite bandwidth observation')
        result[name] = {'patches': len(values), 'blocks': len(medians),
                        'patch_median': float(np.median(values)) if values else None,
                        'block_median': float(np.median(medians)) if medians else None,
                        'block_negative_fraction': float(np.mean(np.array(medians)<0)) if medians else None,
                        'block_positive_fraction': float(np.mean(np.array(medians)>0)) if medians else None}
        signs.append(medians)
    median = result['fit']['block_median'] or 0.
    sign = np.sign(median)
    result['qualified'] = bool(len(signs[0]) >= 3 and len(signs[1]) >= 2 and sign != 0
                               and np.mean(sign*np.array(signs[0])>0) >= .6
                               and np.mean(sign*np.array(signs[1])>0) >= .6)
    return result


def stratify_depth(rows, bins=4):
    """Bin cut points are fit-only; held observations never set them."""
    if bins < 1:
        raise ValueError('Need at least one depth bin')
    fit_depths = [r['depth'] for r in rows if not r['held']]
    if not fit_depths:
        return {'boundaries': [], 'bins': [], 'all': summarize_observations(rows)}
    if not np.isfinite(fit_depths).all() or min(fit_depths) <= 0:
        raise ValueError('Invalid fit depths')
    edges = np.unique(np.quantile(fit_depths, np.arange(1, bins)/bins))
    groups = [[] for _ in range(len(edges)+1)]
    for r in rows:
        if not np.isfinite(r['depth']) or r['depth'] <= 0:
            raise ValueError('Invalid held depth')
        groups[int(np.searchsorted(edges, r['depth'], side='right'))].append(r)
    return {'boundaries': edges.tolist(), 'bins': [summarize_observations(g) for g in groups],
            'all': summarize_observations(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--patch-size', type=int, default=48)
    p.add_argument('--all-pairs', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.stride < 1 or a.patch_size < 24 or a.patch_size%2:
        p.error('Preserve outputs; require positive stride and even patch size >=24')
    footprint = a.patch_size//2+8
    audit_path = a.render/'reprojection_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['uses_eval_rgb_for_prediction'] is not False or audit['uses_masks'] is not False:
        raise ValueError('Only train-only unmasked source warps may be audited')
    data_path = Path(audit['data'])/'transforms.json'
    data = json.loads(data_path.read_text())
    frames = {str(Path(f['file_path']).resolve()): f for f in data['frames']}
    train = {str(Path(f).resolve()) for f in data['train_filenames']}
    forbidden = {'F004_B005_1210O9', 'J004_D005_1210TA', 'L004_B005_12106A'}
    hashes = {str(audit_path): sha256(audit_path), str(data_path): sha256(data_path)}
    sources, gray, valid = [], [], []
    for i, row in enumerate(audit['sources']):
        path = Path(row['source_image']).resolve()
        if str(path) not in train or frames[str(path)]['physical_camera'] in forbidden:
            raise ValueError('Source is not a permitted train camera')
        sources.append(frames[str(path)]['physical_camera'])
        hashes[str(path)] = sha256(path)
        rgb_path = a.render/'source_warps'/f'source_{i:02d}.png'
        valid_path = a.render/'source_warps'/f'valid_{i:02d}.png'
        gray.append(np.asarray(Image.open(rgb_path).convert('RGB'), np.float32)
                    @np.array([.2126,.7152,.0722], np.float32)/255)
        valid.append(np.asarray(Image.open(valid_path)) > 0)
        for path in [rgb_path, valid_path]: hashes[str(path)] = sha256(path)
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate sources')
    depth_manifest = Path(audit['mesh_depth_manifest'])
    if sha256(depth_manifest) != audit['mesh_depth_manifest_sha256']:
        raise ValueError('Changed mesh-depth manifest')
    dm = json.loads(depth_manifest.read_text())
    depth_path = Path(next(r['depth'] for r in dm['images'] if r['image'] == audit['target_image']))
    depth = load_depth(depth_path)
    hashes[str(depth_path)] = sha256(depth_path)
    hashes[str(depth_manifest)] = sha256(depth_manifest)
    rows, summaries = [], []
    for primary, rank in source_pairs(len(sources), a.all_pairs):
        observations = bandwidth_observations(gray[primary], gray[rank], valid[primary], valid[rank],
                                               stride=a.stride, patch_size=a.patch_size)
        kept = []
        for row in observations:
            x, y = row['x'], row['y']
            partition = spatial_partition(x, y, half_width=footprint)
            window = depth[y-footprint:y+footprint, x-footprint:x+footprint]
            if partition is None or not np.isfinite(window).all() or (window <= 0).any():
                continue
            if float(np.log(window.max()/window.min())) > .0075:
                continue
            kept.append({**row, **partition, 'depth': float(depth[y,x]),
                         'primary_rank': primary, 'source_rank': rank,
                         'angular_primary_visible': bool(valid[0][y,x])})
        rows.extend(kept)
        summaries.append({'primary_rank': primary, 'primary_camera': sources[primary],
                          'source_rank': rank, 'physical_camera': sources[rank],
                          'dense_raw_patches': len(observations), 'disjoint_single_layer_patches': len(kept),
                          **stratify_depth(kept)})
        print(json.dumps(summaries[-1]), flush=True)
    atomic_json(a.output, {'uses_eval_rgb': False, 'uses_semantic_masks': False,
                          'changes_prediction': False, 'source_averaging': False,
                          'metric_scope': 'train_source_bandwidth_diagnostic_not_reconstruction_metric',
                          'stride': a.stride, 'patch_size': a.patch_size,
                          'patch_footprint': 2*footprint, 'block_size': 128,
                          'all_pairs': a.all_pairs,
                          'maximum_patch_log_depth_range': .0075,
                          'interpretation': __doc__, 'sources': sources, 'summary': summaries,
                          'observations': rows, 'input_hashes': hashes,
                          'script_sha256': sha256(Path(__file__)),
                          'bandwidth_helper_sha256': sha256(Path(__file__).with_name('source_bandwidth_prior.py'))})


if __name__ == '__main__':
    main()
