#!/usr/bin/env python3
"""Frozen-label control: replace each source's mesh RGB base by a common base.

Reads the retained float prediction, not held-out RGB. Requires normalized
mesh-coordinate camera-path inputs, native pixel centres, and exact visibility.
The off control must replay the original PNG byte for byte.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from PIL import Image
from colmap_patchmatch_tsdf_campaign_common import atomic_json, sha256
from canonical_surface_base import CanonicalSurfaceBase, HELD
from audit_source_detail_control import decode_eight_source_labels


def unproject_mesh_depth(depth, target, depth_manifest):
    """Only accept explicitly normalized camera-path datasets; no double scale."""
    if (depth_manifest['dataparser_scale'] != 1
            or not np.array_equal(depth_manifest['dataparser_transform'], np.eye(4)[:3])
            or depth.shape != (target['h'], target['w']) or not np.isfinite(depth).all()
            or min(target['fl_x'], target['fl_y']) <= 0
            or any(target.get(k, 0) != 0 for k in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2'))):
        raise ValueError('Need finite native depth and undistorted mesh-coordinate cameras')
    pose = np.asarray(target['transform_matrix'], np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Invalid target pose')
    y, x = np.indices(depth.shape)
    q = np.stack(((x+.5-target['cx'])/target['fl_x']*depth,
                  -(y+.5-target['cy'])/target['fl_y']*depth, -depth), -1)
    return (q @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def apply_selected_offsets(baseline, labels, offsets, *, strength=1.):
    """Offsets already match the hard source; no change to support or labels."""
    if (baseline.shape != offsets.shape or baseline.shape != (*labels.shape, 3)
            or not np.isfinite(baseline).all() or not np.isfinite(offsets).all()
            or not np.isfinite(strength) or not 0 <= strength <= 1
            or not np.issubdtype(labels.dtype, np.integer) or labels.min() < -1 or labels.max() > 7):
        raise ValueError('Invalid finite RGB, offsets, source labels or strength')
    delta = np.where((labels >= 0)[..., None], offsets, 0)
    unclamped = baseline + np.float32(strength)*delta
    return unclamped.clip(0, 1), dict(
        clipped_channels=int(((unclamped < 0) | (unclamped > 1)).sum()),
        max_abs_offset=float(np.abs(delta).max()), strength=strength,
        changed_pixels=int((unclamped != baseline).any(-1).sum()))


def main():
    from render_mesh_image_blend import load_depth, load_rgb, fill_small_consistent_depth_holes, write_prediction
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('render', 'base-manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Preserve previous results; output must not exist')
    audit_path = a.render/'reprojection_audit.json'
    audit = json.loads(audit_path.read_text())
    data_path = Path(audit['data'])/'transforms.json'
    data = json.loads(data_path.read_text())
    dm_path = Path(audit['mesh_depth_manifest']); dm = json.loads(dm_path.read_text())
    base = json.loads(a.base_manifest.read_text())
    mesh_path = Path(dm['mesh']); color = audit['camera_color_calibration']; color_path = Path(color['path'])
    if (audit['uses_eval_rgb_for_prediction'] is not False or audit['uses_masks'] is not False
            or not audit['exact_mesh_visibility'] or audit['pixel_center_offset'] != .5
            or audit['eval_rgb_use'] != 'not_read' or dm['masks'] is not False
            or audit['mesh_depth_manifest_sha256'] != sha256(dm_path)
            or dm['mesh_sha256'] != sha256(mesh_path) or color['model'] != 'spatial'
            or color['sha256'] != sha256(color_path) or base['mesh_sha256'] != dm['mesh_sha256']):
        raise ValueError('Baseline provenance, response, visibility or native sampling mismatch')
    frames = {str((data_path.parent/f['file_path']).resolve()): f for f in data['frames']}
    if len(frames) != len(data['frames']) or any(f.get('mask_path') for f in frames.values()):
        raise ValueError('Duplicate frames or semantic mask')
    target = frames[str(Path(audit['target_image']).resolve())]
    train = {str((data_path.parent/name).resolve()) for name in data['train_filenames']}
    source_paths = [Path(row['source_image']).resolve() for row in audit['sources']]
    sources = [frames[str(path)]['physical_camera'] for path in source_paths]
    if (len(train) != 62 or len(sources) != 8 or len(set(sources)) != 8 or set(sources) & HELD
            or not set(map(str, source_paths)) <= train or len(base['source_cameras']) != 62):
        raise ValueError('Need 62 train cameras and eight matching hard detail sources')
    hashes = {}
    for path in map(Path, sorted(train)):
        name = frames[str(path)]['physical_camera']
        hashes[str(path)] = sha256(path)
        if name in HELD or hashes[str(path)] != base['source_hashes'].get(name):
            raise ValueError('Train camera/source hash mismatch')
    depth_path = Path(next(row['depth'] for row in dm['images'] if row['image'] == audit['target_image']))
    hole = audit['depth_hole_fill']
    if (hole['max_area'], hole['boundary_radius'], hole['max_relative_plane_rmse']) != (1000, 4, .015):
        raise ValueError('Unsupported depth hole rule')
    depth = fill_small_consistent_depth_holes(load_depth(depth_path), max_area=1000,
        boundary_radius=4, max_relative_plane_rmse=.015)[0]
    world = unproject_mesh_depth(depth, target, dm)
    variant = a.render/'seam_cut8'
    baseline_path = variant/'eval_pred_0000.exr'; png_path = variant/'eval_pred_0000.png'
    selection_path = variant/'source_selection.png'
    labels = decode_eight_source_labels(np.asarray(Image.open(selection_path).convert('RGB')))
    baseline = load_rgb(baseline_path, torch.device('cpu')).permute(1, 2, 0).numpy()
    if baseline.shape != (1080, 1920, 3) or not np.isfinite(baseline).all():
        raise ValueError('Need finite full-resolution retained float render')
    field = CanonicalSurfaceBase(a.base_manifest, mesh_path, color_path)
    for path in [audit_path, data_path, dm_path, mesh_path, color_path, depth_path,
                 a.base_manifest, a.base_manifest.parent/base['coefficients'], baseline_path,
                 png_path, selection_path, Path(__file__), Path(__file__).with_name('canonical_surface_base.py')]:
        hashes[str(path.resolve())] = sha256(path)
    a.output.mkdir(parents=True)
    atomic_json(a.output/'request.json', dict(input_hashes=hashes, sources=sources,
        uses_eval_rgb=False, uses_semantic_masks=False, query_dependent_base=False,
        source_detail='original float RGB under original hard labels', strengths=[0, 1]))
    field.bind(world, (depth > 0) & (labels >= 0))
    offsets = np.zeros_like(baseline)
    for rank, physical in enumerate(sources):
        sampled = field.sample_offset(physical)
        offsets[labels == rank] = sampled[labels == rank]
    rows = []
    for name, strength in [('off', 0.), ('common_base', 1.)]:
        prediction, stats = apply_selected_offsets(baseline, labels, offsets, strength=strength)
        write_prediction(a.output, name, torch.from_numpy(prediction).permute(2, 0, 1), None)
        shutil.copyfile(selection_path, a.output/name/'source_selection.png')
        rows.append(dict(name=name, **stats))
    if sha256(a.output/'off/eval_pred_0000.png') != sha256(png_path):
        raise RuntimeError('Off control failed byte-identical original PNG replay')
    np.savez_compressed(a.output/'applied_offsets.npz', offsets=offsets)
    result = dict(method='mesh_common_display_base_plus_single_source_detail', uses_eval_rgb=False,
        uses_semantic_masks=False, geometry_unchanged=True, visibility_unchanged=True,
        labels_unchanged=True, query_dependent_base=False, source_rgb_averaging='common base only',
        original_png_byte_identical=True, supported_pixels=int(field.support.sum()),
        far_mesh_pixels=int((~field.near).sum()), sources=sources, variants=rows,
        accepted_surface_recipe=False, input_hashes=hashes,
        output_hashes={str(path.relative_to(a.output)): sha256(path)
                       for path in sorted(a.output.rglob('*')) if path.is_file()})
    atomic_json(a.output/'control_manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'output_hashes')}), flush=True)


if __name__ == '__main__':
    main()
