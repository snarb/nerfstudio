#!/usr/bin/env python3
"""Bind a reviewed offline result to the existing mesh-transfer renderer.

The caller explicitly supplies conversion to the base saved camera-z units.
Physical cameras, intrinsics, poses, train inventory and base manifest are
checked before producing an opt-in override. No dataset is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dec5_hair import VALIDATION, TEST
from run_dec5_hair_campaign import atomic_json, sha


def resolve(reference, data, manifest):
    path = Path(reference)
    if path.is_absolute():
        return path.resolve()
    for root in (data, manifest.parent):
        if (root/path).exists():
            return (root/path).resolve()
    raise FileNotFoundError(reference)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--camera', required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--mesh-manifest', type=Path, required=True)
    p.add_argument('--depth-scale', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        raise ValueError('Explicit positive depth-unit conversion is required')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        raise FileExistsError('Override outputs are immutable')
    source = json.loads((args.campaign/'dataset/transforms.json').read_text())
    target = json.loads((args.data/'transforms.json').read_text())
    source_rows = source['frames']+json.loads((args.campaign/'test_camera_poses.json').read_text())['frames']
    original = next(f for f in source_rows if f['physical_camera'] == args.camera)
    matches = [f for f in target['frames'] if f.get('physical_camera') == args.camera]
    if len(matches) != 1:
        raise ValueError('Expected exactly one matching physical target camera')
    frame = matches[0]
    for key in ('w','h','fl_x','fl_y','cx','cy','k1','k2','p1','p2'):
        if not np.isclose(float(frame.get(key,target.get(key,0))), float(original.get(key,0)), atol=1e-6, rtol=1e-7):
            raise ValueError('Target calibration mismatch: '+key)
    np.testing.assert_allclose(frame['transform_matrix'], original['transform_matrix'], atol=1e-6, rtol=1e-7)
    train = [f['physical_camera'] for f in target['frames'] if f['file_path'] in target['train_filenames']]
    source_train = {f['physical_camera'] for f in source['frames'] if f['file_path'] in source['train_filenames']}
    if set(train) != source_train or len(train) != 59 or set(train) & set(VALIDATION+TEST):
        raise ValueError('Renderer train inventory differs from the campaign')
    mesh = json.loads(args.mesh_manifest.read_text())
    target_image = (args.data/frame['file_path']).resolve()
    depths = [resolve(r['depth'],args.data,args.mesh_manifest) for r in mesh['images']
              if resolve(r['image'],args.data,args.mesh_manifest) == target_image]
    if len(depths) != 1:
        raise ValueError('Base manifest has no unique matching target depth')
    with np.load(args.result, allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in archive.files}
    values = dict(front_depth=arrays['front_depth']*args.depth_scale, roi=arrays['roi'],
                  confidence=((arrays['support'] >= 3) & (arrays['source_id'] >= 0)).astype(np.float32))
    if {'foreground_premultiplied','background_rgb','layer_gate'}.issubset(arrays):
        if any(Path(f['file_path']).suffix.lower() != '.exr' for f in target['frames'] if f['file_path'] in target['train_filenames']):
            raise ValueError('Scene-linear layers require raw linear EXR renderer inputs')
        values.update(back_depth=arrays['layer_back_depth']*args.depth_scale, alpha=arrays['layer_alpha'],
                      foreground_premultiplied=arrays['foreground_premultiplied'], background_rgb=arrays['background_rgb'],
                      confidence=arrays['layer_gate'].astype(np.float32))
    path = args.output.with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    atomic_json(args.output, dict(archive=path.name, archive_sha256=sha(path),
        base_depth_sha256=sha(depths[0]), mesh_depth_manifest_sha256=sha(args.mesh_manifest),
        target_image_name=target_image.name, depth_convention='OpenCV camera-z', color_space='scene_linear',
        minimum_confidence=.75, source_result_sha256=sha(args.result), explicit_depth_scale=args.depth_scale,
        renderer_data_sha256=sha(args.data/'transforms.json'), camera=args.camera,
        implementation_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
