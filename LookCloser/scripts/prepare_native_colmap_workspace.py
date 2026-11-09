#!/usr/bin/env python3
"""Stage already-undistorted PINHOLE RGB without COLMAP image resampling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from import_colmap_mvs_depth_dataset import load_binary_pinhole_calibration
from run_dec5_hair_campaign import atomic_json, sha


def prepare_native_workspace(data, model, output, colmap):
    if output.exists():
        raise FileExistsError('Native MVS workspace already exists')
    manifest = json.loads((model/'export_manifest.json').read_text())
    if manifest['camera_model'] != 'PINHOLE' or manifest['uses_eval_images']:
        raise ValueError('Only split-clean, already-undistorted PINHOLE models are supported')
    for folder in ('sparse', 'images', 'stereo/depth_maps', 'stereo/normal_maps', 'stereo/consistency_graphs'):
        (output/folder).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    private = colmap.parent.parent/'lib'
    if private.is_dir():
        env['LD_LIBRARY_PATH'] = str(private)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    subprocess.run([str(colmap), 'model_converter', '--input_path', str(model),
                    '--output_path', str(output/'sparse'), '--output_type', 'BIN'], check=True, env=env)
    calibration = load_binary_pinhole_calibration(output/'sparse')
    rows = []
    for row in manifest['images']:
        relative = Path(row['name'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Nonlocal source image path')
        source = data/relative
        target = output/'images'/relative
        with Image.open(source) as image:
            camera = calibration[row['name']]
            if image.size != (camera['w'], camera['h']):
                raise ValueError('Source raster does not match its fixed camera')
        target.parent.mkdir(parents=True, exist_ok=True)
        # COLMAP does not create nested parents for registered names like
        # "images/frame_train_X.png" when writing its stereo output arrays.
        for folder in ('depth_maps', 'normal_maps', 'consistency_graphs'):
            (output/'stereo'/folder/relative.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        expected = sha(source)
        if sha(target) != expected:
            raise ValueError('Native RGB bytes changed')
        rows.append(dict(name=row['name'], sha256=expected, raster=[camera['w'],camera['h']]))
    atomic_json(output/'native_rgb_receipt.json', dict(images=rows, unmodified=True,
                export_manifest_sha256=sha(model/'export_manifest.json'),
                image_undistorter_used=False, source_rgb_resampled=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--colmap', type=Path, required=True)
    args = p.parse_args()
    prepare_native_workspace(args.data, args.model, args.output, args.colmap)


if __name__ == '__main__':
    main()
