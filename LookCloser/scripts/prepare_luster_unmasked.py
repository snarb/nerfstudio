#!/usr/bin/env python3
"""Prepare calibrated, unmasked HD RGB for one Luster frame."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text, read_images_text

EVAL_IDS = {12, 95, 150}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(source: Path, output: Path, long_side: int = 1920) -> dict:
    if (output / 'transforms.json').exists():
        raise FileExistsError(f'Refusing to replace a prepared dataset: {output}')
    cameras = read_cameras_text(source / 'sparse/text/cameras.txt')
    images = read_images_text(source / 'sparse/text/images.txt')
    if len(images) != 165:
        raise ValueError(f'Expected 165 cameras, got {len(images)}')
    cv2.setNumThreads(1)
    (output / 'images').mkdir(parents=True, exist_ok=True)

    def convert(item):
        _, image = item
        camera = cameras[image.camera_id]
        source_image = source / 'images' / image.name
        cid = int(Path(image.name).stem.split('_')[1])
        with Image.open(source_image) as im:
            if camera.model != 'PINHOLE' or im.size != (camera.width, camera.height):
                raise ValueError(f'Calibration/image mismatch: {image.name}')
            rgb = np.asarray(im.convert('RGB'))
        size = tuple(round(v * long_side / max(camera.width, camera.height))
                     for v in (camera.width, camera.height))
        hd = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        name = f'cam_{cid:03d}_{Path(source).name}.png'
        Image.fromarray(hd).save(output / 'images' / name, compress_level=1)
        sx, sy = size[0] / camera.width, size[1] / camera.height
        fx, fy, cx, cy = camera.params
        w2c = np.eye(4)
        w2c[:3, :3] = image.qvec2rotmat()
        w2c[:3, 3] = image.tvec
        pose = np.linalg.inv(w2c)
        pose[:3, 1:3] *= -1
        row = dict(file_path=f'images/{name}', physical_camera=f'cam_{cid:03d}',
                   camera_id=cid, w=size[0], h=size[1], fl_x=float(fx*sx), fl_y=float(fy*sy),
                   cx=float(cx*sx), cy=float(cy*sy), transform_matrix=pose.tolist())
        receipt = dict(camera_id=cid, source=str(source_image), source_sha256=sha(source_image),
                       hd_sha256=sha(output / 'images' / name), original_size=[camera.width, camera.height],
                       size=list(size), scale_xy=[sx, sy])
        return row, receipt

    with ThreadPoolExecutor(max_workers=8) as pool:
        converted = list(pool.map(convert, sorted(images.items())))
    rows, receipts = map(list, zip(*converted))
    if len({r['camera_id'] for r in rows}) != 165:
        raise ValueError('Duplicate physical camera identifiers')
    poses = torch.tensor([r['transform_matrix'] for r in rows], dtype=torch.float32)
    train = [i for i, row in enumerate(rows) if row['camera_id'] not in EVAL_IDS]
    normalized, transform = auto_orient_and_center_poses(poses[train], method='up', center_method='focus')
    scale = 1 / float(normalized[:, :3, 3].abs().max())
    transform4 = torch.eye(4)
    transform4[:3] = transform
    normalized_all = transform4 @ poses
    normalized_all[:, :3, 3] *= scale
    for row, pose in zip(rows, normalized_all):
        row['transform_matrix'] = pose.tolist()
    centers = normalized_all[train, :3, 3]
    low, high = centers.min(0).values, centers.max(0).values
    padding = (high-low).clamp_min(.1) * .5
    bounds = torch.stack([low-padding, high+padding]).tolist()
    meta = dict(camera_model='OPENCV', orientation_override='none', coordinate_system='train_normalized_opengl',
                opacity_aabb=bounds, depth_unit_scale_factor=1.0, frames=rows,
                train_filenames=[r['file_path'] for r in rows if r['camera_id'] not in EVAL_IDS],
                val_filenames=[r['file_path'] for r in rows if r['camera_id'] in EVAL_IDS],
                test_filenames=[r['file_path'] for r in rows if r['camera_id'] in EVAL_IDS])
    write(output / 'transforms.json', meta)
    audit = dict(source=str(source), images=165, train=162, eval=3, masks='forbidden',
                 resizing='area; long-side; no crop', long_side=long_side,
                 normalization=transform4.tolist(), scale=scale, bounds=bounds,
                 bounds_policy='train-camera envelope plus 50% of each axis extent',
                 train_review_ids=[11, 97, 151], receipts=receipts,
                 calibration_sha256={n:sha(source / 'sparse/text' / n) for n in ['cameras.txt','images.txt']},
                 transforms_sha256=sha(output / 'transforms.json'),
                 shape_counts={str(k):v for k,v in Counter(tuple(r['size']) for r in receipts).items()})
    write(output / 'audit.json', audit)
    sheet = Image.new('RGB', (6*220, 400), (35,35,35))
    draw = ImageDraw.Draw(sheet)
    for col, cid in enumerate([11,12,95,97,150,151]):
        row = next(r for r in rows if r['camera_id'] == cid)
        with Image.open(output / row['file_path']) as im:
            im.thumbnail((220,370))
            sheet.paste(im, (col*220,25))
        draw.text((col*220+3,5),f'cam{cid:03d}',fill='white')
    sheet.save(output / 'initial_contact.jpg')
    return {k:audit[k] for k in ['images','train','eval','bounds','shape_counts','transforms_sha256']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/fsx/tmp/luster/root_8s/working/fullres/000470'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--long-side', type=int, default=1920)
    args = p.parse_args()
    if args.long_side <= 0:
        p.error('--long-side must be positive')
    torch.set_num_threads(2)
    print(json.dumps(prepare(args.source, args.output, args.long_side)), flush=True)
