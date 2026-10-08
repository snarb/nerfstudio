#!/usr/bin/env python3
"""Transfer low-resolution LookCloser map locations to pixel-identical 4K RGB.

This preserves the fitted scalar-resolution values and interpolates only their
image-plane locations. It is a cheap full-resolution training gate, not a new
frequency fit; the receipt makes that limitation explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def grid_shape(height: int, width: int, patch: int, stride: int) -> tuple[int,int]:
    if min(height,width,patch,stride) <= 0 or min(height,width) < patch:
        raise ValueError('Invalid image or frequency patch dimensions')
    return ((height-patch)//stride+1, (width-patch)//stride+1)


def transfer(low_data: Path, full_data: Path) -> dict:
    old = json.loads((low_data/'transforms.json').read_text())
    new = json.loads((full_data/'transforms.json').read_text())
    old_frames = {int(row['camera_id']):row for row in old['frames']}
    new_frames = {int(row['camera_id']):row for row in new['frames']}
    if set(old_frames) != set(new_frames):
        raise ValueError('Full-resolution camera inventory changed')
    old_names = {Path(name).stem for name in old['train_filenames']}
    new_names = {Path(name).stem for name in new['train_filenames']}
    if old_names != new_names or len(old_names)!=162:
        raise ValueError('Train split or frequency-map stems changed')
    destination = full_data/'lookcloser_frequencies'
    if destination.exists() or (full_data/'frequency_ready.json').exists():
        raise FileExistsError('Preserve earlier full-resolution frequency transfer')
    destination.mkdir()
    rows = []
    for camera_id in sorted(old_frames):
        low,full = old_frames[camera_id],new_frames[camera_id]
        if (low['physical_camera'] != full['physical_camera']
                or low['transform_matrix'] != full['transform_matrix']):
            raise ValueError(f'Camera {camera_id} pose or identity changed')
        if Path(low['file_path']).stem not in old_names:
            continue
        source_map = low_data/'lookcloser_frequencies'/f'{Path(low["file_path"]).stem}.pt'
        source_meta = source_map.with_suffix('.json')
        if not source_map.is_file() or not source_meta.is_file():
            raise FileNotFoundError(source_map)
        metadata = json.loads(source_meta.read_text())
        patch, stride = int(metadata['patch_size']),int(metadata['stride'])
        low_shape = grid_shape(int(low['h']),int(low['w']),patch,stride)
        full_shape = grid_shape(int(full['h']),int(full['w']),patch,stride)
        values = torch.load(source_map,map_location='cpu',weights_only=True).float()
        if tuple(values.shape)!=low_shape or metadata['image_shape']!=[low['h'],low['w']]:
            raise ValueError(f'Low-resolution map geometry disagrees for camera {camera_id}')
        transferred = F.interpolate(values[None,None],size=full_shape,
                                    mode='bilinear',align_corners=True)[0,0].contiguous()
        if not torch.isfinite(transferred).all() or transferred.min() <= 0:
            raise ValueError(f'Invalid transferred map for camera {camera_id}')
        output_map = destination/f'{Path(full["file_path"]).stem}.pt'
        torch.save(transferred,output_map)
        full_image = full_data/full['file_path']
        updated = dict(metadata,image=full_image.name,image_shape=[full['h'],full['w']],
                       source_sha256=sha(full_image),
                       derivation='bilinear spatial transfer of fitted 1920px scalar-resolution map; no 4K refit',
                       parent_map_sha256=sha(source_map),parent_metadata_sha256=sha(source_meta),
                       parent_image_sha256=sha(low_data/low['file_path']))
        write(output_map.with_suffix('.json'),updated)
        rows.append(dict(camera_id=camera_id,low_shape=list(low_shape),full_shape=list(full_shape),
                         map_sha256=sha(output_map),source_map_sha256=sha(source_map)))
    if len(rows)!=162:
        raise ValueError(f'Expected 162 maps, transferred {len(rows)}')
    result=dict(scope='Transfer of 1920px map locations only; not a full-resolution frequency fit',
                masks_used=False,low_transforms_sha256=sha(low_data/'transforms.json'),
                full_transforms_sha256=sha(full_data/'transforms.json'),
                maps=len(rows),rows=rows)
    write(full_data/'frequency_transfer_audit.json',result)
    write(full_data/'frequency_ready.json',dict(maps=len(rows),kind='spatial_transfer_from_1920',
          audit_sha256=sha(full_data/'frequency_transfer_audit.json'),full_resolution_fit=False))
    return dict(maps=len(rows),full_resolution_fit=False,audit=str(full_data/'frequency_transfer_audit.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-data',type=Path,required=True)
    parser.add_argument('--full-data',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(transfer(args.low_data,args.full_data)))
