"""Frozen-input utilities for the Luster 000540 native hybrid experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ORIGIN = Path('/fsx/tmp/luster_000540_body_opt_20261016')
SOURCE = Path('/fsx/tmp/luster/root_8s/working/fullres/000540')
CHECKPOINT = ORIGIN / 'runs/cold_body80_fas50_threshold095_zero/G106316/field/lookcloser/run/nerfstudio_models/step-000106316.ckpt'
DIGEST = 'f07369d46d29852d9816a9681ac5758e9ecf22fad087ac3aece78e443ae8413b'
FRAME = 540
# Explicit per-process context; original frame540 defaults remain unchanged.
# Never rebind these globals in a long-lived process that has imported consumers.
if os.environ.get('LUSTER_HYBRID_FRAME_MANIFEST'):
    _context = json.loads(Path(os.environ['LUSTER_HYBRID_FRAME_MANIFEST']).read_text())
    FRAME = int(_context['frame'])
    ORIGIN = Path(_context['origin'])
    SOURCE = Path(_context['source'])
    CHECKPOINT = Path(_context['checkpoint'])
    DIGEST = _context['checkpoint_sha256']
    if not (len(DIGEST) == 64 and all(c in '0123456789abcdef' for c in DIGEST)):
        raise ValueError('Invalid pinned checkpoint digest')
    if SOURCE.name != f'{FRAME:06d}' or not all(p.is_absolute() for p in (ORIGIN,SOURCE,CHECKPOINT)):
        raise ValueError('Invalid frame context')
TARGETS = (7, 12, 95, 150)
HELDOUT = (12, 95, 150)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def log(root, event, **values):
    row = dict(time=time.time(), event=event, **values)
    with (Path(root) / 'logs/events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def bootstrap(root):
    runtime = Path(root) / 'runtime'
    sys.path[:0] = [str(runtime), str(runtime / 'LookCloser/scripts')]
    return runtime


def source_path(cid):
    return SOURCE / 'images' / f'cam_{cid:03d}_{FRAME:06d}.jpg'


def native_rows():
    meta = read(ORIGIN / 'data/unmasked/transforms.json')
    rows = {}
    for original in meta['frames']:
        row = dict(original)
        with Image.open(source_path(row['camera_id'])) as im:
            w, h = im.size
        for key in ('fl_x', 'cx'):
            row[key] *= w / row['w']
        for key in ('fl_y', 'cy'):
            row[key] *= h / row['h']
        row.update(w=w, h=h, train=original['file_path'] in meta['train_filenames'])
        rows[row['camera_id']] = row
    return rows


def allowed_sources(rows, target):
    return [cid for cid, row in rows.items() if row['train'] and cid not in HELDOUT and cid != target]


def assert_source(rows, target, cid):
    if cid not in allowed_sources(rows, target):
        raise ValueError(f'Forbidden texture source {cid} for target {target}')


def project(world, row):
    pose = np.asarray(row['transform_matrix'], np.float32)
    local = (world - pose[:3, 3]) @ pose[:3, :3]
    z = -local[..., 2]
    den = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    uv = np.stack([row['fl_x'] * local[..., 0] / den + row['cx'] - .5,
                   -row['fl_y'] * local[..., 1] / den + row['cy'] - .5], -1)
    return uv, z


def unproject(z, row, xy=None):
    if xy is None:
        yy, xx = np.indices(z.shape, dtype=np.float32)
        xy = np.stack([xx, yy], -1)
    local = np.stack([(xy[..., 0] + .5 - row['cx']) / row['fl_x'],
                      -(xy[..., 1] + .5 - row['cy']) / row['fl_y'],
                      -np.ones_like(z)], -1) * z[..., None]
    pose = np.asarray(row['transform_matrix'], np.float32)
    return local @ pose[:3, :3].T + pose[:3, 3]


def image(path, rgb):
    Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def geometry(root, cid):
    directory = Path(root) / 'geometry' / f'cam{cid:03d}'
    receipt = read(directory / 'receipt.json')
    if receipt['state'] != 'complete' or receipt['checkpoint_sha256'] != DIGEST:
        raise ValueError('Incomplete or foreign geometry')
    return {k: np.load(directory / f'{k}.npy', mmap_mode='r') for k in receipt['arrays']}


def prepare(root):
    root = Path(root)
    assert sha(CHECKPOINT) == DIGEST
    rows = native_rows()
    assert len(rows) == 165 and sum(r['train'] for r in rows.values()) == 162
    cameras = {}
    for line in (SOURCE / 'sparse/text/cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        tokens = line.split()
        assert tokens[1] == 'PINHOLE'
        cameras[int(tokens[0])] = list(map(float, tokens[2:]))
    # Match images by filename, not by accidental camera/image ID equality.
    photo_cameras = {}
    lines = (SOURCE / 'sparse/text/images.txt').read_text().splitlines()
    for line in lines:
        tokens = line.split()
        if len(tokens) == 10 and tokens[-1].endswith('.jpg'):
            photo_cameras[tokens[-1]] = int(tokens[8])
    errors = []
    for cid, row in rows.items():
        native = cameras[photo_cameras[source_path(cid).name]]
        expected = [row[k] for k in ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy')]
        errors.append(max(abs(a-b) for a,b in zip(native, expected)))
    assert max(errors) < 1e-5, max(errors)
    runtime = root / 'runtime'
    runtime_hashes = {str(p.relative_to(runtime)): sha(p) for p in sorted(runtime.rglob('*'))
                      if p.is_file() and '__pycache__' not in p.parts and p.name != '.git'}
    inputs = [CHECKPOINT, CHECKPOINT.parent.parent / 'config.yml',
              ORIGIN / 'data/unmasked/transforms.json', ORIGIN / 'preflight/sequence_aabb.json',
              SOURCE / 'sparse/text/cameras.txt', SOURCE / 'sparse/text/images.txt']
    write(root / 'preflight/manifest.json', dict(created=time.time(), checkpoint_sha256=DIGEST,
          inputs={str(p):sha(p) for p in inputs}, runtime_hashes=runtime_hashes,
          source_hashes={str(source_path(cid)):sha(source_path(cid)) for cid in rows},
          target_cameras=TARGETS, heldout=HELDOUT, native_calibration_max_error=max(errors),
          goal='validation person native LPIPS <= 0.60 * frozen LookCloser; no local regression',
          deadline=time.time()+24*3600, rows=rows))
    log(root, 'prepared', cameras=len(rows), calibration_max_error=max(errors))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    prepare(p.parse_args().root)
