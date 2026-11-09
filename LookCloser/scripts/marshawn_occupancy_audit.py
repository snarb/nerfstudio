#!/usr/bin/env python3
"""Offline occupancy review: every cascade/slice, point lookups and exact ray cells.

Input NPZ: occupied bool[L,X,Y,Z], aabbs float[L,6], optional occs.
A checkpoint can instead supply keys ending in occupancy_grid.{binaries,aabbs,occs}.
Coordinates are normalized model-world coordinates, NOT raw solve coordinates.
The atlas describes traversal admission, not geometry ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def validate_grid(occupied, aabbs):
    grid, boxes = np.asarray(occupied), np.asarray(aabbs, dtype=np.float64)
    if grid.dtype != np.bool_ or grid.ndim != 4 or min(grid.shape) < 1:
        raise ValueError('occupied must be nonempty bool[L,X,Y,Z]')
    if boxes.shape != (len(grid), 6) or not np.isfinite(boxes).all() or (boxes[:, 3:] <= boxes[:, :3]).any():
        raise ValueError('aabbs must have finite, increasing [L,6] bounds')
    return grid, boxes


def dilate_one(grid):
    """One non-cumulative cube radius1; zero padding cannot wrap at boundaries."""
    padded = np.pad(grid, ((0, 0), (1, 1), (1, 1), (1, 1)))
    out = np.zeros_like(grid)
    for x in range(3):
        for y in range(3):
            for z in range(3):
                out |= padded[:, x:x+grid.shape[1], y:y+grid.shape[2], z:z+grid.shape[3]]
    return out


def lookup(grid, boxes, point):
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('point must contain three finite coordinates')
    result = []
    for level, box in enumerate(boxes):
        rel = (point-box[:3])/(box[3:]-box[:3])
        inside = bool(((rel >= 0) & (rel < 1)).all())
        row = {'level': level, 'inside': inside}
        if inside:
            index = np.floor(rel*np.asarray(grid.shape[1:])).astype(int)
            lower = box[:3]+index/np.asarray(grid.shape[1:])*(box[3:]-box[:3])
            upper = lower+(box[3:]-box[:3])/np.asarray(grid.shape[1:])
            row.update(index_xyz=index.tolist(), occupied=bool(grid[(level, *index)]), bounds=[*lower.tolist(), *upper.tolist()])
        result.append(row)
    return result



def cell_record(grid, boxes, level, index_xyz):
    index = np.asarray(index_xyz)
    if not isinstance(level, int) or not 0 <= level < len(grid) or index.shape != (3,) or not np.issubdtype(index.dtype, np.integer) or (index < 0).any() or (index >= np.asarray(grid.shape[1:])).any():
        raise ValueError('Cell requires a valid integer level and XYZ indices')
    box = boxes[level]
    center = box[:3]+(index+.5)/np.asarray(grid.shape[1:])*(box[3:]-box[:3])
    return {'center_xyz': center.tolist(), **lookup(grid, boxes, center)[level]}


def ray_intervals(grid, boxes, origin, direction, near=0., far=float('inf')):
    """Exact plane partitions, per cascade; p(t)=origin+t*direction.

    Direction is not normalized silently. t is distance only for unit directions.
    Positive-length intervals use half-open voxel membership; tangencies omitted.
    """
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([origin, direction]).all() or not np.any(direction):
        raise ValueError('ray must have finite origin and nonzero direction')
    if not np.isfinite(near) or np.isnan(far) or near < 0 or far <= near:
        raise ValueError('require finite near >= 0 and far > near')
    reports = []
    for level, box in enumerate(boxes):
        entry, exit = float(near), float(far)
        hit = True
        for axis in range(3):
            if direction[axis] == 0:
                if not box[axis] <= origin[axis] < box[axis+3]:
                    hit = False
                    break
            else:
                t = (box[[axis, axis+3]]-origin[axis])/direction[axis]
                entry, exit = max(entry, float(t.min())), min(exit, float(t.max()))
        row = {'level': level, 'hit': bool(hit and exit > entry), 'intervals': []}
        if row['hit']:
            partitions = [entry, exit]
            for axis, size in enumerate(grid.shape[1:]):
                if direction[axis] != 0:
                    planes = np.linspace(box[axis], box[axis+3], size+1)
                    ts = (planes-origin[axis])/direction[axis]
                    partitions.extend(ts[(ts > entry) & (ts < exit)].tolist())
            ts = np.unique(partitions)
            for start, end in zip(ts[:-1], ts[1:]):
                if end <= start:
                    continue
                point = origin+direction*(start+(end-start)/2)
                index = np.floor((point-box[:3])/(box[3:]-box[:3])*np.asarray(grid.shape[1:])).astype(int)
                index = np.clip(index, 0, np.asarray(grid.shape[1:])-1)
                row['intervals'].append({'t_start': float(start), 't_end': float(end), 'index_xyz': index.tolist(), 'occupied': bool(grid[(level, *index)])})
            row.update(t_enter=entry, t_exit=exit)
        reports.append(row)
    return reports


def load_grid_details(path):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as source:
            grid, boxes = validate_grid(source['occupied'].copy(), source['aabbs'].copy())
            return grid, boxes, source['occs'].copy().reshape(grid.shape) if 'occs' in source else None
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('pipeline', state)
    def find(suffix):
        keys = [key for key in state if key.endswith(suffix)]
        if not keys:
            raise ValueError(f'Missing {suffix} tensor')
        value = state[keys[0]].detach().cpu().numpy()
        if any(not np.array_equal(value, state[key].detach().cpu().numpy()) for key in keys[1:]):
            raise ValueError(f'Conflicting shared-grid aliases for {suffix}: {keys}')
        return value
    grid, boxes = validate_grid(find('occupancy_grid.binaries'), find('occupancy_grid.aabbs'))
    return grid, boxes, find('occupancy_grid.occs').reshape(grid.shape)


def load_grid(path):
    grid, boxes, _ = load_grid_details(path)
    return grid, boxes


def classify_cells(grid, occs, radius, cap=.01, mean_mult=1.):
    """EMA threshold evidence, not ground truth; skipped cells remain unverified."""
    if occs is None:
        return None, None
    values = np.asarray(occs)
    if values.shape != grid.shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative occupancy EMA matching grid')
    if not np.isfinite([cap, mean_mult]).all() or cap <= 0 or mean_mult <= 0:
        raise ValueError('Threshold cap and mean multiplier must be positive')
    threshold = min(float(values.mean())*mean_mult, cap)
    core = values > threshold
    halo = dilate_one(core) if radius else core
    labels = np.zeros(grid.shape, np.uint8)
    labels[grid & ~core] = 3  # admission unsupported by the declared local halo
    labels[grid & ~core & halo] = 2
    labels[grid & core] = 1
    labels[~grid & core] = 4
    return labels, threshold


def write_atlas(grid, boxes, output, cell_pixels=2, labels=None):
    """Every slice of each axis, grouped in 16-panel lossless pages."""
    pages = []
    for level in range(len(grid)):
        for axis in range(3):
            remaining = [i for i in range(3) if i != axis]
            width, height = [grid.shape[i+1]*cell_pixels for i in remaining]
            for start in range(0, grid.shape[axis+1], 16):
                canvas = Image.new('RGB', (4*(width+8), 4*(height+26)), '#242424')
                draw = ImageDraw.Draw(canvas)
                for local, index in enumerate(range(start, min(start+16, grid.shape[axis+1]))):
                    # Horizontal = first remaining axis; vertical = second, increasing down.
                    pixels = np.take(grid[level], index, axis=axis).T.astype(np.uint8)*255
                    if labels is not None:
                        palette = np.array([[0,0,0],[70,170,255],[255,190,40],[150,150,150],[255,40,70]], np.uint8)
                        pixels = palette[np.take(labels[level], index, axis=axis).T]
                    panel = Image.fromarray(pixels).resize((width, height), Image.Resampling.NEAREST).convert('RGB')
                    x, y = (local % 4)*(width+8), (local//4)*(height+26)
                    canvas.paste(panel, (x, y+20))
                    draw.text((x+2, y+3), f'{"XYZ"[axis]}={index}', fill='white')
                name = f'level{level}_{"xyz"[axis]}_{start:04d}.png'
                canvas.save(output/name)
                pages.append({'level': level, 'axis': 'xyz'[axis], 'slice_start': start, 'slice_end': min(start+16, grid.shape[axis+1])-1, 'path': name})
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Occupancy atlas</title><h1>Occupancy atlas</h1><p>Binary: white admitted. With EMA labels: blue admitted threshold core; amber declared radius1 guard; gray other admission; red core skipped by traversal; black skipped/unverified. Guard and core are algorithmic evidence, not proven geometry. All cascades are shown independently. Horizontal/vertical axes are the remaining XYZ axes in order; both increase right/down.</p>'+''.join(f'<h2>Level {p["level"]} {p["axis"]} {p["slice_start"]}–{p["slice_end"]}</h2><a href="{html.escape(p["path"])}"><img loading="lazy" style="max-width:100%" src="{html.escape(p["path"])}"></a>' for p in pages))
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, required=True, help='NPZ or trusted local checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=Path, help='JSON: cells [{level,index_xyz}], points [[x,y,z]], rays [{origin,direction,near?,far?}]')
    parser.add_argument('--dilation-radius', type=int, choices=[0, 1], default=0)
    parser.add_argument('--source-dilation-radius', type=int, choices=[0,1], default=0, help='Declared train/eval dilation already present in checkpoint; only for evidence labels')
    parser.add_argument('--threshold-cap', type=float, default=.01)
    parser.add_argument('--threshold-mean-mult', type=float, default=1.)
    parser.add_argument('--cell-pixels', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.cell_pixels <= 8:
        parser.error('--cell-pixels must be between 1 and 8')
    grid, boxes, occs = load_grid_details(args.grid)
    if args.dilation_radius:
        grid = dilate_one(grid)
    args.output.mkdir(parents=True, exist_ok=False)
    labels, threshold = classify_cells(grid, occs, max(args.dilation_radius, args.source_dilation_radius), args.threshold_cap, args.threshold_mean_mult)
    pages = write_atlas(grid, boxes, args.output, args.cell_pixels, labels=labels)
    label_names = ['skipped_unverified', 'admitted_ema_core', 'declared_radius1_guard', 'other_admission_unknown', 'ema_core_skipped']
    if labels is not None:
        np.savez_compressed(args.output/'evidence_labels.npz', labels=labels, aabbs=boxes)
    query = json.loads(args.queries.read_text()) if args.queries else {}
    digest = hashlib.sha256()
    with args.grid.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    report = {'source': str(args.grid.resolve()), 'source_sha256': digest.hexdigest(), 'shape_lxyz': list(grid.shape), 'aabbs': boxes.tolist(), 'dilation_radius': args.dilation_radius, 'source_dilation_radius': args.source_dilation_radius, 'ema_threshold': threshold, 'evidence_counts': None if labels is None else {name: int((labels==i).sum()) for i,name in enumerate(label_names)}, 'occupied_per_level': [int(g.sum()) for g in grid], 'cell_size_xyz': ((boxes[:,3:]-boxes[:,:3])/np.asarray(grid.shape[1:])).tolist(), 'pages': pages, 'cells': [cell_record(grid, boxes, **cell) for cell in query.get('cells', [])], 'points': [{'point': p, 'levels': lookup(grid, boxes, p)} for p in query.get('points', [])], 'rays': [{'query': ray, 'levels': ray_intervals(grid, boxes, **ray)} for ray in query.get('rays', [])], 'limitations': ['Normalized model-world coordinates required.', 'No geometric truth or false-positive/false-negative classification inferred.', 'No learned density/alpha trace; ray intervals show occupancy traversal only.', 'Every cascade shown independently; combined nerfacc cascade selection is not emulated.']}
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'pages': len(pages), 'shape': list(grid.shape)}))


if __name__ == '__main__':
    main()
