"""Actual-core plus fixed semantic admission snapshots and read-only audits.

Attach after model setup, before the first training callback. No CPU threshold
reconstruction is authoritative. A skipped cell means traversal skips it, NOT
known empty geometry. This module never changes the training grid.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import types

import numpy as np

from marshawn_occupancy_audit import dilate_one, lookup, ray_intervals, validate_grid, write_atlas


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def install_capture_hook(model):
    """The opt-in subclass captures actual core before applying fixed admission."""
    if not hasattr(model, 'semantic_receipt') or not hasattr(model, 'semantic_learned_core'):
        raise ValueError('Semantic model subclass required; generic inference reload is unsafe')
    return lambda: None


def validate_policy(core, active, radius, binary_warmup=False, admission=None):
    if core.dtype != np.bool_ or active.dtype != np.bool_ or core.shape != active.shape:
        raise ValueError('Core and active must have identical boolean shapes')
    if radius not in (0, 1):
        raise ValueError('Only radius0/1 supported')
    expected = np.ones_like(core) if binary_warmup else (core if radius == 0 else dilate_one(core))
    if admission is None or admission.dtype != np.bool_ or admission.shape != core.shape:
        raise ValueError('Explicit fixed admission required')
    expected = expected & admission
    if not np.array_equal(active, expected):
        raise ValueError('Active grid violates actual-core noncumulative policy')


def export_snapshot(model, checkpoint, step, output_dir, *, sources=None):
    """Write actual GPU masks/EMA bound to an already saved checkpoint."""
    capture = getattr(model, '_tuning_occupancy_capture', None)
    if capture is None:
        raise ValueError('No actual core captured; wait for a fresh native update')
    period = int(model.config.occupancy_update_interval)
    if not 0 <= int(step) - capture['step'] < period:
        raise ValueError('Captured core is stale for checkpoint')
    grid = model.occupancy_grid
    core, active = _numpy(capture['core']), _numpy(grid.binaries)
    if not np.array_equal(active, _numpy(capture['active'])):
        raise ValueError('Grid changed after captured native policy')
    admission = _numpy(model.semantic_admission)[None]
    validate_policy(core, active, capture['radius'], capture['binary_warmup'], admission)
    active, boxes = validate_grid(active, _numpy(grid.aabbs))
    ema = _numpy(grid.occs).reshape(core.shape)
    if not np.isfinite(ema).all():
        raise ValueError('Nonfinite occupancy EMA')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f'step-{int(step):09d}_grid.npz'
    receipt = archive.with_suffix('.json')
    if archive.exists() or receipt.exists():
        raise FileExistsError('Snapshot paths are immutable')
    temporary = archive.with_name(archive.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, core=core, active=active, ema=ema, admission=admission,
                            cascade_aabbs=boxes, resolution=np.asarray(core.shape[1:]))
    os.replace(temporary, archive)
    record = dict(schema=2, semantic_receipt=model.semantic_receipt, admission_cells=int(admission.sum()), checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=sha(checkpoint), checkpoint_step=int(step),
                  grid_archive_sha256=sha(archive), core_update_step=capture['step'],
                  native_period=period, radius=capture['radius'],
                  binary_warmup=capture['binary_warmup'], core_cells=int(core.sum()),
                  active_cells=int(active.sum()), sources_sha256=sources or {},
                  exporter_sha256=sha(__file__), authority='actual GPU core before postprocess',
                  cpu_reconstruction_used=False,
                  geometry_claim='Traversal admission only; unseen/skipped cells are not certified free')
    atomic_json(receipt, record)
    return receipt


def load_snapshot(receipt, checkpoint=None):
    receipt = Path(receipt)
    record = json.loads(receipt.read_text())
    archive = receipt.with_suffix('.npz')
    if sha(archive) != record['grid_archive_sha256']:
        raise ValueError('Grid archive hash mismatch')
    if checkpoint is not None and sha(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    with np.load(archive, allow_pickle=False) as src:
        core, active = src['core'].copy(), src['active'].copy()
        active, boxes = validate_grid(active, src['cascade_aabbs'])
        ema = src['ema'].copy()
        admission = src['admission'].copy()
        if not np.array_equal(src['resolution'], core.shape[1:]):
            raise ValueError('Resolution mismatch')
    validate_policy(core, active, record['radius'], record['binary_warmup'], admission)
    if ema.shape != core.shape or not np.isfinite(ema).all():
        raise ValueError('Invalid EMA')
    if (int(core.sum()) != record['core_cells'] or int(active.sum()) != record['active_cells'] or
            not 0 <= record['checkpoint_step'] - record['core_update_step'] < record['native_period']):
        raise ValueError('Counts or update freshness disagree')
    return core, active, ema, boxes, record


def audit_snapshot(receipt, output_dir, *, checkpoint=None, queries=None):
    """Every slice, actual admission labels and exact per-cascade ray intervals."""
    core, active, ema, boxes, source = load_snapshot(receipt, checkpoint)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = np.zeros(core.shape, np.uint8)
    labels[core & active] = 1
    labels[active & ~core] = 3 if source['binary_warmup'] else 2
    # Existing atlas colors: 0 skipped,1 core,2 halo. Warmup is explicitly named.
    (out / 'atlas').mkdir(exist_ok=True)
    pages = write_atlas(active, boxes, out / 'atlas', 1, labels=labels)
    queries = queries or {}
    points = [{'xyz': p, 'core': lookup(core, boxes, p), 'active': lookup(active, boxes, p)}
              for p in queries.get('points', [])]
    rays = []
    for ray in queries.get('rays', []):
        kwargs = {k: ray[k] for k in ['near', 'far'] if k in ray}
        rays.append(dict(query=ray, core=ray_intervals(core, boxes, ray['origin'], ray['direction'], **kwargs),
                         active=ray_intervals(active, boxes, ray['origin'], ray['direction'], **kwargs)))
    report = dict(source=source, receipt_sha256=sha(receipt), all_cells_scanned=int(core.size),
                  counts={'core': int(core.sum()), 'admitted_noncore': int((active & ~core).sum()),
                          'skipped_unknown_geometry': int((~active).sum())},
                  ema_min=float(ema.min()), ema_max=float(ema.max()), pages=pages,
                  points=points, rays=rays, complete_manual_review=False,
                  limitations=['Labels describe traversal, not true geometry.',
                               'Per-cascade ray intervals do not emulate combined nerfacc cascade selection.',
                               'Fixed TRAIN semantic admission is not certified physical free space; learned core and EMA remain separate.',
                               'During binary warmup noncore admitted cells are warmup, not halo.'])
    atomic_json(out / 'audit.json', report)
    return report


def scan_subcell_density(core, active, boxes, density_fn, *, chunk_size=32768):
    """Probe center and eight quarter-cell positions of EVERY cell.

    density_fn takes float64 numpy[N,3] model-world coordinates and returns
    nonnegative numpy[N] sigma. Caller handles device/dtype/no_grad. This is
    independent of traversal and does not classify low-density cells as free.
    Returned max sigma is a finite-sampling diagnostic, not continuous support.
    """
    active, boxes = validate_grid(active, boxes)
    if core.shape != active.shape or core.dtype != np.bool_ or chunk_size < 1:
        raise ValueError('Invalid core or chunk size')
    offsets = np.array([(0.5, 0.5, 0.5), *itertools.product((0.25, 0.75), repeat=3)])
    dims = np.array(active.shape[1:])
    count = int(np.prod(dims))
    maximum = np.zeros(active.shape, np.float32)
    for level, box in enumerate(boxes):
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            indices = np.array(np.unravel_index(np.arange(start, stop), dims)).T
            peak = np.zeros(stop - start)
            for offset in offsets:
                xyz = box[:3] + (indices + offset) / dims * (box[3:] - box[:3])
                values = np.asarray(density_fn(xyz)).reshape(-1)
                if values.shape != peak.shape or not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Density callback must return finite nonnegative sigma per point')
                peak = np.maximum(peak, values)
            maximum[level].reshape(-1)[start:stop] = peak
    return maximum, dict(probes=int(active.size * len(offsets)), cells=int(active.size),
                         offsets=offsets.tolist(), skipped_cells=int((~active).sum()),
                         skipped_sigma_max=float(maximum[~active].max()) if (~active).any() else 0.,
                         geometry_certificate=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=Path)
    args = parser.parse_args()
    report = audit_snapshot(args.receipt, args.output, checkpoint=args.checkpoint,
                            queries=json.loads(args.queries.read_text()) if args.queries else None)
    print(json.dumps({'cells': report['all_cells_scanned'], 'counts': report['counts']}))
