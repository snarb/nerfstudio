#!/usr/bin/env python3
"""Propose a small render-only R1 repair on failed current actor contour rays.

No checkpoint, learned EMA, MLP, training grid or source archive is changed.
This does not infer true surface depth: every proposed cell must still lie in
the existing source admission and the one-cell neighborhood of actual R0.
Actual render/metric review is required before using a candidate for delivery.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from tune_marshawn_hybrid_cache import sha, write


def intersect_cells(lower, upper, origin, direction):
    parallel = np.abs(direction) < 1e-14
    outside = (parallel & ((origin < lower) | (origin >= upper))).any(1)
    inv = np.divide(1., direction, out=np.zeros(3), where=~parallel)
    a, b = (lower - origin) * inv, (upper - origin) * inv
    near = np.where(parallel, -np.inf, np.minimum(a, b)).max(1)
    far = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
    return ~outside & (far > np.maximum(near, 0.))


def build(snapshot, audit_path, output, max_added_fraction=.05, cell_halo=0):
    snapshot, audit_path, output = Path(snapshot), Path(audit_path), Path(output)
    report = json.loads(audit_path.read_text())
    if sha(snapshot) != report['input_sha256']['runtime_snapshot']:
        raise ValueError('Audit belongs to a different learned grid')
    data = Path(report['data'])
    for key, filename in [('ready', 'ready.json'), ('transforms', 'transforms.json')]:
        if sha(data / filename) != report['input_sha256'][key]:
            raise ValueError('Source inputs changed since contour audit')
    meta = json.loads(snapshot.with_suffix('.json').read_text())
    if meta['grid_archive_sha256'] != sha(snapshot):
        raise ValueError('Unbound grid snapshot metadata')
    if int(meta['checkpoint_step']) + 1 > 140000:
        raise ValueError('Over-cap checkpoint rejected')
    transforms = json.loads((data / 'transforms.json').read_text())
    rows = {r['physical_camera']: r for r in transforms['frames']}
    with np.load(snapshot) as z:
        active, admission = z['active'].copy(), z['admission'].copy()
        bounds = z['cascade_aabbs'].reshape(-1, 2, 3)
    if active.shape != (1, 128, 128, 128) or active.dtype != bool or len(bounds) != 1:
        raise ValueError('One bounded actual 128-cubed runtime grid required')
    if np.any(active & ~admission):
        raise ValueError('R0 already lies outside the source admission')
    available = maximum_filter(active, size=(1, 3, 3, 3), mode='constant', cval=0) & admission & ~active
    ijk = np.argwhere(available[0])
    pitch = (bounds[0, 1] - bounds[0, 0]) / 128
    lower, upper = bounds[0, 0] + ijk * pitch, bounds[0, 0] + (ijk + 1) * pitch
    selected = np.zeros(len(ijk), bool)
    rays, unrepairable = [], []
    for record in report['records']:
        if record['grid_kind'] != 'runtime_active' or not record['failed']:
            continue
        semantic = data / 'semantics' / (record['camera'] + '.npz')
        failed = Path(record['failed_pixels'])
        if sha(semantic) != record['semantics_sha256'] or sha(failed) != record['failed_pixels_sha256']:
            raise ValueError('Current actor evidence changed')
        with np.load(failed) as z:
            pixels = z['pixel_HD_xy'].copy()
        if len(pixels) != record['failed']:
            raise ValueError('Failure count differs from bound pixel archive')
        with np.load(semantic) as z:
            actor = z[record['actor']].copy()
        row = rows[record['camera']]
        pose = np.asarray(row['transform_matrix'])
        for u, v in pixels:
            if not actor[v, u]:
                raise ValueError('Failed ray is no longer current actor support')
            direction = pose[:3, :3] @ np.array([(u + .5 - row['cx']) / row['fl_x'],
                                                -(v + .5 - row['cy']) / row['fl_y'], -1.])
            hit = intersect_cells(lower, upper, pose[:3, 3], direction)
            selected |= hit
            ray = dict(camera=record['camera'], actor=record['actor'], pixel_HD=[int(u), int(v)],
                       proposed_cells=int(hit.sum()))
            rays.append(ray)
            if not hit.any():
                unrepairable.append(ray)
    added = np.zeros_like(active)
    ids = ijk[selected]
    if len(ids):
        added[0, ids[:, 0], ids[:, 1], ids[:, 2]] = True
    if cell_halo not in (0, 1):
        raise ValueError('Only zero or one local pixel-footprint guard cell is qualified')
    if cell_halo:
        # Keep the GLOBAL R1 boundary. This is a local guard for the finite
        # source pixel footprint; it never turns the proposal into global R2.
        added = maximum_filter(added, size=(1, 3, 3, 3), mode='constant', cval=0) & available
    candidate = active | added
    assert np.all(candidate[active]) and not np.any(added & ~admission)
    fraction = int(added.sum()) / max(1, int(active.sum()))
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        kind='CURRENT_CONTOUR_LOCAL_R1_RENDER_CANDIDATE_V1', frame=report['frame'],
        checkpoint_sha256=meta['checkpoint_sha256'], checkpoint_step=meta['checkpoint_step'],
        runtime_snapshot_sha256=sha(snapshot), audit_sha256=sha(audit_path),
        data_ready_sha256=sha(data / 'ready.json'),
        code_sha256=sha(__file__), original_cells=int(active.sum()), added_cells=int(added.sum()),
        added_fraction=fraction, candidate_cells=int(candidate.sum()), local_cell_halo=cell_halo,
        failed_source_rays=rays,
        unrepairable_rays=unrepairable, size_budget_fraction=max_added_fraction,
        eligible_for_render_experiment=not unrepairable and fraction <= max_added_fraction,
        production_approved=False, model_or_source_changed=False,
        actual_surface_depth_certified=False,
        limitation='Positive contour rays constrain traversal; they do not establish hidden surface depth.')
    with (output / 'creation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination, metadata = output / 'candidate.npz', output / 'candidate.json'
        if metadata.exists():
            old = json.loads(metadata.read_text())
            if any(old.get(k) != receipt[k] for k in receipt) or sha(destination) != old['candidate_sha256']:
                raise ValueError('Preserve the existing candidate; use a new versioned output directory')
            return
        if destination.exists():
            # Recover a crash after NPZ commit without rewriting bound bytes.
            with np.load(destination) as old:
                if not np.array_equal(old['active'], candidate) or not np.array_equal(old['added'], added):
                    raise ValueError('Interrupted candidate differs from the current proposal')
        else:
            temporary = output / 'candidate.pending.npz'
            np.savez_compressed(temporary, active=candidate, added=added)
            temporary.replace(destination)
        write(metadata, dict(receipt, candidate_sha256=sha(destination)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-added-fraction', type=float, default=.05)
    parser.add_argument('--cell-halo', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    build(args.snapshot, args.audit, args.output, args.max_added_fraction, args.cell_halo)
