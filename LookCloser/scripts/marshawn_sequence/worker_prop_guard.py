"""Isolated prop-carving ablation; original production modules stay immutable.

Exclude conservative projected prop capsules from the empty-background prior.
This does not synthesize RGB/alpha or force any ray to be opaque.
"""
import argparse
from pathlib import Path
import numpy as np
from common import read, sha, atomic
import worker_background as background


def exclude_capsule(mask, endpoints, radius):
    """Remove a clipped 2-D capsule from a background-eligibility raster in place."""
    p, q = np.asarray(endpoints, dtype=np.float64)
    radius = float(radius)
    if not np.isfinite([*p, *q, radius]).all() or radius <= 0:
        raise ValueError('Finite capsule and positive radius required')
    h, w = mask.shape
    lo = np.maximum(0, np.floor(np.minimum(p, q)-radius).astype(int))
    hi = np.minimum([w, h], np.ceil(np.maximum(p, q)+radius+1).astype(int))
    if np.any(hi <= lo): return 0
    x, y = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
    v = q-p
    t = np.clip(((x-p[0])*v[0]+(y-p[1])*v[1])/max(float(v@v), 1e-12), 0, 1)
    protected = (x-p[0]-t*v[0])**2+(y-p[1]-t*v[1])**2 <= radius**2
    patch = mask[lo[1]:hi[1], lo[0]:hi[0]]
    removed = int(np.count_nonzero(patch & protected))
    patch[protected] = False
    return removed


def train(spec_path):
    spec = read(spec_path)
    path = Path(spec['prop_guard_manifest'])
    if sha(path) != spec['prop_guard_manifest_sha256']:
        raise ValueError('Prop geometry manifest changed')
    manifest = read(path)
    if manifest['kind'] != 'PROJECTED_PROP_BACKGROUND_GUARD_V1':
        raise ValueError('Unexpected prop guard schema')
    original = background.trusted_background
    records = []
    def guarded(semantic, unknown):
        mask = original(semantic, unknown)
        name = Path(semantic.zip.filename).stem
        row = manifest['cameras'][name]
        if list(mask.shape) != row['shape_hw']:
            raise ValueError('Prop guard camera dimensions differ')
        removed = exclude_capsule(mask, row['endpoints_xy'], row['radius_px'])
        records.append(dict(camera=name, background_pixels_excluded=removed))
        return mask
    background.trusted_background = guarded
    original_install = background.install
    def install(pipe, parameters):
        original_install(pipe, parameters)
        atomic(Path(parameters['output'])/'prop_guard_audit.json', dict(
            manifest_sha256=sha(path), records=records,
            changes='Only empty-background prior eligibility; no RGB/alpha synthesis'))
    background.worker.install_core = install
    background.worker.train(spec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    train(parser.parse_args().spec)
