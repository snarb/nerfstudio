"""Correctness helpers for LookCloser occupancy-grid updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def admit_occupancy_points_(grid: Tensor, aabbs: Tensor, points: Tensor) -> None:
    """Conservatively add point-containing cells; retain all existing occupancy."""
    if grid.dtype != torch.bool or grid.ndim != 4 or aabbs.shape != (len(grid), 6):
        raise ValueError('Expected boolean cascaded occupancy and matching AABBs')
    resolution = points.new_tensor(grid.shape[1:])
    for level, box in enumerate(aabbs):
        normalized = (points-box[:3])/(box[3:]-box[:3])
        inside = torch.isfinite(normalized).all(-1) & ((normalized >= 0) & (normalized < 1)).all(-1)
        ijk = torch.floor(normalized[inside]*resolution).long()
        if len(ijk):
            grid[level, ijk[:,0], ijk[:,1], ijk[:,2]] = True


def load_occupancy_prior(manifest_path, aabbs: Tensor, shape, transforms_sha256: str) -> Tensor:
    """Load a hash-qualified train-only spatial admission grid; no density veto."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if not manifest.get('train_only') or not transforms_sha256 or manifest.get('transforms_sha256') != transforms_sha256:
        raise ValueError('Occupancy prior requires matching train-only calibration')
    archive = path.parent/manifest['grid_path']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest['grid_sha256']:
        raise ValueError('Occupancy prior grid hash mismatch')
    with np.load(archive, allow_pickle=False) as values:
        grid = values['occupied'].copy()
        boxes = values['aabbs'].copy()
    if grid.dtype != np.bool_ or tuple(grid.shape) != tuple(shape):
        raise ValueError('Occupancy prior resolution/levels mismatch')
    if not np.array_equal(boxes, aabbs.detach().cpu().numpy()):
        raise ValueError('Occupancy prior AABB mismatch')
    return torch.from_numpy(grid).to(aabbs.device)


@torch.no_grad()
def stable_ema_max_update_(
    state: Tensor,
    cell_ids: Tensor,
    candidates: Tensor,
    ema_decay: float,
) -> None:
    """Apply one max-with-decay update per touched cell, independent of duplicate order."""

    if state.ndim != 1 or cell_ids.ndim != 1 or candidates.ndim != 1:
        raise ValueError("state, cell_ids, and candidates must be one-dimensional.")
    if cell_ids.numel() != candidates.numel():
        raise ValueError("cell_ids and candidates must contain the same number of values.")
    if not 0.0 < float(ema_decay) <= 1.0:
        raise ValueError("ema_decay must be in (0, 1].")
    if cell_ids.numel() == 0:
        return
    if cell_ids.dtype != torch.long:
        raise ValueError("cell_ids must use torch.long indices.")
    if candidates.device != state.device or cell_ids.device != state.device:
        raise ValueError("state, cell_ids, and candidates must be on the same device.")
    if candidates.dtype != state.dtype:
        candidates = candidates.to(dtype=state.dtype)

    per_cell = torch.full_like(state, -torch.inf)
    per_cell.scatter_reduce_(0, cell_ids, candidates, reduce="amax", include_self=True)
    touched = ~torch.isneginf(per_cell)
    state[touched] = torch.maximum(state[touched] * float(ema_decay), per_cell[touched])
