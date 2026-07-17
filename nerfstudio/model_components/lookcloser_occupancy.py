"""Correctness helpers for LookCloser occupancy-grid updates."""

from __future__ import annotations

import torch
from torch import Tensor


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
