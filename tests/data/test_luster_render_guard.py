"""A render envelope may remove occupied cells, never enable or move them."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "LookCloser" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from luster_render_guard import apply_guard


def fixture(tmp_path, **overrides):
    grid = SimpleNamespace(
        binaries=torch.tensor([[[[True, False], [True, True]]]]),
        aabbs=torch.tensor([[0., 0., 0., 1., 1., 1.]]),
        resolution=torch.tensor([1, 2, 2]),
    )
    values = dict(allowed=np.array([[[[True, True], [False, True]]]]),
                  aabbs=grid.aabbs.numpy(), resolution=grid.resolution.numpy())
    values.update(overrides)
    path = tmp_path / "guard.npz"
    np.savez_compressed(path, **values)
    record = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return SimpleNamespace(model=SimpleNamespace(occupancy_grid=grid)), record


def test_guard_only_removes_cells_and_is_idempotent(tmp_path):
    pipe, record = fixture(tmp_path)
    before = pipe.model.occupancy_grid.binaries.clone()
    stats = apply_guard(pipe, record)
    after = pipe.model.occupancy_grid.binaries.clone()
    assert stats['occupied_before'] == 3 and stats['occupied_after'] == 2
    assert not (after & ~before).any()
    apply_guard(pipe, record)
    assert torch.equal(after, pipe.model.occupancy_grid.binaries)


@pytest.mark.parametrize('mismatch', ['hash', 'shape', 'coordinates'])
def test_guard_rejects_mismatched_provenance_before_mutation(tmp_path, mismatch):
    overrides = {}
    if mismatch == 'shape':
        overrides['allowed'] = np.ones((2, 2), dtype=bool)
    if mismatch == 'coordinates':
        overrides['aabbs'] = np.array([[0., 0., 0., 2., 1., 1.]])
    pipe, record = fixture(tmp_path, **overrides)
    if mismatch == 'hash':
        record['sha256'] = 'wrong'
    before = pipe.model.occupancy_grid.binaries.clone()
    with pytest.raises(ValueError):
        apply_guard(pipe, record)
    assert torch.equal(before, pipe.model.occupancy_grid.binaries)
