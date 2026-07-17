"""Stable occupancy-grid reduction tests."""

import pytest
import torch

from nerfstudio.model_components.lookcloser_occupancy import stable_ema_max_update_


def test_duplicate_order_does_not_change_update() -> None:
    initial = torch.tensor([10.0, 20.0, 30.0])
    ids = torch.tensor([0, 0, 1, 0], dtype=torch.long)
    values = torch.tensor([8.0, 12.0, 5.0, 11.0])

    forward = initial.clone()
    stable_ema_max_update_(forward, ids, values, ema_decay=0.5)
    reverse = initial.clone()
    order = torch.arange(ids.numel() - 1, -1, -1)
    stable_ema_max_update_(reverse, ids[order], values[order], ema_decay=0.5)

    assert torch.equal(forward, torch.tensor([12.0, 10.0, 30.0]))
    assert torch.equal(reverse, forward)


def test_empty_update_leaves_state_unchanged() -> None:
    state = torch.tensor([1.0, 2.0])
    stable_ema_max_update_(
        state,
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        ema_decay=0.95,
    )
    assert torch.equal(state, torch.tensor([1.0, 2.0]))


def test_unique_ids_match_legacy_indexed_update() -> None:
    state = torch.tensor([3.0, 6.0, 9.0, 12.0])
    ids = torch.tensor([0, 2, 3], dtype=torch.long)
    values = torch.tensor([4.0, 1.0, 20.0])
    legacy = state.clone()
    legacy[ids] = torch.maximum(legacy[ids] * 0.5, values)

    stable = state.clone()
    stable_ema_max_update_(stable, ids, values, ema_decay=0.5)
    assert torch.equal(stable, legacy)


def test_random_duplicates_match_slow_per_cell_reference() -> None:
    generator = torch.Generator().manual_seed(1701)
    for num_cells, num_updates in ((1, 17), (7, 128), (64, 4096)):
        initial = torch.rand(num_cells, generator=generator)
        ids = torch.randint(num_cells, (num_updates,), generator=generator)
        values = torch.rand(num_updates, generator=generator)
        expected = initial.clone()
        for cell_id in torch.unique(ids, sorted=True):
            mask = ids == cell_id
            expected[cell_id] = torch.maximum(expected[cell_id] * 0.95, values[mask].max())

        actual = initial.clone()
        stable_ema_max_update_(actual, ids, values, ema_decay=0.95)
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_parity_with_duplicates() -> None:
    generator = torch.Generator().manual_seed(42)
    ids = torch.randint(64, (4096,), generator=generator)
    values = torch.rand(4096, generator=generator)
    cpu = torch.rand(64, generator=generator)
    cuda = cpu.cuda()

    stable_ema_max_update_(cpu, ids, values, ema_decay=0.95)
    stable_ema_max_update_(cuda, ids.cuda(), values.cuda(), ema_decay=0.95)
    assert torch.equal(cuda.cpu(), cpu)
