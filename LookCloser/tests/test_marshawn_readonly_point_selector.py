import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location('readonly_selector', Path(__file__).parents[1] / 'scripts/marshawn_readonly_point_selector.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Field:
    def _normalize_positions(self, positions):
        return positions * 2, positions[..., 0] >= 0, positions.shape, positions


def test_identity_and_exact_bound_method_restore():
    field = Field()
    positions = torch.tensor([[-1., 0., 0.], [1., 0., 0.]], requires_grad=True)
    before = field._normalize_positions(positions)
    original = module.compose_normalization(field, lambda p: torch.ones(p.shape[:-1], dtype=torch.bool))
    after = field._normalize_positions(positions)
    assert torch.equal(after[0], before[0]) and torch.equal(after[1], before[1])
    assert after[2] == before[2] and after[3] is positions
    after[0].sum().backward()
    assert torch.equal(positions.grad, torch.full_like(positions, 2))
    module.restore_normalization(field, original)
    assert field._normalize_positions is original


def test_lookup_can_only_remove_native_admission():
    field = Field()
    positions = torch.tensor([[-1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    module.compose_normalization(field, lambda p: torch.tensor([True, False, True]))
    assert field._normalize_positions(positions)[1].tolist() == [False, False, True]


@pytest.mark.parametrize('lookup', [lambda p: torch.ones(2), lambda p: torch.ones((2, 1), dtype=torch.bool)])
def test_selector_shape_or_dtype_mismatch_rejected(lookup):
    field = Field()
    module.compose_normalization(field, lookup)
    with pytest.raises(ValueError, match='Boolean selector'):
        field._normalize_positions(torch.zeros((2, 3)))
