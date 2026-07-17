"""Exact parity properties for discrete feature-reweighting lookup."""

import pytest
import torch
from torch import nn

import nerfstudio.fields.lookcloser_field as lookcloser_field_module
from nerfstudio.fields.lookcloser_field import LookCloserField


def _field(strength: float) -> LookCloserField:
    field = LookCloserField.__new__(LookCloserField)
    nn.Module.__init__(field)
    field.num_levels = 16
    field.features_per_level = 2
    field.l_min = 0.0
    field.l_max = 15.0
    field.feature_reweighting_strength = strength
    field.register_buffer(
        "_feature_weight_lut", torch.empty((0, 32), dtype=torch.float32), persistent=False
    )
    field._feature_weight_lut_strength = None
    return field


@pytest.mark.parametrize("strength", [0.3, 1.0])
def test_discrete_lut_is_bit_exact_for_every_grid_level(strength: float) -> None:
    field = _field(strength)
    levels = torch.arange(16, dtype=torch.float32).repeat_interleave(7).unsqueeze(-1)
    analytical = field.get_weights(levels, batch_size=levels.shape[0])
    looked_up = field._get_discrete_feature_weights(levels)
    assert torch.equal(looked_up, analytical)


def test_discrete_lut_is_rebuilt_when_live_strength_changes() -> None:
    field = _field(1.0)
    levels = torch.tensor([[0.0], [7.0], [15.0]])
    initial = field._get_discrete_feature_weights(levels).clone()
    field.feature_reweighting_strength = 0.3
    switched = field._get_discrete_feature_weights(levels)
    analytical = field.get_weights(levels, batch_size=levels.shape[0])
    assert not torch.equal(switched, initial)
    assert torch.equal(switched, analytical)


class _FakeTCNNNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.jit_fusion = False


def test_live_tcnn_jit_switch_preserves_networks_and_parameters(monkeypatch) -> None:
    field = LookCloserField.__new__(LookCloserField)
    nn.Module.__init__(field)
    field.mlp_geo = _FakeTCNNNetwork()
    field.mlp_color = _FakeTCNNNetwork()
    monkeypatch.setattr(lookcloser_field_module.tcnn, "supports_jit_fusion", lambda: True)

    network_ids = (id(field.mlp_geo), id(field.mlp_color))
    parameter_ids = tuple(id(parameter) for parameter in field.parameters())
    parameter_values = tuple(parameter.detach().clone() for parameter in field.parameters())

    field.set_tcnn_network_jit(True)

    assert field.get_tcnn_network_jit() is True
    assert (id(field.mlp_geo), id(field.mlp_color)) == network_ids
    assert tuple(id(parameter) for parameter in field.parameters()) == parameter_ids
    for parameter, expected in zip(field.parameters(), parameter_values):
        torch.testing.assert_close(parameter, expected, rtol=0.0, atol=0.0)
