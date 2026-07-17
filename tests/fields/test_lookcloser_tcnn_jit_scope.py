"""CPU-only properties for scoped TCNN network JIT state."""

import pytest
import torch
from torch import nn

import nerfstudio.fields.lookcloser_field as lookcloser_field_module
from nerfstudio.fields.lookcloser_field import LookCloserField
from nerfstudio.models.lookcloser import LookCloserModelConfig


class _FakeTCNNNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.jit_fusion = False


def _field() -> LookCloserField:
    field = LookCloserField.__new__(LookCloserField)
    nn.Module.__init__(field)
    field.mlp_geo = _FakeTCNNNetwork()
    field.mlp_color = _FakeTCNNNetwork()
    return field


def test_model_config_preserves_default_off_legacy_both_behavior() -> None:
    config = LookCloserModelConfig()
    assert config.tcnn_network_jit is False
    assert config.tcnn_network_jit_scope == "both"


@pytest.mark.parametrize(
    ("scope", "expected_geo", "expected_color"),
    [("both", True, True), ("geometry", True, False), ("color", False, True)],
)
def test_scoped_jit_changes_only_selected_networks(
    monkeypatch, scope: str, expected_geo: bool, expected_color: bool
) -> None:
    field = _field()
    monkeypatch.setattr(lookcloser_field_module.tcnn, "supports_jit_fusion", lambda: True)

    field.set_tcnn_network_jit(True, scope=scope)

    assert bool(field.mlp_geo.jit_fusion) is expected_geo
    assert bool(field.mlp_color.jit_fusion) is expected_color
    assert field.get_tcnn_network_jit(scope=scope) is True


def test_legacy_both_scope_and_scoped_getters_handle_mixed_state(monkeypatch) -> None:
    field = _field()
    monkeypatch.setattr(lookcloser_field_module.tcnn, "supports_jit_fusion", lambda: True)

    field.set_tcnn_network_jit(True, scope="color")

    assert field.get_tcnn_network_jit(scope="geometry") is False
    assert field.get_tcnn_network_jit(scope="color") is True
    with pytest.raises(RuntimeError, match="states disagree"):
        field.get_tcnn_network_jit()


def test_scoped_setter_is_idempotent(monkeypatch) -> None:
    field = _field()
    support_checks = 0

    def _supports_jit_fusion() -> bool:
        nonlocal support_checks
        support_checks += 1
        return True

    monkeypatch.setattr(lookcloser_field_module.tcnn, "supports_jit_fusion", _supports_jit_fusion)

    field.set_tcnn_network_jit(True, scope="geometry")
    field.set_tcnn_network_jit(True, scope="geometry")

    assert support_checks == 1
    assert field.get_tcnn_network_jit(scope="geometry") is True
    assert field.get_tcnn_network_jit(scope="color") is False


def test_invalid_jit_scope_fails_closed() -> None:
    field = _field()
    with pytest.raises(ValueError, match="Unsupported TCNN network JIT scope"):
        field.set_tcnn_network_jit(True, scope="invalid")
