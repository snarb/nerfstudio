"""Portable regressions for the validated opt-in density and SH paths."""
from types import SimpleNamespace
import pytest
import torch
from nerfstudio.fields.lookcloser_field import LookCloserField


def test_legacy_softplus_exact():
    x = torch.linspace(-12, 12, 65, dtype=torch.float16)
    actual = LookCloserField.activate_density(SimpleNamespace(density_normalization="none", density_activation="softplus"), x)
    assert actual.dtype == x.dtype
    assert torch.equal(actual, torch.nn.functional.softplus(x + 1))


def test_exp_cast_precedes_bias_and_activation():
    x = torch.tensor([10.703], dtype=torch.float16, requires_grad=True)
    actual = LookCloserField.activate_density(SimpleNamespace(density_normalization="none", density_activation="trunc_exp"), x)
    assert actual.dtype == torch.float32 and actual.isfinite().all()
    torch.testing.assert_close(actual, torch.exp(x.float() + 1))
    (actual * 1e-4).sum().backward()
    assert x.grad.isfinite().all() and (x.grad > 0).all()


@pytest.mark.parametrize("correct", [False, True])
def test_sh_encoder_domain(correct):
    directions = torch.tensor([[-1., 0., 1.], [0., 1., 0.]])
    field = SimpleNamespace(correct_sh_directions=correct, direction_encoding=lambda x: x)
    expected = (directions + 1) / 2 if correct else directions
    assert torch.equal(LookCloserField.encode_directions(field, directions), expected)
