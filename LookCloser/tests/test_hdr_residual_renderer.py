"""Unit checks for the bounded HDR residual renderer."""

from LookCloser.scripts.train_hdr_residual_renderer import HDRResidualRenderer

import torch


def test_zero_initialized_renderer_preserves_primary_input():
    model = HDRResidualRenderer(channels=8, correction_limit=0.04)
    primary = torch.rand(2, 3, 24, 32)
    auxiliary = torch.rand_like(primary)

    prediction = model(primary, auxiliary)

    torch.testing.assert_close(prediction, primary)


def test_renderer_correction_is_bounded():
    model = HDRResidualRenderer(channels=8, correction_limit=0.04)
    torch.nn.init.normal_(model.tail.weight)
    primary = torch.full((1, 3, 24, 32), 0.5)
    auxiliary = torch.rand_like(primary)

    prediction = model(primary, auxiliary)

    assert torch.max(torch.abs(prediction - primary)) <= 0.040001
