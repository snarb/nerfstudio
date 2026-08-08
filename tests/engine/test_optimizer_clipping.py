"""Regression tests for optimizer clipping used by HDR weighted reconstruction."""

import torch

from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers


def test_elementwise_gradient_clip_is_applied_and_not_forwarded_to_adam() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    config = AdamOptimizerConfig(lr=1e-2, max_value=0.1)
    optimizers = Optimizers(
        {"fields": {"optimizer": config, "scheduler": None}},
        {"fields": [parameter]},
    )
    parameter.grad = torch.tensor([-5.0, -0.05, 0.2, 7.0])
    optimizers.optimizer_step_all()
    torch.testing.assert_close(parameter.grad, torch.tensor([-0.1, -0.05, 0.1, 0.1]))
