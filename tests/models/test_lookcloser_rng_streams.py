"""Model-side independent occupancy RNG stream tests."""

from types import SimpleNamespace

import torch
from torch import nn

import nerfstudio.models.lookcloser as lookcloser_model_module
from nerfstudio.models.lookcloser import LookCloserModel
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng


def _stable_model():
    model = LookCloserModel.__new__(LookCloserModel)
    nn.Module.__init__(model)
    model.device_indicator_param = nn.Parameter(torch.empty(0))
    model.config = SimpleNamespace(
        ray_sampling_mode="adaptive",
        enable_adaptive_ray_marching=True,
        render_step_size=0.01,
        occupancy_update_step_size=None,
        stable_occupancy_reduction=True,
        occupancy_update_interval=4,
        independent_rng_streams=True,
        training_seed=42,
    )
    model.field = SimpleNamespace(density_fn=lambda positions: positions)
    draws = []
    postprocess_steps = []
    model._stable_update_occupancy_grid = lambda **_kwargs: draws.append(torch.rand(16))
    model._postprocess_occupancy_grid = lambda step: postprocess_steps.append(step)
    return model, draws, postprocess_steps


def test_occupancy_callback_uses_boundary_step_stream_and_restores_global_rng() -> None:
    model, draws, postprocess_steps = _stable_model()

    callback = model.get_training_callbacks(SimpleNamespace())[-1]
    torch.manual_seed(7919)
    global_before = torch.random.get_rng_state().clone()
    with fork_seeded_rng(42, "occupancy", 16, "cpu"):
        expected = torch.rand(16)

    callback.run_callback(16)

    assert len(draws) == 1
    assert torch.equal(draws[0], expected)
    assert postprocess_steps == [16]
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_stable_occupancy_non_boundary_skips_fork_update_and_postprocess(monkeypatch) -> None:
    model, draws, postprocess_steps = _stable_model()

    def forbidden_fork(*_args, **_kwargs):
        raise AssertionError("stable non-boundary step must not fork RNG")

    monkeypatch.setattr(lookcloser_model_module, "fork_seeded_rng", forbidden_fork)
    callback = model.get_training_callbacks(SimpleNamespace())[-1]
    torch.manual_seed(7920)
    global_before = torch.random.get_rng_state().clone()

    callback.run_callback(17)

    assert draws == []
    assert postprocess_steps == []
    assert torch.equal(torch.random.get_rng_state(), global_before)
