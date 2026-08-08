"""Exact training-state parity for optional occupancy-grid diagnostics."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from nerfstudio.models.lookcloser import LookCloserModel, LookCloserModelConfig


@pytest.fixture(params=("cpu", "cuda"))
def device(request) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


def _training_probe(self, _input: torch.Tensor):
    active = self.occupancy_grid.occs[self.occupancy_grid.binaries.reshape(-1)]
    return {"probe": active.sum() * self.probe_weight}


def _model(
    device: torch.device,
    *,
    diagnostics: bool,
    clamp_mult: float,
    dilation_radius: int,
    binary_warmup_steps: int,
) -> LookCloserModel:
    model = LookCloserModel.__new__(LookCloserModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        rgb_output_parameterization="sigmoid",
        occupancy_diagnostics=diagnostics,
        occupancy_occ_thre=0.45,
        occupancy_thre_clamp_mult=clamp_mult,
        occupancy_dilation_radius=dilation_radius,
        occupancy_binary_warmup_steps=binary_warmup_steps,
        alpha_thre=0.02,
    )
    model.device_indicator_param = nn.Parameter(torch.empty(0, device=device), requires_grad=False)
    occs = torch.tensor(
        [
            0.00,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            0.42,
            0.17,
            0.73,
            0.29,
            0.61,
            0.08,
        ],
        device=device,
    )
    binaries = torch.tensor(
        [
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
        ],
        device=device,
    ).view(1, 3, 3, 3)
    grid = SimpleNamespace(occs=occs, binaries=binaries)
    model.occupancy_grid = grid
    model.adaptive_sampler = SimpleNamespace(occupancy_grid=grid)
    model._last_occupancy_binaries = ~binaries.clone()
    model._last_occupancy_stats = {"stale_occupancy_metric": -1.0}
    model.probe_weight = nn.Parameter(torch.tensor(1.25, device=device))
    model.collider = None
    model.get_outputs = MethodType(_training_probe, model)
    model.train()
    return model


@pytest.mark.parametrize(
    "clamp_mult,dilation_radius,binary_warmup_steps,step",
    [
        pytest.param(1.0, 0, 0, 4096, id="historical-defaults"),
        pytest.param(0.55, 1, 0, 4096, id="clamp-and-dilation"),
        pytest.param(0.55, 1, 32, 16, id="clamp-dilation-and-binary-warmup"),
    ],
)
def test_diagnostics_toggle_is_exact_for_training_state_and_forward(
    device: torch.device,
    clamp_mult: float,
    dilation_radius: int,
    binary_warmup_steps: int,
    step: int,
) -> None:
    enabled = _model(
        device,
        diagnostics=True,
        clamp_mult=clamp_mult,
        dilation_radius=dilation_radius,
        binary_warmup_steps=binary_warmup_steps,
    )
    disabled = _model(
        device,
        diagnostics=False,
        clamp_mult=clamp_mult,
        dilation_radius=dilation_radius,
        binary_warmup_steps=binary_warmup_steps,
    )
    initial_occs = enabled.occupancy_grid.occs.clone()
    rng_before = torch.random.get_rng_state().clone()

    enabled._postprocess_occupancy_grid(step)
    disabled._postprocess_occupancy_grid(step)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert torch.equal(enabled.occupancy_grid.occs, initial_occs)
    assert torch.equal(disabled.occupancy_grid.occs, initial_occs)
    assert torch.equal(disabled.occupancy_grid.binaries, enabled.occupancy_grid.binaries)
    assert enabled.adaptive_sampler.occupancy_grid is enabled.occupancy_grid
    assert disabled.adaptive_sampler.occupancy_grid is disabled.occupancy_grid
    assert enabled._last_occupancy_binaries is not None
    assert enabled._last_occupancy_stats.keys() == {
        "occupancy_ratio",
        "occupancy_ratio_level0",
        "occupancy_occs_mean",
        "occupancy_occs_max",
        "occupancy_effective_threshold",
        "occupancy_default_threshold",
        "occupancy_effective_alpha_thre",
        "occupancy_flipped_on",
        "occupancy_flipped_off",
    }
    assert disabled._last_occupancy_binaries is None
    assert disabled._last_occupancy_stats == {}

    enabled_output = enabled(torch.ones((), device=device))["probe"]
    disabled_output = disabled(torch.ones((), device=device))["probe"]
    assert enabled.training and disabled.training
    assert torch.equal(disabled_output, enabled_output)
    enabled_output.backward()
    disabled_output.backward()
    assert torch.equal(disabled.probe_weight.grad, enabled.probe_weight.grad)

    enabled.psnr = lambda actual, expected: torch.mean((actual - expected) ** 2)
    disabled.psnr = enabled.psnr
    image = torch.zeros((2, 3), device=device)
    enabled_metrics = enabled.get_metrics_dict({"rgb": image}, {"image": image})
    disabled_metrics = disabled.get_metrics_dict({"rgb": image}, {"image": image})
    assert any(name.startswith("occupancy_") for name in enabled_metrics)
    assert not any(name.startswith("occupancy_") for name in disabled_metrics)


def test_disabled_default_policy_does_not_read_occupancy_values() -> None:
    class ForbiddenDiagnosticReductions:
        def mean(self):
            raise AssertionError("disabled default policy must not reduce occs.mean")

        def max(self):
            raise AssertionError("disabled default policy must not reduce occs.max")

    model = LookCloserModel.__new__(LookCloserModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        occupancy_diagnostics=False,
        occupancy_occ_thre=0.01,
        occupancy_thre_clamp_mult=1.0,
        occupancy_dilation_radius=0,
        occupancy_binary_warmup_steps=0,
        alpha_thre=0.0,
    )
    model.occupancy_grid = SimpleNamespace(
        occs=ForbiddenDiagnosticReductions(),
        binaries=torch.tensor([[[[True]]]]),
    )
    model._last_occupancy_binaries = torch.tensor([[[[False]]]])
    model._last_occupancy_stats = {"stale": 1.0}

    model._postprocess_occupancy_grid(step=1)

    assert model._last_occupancy_binaries is None
    assert model._last_occupancy_stats == {}


def test_model_config_keeps_occupancy_diagnostics_enabled_by_default() -> None:
    config = LookCloserModelConfig()
    assert config.occupancy_diagnostics is True
    assert config.stable_occupancy_reduction is True
    assert config.adaptive_warmup_steps == 4096
    assert config.occupancy_eval_dilation_radius == 0
    assert config.occupancy_eval_dilation_min_frequency_level == 0.0
    assert config.occupancy_eval_dilation_frequency_quantile is None
    assert config.occupancy_eval_dilation_frequency_halo == 0


def test_eval_only_dilation_is_restored_before_training() -> None:
    model = LookCloserModel.__new__(LookCloserModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(occupancy_eval_dilation_radius=1)
    original = torch.zeros((1, 3, 3, 3), dtype=torch.bool)
    original[0, 1, 1, 1] = True
    model.occupancy_grid = SimpleNamespace(binaries=original.clone())
    model._eval_occupancy_backup = None

    model.eval()
    model._ensure_eval_occupancy_dilation()

    assert model.occupancy_grid.binaries.all()
    assert torch.equal(model._eval_occupancy_backup, original)

    model.train()

    assert model._eval_occupancy_backup is None
    assert torch.equal(model.occupancy_grid.binaries, original)


def test_eval_dilation_can_use_scene_adaptive_frequency_quantile() -> None:
    model = LookCloserModel.__new__(LookCloserModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        occupancy_eval_dilation_radius=1,
        occupancy_eval_dilation_min_frequency_level=0.0,
        occupancy_eval_dilation_frequency_quantile=0.75,
        occupancy_eval_dilation_frequency_halo=0,
    )
    original = torch.zeros((1, 3, 3, 3), dtype=torch.bool)
    original[0, 1, 1, 1] = True
    frequency = torch.zeros((3, 3, 3))
    frequency[0, 0, 0] = 1.0
    frequency[0, 0, 1] = 5.0
    frequency[0, 1, 0] = 10.0
    frequency[2, 2, 2] = 15.0
    model.occupancy_grid = SimpleNamespace(binaries=original.clone())
    model.freq_grid = SimpleNamespace(grid=frequency)
    model._eval_occupancy_backup = None

    model.eval()
    model._ensure_eval_occupancy_dilation()

    expected = original.clone()
    expected[0, 2, 2, 2] = True
    assert torch.equal(model.occupancy_grid.binaries, expected)
