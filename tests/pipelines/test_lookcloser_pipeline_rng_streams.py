"""Static LookCloser pipeline-level independent RNG stream tests."""

from types import SimpleNamespace

import torch
from torch import nn

import nerfstudio.pipelines.lookcloser_pipeline as lookcloser_pipeline_module
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device_indicator_param = nn.Parameter(torch.empty(0))
        self.config = SimpleNamespace(fixed_num_samples_per_ray=3)
        self.field = SimpleNamespace(feature_reweighting_strength=1.0)
        self.current_train_step = -1

    @property
    def device(self) -> torch.device:
        return self.device_indicator_param.device

    def forward(self, _ray_bundle):
        return {"rgb": torch.zeros((2, 3), device=self.device)}

    def get_metrics_dict(self, _model_outputs, _batch):
        return {}

    def get_loss_dict(self, _model_outputs, _batch, _metrics_dict):
        return {"loss": torch.zeros((), device=self.device)}


class _FakeDataManager:
    def __init__(self, extra_draws: int) -> None:
        self.config = SimpleNamespace(cpu_fas_prefetch=False)
        self.train_pixel_sampler = SimpleNamespace(sample_count=0)
        self.extra_draws = extra_draws
        self.pixel_draws = []

    def next_train(self, _step: int):
        self.pixel_draws.append(torch.rand(self.extra_draws))
        self.train_pixel_sampler.sample_count += 1
        return object(), {}

    def get_train_rays_per_batch(self) -> int:
        return 4


def _pipeline(*, independent: bool, extra_draws: int, grid_draws):
    pipeline = LookCloserPipeline.__new__(LookCloserPipeline)
    nn.Module.__init__(pipeline)
    model = _FakeModel()
    datamanager = _FakeDataManager(extra_draws)
    pipeline._model = model
    pipeline.datamanager = datamanager
    pipeline.config = SimpleNamespace(
        train_rays_switch_step=None,
        train_rays_after_switch=None,
        feature_reweighting_switch_step=None,
        feature_reweighting_after_switch=None,
        independent_rng_streams=independent,
        training_seed=42,
        datamanager=datamanager.config,
        grid_update_interval=4,
        enable_frequency_grid=True,
        target_num_samples_switch_step=None,
        target_num_samples_after_switch=None,
        target_num_samples_per_batch=0,
    )
    pipeline._train_rays_switch_applied = False
    pipeline._feature_reweighting_switch_applied = False
    pipeline.fas_sample_count_state = torch.zeros((), dtype=torch.int64)
    pipeline.cumulative_point_samples = torch.zeros((), dtype=torch.int64)
    pipeline.dynamic_samples_per_ray_ema = torch.zeros((), dtype=torch.float64)
    pipeline._apply_live_tcnn_jit_switch = lambda _step: None
    pipeline._update_dynamic_rays = lambda _step, _points, _rays: None
    pipeline._update_frequency_grid = lambda step: grid_draws.append((step, torch.rand(11)))
    return pipeline, datamanager


def test_independent_train_step_isolates_pixel_and_boundary_frequency_streams() -> None:
    step = 8
    with fork_seeded_rng(42, "frequency_grid", step, "cpu"):
        expected_grid = torch.rand(11)
    observed_grid = []

    for extra_draws in (1, 37):
        grid_draws = []
        pipeline, datamanager = _pipeline(
            independent=True,
            extra_draws=extra_draws,
            grid_draws=grid_draws,
        )
        with fork_seeded_rng(42, "pixel", step, "cpu"):
            expected_pixel = torch.rand(extra_draws)
        torch.manual_seed(9101)
        outer_before = torch.random.get_rng_state().clone()

        pipeline.get_train_loss_dict(step)

        assert len(datamanager.pixel_draws) == 1
        assert torch.equal(datamanager.pixel_draws[0], expected_pixel)
        assert len(grid_draws) == 1
        assert grid_draws[0][0] == step
        assert torch.equal(grid_draws[0][1], expected_grid)
        assert torch.equal(torch.random.get_rng_state(), outer_before)
        observed_grid.append(grid_draws[0][1])

    assert torch.equal(observed_grid[0], observed_grid[1])


def test_independent_frequency_stream_runs_only_on_update_boundary() -> None:
    grid_draws = []
    pipeline, _datamanager = _pipeline(independent=True, extra_draws=13, grid_draws=grid_draws)
    torch.manual_seed(9102)
    outer_before = torch.random.get_rng_state().clone()

    pipeline.get_train_loss_dict(9)

    assert grid_draws == []
    assert torch.equal(torch.random.get_rng_state(), outer_before)


def test_frequency_grid_update_cadence_is_local_after_resume_reset() -> None:
    grid_draws = []
    pipeline, _datamanager = _pipeline(independent=True, extra_draws=13, grid_draws=grid_draws)
    pipeline._frequency_grid_warmup_start_step = 91_129

    pipeline.get_train_loss_dict(91_132)
    assert grid_draws == []

    pipeline.get_train_loss_dict(91_133)
    assert len(grid_draws) == 1
    assert grid_draws[0][0] == 4


def test_default_path_uses_vanilla_pipeline_and_never_forks(monkeypatch) -> None:
    def forbidden_stream_helper(*_args, **_kwargs):
        raise AssertionError("default-off pipeline must not use independent RNG helpers")

    monkeypatch.setattr(lookcloser_pipeline_module, "fork_seeded_rng", forbidden_stream_helper)
    monkeypatch.setattr(lookcloser_pipeline_module, "stream_seed", forbidden_stream_helper)
    original_get_train_loss_dict = VanillaPipeline.get_train_loss_dict
    vanilla_calls = []

    def tracked_vanilla(self, step: int):
        vanilla_calls.append(step)
        return original_get_train_loss_dict(self, step)

    monkeypatch.setattr(VanillaPipeline, "get_train_loss_dict", tracked_vanilla)
    grid_draws = []
    pipeline, datamanager = _pipeline(independent=False, extra_draws=23, grid_draws=grid_draws)
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(9103)
    expected_pixel = torch.rand(23, generator=expected_generator)
    expected_grid = torch.rand(11, generator=expected_generator)
    expected_final_state = expected_generator.get_state()
    torch.manual_seed(9103)

    pipeline.get_train_loss_dict(8)

    assert vanilla_calls == [8]
    assert torch.equal(datamanager.pixel_draws[0], expected_pixel)
    assert grid_draws[0][0] == 8
    assert torch.equal(grid_draws[0][1], expected_grid)
    assert torch.equal(torch.random.get_rng_state(), expected_final_state)
