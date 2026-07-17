"""Independent LookCloser RNG stream tests."""

import pytest
import torch

from nerfstudio.models.lookcloser import LookCloserModelConfig
from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline, LookCloserPipelineConfig
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng, stream_seed


def test_independent_stream_configs_are_default_off() -> None:
    pipeline = LookCloserPipelineConfig()
    model = LookCloserModelConfig()

    assert pipeline.independent_rng_streams is False
    assert model.independent_rng_streams is False
    assert pipeline.training_seed == 42
    assert model.training_seed == 42
    LookCloserPipeline._validate_independent_rng_config(pipeline)
    pipeline.model = model
    pipeline.independent_rng_streams = True
    pipeline.model.independent_rng_streams = True
    pipeline.training_seed = 314
    pipeline.model.training_seed = 314
    LookCloserPipeline._validate_independent_rng_config(pipeline)


@pytest.mark.parametrize(
    ("pipeline_enabled", "model_enabled", "pipeline_seed", "model_seed"),
    [(True, False, 42, 42), (False, True, 42, 42), (True, True, 42, 43)],
)
def test_partial_or_mismatched_stream_config_fails_closed(
    pipeline_enabled: bool,
    model_enabled: bool,
    pipeline_seed: int,
    model_seed: int,
) -> None:
    config = LookCloserPipelineConfig()
    config.model = LookCloserModelConfig()
    config.independent_rng_streams = pipeline_enabled
    config.model.independent_rng_streams = model_enabled
    config.training_seed = pipeline_seed
    config.model.training_seed = model_seed

    with pytest.raises(ValueError):
        LookCloserPipeline._validate_independent_rng_config(config)


def test_stream_seeds_are_stable_and_independent() -> None:
    assert stream_seed(42, "pixel", 10) == stream_seed(42, "pixel", 10)
    assert stream_seed(42, "pixel", 10) != stream_seed(42, "occupancy", 10)
    assert stream_seed(42, "pixel", 10) != stream_seed(42, "pixel", 11)


@pytest.mark.parametrize(
    ("stream", "step", "message"),
    [("unknown", 0, "Unknown"), ("pixel", -1, "non-negative")],
)
def test_invalid_stream_requests_fail_closed(stream: str, step: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stream_seed(42, stream, step)


def test_forked_stream_repeats_without_changing_global_rng() -> None:
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    with fork_seeded_rng(42, "pixel", 100, "cpu"):
        first = torch.rand(16)
    assert torch.equal(torch.random.get_rng_state(), before)
    with fork_seeded_rng(42, "pixel", 100, "cpu"):
        second = torch.rand(16)
    assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_forked_cuda_stream_repeats_and_restores_cpu_and_device_rng() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(7)
    torch.cuda.manual_seed(11)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()

    with fork_seeded_rng(42, "occupancy", 100, device):
        first_cpu = torch.rand(16)
        first_cuda = torch.rand(16, device=device)

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
    with fork_seeded_rng(42, "occupancy", 100, device):
        second_cpu = torch.rand(16)
        second_cuda = torch.rand(16, device=device)
    assert torch.equal(first_cpu, second_cpu)
    assert torch.equal(first_cuda, second_cuda)
