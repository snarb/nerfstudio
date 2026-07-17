"""Properties for the default-off in-process dynamic point-target schedule."""

from types import SimpleNamespace

import pytest

from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline


def _pipeline_config(**overrides):
    values = {
        "target_num_samples_per_batch": 2**21,
        "target_num_samples_switch_step": 30_376,
        "target_num_samples_after_switch": 2**20,
        "dynamic_rays_ema": 0.9,
        "dynamic_rays_min": 256,
        "dynamic_rays_max": 32_768,
        "dynamic_rays_change_limit": 1.25,
        "dynamic_rays_start_step": 4_096,
    }
    values.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**values))


def test_dynamic_target_switch_applies_after_the_named_update_without_state_reset() -> None:
    pipeline = _pipeline_config()
    assert LookCloserPipeline._target_num_samples_for_step(pipeline, 30_375) == 2**21
    assert LookCloserPipeline._target_num_samples_for_step(pipeline, 30_376) == 2**20
    assert LookCloserPipeline._target_num_samples_for_step(pipeline, 75_940) == 2**20


def test_dynamic_target_schedule_is_default_off() -> None:
    pipeline = _pipeline_config(
        target_num_samples_per_batch=0,
        target_num_samples_switch_step=None,
        target_num_samples_after_switch=None,
    )
    assert LookCloserPipeline._target_num_samples_for_step(pipeline, 0) == 0
    assert LookCloserPipeline._target_num_samples_for_step(pipeline, 100_000) == 0
    LookCloserPipeline._validate_dynamic_batch_config(pipeline)


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_num_samples_after_switch": None},
        {"target_num_samples_switch_step": None},
        {"target_num_samples_switch_step": -1},
        {"target_num_samples_after_switch": 0},
        {"target_num_samples_per_batch": 0},
    ],
)
def test_invalid_dynamic_target_schedules_fail_closed(overrides) -> None:
    pipeline = _pipeline_config(**overrides)
    with pytest.raises(ValueError):
        LookCloserPipeline._validate_dynamic_batch_config(pipeline)
