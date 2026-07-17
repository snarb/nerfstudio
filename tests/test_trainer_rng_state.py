"""Tests for exact training RNG checkpoint continuation."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nerfstudio.engine.trainer as trainer_module
from nerfstudio.engine.trainer import (
    Trainer,
    TrainerConfig,
    _capture_rng_state,
    _restore_rng_state,
    _validate_grad_scaler_checkpoint_config,
)


def test_grad_scaler_config_preserves_pytorch_defaults() -> None:
    config = TrainerConfig()

    assert config.grad_scaler_init_scale == 65536.0
    assert config.grad_scaler_growth_interval == 2000


def test_trainer_forwards_explicit_grad_scaler_controls(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeGradScaler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = TrainerConfig(
        mixed_precision=True,
        grad_scaler_init_scale=8192.0,
        grad_scaler_growth_interval=1_000_000,
    )
    config.machine.device_type = "cpu"
    config.output_dir = tmp_path
    config.method_name = "test"
    config.experiment_name = "fixed-scaler"
    config.timestamp = "test"
    monkeypatch.setattr(trainer_module, "GradScaler", FakeGradScaler)

    Trainer(config)

    assert captured == {
        "enabled": True,
        "init_scale": 8192.0,
        "growth_interval": 1_000_000,
    }


def test_trainer_rejects_invalid_grad_scaler_controls(tmp_path) -> None:
    for init_scale, growth_interval, match in (
        (0.0, 2000, "init_scale"),
        (float("nan"), 2000, "init_scale"),
        (65536.0, 0, "growth_interval"),
    ):
        config = TrainerConfig(
            grad_scaler_init_scale=init_scale,
            grad_scaler_growth_interval=growth_interval,
        )
        config.machine.device_type = "cpu"
        config.output_dir = tmp_path
        config.method_name = "test"
        config.experiment_name = "invalid-scaler"
        config.timestamp = "test"
        with pytest.raises(ValueError, match=match):
            Trainer(config)


def test_grad_scaler_checkpoint_policy_match_is_required() -> None:
    _validate_grad_scaler_checkpoint_config({"growth_interval": 1_000_000}, 1_000_000)

    with pytest.raises(ValueError, match="does not match"):
        _validate_grad_scaler_checkpoint_config({"growth_interval": 2000}, 1_000_000)
    with pytest.raises(ValueError, match="has no growth_interval"):
        _validate_grad_scaler_checkpoint_config({}, 2000)


def test_rng_state_round_trip_cpu() -> None:
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)
    state = _capture_rng_state()

    expected_python = random.random()
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    _restore_rng_state(state)

    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)


def test_rng_state_round_trip_cuda() -> None:
    if not torch.cuda.is_available():
        return

    torch.cuda.manual_seed_all(31)
    state = _capture_rng_state()
    expected = torch.rand(4, device="cuda")

    _restore_rng_state(state)

    torch.testing.assert_close(torch.rand(4, device="cuda"), expected, rtol=0, atol=0)


def test_eval_trajectory_replay_matches_scheduled_side_effect_order() -> None:
    calls = []

    class FixedLoader:
        def __iter__(self):
            calls.append("all_start")
            yield "camera0", "batch0"
            yield "camera1", "batch1"
            calls.append("all_end")

    datamanager = SimpleNamespace(
        next_eval=lambda step: calls.append(("batch", step)),
        next_eval_image=lambda step: calls.append(("image", step)),
        fixed_indices_eval_dataloader=FixedLoader(),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.pipeline = SimpleNamespace(datamanager=datamanager, eval=lambda: None, train=lambda: None)
    trainer.config = SimpleNamespace(
        steps_per_eval_batch=10,
        steps_per_eval_image=10,
        steps_per_eval_all_images=10,
    )

    trainer._replay_eval_trajectory(9)
    assert calls == []
    trainer._replay_eval_trajectory(10)
    assert calls == [("batch", 10), ("image", 10), "all_start", "all_end"]


def test_train_batch_prefetch_barrier_and_close_delegate_to_datamanager() -> None:
    calls = []
    trainer = Trainer.__new__(Trainer)
    trainer.pipeline = SimpleNamespace(
        datamanager=SimpleNamespace(
            train_batch_prefetch_barrier=lambda: calls.append("barrier"),
            close_train_batch_prefetch=lambda: calls.append("close"),
        )
    )

    trainer._train_batch_prefetch_barrier()
    trainer._close_train_batch_prefetch()

    assert calls == ["barrier", "close"]
