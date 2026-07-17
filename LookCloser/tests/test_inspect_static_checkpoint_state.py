from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts import inspect_static_checkpoint_state as inspector


PYTHON_RNG_SHA256 = "cf4fb75f755c5a97ccf42950a5a1c267b8e9cc3b098065f06e191e963f0f215a"
NUMPY_RNG_SHA256 = "e8d5b74c789b2df2649da46a096d4f10c59c0fd7edf88ea26bade410ab0012db"
TORCH_CPU_RNG_SHA256 = "da2cb6ad175bc966de5e79c6e16777f8a98b610c2424a894132df2815be50677"
TORCH_CUDA_RNG_SHA256 = [
    "06df4f7e1394f1c57cc6583fba4d8060a5a66f4f4771c14aeff6b9af8a28c9b3",
    "678feb3f747a2d2f550e94426e93e05c4701e4ee09543ff89eb06986772c261c",
]


def _required_training_state() -> dict:
    return {
        "step": 7,
        "optimizers": {
            "fields": {
                "state": {
                    0: {"step": torch.tensor(6.0)},
                    1: {"step": 6},
                },
                "param_groups": [{"lr": 0.01}, {"lr": 0.0025}],
            }
        },
        "schedulers": {
            "fields": {
                "last_epoch": 6,
                "_step_count": 7,
                "_last_lr": [0.01, 0.0025],
            }
        },
    }


def test_inspect_reports_iteration_skip_scaler_pipeline_and_rng_state(tmp_path: Path) -> None:
    checkpoint = _required_training_state()
    checkpoint.update(
        {
            "pipeline": {
                "cumulative_point_samples": torch.tensor(987_654_321, dtype=torch.int64),
                "fas_sample_count_state": torch.tensor(12_345, dtype=torch.int64),
            },
            "scalers": {
                "scale": 8192.0,
                "_growth_tracker": 37,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 1_000_000,
            },
            "rng_state": {
                "python": (3, (1, 2, 3, 4), None),
                "numpy": (
                    "MT19937",
                    np.array([1, 2, 3, 4_294_967_295], dtype=np.uint32),
                    2,
                    1,
                    0.125,
                ),
                "torch_cpu": torch.tensor([0, 1, 2, 127, 128, 255], dtype=torch.uint8),
                "torch_cuda": [
                    torch.tensor([9, 8, 7], dtype=torch.uint8),
                    torch.tensor([6, 5], dtype=torch.uint8),
                ],
            },
        }
    )
    path = tmp_path / "synthetic.ckpt"
    torch.save(checkpoint, path)

    first = inspector.inspect(path)
    second = inspector.inspect(path)

    assert first == second
    assert first["checkpoint"] == str(path)
    assert first["bytes"] == path.stat().st_size
    assert first["trainer_step"] == 7
    assert first["training_iterations"] == 8
    assert first["adam_steps"] == [6]
    assert first["optimizer_updates"] == 6
    # The legacy gap is indexed from trainer step, whereas skipped updates is
    # indexed from the number of attempted iterations and is therefore +1.
    assert first["trainer_optimizer_gap"] == 1
    assert first["skipped_optimizer_updates"] == 2
    assert first["optimizer_lrs"] == [0.01, 0.0025]
    assert first["scheduler_last_epoch"] == 6
    assert first["scheduler_step_count"] == 7
    assert first["scheduler_last_lrs"] == [0.01, 0.0025]
    assert first["cumulative_point_samples"] == 987_654_321
    assert first["fas_sample_count_state"] == 12_345
    assert first["grad_scaler_scale"] == 8192.0
    assert first["grad_scaler_growth_tracker"] == 37
    assert first["grad_scaler_growth_factor"] == 2.0
    assert first["grad_scaler_backoff_factor"] == 0.5
    assert first["grad_scaler_growth_interval"] == 1_000_000
    assert first["rng_state_present"] is True
    assert first["rng_state"] == {
        "python_sha256": PYTHON_RNG_SHA256,
        "numpy_sha256": NUMPY_RNG_SHA256,
        "torch_cpu_sha256": TORCH_CPU_RNG_SHA256,
        "torch_cuda_sha256": TORCH_CUDA_RNG_SHA256,
    }


def test_inspect_handles_missing_optional_rng_scaler_and_pipeline_counters(tmp_path: Path) -> None:
    checkpoint = _required_training_state()
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)

    result = inspector.inspect(path)

    assert result["cumulative_point_samples"] is None
    assert result["fas_sample_count_state"] is None
    assert result["grad_scaler_scale"] is None
    assert result["grad_scaler_growth_tracker"] is None
    assert result["grad_scaler_growth_factor"] is None
    assert result["grad_scaler_backoff_factor"] is None
    assert result["grad_scaler_growth_interval"] is None
    assert result["rng_state_present"] is False
    assert result["rng_state"] is None
