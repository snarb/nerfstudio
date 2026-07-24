from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import (
    Trainer,
    _load_model_parameters_only,
    _reset_lookcloser_occupancy_for_resume,
)


class _Field(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.register_buffer("aabb", torch.tensor([7.0]))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.field = _Field()
        self.register_buffer("occupancy", torch.zeros(2))
        self.occupancy_grid = SimpleNamespace(
            occs=torch.ones(4),
            binaries=torch.zeros(2, 2, dtype=torch.bool),
        )
        self.lpips = nn.Linear(1, 1, bias=False)


class _Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model = _Model()
        self.register_buffer("cumulative_point_samples", torch.zeros((), dtype=torch.int64))
        self.loaded = False

    def load_pipeline(self, state, step) -> None:
        self.loaded = True
        self.load_state_dict(state)


def _model_only_checkpoint(path: Path) -> None:
    torch.save(
        {
            "step": 91_128,
            "pipeline": {
                "_model.field.weight": torch.tensor([2.0, 3.0]),
                "_model.field.aabb": torch.tensor([99.0]),
                "_model.occupancy": torch.ones(2),
                "_model.lpips.weight": torch.ones(1, 1),
                "cumulative_point_samples": torch.tensor(1234, dtype=torch.int64),
            },
            "optimizers": {"fields": {"should_not_load": True}},
            "schedulers": {"fields": {"should_not_load": True}},
            "scalers": {"scale": 4.0},
            "rng_state": {"torch_cpu": torch.ones(1)},
        },
        path,
    )


def test_model_parameters_only_copies_fields_and_leaves_buffers_fresh(tmp_path: Path) -> None:
    pipeline = _Pipeline()
    checkpoint = tmp_path / "step-000091128.ckpt"
    _model_only_checkpoint(checkpoint)
    state = torch.load(checkpoint, weights_only=False)["pipeline"]

    names = _load_model_parameters_only(pipeline, [pipeline._model.field.weight], state)

    assert names == ["_model.field.weight"]
    assert pipeline._model.field.weight.tolist() == [2.0, 3.0]
    assert pipeline._model.field.aabb.tolist() == [7.0]
    assert pipeline._model.occupancy.tolist() == [0.0, 0.0]
    assert pipeline.cumulative_point_samples.item() == 0
    assert pipeline._model.lpips.weight.item() != 1.0


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.ones(3), "shape"),
        (torch.ones(2, dtype=torch.float64), "dtype"),
    ],
)
def test_model_parameters_only_rejects_shape_or_dtype_mismatch(replacement, message) -> None:
    pipeline = _Pipeline()
    with pytest.raises(RuntimeError, match=message):
        _load_model_parameters_only(
            pipeline,
            [pipeline._model.field.weight],
            {"_model.field.weight": replacement},
        )


def test_model_parameters_only_rejects_an_unknown_source_field_parameter() -> None:
    pipeline = _Pipeline()
    with pytest.raises(RuntimeError, match="unexpected field"):
        _load_model_parameters_only(
            pipeline,
            [pipeline._model.field.weight],
            {
                "_model.field.weight": torch.ones(2),
                "_model.field.aabb": torch.ones(1),
                "_model.field.new_trainable_parameter": torch.ones(1),
            },
        )


def test_model_only_parent_step_91128_starts_local_step_zero_with_fresh_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-000091128.ckpt"
    _model_only_checkpoint(checkpoint)
    pipeline = _Pipeline()
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            load_dir=None,
            load_checkpoint=checkpoint,
            load_step=None,
            checkpoint_load_mode="model_parameters_only",
            resume_fields_lr_override=None,
        ),
        pipeline=pipeline,
        optimizers=SimpleNamespace(parameters={"fields": [pipeline._model.field.weight]}),
        _start_step=-1,
        _loaded_rng_state={"unexpected": True},
        checkpoint_load_audit={},
    )

    Trainer._load_checkpoint(trainer)

    assert trainer._start_step == 0
    assert trainer._loaded_rng_state is None
    assert not pipeline.loaded
    assert trainer.checkpoint_load_audit["source_step"] == 91_128
    assert trainer.checkpoint_load_audit["optimizer_loaded"] is False
    assert trainer.checkpoint_load_audit["pipeline_buffers_loaded"] is False


def test_resume_occupancy_reset_preserves_model_and_sets_local_warmup_origin() -> None:
    pipeline = _Pipeline()
    pipeline._model.field.weight.data.copy_(torch.tensor([2.0, 3.0]))
    before = pipeline._model.field.weight.detach().clone()

    audit = _reset_lookcloser_occupancy_for_resume(
        pipeline, warmup_start_step=91_129
    )

    assert torch.equal(pipeline._model.field.weight, before)
    assert torch.count_nonzero(pipeline._model.occupancy_grid.occs).item() == 0
    assert pipeline._model.occupancy_grid.binaries.all()
    assert pipeline._model._occupancy_warmup_start_step == 91_129
    assert audit == {
        "occs_zero": True,
        "binaries_true_count": 4,
        "binaries_numel": 4,
        "warmup_start_step": 91_129,
    }


def test_lr_override_preserves_adam_moments_and_scheduler_progress() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizers = Optimizers(
        {
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=100),
            }
        },
        {"fields": [parameter]},
    )
    parameter.grad = torch.tensor([0.5])
    optimizers.optimizers["fields"].step()
    optimizers.schedulers["fields"].step()
    before_state = copy.deepcopy(optimizers.optimizers["fields"].state_dict()["state"])
    before_epoch = optimizers.schedulers["fields"].last_epoch

    optimizers.override_learning_rate("fields", 2.5e-4)

    after_state = optimizers.optimizers["fields"].state_dict()["state"]
    assert before_state.keys() == after_state.keys()
    for key in before_state:
        for state_name in before_state[key]:
            assert torch.equal(before_state[key][state_name], after_state[key][state_name])
    assert optimizers.schedulers["fields"].last_epoch == before_epoch
    assert optimizers.optimizers["fields"].param_groups[0]["lr"] == 2.5e-4
    optimizers.schedulers["fields"].step()
    assert optimizers.optimizers["fields"].param_groups[0]["lr"] == pytest.approx(2.5e-4)
