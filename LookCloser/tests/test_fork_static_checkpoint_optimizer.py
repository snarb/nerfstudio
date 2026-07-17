from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts import fork_static_checkpoint_optimizer as fork


def _checkpoint(path: Path) -> None:
    torch.save(
        {
            "step": 7,
            "pipeline": {"weight": torch.tensor([1.0])},
            "optimizers": {
                "fields": {
                    "state": {0: {"step": torch.tensor(7.0)}},
                    "param_groups": [{"lr": 0.1, "initial_lr": 0.1}],
                }
            },
            "schedulers": {
                "fields": {
                    "base_lrs": [0.1],
                    "_last_lr": [0.1],
                    "last_epoch": 7,
                    "_step_count": 8,
                }
            },
            "scalers": {},
            "rng_state": {
                "python": (3, (), None),
                "numpy": ("MT19937", torch.arange(4).numpy(), 0, 0, 0.0),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": [],
            },
        },
        path,
    )


def test_drop_rng_state_preserves_training_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.ckpt"
    output = tmp_path / "fork.ckpt"
    _checkpoint(source)
    monkeypatch.setattr(sys, "argv", ["fork", str(source), str(output), "--drop-rng-state"])

    assert fork.main() == 0

    source_state = torch.load(source, map_location="cpu", weights_only=False)
    output_state = torch.load(output, map_location="cpu", weights_only=False)
    assert "rng_state" in source_state
    assert "rng_state" not in output_state
    assert output_state["step"] == source_state["step"]
    assert output_state["optimizers"] == source_state["optimizers"]
    assert output_state["schedulers"] == source_state["schedulers"]
    provenance = json.loads(output.with_suffix(".ckpt.fork.json").read_text(encoding="utf-8"))
    assert provenance["actions"]["drop_rng_state"] is True
    assert provenance["before"]["rng_state_present"] is True
    assert provenance["after"]["rng_state_present"] is False


def test_drop_rng_state_rejects_selective_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.ckpt"
    _checkpoint(source)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fork", str(source), str(tmp_path / "fork.ckpt"), "--drop-rng-state", "--reset-torch-cpu-rng-seed", "42"],
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        fork.main()


def test_scheduler_time_scale_changes_only_scheduler_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.ckpt"
    output = tmp_path / "fork.ckpt"
    _checkpoint(source)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fork", str(source), str(output), "--scheduler-time-scale", "1.5"],
    )

    assert fork.main() == 0

    original = torch.load(source, map_location="cpu", weights_only=False)
    remapped = torch.load(output, map_location="cpu", weights_only=False)
    before = original["schedulers"]["fields"]
    after = remapped["schedulers"]["fields"]

    assert after["last_epoch"] == round(before["last_epoch"] * 1.5) == 10
    assert after["_step_count"] - after["last_epoch"] == before["_step_count"] - before["last_epoch"]
    assert after["base_lrs"] == before["base_lrs"]
    assert after["_last_lr"] == before["_last_lr"]
    assert remapped["step"] == original["step"]
    assert remapped["optimizers"] == original["optimizers"]
    assert remapped["scalers"] == original["scalers"]
    assert torch.equal(remapped["pipeline"]["weight"], original["pipeline"]["weight"])
    assert torch.equal(remapped["rng_state"]["torch_cpu"], original["rng_state"]["torch_cpu"])

    provenance = json.loads(output.with_suffix(".ckpt.fork.json").read_text(encoding="utf-8"))
    assert provenance["actions"]["scheduler_time_scale"] == 1.5
    assert provenance["before"]["scheduler_last_epoch"] == 7
    assert provenance["after"]["scheduler_last_epoch"] == 10
    assert provenance["before"]["optimizer_lrs"] == provenance["after"]["optimizer_lrs"]
