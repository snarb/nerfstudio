from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from scripts import compare_static_checkpoint_drift as drift


def _pipeline(offset: float = 0.0) -> dict[str, torch.Tensor]:
    pipeline = {
        key: torch.tensor([[1.0 + offset, 0.0], [0.0, 1.0 - offset]], dtype=torch.float32)
        for key in drift.FIELD_KEYS
    }
    occupancy = {
        "occs": torch.tensor([1.0 + offset, 0.0, 0.5, 0.25], dtype=torch.float32),
        "binaries": torch.tensor([True, False, True, False]),
        "aabbs": torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
    }
    for prefix in drift.OCCUPANCY_PREFIXES:
        pipeline.update({prefix + suffix: tensor.clone() for suffix, tensor in occupancy.items()})
    return pipeline


def _save(path: Path, *, step: int, offset: float = 0.0) -> None:
    torch.save({"step": step, "pipeline": _pipeline(offset)}, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compare_multiple_checkpoints_is_chunked_read_only_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.ckpt"
    candidate = tmp_path / "candidate.ckpt"
    identical = tmp_path / "identical.ckpt"
    output = tmp_path / "report.json"
    _save(reference, step=10)
    _save(candidate, step=12, offset=0.25)
    _save(identical, step=10)
    hashes_before = {path: _sha256(path) for path in (reference, candidate, identical)}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            str(reference),
            str(candidate),
            str(identical),
            "--output",
            str(output),
            "--chunk-elements",
            "2",
        ],
    )

    assert drift.main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["chunk_elements"] == 2
    assert payload["reference"]["trainer_step"] == 10
    assert len(payload["comparisons"]) == 2
    changed, same = payload["comparisons"]
    assert changed["candidate"]["trainer_step"] == 12
    assert changed["trainer_step_delta"] == 2
    assert changed["validation"] == {
        "keys": "exact_match",
        "shapes": "exact_match",
        "dtypes": "exact_match",
        "finite": True,
    }
    assert changed["groups"]["field"]["aggregate"]["symmetric_relative_l2"] > 0.0
    occs = changed["groups"]["occupancy"]["by_suffix"]["occs"]
    assert len(occs["keys"]) == 2
    assert occs["aggregate"]["symmetric_relative_l2"] > 0.0
    assert same["groups"]["field"]["aggregate"]["symmetric_relative_l2"] == 0.0
    assert same["groups"]["occupancy"]["aggregate"]["symmetric_relative_l2"] == 0.0
    assert all(payload["reference"]["occupancy_duplicate_equal"].values())
    assert hashes_before == {path: _sha256(path) for path in (reference, candidate, identical)}


def test_symmetric_relative_l2_has_expected_value_and_zero_convention() -> None:
    orthogonal_left = torch.tensor([1.0, 0.0])
    orthogonal_right = torch.tensor([0.0, 1.0])
    sums = drift._tensor_sums(
        orthogonal_left,
        orthogonal_right,
        chunk_elements=1,
        key="orthogonal",
    )
    assert drift._norm_payload(*sums)["symmetric_relative_l2"] == pytest.approx(math.sqrt(2.0))

    zero_sums = drift._tensor_sums(
        torch.zeros(5),
        torch.zeros(5),
        chunk_elements=2,
        key="zero",
    )
    assert drift._norm_payload(*zero_sums)["symmetric_relative_l2"] == 0.0


@pytest.mark.parametrize("mismatch", ["missing", "shape", "dtype"])
def test_exact_occupancy_checks_fail_closed(tmp_path: Path, mismatch: str) -> None:
    reference = tmp_path / "reference.ckpt"
    candidate = tmp_path / "candidate.ckpt"
    _save(reference, step=1)
    candidate_pipeline = _pipeline()
    key = drift.OCCUPANCY_PREFIXES[1] + "occs"
    if mismatch == "missing":
        del candidate_pipeline[key]
    elif mismatch == "shape":
        candidate_pipeline[key] = torch.ones(5, dtype=torch.float32)
    else:
        candidate_pipeline[key] = candidate_pipeline[key].to(torch.float64)
    torch.save({"step": 1, "pipeline": candidate_pipeline}, candidate)

    with pytest.raises(drift.CheckpointValidationError):
        drift.compare_checkpoints([reference, candidate], chunk_elements=2)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_field_shape_and_dtype_are_checked(tmp_path: Path, mismatch: str) -> None:
    reference = tmp_path / "reference.ckpt"
    candidate = tmp_path / "candidate.ckpt"
    _save(reference, step=1)
    candidate_pipeline = _pipeline()
    if mismatch == "shape":
        candidate_pipeline[drift.FIELD_KEYS[0]] = torch.ones(5, dtype=torch.float32)
    else:
        candidate_pipeline[drift.FIELD_KEYS[0]] = candidate_pipeline[drift.FIELD_KEYS[0]].to(torch.float64)
    torch.save({"step": 1, "pipeline": candidate_pipeline}, candidate)

    with pytest.raises(drift.CheckpointValidationError, match=rf"{mismatch} mismatch"):
        drift.compare_checkpoints([reference, candidate], chunk_elements=2)


def test_invalid_cli_domain_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    _save(checkpoint, step=1)
    with pytest.raises(ValueError, match="At least two"):
        drift.compare_checkpoints([checkpoint], chunk_elements=2)
    with pytest.raises(ValueError, match="positive"):
        drift.compare_checkpoints([checkpoint, tmp_path / "other.ckpt"], chunk_elements=0)
    with pytest.raises(ValueError, match="unique"):
        drift.compare_checkpoints([checkpoint, checkpoint], chunk_elements=2)


def test_non_finite_tensor_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.ckpt"
    candidate = tmp_path / "candidate.ckpt"
    _save(reference, step=1)
    candidate_pipeline = _pipeline()
    candidate_pipeline[drift.FIELD_KEYS[1]][0, 0] = float("nan")
    torch.save({"step": 1, "pipeline": candidate_pipeline}, candidate)

    with pytest.raises(drift.CheckpointValidationError, match="non-finite"):
        drift.compare_checkpoints([reference, candidate], chunk_elements=2)
