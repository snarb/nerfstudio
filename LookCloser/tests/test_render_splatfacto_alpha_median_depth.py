from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_splatfacto_alpha_median_depth.py"
SPEC = importlib.util.spec_from_file_location("render_splatfacto_alpha_median_depth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_alpha_quantile_selects_first_crossing_and_marks_missing() -> None:
    pixel_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    alphas = torch.tensor([0.2, 0.5, 0.1, 0.2])

    selected, valid = MODULE.first_alpha_quantile_indices(
        pixel_ids,
        alphas,
        num_pixels=3,
        quantile=0.5,
    )

    assert valid.tolist() == [True, False, False]
    assert selected[0].item() == 1


def test_first_alpha_quantile_supports_different_quantile() -> None:
    pixel_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    alphas = torch.tensor([0.2, 0.5, 0.5])

    selected, valid = MODULE.first_alpha_quantile_indices(
        pixel_ids,
        alphas,
        num_pixels=1,
        quantile=0.75,
    )

    assert valid.item()
    assert selected.item() == 2


def test_first_alpha_quantile_rejects_ungrouped_pixels() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        MODULE.first_alpha_quantile_indices(
            torch.tensor([1, 0], dtype=torch.long),
            torch.tensor([0.6, 0.6]),
            num_pixels=2,
            quantile=0.5,
        )
