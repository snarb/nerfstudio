from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_edge_aware_frequency_variants import (  # noqa: E402
    levels_to_resolutions,
    resolutions_to_levels,
    variants,
)


def test_level_resolution_roundtrip() -> None:
    levels = torch.arange(16).reshape(4, 4)
    restored = resolutions_to_levels(levels_to_resolutions(levels, 16, 8192, 16), 16, 8192, 16)
    torch.testing.assert_close(restored, levels)


def test_edge_floor_and_union_are_conservative() -> None:
    knee = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    calibrated = torch.tensor([[2, 1, 4], [3, 8, 5], [9, 7, 10]])
    proxy = torch.tensor([[0.0, 0.1, 0.2], [0.3, 1.0, 0.4], [0.5, 0.6, 0.7]])
    output = variants(
        knee,
        calibrated,
        proxy,
        structural_fraction=0.2,
        floor_level=12,
        dilation_radius=0,
        n_levels=16,
    )
    assert bool((output["knee_plus1"] >= knee).all())
    assert bool((output["knee_calibrated_union"] >= knee).all())
    assert output["knee_edge_floor"][1, 1] == 12
    assert output["knee_edge_floor"][0, 0] == knee[0, 0]
