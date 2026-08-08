from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_geometry_support_maps import rank_normalize, structural_maps  # noqa: E402


def test_rank_normalize_does_not_create_structure_from_ties() -> None:
    values = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    ranks = rank_normalize(values)
    assert torch.unique(ranks[values == 0.0]).numel() == 1
    assert torch.unique(ranks[values == 2.0]).numel() == 1
    assert ranks[0, 0] < ranks[1, 0] < ranks[1, 1]


def test_dark_thin_line_is_selected_by_edge_and_ridge_maps() -> None:
    image = torch.full((32, 32, 3), 0.5)
    image[:, 15:17] = 0.05
    maps = structural_maps(image, patch_size=4, ridge_scales=(5, 9))

    assert maps.keys() == {"edge", "edge_ridge", "ridge"}
    assert all(value.shape == (8, 8) for value in maps.values())
    line_columns = torch.tensor([3, 4])
    away_columns = torch.tensor([0, 7])
    assert maps["ridge"][:, line_columns].mean() > maps["ridge"][:, away_columns].mean()
    assert maps["edge_ridge"][:, line_columns].mean() > maps["edge_ridge"][:, away_columns].mean()
