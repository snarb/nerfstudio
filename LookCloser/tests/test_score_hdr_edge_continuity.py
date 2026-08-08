from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_hdr_edge_continuity import metrics, parse_rois  # noqa: E402


def test_continuity_metric_penalizes_broken_line() -> None:
    gt = np.zeros((32, 32), dtype=bool)
    gt[16, 3:29] = True
    intact = gt.copy()
    broken = gt.copy()
    broken[16, 13:20] = False

    intact_result = metrics(gt, intact, tolerance=1.0, long_gap_min_pixels=3)
    broken_result = metrics(gt, broken, tolerance=1.0, long_gap_min_pixels=3)

    assert intact_result["edge_recall"] == 1.0
    assert intact_result["long_gap_fraction"] == 0.0
    assert broken_result["edge_recall"] < intact_result["edge_recall"]
    assert broken_result["long_gap_fraction"] > 0.0
    assert broken_result["largest_gap_pixels"] >= 3


def test_parse_custom_roi() -> None:
    assert parse_rois(["wire:2:1:2:20:30"]) == [("wire", 2, (1, 2, 20, 30))]
