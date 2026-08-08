"""Focused tests for ordered thin-cable gap detection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_thin_cable_gaps.py"
SPEC = importlib.util.spec_from_file_location("score_thin_cable_gaps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_detects_a_long_missing_run_on_an_ordered_dark_cable() -> None:
    gt = np.full((100, 64), 0.5, dtype=np.float32)
    prediction = gt.copy()
    gt[10:90, 29:32] = 0.1
    prediction[10:90, 29:32] = 0.1
    prediction[40:63, 29:32] = 0.5
    route = [(30, y) for y in range(10, 90)]
    config = MODULE.DetectorConfig(min_gap_length=10)

    missing, gaps, summary = MODULE.detect_gaps(gt, prediction, route, config)

    assert missing.any()
    assert len(gaps) == 1
    # The detector deliberately tolerates a three-pixel prediction shift on
    # both ends, so 23 missing source pixels produce at least 17 unsupported
    # ordered centerline samples.
    assert summary["longest_gap_pixels"] >= 17
    assert gaps[0]["start_xy"][1] <= 43
    assert gaps[0]["end_xy"][1] >= 59


def test_parse_cables_requires_an_ordered_corridor() -> None:
    parsed = MODULE.parse_cables(["cable:2:10,20;30,40;50,60"])
    assert parsed == [("cable", 2, ((10, 20), (30, 40), (50, 60)))]
