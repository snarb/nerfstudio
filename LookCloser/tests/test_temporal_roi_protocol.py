from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import temporal_roi_protocol as roi


def test_forward_backward_tracking_moves_seed_box_with_confidence() -> None:
    rng = np.random.default_rng(42)
    previous = rng.integers(0, 256, size=(180, 240, 3), dtype=np.uint8)
    transform = np.float32([[1, 0, 6], [0, 1, 4]])
    current = cv2.warpAffine(previous, transform, (240, 180), borderMode=cv2.BORDER_REFLECT)

    tracked = roi.track_box(previous, current, (50, 40, 170, 140))

    assert tracked.confidence >= 0.60
    assert tracked.box == (56, 44, 176, 144)
    assert tracked.median_forward_backward_error < 0.2


def test_exposure_compensated_motion_and_hole_boxes_are_never_empty() -> None:
    previous = np.full((180, 240, 3), 90, dtype=np.uint8)
    current = np.full((180, 240, 3), 115, dtype=np.uint8)
    current[60:130, 80:160] = 10

    broad, holes = roi.motion_boxes(previous, current)

    assert broad
    assert holes
    for box in broad + holes:
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= 240
        assert 0 <= y0 < y1 <= 180
