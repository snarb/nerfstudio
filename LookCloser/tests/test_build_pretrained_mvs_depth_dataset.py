from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_pretrained_mvs_depth_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_pretrained_mvs_depth_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_camera_converts_opengl_c2w_to_opencv() -> None:
    frame = {
        "file_path": "images/camera.jpg",
        "transform_matrix": np.eye(4).tolist(),
        "fl_x": 1000.0,
        "fl_y": 900.0,
        "cx": 500.0,
        "cy": 300.0,
    }

    c2w, intrinsic = MODULE.source_camera(frame, {})

    np.testing.assert_allclose(c2w, np.diag([1.0, -1.0, -1.0, 1.0]))
    np.testing.assert_allclose(
        intrinsic,
        np.asarray([[1000.0, 0.0, 500.0], [0.0, 900.0, 300.0], [0.0, 0.0, 1.0]]),
    )


def test_resize_dimensions_preserves_aspect_and_multiple_of_64() -> None:
    width, height = MODULE.resize_dimensions(1920, 1080, 1152, 864)

    assert (width, height) == (1152, 640)
    assert width % 64 == 0 and height % 64 == 0


def test_source_view_order_starts_with_reference_then_closest() -> None:
    cameras = []
    for center in (0.0, 3.0, 1.0, 2.0):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = center
        cameras.append((pose, np.eye(3, dtype=np.float32)))

    assert MODULE.source_view_order(cameras, 0) == [0, 2, 3, 1]


def test_selected_train_frames_respects_declared_order_and_rejects_masks() -> None:
    payload = {
        "frames": [
            {"file_path": "images/a.jpg"},
            {"file_path": "images/b.jpg"},
            {"file_path": "images/c.jpg"},
        ],
        "train_filenames": ["images/c.jpg", "images/a.jpg"],
    }
    assert [row["file_path"] for row in MODULE.selected_train_frames(payload)] == [
        "images/c.jpg",
        "images/a.jpg",
    ]

    payload["frames"][0]["mask_path"] = "masks/a.png"
    with pytest.raises(ValueError, match="forbids person/image masks"):
        MODULE.selected_train_frames(payload)
