from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_mapanything_pose_depth_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_mapanything_pose_depth_dataset", SCRIPT)
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


def test_robust_baseline_scale_rejects_one_bad_pair() -> None:
    predicted = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    source = predicted * 2.5
    source[2, 1] = 0.1

    assert MODULE.robust_baseline_scale(predicted, source) == pytest.approx(2.50049995, rel=1e-5)


def test_remap_pinhole_scalar_is_identity_for_matching_calibration() -> None:
    image = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    intrinsic = np.asarray([[9.0, 0.0, 3.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]])

    remapped = MODULE.remap_pinhole_scalar(
        image,
        predicted_intrinsic=intrinsic,
        source_intrinsic=intrinsic,
        width=7,
        height=5,
        interpolation=cv2.INTER_NEAREST,
    )

    np.testing.assert_array_equal(remapped, image)


def test_selected_train_frames_rejects_masks() -> None:
    payload = {
        "frames": [{"file_path": "images/a.jpg", "mask_path": "masks/a.png"}],
        "train_filenames": ["images/a.jpg"],
    }

    with pytest.raises(ValueError, match="forbids person/image masks"):
        MODULE.selected_train_frames(payload)
