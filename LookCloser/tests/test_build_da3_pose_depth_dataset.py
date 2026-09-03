from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_da3_pose_depth_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_da3_pose_depth_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_nerfstudio_camera_to_da3_converts_opengl_c2w_to_opencv_w2c() -> None:
    frame = {
        "file_path": "images/camera.jpg",
        "transform_matrix": np.eye(4).tolist(),
        "fl_x": 1000.0,
        "fl_y": 900.0,
        "cx": 500.0,
        "cy": 300.0,
    }

    extrinsic, intrinsic = MODULE.nerfstudio_camera_to_da3(frame, {})

    np.testing.assert_allclose(extrinsic, np.diag([1.0, -1.0, -1.0, 1.0]))
    np.testing.assert_allclose(
        intrinsic,
        np.asarray([[1000.0, 0.0, 500.0], [0.0, 900.0, 300.0], [0.0, 0.0, 1.0]]),
    )


def test_nerfstudio_camera_to_da3_preserves_camera_center() -> None:
    c2w = np.eye(4)
    c2w[:3, 3] = [1.25, -0.5, 3.0]
    frame = {
        "file_path": "images/camera.jpg",
        "transform_matrix": c2w.tolist(),
    }
    payload = {"fl_x": 1000.0, "fl_y": 1000.0, "cx": 500.0, "cy": 300.0}

    extrinsic, _ = MODULE.nerfstudio_camera_to_da3(frame, payload)
    recovered_c2w = np.linalg.inv(extrinsic)

    np.testing.assert_allclose(recovered_c2w[:3, 3], c2w[:3, 3], atol=1e-6)


def test_selected_train_frames_respects_declared_order() -> None:
    payload = {
        "frames": [
            {"file_path": "images/a.jpg"},
            {"file_path": "images/b.jpg"},
            {"file_path": "images/c.jpg"},
        ],
        "train_filenames": ["images/c.jpg", "images/a.jpg"],
    }

    selected = MODULE.selected_train_frames(payload)

    assert [frame["file_path"] for frame in selected] == ["images/c.jpg", "images/a.jpg"]


def test_selected_train_frames_rejects_masks() -> None:
    payload = {
        "frames": [
            {"file_path": "images/a.jpg", "mask_path": "masks/a.png"},
        ],
        "train_filenames": ["images/a.jpg"],
    }

    with pytest.raises(ValueError, match="forbids image/person masks"):
        MODULE.selected_train_frames(payload)
