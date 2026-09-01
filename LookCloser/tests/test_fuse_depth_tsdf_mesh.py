from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "fuse_depth_tsdf_mesh.py"
SPEC = importlib.util.spec_from_file_location("fuse_depth_tsdf_mesh", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_nerfstudio_identity_camera_becomes_opencv_axis_flip() -> None:
    c2w = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    result = MODULE.nerfstudio_c2w_to_opencv_extrinsic(c2w)
    np.testing.assert_allclose(result, np.diag([1.0, -1.0, -1.0, 1.0]))


def test_camera_conversion_preserves_world_camera_roundtrip() -> None:
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    world_to_cv = MODULE.nerfstudio_c2w_to_opencv_extrinsic(c2w)
    camera_origin_world = np.linalg.inv(world_to_cv) @ np.asarray([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(camera_origin_world[:3], [1.0, 2.0, 3.0])
