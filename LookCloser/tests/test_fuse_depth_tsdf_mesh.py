from __future__ import annotations

import importlib.util
import gzip
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


def test_non_manifold_cleanup_is_default_on_and_can_be_disabled(tmp_path: Path) -> None:
    default = MODULE.parse_args(["--data", str(tmp_path), "--output", str(tmp_path / "default.ply")])
    disabled = MODULE.parse_args(
        [
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "disabled.ply"),
            "--no-remove-non-manifold-edges",
        ]
    )

    assert default.remove_non_manifold_edges is True
    assert disabled.remove_non_manifold_edges is False


def test_load_compressed_numpy_depth_applies_scene_scale(tmp_path: Path) -> None:
    source = tmp_path / "depth.npy.gz"
    with gzip.open(source, "wb") as stream:
        np.save(stream, np.asarray([[2.0, 0.0]], dtype=np.float32), allow_pickle=False)

    np.testing.assert_allclose(MODULE.load_depth(source, scale_factor=0.25), [[0.5, 0.0]])


def test_component_threshold_combines_absolute_and_relative_gates() -> None:
    counts = np.asarray([158_081, 228], dtype=np.int64)

    assert MODULE.component_triangle_threshold(
        counts, minimum_triangles=100, minimum_fraction=0.0
    ) == 100
    assert MODULE.component_triangle_threshold(
        counts, minimum_triangles=100, minimum_fraction=0.002
    ) == 317
    assert MODULE.component_triangle_threshold(
        counts, minimum_triangles=500, minimum_fraction=0.002
    ) == 500
