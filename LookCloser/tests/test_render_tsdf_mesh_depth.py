from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "render_tsdf_mesh_depth.py"
SPEC = importlib.util.spec_from_file_location("render_tsdf_mesh_depth", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_barycentric_vertex_color_interpolation_and_miss() -> None:
    miss = np.iinfo(np.uint32).max
    result = MODULE.barycentric_vertex_colors(
        primitive_ids=np.asarray([[0, miss]], dtype=np.uint32),
        primitive_uvs=np.asarray([[[0.25, 0.25], [0.0, 0.0]]], dtype=np.float32),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        vertex_colors=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )
    np.testing.assert_allclose(result[0, 0], [0.5, 0.25, 0.25])
    np.testing.assert_array_equal(result[0, 1], [0.0, 0.0, 0.0])


def test_plane_fallback_fills_miss_and_preserves_foreground() -> None:
    origins = np.zeros((1, 3, 3), dtype=np.float64)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    combined, replaced, valid = MODULE.apply_plane_fallback(
        ray_origins=origins,
        ray_directions=directions,
        mesh_t_hit=np.asarray([[np.inf, 1.0, 1.98]], dtype=np.float64),
        plane=np.asarray([0.0, 0.0, 1.0, -2.0]),
        replace_band=0.05,
    )
    np.testing.assert_allclose(combined, [[2.0, 1.0, 2.0]])
    np.testing.assert_array_equal(replaced, [[True, False, True]])
    np.testing.assert_array_equal(valid, [[True, True, True]])


def test_manifest_reference_is_relative_to_manifest_directory(tmp_path: Path) -> None:
    anchor = tmp_path / "receipt" / "depth"
    path = tmp_path / "receipt" / "actor.ply"
    assert MODULE.manifest_reference(path, anchor=anchor, portable=True) == "../actor.ply"
    assert MODULE.manifest_reference(path, anchor=anchor, portable=False) == str(path.resolve())
