from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "import_colmap_mvs_depth_dataset.py"
SPEC = importlib.util.spec_from_file_location("import_colmap_mvs_depth_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_colmap_dense(path: Path, values: np.ndarray) -> None:
    height, width, channels = values.shape
    with path.open("wb") as handle:
        handle.write(f"{width}&{height}&{channels}&".encode())
        values.transpose(1, 0, 2).reshape(-1, order="F").astype(np.float32).tofile(handle)


def test_read_colmap_dense_array_preserves_pixel_layout(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    path = tmp_path / "depth.bin"
    write_colmap_dense(path, values)

    loaded = MODULE.read_colmap_dense_array(path)

    assert loaded.shape == values.shape
    assert np.array_equal(loaded, values)


def test_normalized_name_removes_dot_prefix() -> None:
    assert MODULE.normalized_name("./images/a.jpg") == "images/a.jpg"
