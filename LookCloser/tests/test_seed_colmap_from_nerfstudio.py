from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "seed_colmap_from_nerfstudio.py"
SPEC = importlib.util.spec_from_file_location("seed_colmap_from_nerfstudio", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rotation_matrix_to_qvec_identity() -> None:
    assert np.allclose(MODULE.rotation_matrix_to_qvec(np.eye(3)), [1.0, 0.0, 0.0, 0.0])


def test_camera_matrix_inherits_global_values() -> None:
    payload = {"w": 1920, "h": 1080, "fl_x": 1000.0, "fl_y": 1001.0, "cx": 960.0, "cy": 540.0}
    width, height, params = MODULE.camera_matrix({"file_path": "images/a.jpg", "camera_model": "OPENCV"}, payload)
    assert (width, height) == (1920, 1080)
    assert np.allclose(params, [1000.0, 1001.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0])


def test_colmap_opencv_model_id_matches_database_convention(tmp_path: Path) -> None:
    database = sqlite3.connect(tmp_path / "database.db")
    database.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, model INTEGER)")
    database.execute("INSERT INTO cameras VALUES(1, ?)", (MODULE.COLMAP_OPENCV_MODEL_ID,))
    assert database.execute("SELECT model FROM cameras WHERE camera_id=1").fetchone()[0] == 4


def test_table_columns_supports_colmap_313_without_pose_priors(tmp_path: Path) -> None:
    database = sqlite3.connect(tmp_path / "database.db")
    database.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")

    columns = MODULE.table_columns(database, "images")

    assert columns == {"image_id", "name", "camera_id"}


def test_train_split_names_are_normalized() -> None:
    payload = {"train_filenames": ["./images/a.jpg", "images/b.jpg"]}

    selected = {MODULE.normalized_name(value) for value in payload["train_filenames"]}

    assert selected == {"images/a.jpg", "images/b.jpg"}
