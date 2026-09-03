from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_angular_camera_subset.py"
SPEC = importlib.util.spec_from_file_location("build_angular_camera_subset_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def camera_frame(index: int, angle: float, split: str = "train") -> dict:
    radians = np.radians(angle)
    center = np.asarray([np.cos(radians), np.sin(radians), 0.0])
    forward = -center
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = center
    return {
        "file_path": f"images/frame_{split}_{index:05d}.jpg",
        "physical_camera": f"camera_{index}",
        "transform_matrix": matrix.tolist(),
    }


def test_farthest_point_subset_is_spread_and_keeps_required_camera() -> None:
    frames = [camera_frame(index, index * 45.0) for index in range(8)]
    selected, stats = MODULE.select_angular_frames(frames, 4, "camera_1")
    assert 1 in selected
    assert len(selected) == 4
    assert stats["nearest_selected_angle_deg"]["min"] >= 45.0
    assert stats["maximum_selected_pair_angle_deg"] >= 135.0


def test_farthest_point_ladder_is_nested() -> None:
    """Camera-count A/Bs must add views rather than replace earlier views."""
    frames = [camera_frame(index, index * 15.0) for index in range(24)]
    selected_four, _ = MODULE.select_angular_frames(frames, 4, "camera_3")
    selected_sixteen, _ = MODULE.select_angular_frames(frames, 16, "camera_3")
    assert set(selected_four) < set(selected_sixteen)


def test_main_preserves_context_and_writes_explicit_split(tmp_path: Path) -> None:
    source = tmp_path / "source"
    images = source / "images"
    images.mkdir(parents=True)
    frames = [camera_frame(index, index * 45.0) for index in range(8)] + [camera_frame(8, 22.5, "eval")]
    for frame in frames:
        (source / frame["file_path"]).write_bytes(b"jpg")
    (source / "transforms.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    output = tmp_path / "output"
    assert MODULE.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--train-count",
            "4",
            "--required-camera",
            "camera_1",
        ]
    ) == 0
    result = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert len(result["frames"]) == 9
    assert len(result["train_filenames"]) == 4
    assert result["val_filenames"] == ["images/frame_eval_00008.jpg"]
    manifest = json.loads((output / "angular_subset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection_uses_image_pixels"] is False
    assert any(row["physical_camera"] == "camera_1" for row in manifest["selected_train_frames"])


def test_nearest_eval_strategy_selects_geometric_neighbors() -> None:
    frames = [camera_frame(index, index * 45.0) for index in range(8)]
    evaluation = camera_frame(8, 22.5, "eval")
    selected, stats = MODULE.select_nearest_frames(frames, evaluation, 2, "camera_0")
    assert selected == [0, 1]
    assert stats["distance_to_eval"]["max"] < 1.0
