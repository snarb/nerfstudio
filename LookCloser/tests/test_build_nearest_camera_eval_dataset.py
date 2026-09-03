from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_nearest_camera_eval_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_nearest_camera_eval_dataset_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frame(path: str, x: float, *, mask: bool = False) -> dict:
    row = {
        "file_path": path,
        "transform_matrix": [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    if mask:
        row["mask_path"] = "masks/person.png"
    return row


def make_source(tmp_path: Path, frames: list[dict], *, ply: bool = False) -> Path:
    source = tmp_path / "source"
    for row in frames:
        image = source / row["file_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(row["file_path"].encode())
    payload = {"frames": frames}
    if ply:
        (source / "geometry").mkdir()
        (source / "geometry" / "sparse.ply").write_bytes(b"ply")
        payload["ply_file_path"] = "geometry/sparse.ply"
    (source / "transforms.json").write_text(json.dumps(payload), encoding="utf-8")
    return source


def test_selects_nearest_train_views_and_keeps_eval_held_out(tmp_path: Path) -> None:
    frames = [
        frame("images/train_far.png", 5.0),
        frame("images/train_near.jpg", 0.2),
        frame("images/train_mid.webp", 1.0),
        frame("images/chosen_eval.exr", 0.0),
        frame("images/other_eval.jpg", 0.1),
    ]
    source = make_source(tmp_path, frames, ply=True)
    output = tmp_path / "output"

    assert (
        MODULE.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--eval-frame-file",
                "images/chosen_eval.exr",
                "--train-count",
                "2",
            ]
        )
        == 0
    )
    result = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    receipt = result["nearest_camera_eval"]
    assert [row["source_file_path"] for row in receipt["train_source_frames"]] == [
        "images/train_near.jpg",
        "images/train_mid.webp",
    ]
    assert receipt["eval_source_file_path"] == "images/chosen_eval.exr"
    assert receipt["eval_is_held_out"] is True
    assert [Path(row["file_path"]).suffix for row in result["frames"]] == [".jpg", ".webp", ".exr"]
    assert (output / "geometry" / "sparse.ply").read_bytes() == b"ply"


def test_rejects_masks_without_materializing_output(tmp_path: Path) -> None:
    frames = [frame("images/train.jpg", 0.2, mask=True), frame("images/eval.jpg", 0.0)]
    source = make_source(tmp_path, frames)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="forbids masks"):
        MODULE.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--eval-frame-file",
                "images/eval.jpg",
                "--train-count",
                "1",
            ]
        )
    assert not output.exists()


def test_rejects_too_many_train_cameras(tmp_path: Path) -> None:
    source = make_source(tmp_path, [frame("images/train.jpg", 1.0), frame("images/eval.jpg", 0.0)])
    with pytest.raises(ValueError, match="only 1 are eligible"):
        MODULE.main(
            [
                "--input",
                str(source),
                "--output",
                str(tmp_path / "output"),
                "--eval-frame-file",
                "images/eval.jpg",
                "--train-count",
                "2",
            ]
        )


def test_preserves_normalization_context_but_holds_target_out_of_train(tmp_path: Path) -> None:
    frames = [
        frame("images/train_far.png", 4.0),
        frame("images/target_train.jpg", 0.0),
        frame("images/train_near.webp", 0.2),
        frame("images/original_eval.jpg", 0.1),
    ]
    source = make_source(tmp_path, frames)
    output = tmp_path / "leave_one_out"

    assert (
        MODULE.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--eval-frame-file",
                "images/target_train.jpg",
                "--train-count",
                "2",
                "--preserve-normalization-context",
            ]
        )
        == 0
    )
    result = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    assert [row["file_path"] for row in result["frames"]] == [row["file_path"] for row in frames]
    assert result["train_filenames"] == ["images/train_near.webp", "images/train_far.png"]
    assert result["val_filenames"] == ["images/target_train.jpg"]
    assert result["test_filenames"] == ["images/target_train.jpg"]
    assert "images/target_train.jpg" not in result["train_filenames"]
    assert result["nearest_camera_eval"]["preserve_normalization_context"] is True
    assert result["nearest_camera_eval"]["normalization_frame_count"] == 4
    assert (output / "images" / "target_train.jpg").read_bytes() == b"images/target_train.jpg"
