from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_colmap_patchmatch_tsdf.py"
SPEC = importlib.util.spec_from_file_location("run_colmap_patchmatch_tsdf", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dry_run_uses_full_resolution_strict_recipe(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data"
    data.mkdir()
    payload = {
        "frames": [
            {"file_path": "images/a.jpg", "transform_matrix": [[1, 0, 0, 0]] * 4},
            {"file_path": "images/b.jpg", "transform_matrix": [[1, 0, 0, 0]] * 4},
            {"file_path": "images/eval.jpg", "transform_matrix": [[1, 0, 0, 0]] * 4},
        ],
        "train_filenames": ["images/a.jpg", "images/b.jpg"],
        "val_filenames": ["images/eval.jpg"],
    }
    (data / "transforms.json").write_text(json.dumps(payload), encoding="utf-8")
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")
    output = tmp_path / "output"

    assert MODULE.main(
        [
            "--data", str(data),
            "--output-dir", str(output),
            "--colmap-bin", str(fake_colmap),
            "--dry-run",
        ]
    ) == 0

    printed = capsys.readouterr().out
    assert "stage=texture-subset status=dry-run" in printed
    assert "build_angular_camera_subset.py" in printed
    assert "train-count 2" in printed
    assert "copy_policy soft-link" in printed
    assert "PatchMatchStereo.max_image_size 1920" in printed
    assert "PatchMatchStereo.num_iterations 3" in printed
    assert "stage=patchmatch-photometric status=dry-run" in printed
    assert "PatchMatchStereo.geom_consistency 0" in printed
    assert "stage=patchmatch-geometric status=dry-run" in printed
    assert "PatchMatchStereo.geom_consistency 1" in printed
    assert "PatchMatchStereo.filter_min_triangulation_angle 1.0" in printed
    assert "tensor-weight-threshold 2.0" in printed
    assert "min-component-fraction 0.002" in printed
    assert "aggregation-modes nearest-fill" in printed
    assert "stage=hard-texture-render status=dry-run" in printed
    assert not output.exists()


def test_masks_are_rejected_before_running(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    payload = {
        "frames": [
            {"file_path": "images/a.jpg", "mask_path": "masks/a.png"},
            {"file_path": "images/b.jpg"},
        ],
        "train_filenames": ["images/a.jpg", "images/b.jpg"],
    }
    (data / "transforms.json").write_text(json.dumps(payload), encoding="utf-8")

    try:
        MODULE.train_count(data)
    except ValueError as error:
        assert "forbids image/person masks" in str(error)
    else:
        raise AssertionError("mask-bearing input must be rejected")


def test_verified_colmap_build_is_pinned() -> None:
    probe = "COLMAP 3.13.0.dev0 -- Structure-from-Motion\n(Commit 5509fffe on 2025-07-06 with CUDA)\n"
    assert "5509fffe" in MODULE.validate_colmap_build(probe, allow_unverified=False)


def test_unverified_colmap_build_fails_closed() -> None:
    probe = "COLMAP 4.1.1 (Commit Unknown on Unknown with CUDA)\n"
    try:
        MODULE.validate_colmap_build(probe, allow_unverified=False)
    except RuntimeError as error:
        assert "Unverified COLMAP" in str(error)
    else:
        raise AssertionError("unverified dense-MVS build must require an explicit override")
    assert "COLMAP 4.1.1" in MODULE.validate_colmap_build(probe, allow_unverified=True)


def test_cpu_colmap_is_rejected_even_with_override() -> None:
    try:
        MODULE.validate_colmap_build("COLMAP 3.13.0.dev0 (Commit 5509fffe)", allow_unverified=True)
    except RuntimeError as error:
        assert "CUDA" in str(error)
    else:
        raise AssertionError("PatchMatch requires CUDA")
