#!/usr/bin/env python3
"""Export fixed Nerfstudio cameras as a registered COLMAP text model.

This is the dense-MVS entry point for calibrated multi-camera captures.  It
does not run feature matching or bundle adjustment: the supplied calibration
is copied exactly, eval images can be excluded, and the sparse point file is
intentionally empty.  COLMAP's image undistorter and PatchMatch stereo only
need registered cameras/images when an explicit ``patch-match.cfg`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np

try:
    from seed_colmap_from_nerfstudio import (
        camera_matrix,
        format_values,
        rotation_matrix_to_qvec,
        world_to_camera,
    )
except ModuleNotFoundError:  # Imported as ``scripts.export_...`` in tests/tools.
    _seed_path = Path(__file__).resolve().with_name("seed_colmap_from_nerfstudio.py")
    _seed_spec = importlib.util.spec_from_file_location("lookcloser_seed_colmap", _seed_path)
    if _seed_spec is None or _seed_spec.loader is None:
        raise ImportError(f"Cannot import camera conversion helpers from {_seed_path}")
    _seed_module = importlib.util.module_from_spec(_seed_spec)
    _seed_spec.loader.exec_module(_seed_module)
    camera_matrix = _seed_module.camera_matrix
    format_values = _seed_module.format_values
    rotation_matrix_to_qvec = _seed_module.rotation_matrix_to_qvec
    world_to_camera = _seed_module.world_to_camera


def normalized_name(value: str) -> str:
    name = PurePosixPath(value).as_posix()
    while name.startswith("./"):
        name = name[2:]
    return name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("all", "train"),
        default="train",
        help="Export all frames or only explicit train_filenames. Dense production defaults to train-only.",
    )
    args = parser.parse_args(argv)
    args.data = args.data.expanduser().resolve()
    args.output_model = args.output_model.expanduser().resolve()
    if not args.data.is_dir():
        parser.error(f"Dataset does not exist: {args.data}")
    if args.output_model.exists():
        parser.error(f"Output model already exists: {args.output_model}")
    return args


def selected_frames(payload: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.json contains no frames")
    if not all(isinstance(frame, dict) and isinstance(frame.get("file_path"), str) for frame in frames):
        raise ValueError("Every frame must contain a string file_path")
    if any("mask_path" in frame for frame in frames):
        raise ValueError("Fixed-calibration dense MVS export forbids image/person masks")
    if split == "all":
        return frames
    raw_train = payload.get("train_filenames")
    if not isinstance(raw_train, list) or not raw_train or not all(isinstance(value, str) for value in raw_train):
        raise ValueError("--split train requires a non-empty explicit train_filenames list")
    train_names = {normalized_name(value) for value in raw_train}
    selected = [frame for frame in frames if normalized_name(str(frame["file_path"])) in train_names]
    selected_names = {normalized_name(str(frame["file_path"])) for frame in selected}
    if selected_names != train_names:
        raise ValueError(f"train_filenames references missing frames: {sorted(train_names - selected_names)[:8]}")
    return selected


def export_model(data: Path, output: Path, *, split: str) -> dict[str, Any]:
    transforms = data / "transforms.json"
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    frames = selected_frames(payload, split=split)
    names = [normalized_name(str(frame["file_path"])) for frame in frames]
    if len(set(names)) != len(names):
        raise ValueError("Frame paths must be unique")
    for name in names:
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)

    requested_ids = [frame.get("colmap_im_id") for frame in frames]
    use_requested_ids = all(isinstance(value, int) and value > 0 for value in requested_ids)
    if use_requested_ids and len(set(int(value) for value in requested_ids)) != len(frames):
        raise ValueError("Positive colmap_im_id values must be unique")
    image_ids = [int(value) for value in requested_ids] if use_requested_ids else list(range(1, len(frames) + 1))

    camera_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    manifest_rows: list[dict[str, Any]] = []
    for frame, name, image_id in zip(frames, names, image_ids):
        width, height, parameters = camera_matrix(frame, payload)
        w2c = world_to_camera(frame)
        quaternion = rotation_matrix_to_qvec(w2c[:3, :3])
        translation = w2c[:3, 3]
        camera_id = image_id
        camera_lines.append(f"{camera_id} OPENCV {width} {height} {format_values(parameters)}")
        image_lines.extend(
            (
                f"{image_id} {format_values(quaternion)} {format_values(translation)} {camera_id} {name}",
                "",
            )
        )
        manifest_rows.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "physical_camera": frame.get("physical_camera"),
                "width": width,
                "height": height,
            }
        )

    output.mkdir(parents=True)
    (output / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (output / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (output / "points3D.txt").write_text("# Empty fixed-calibration model for dense MVS.\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "fixed_nerfstudio_calibration_to_colmap_text",
        "data": str(data),
        "split": split,
        "image_count": len(frames),
        "camera_model": "OPENCV",
        "uses_sparse_points": False,
        "uses_eval_images": split == "all",
        "uses_masks": False,
        "image_id_policy": "colmap_im_id" if use_requested_ids else "sequential",
        "coordinate_conversion": "world_to_camera_cv = diag(1,-1,-1,1) @ inv(camera_to_world_nerfstudio)",
        "transforms_sha256": hashlib.sha256(transforms.read_bytes()).hexdigest(),
        "images": manifest_rows,
    }
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_model(args.data, args.output_model, split=args.split)
    print(
        f"exported images={manifest['image_count']} split={manifest['split']} "
        f"output={args.output_model}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
