#!/usr/bin/env python3
"""Seed a COLMAP database/text model from a Nerfstudio ``transforms.json``.

The utility is dataset-agnostic: images are joined by their relative paths,
and every image keeps its own intrinsics.  A typical calibration audit first
runs COLMAP feature extraction with ``single_camera_per_image=1``, then invokes
this script before matching, triangulation, and bundle adjustment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


COLMAP_OPENCV_MODEL_ID = 4
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("all", "train"),
        default="all",
        help="Seed every frame or only the explicit transforms.json train_filenames split.",
    )
    return parser.parse_args()


def inherited(row: dict[str, Any], payload: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, payload.get(key, default))
    if value is None:
        raise ValueError(f"Missing required camera value {key!r} for {row.get('file_path')!r}")
    return value


def normalized_name(value: str) -> str:
    name = PurePosixPath(value).as_posix()
    while name.startswith("./"):
        name = name[2:]
    return name


def camera_matrix(row: dict[str, Any], payload: dict[str, Any]) -> tuple[int, int, np.ndarray]:
    width = int(inherited(row, payload, "w"))
    height = int(inherited(row, payload, "h"))
    params = np.asarray(
        [
            float(inherited(row, payload, "fl_x")),
            float(inherited(row, payload, "fl_y")),
            float(inherited(row, payload, "cx")),
            float(inherited(row, payload, "cy")),
            float(inherited(row, payload, "k1", 0.0)),
            float(inherited(row, payload, "k2", 0.0)),
            float(inherited(row, payload, "p1", 0.0)),
            float(inherited(row, payload, "p2", 0.0)),
        ],
        dtype=np.float64,
    )
    if width <= 0 or height <= 0 or not np.isfinite(params).all() or min(params[:2]) <= 0:
        raise ValueError(f"Invalid OPENCV camera for {row.get('file_path')!r}: {width}x{height}, {params}")
    model = str(row.get("camera_model", payload.get("camera_model", "OPENCV"))).upper()
    if model != "OPENCV":
        raise ValueError(f"Only OPENCV cameras are supported, got {model!r} for {row.get('file_path')!r}")
    return width, height, params


def world_to_camera(row: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(row["transform_matrix"], dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0.0, 0.0, 0.0, 1.0]])
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Invalid transform for {row.get('file_path')!r}: shape={transform.shape}")
    result = GL_TO_CV @ np.linalg.inv(transform)
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"Non-rigid homogeneous transform for {row.get('file_path')!r}")
    return result


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """COLMAP-compatible scalar-first quaternion conversion."""

    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.asarray(
        [
            [rxx - ryy - rzz, ryx + rxy, rzx + rxz, ryz - rzy],
            [ryx + rxy, ryy - rxx - rzz, rzy + ryz, rzx - rxz],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, rxy - ryx],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    quaternion = eigenvectors[[3, 0, 1, 2], int(np.argmax(eigenvalues))]
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def format_values(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Return database columns without assuming a particular COLMAP major version."""

    if not table.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQLite table name {table!r}")
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def main() -> int:
    args = parse_args()
    data = args.data.expanduser().resolve()
    database = args.database.expanduser().resolve()
    output = args.output_model.expanduser().resolve()
    payload = json.loads((data / "transforms.json").read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in {data / 'transforms.json'}")
    selected_names: set[str] | None = None
    if args.split == "train":
        raw_train = payload.get("train_filenames")
        if not isinstance(raw_train, list) or not raw_train or not all(isinstance(value, str) for value in raw_train):
            raise ValueError("--split train requires a non-empty explicit train_filenames list")
        selected_names = {normalized_name(value) for value in raw_train}
    by_name: dict[str, dict[str, Any]] = {}
    for row in frames:
        if not isinstance(row, dict) or not isinstance(row.get("file_path"), str):
            raise ValueError("Every frame must contain a string file_path")
        name = normalized_name(row["file_path"])
        if selected_names is not None and name not in selected_names:
            continue
        if name in by_name:
            raise ValueError(f"Duplicate frame path {name!r}")
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
        by_name[name] = row
    if selected_names is not None and set(by_name) != selected_names:
        missing = sorted(selected_names - set(by_name))
        raise ValueError(f"train_filenames references missing frames: {missing[:8]}")

    connection = sqlite3.connect(database)
    try:
        image_columns = table_columns(connection, "images")
        pose_prior_columns = {
            "prior_qw",
            "prior_qx",
            "prior_qy",
            "prior_qz",
            "prior_tx",
            "prior_ty",
            "prior_tz",
        }
        writes_pose_priors = pose_prior_columns.issubset(image_columns)
        all_db_images = connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
        database_names = {str(row[1]) for row in all_db_images}
        missing = sorted(set(by_name) - database_names)
        if missing:
            raise ValueError(f"Database is missing selected transforms images: {missing[:8]}")
        db_images = [row for row in all_db_images if str(row[1]) in by_name]
        camera_ids = [int(row[2]) for row in db_images]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Calibration audit requires one COLMAP camera per image")

        cameras_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
        images_lines = [
            "# Image list with two lines of data per image:",
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "# POINTS2D[] as (X, Y, POINT3D_ID)",
        ]
        rows_manifest = []
        for image_id, raw_name, camera_id in db_images:
            name = str(raw_name)
            row = by_name[name]
            width, height, params = camera_matrix(row, payload)
            w2c = world_to_camera(row)
            quaternion = rotation_matrix_to_qvec(w2c[:3, :3])
            translation = w2c[:3, 3]
            connection.execute(
                "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                (COLMAP_OPENCV_MODEL_ID, width, height, params.tobytes(), int(camera_id)),
            )
            if writes_pose_priors:
                connection.execute(
                    "UPDATE images SET prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?, "
                    "prior_tx=?, prior_ty=?, prior_tz=? WHERE image_id=?",
                    (
                        *[float(value) for value in quaternion],
                        *[float(value) for value in translation],
                        int(image_id),
                    ),
                )
            cameras_lines.append(f"{int(camera_id)} OPENCV {width} {height} {format_values(params)}")
            images_lines.extend(
                [
                    f"{int(image_id)} {format_values(quaternion)} {format_values(translation)} "
                    f"{int(camera_id)} {name}",
                    "",
                ]
            )
            rows_manifest.append(
                {
                    "image_id": int(image_id),
                    "camera_id": int(camera_id),
                    "name": name,
                    "physical_camera": row.get("physical_camera"),
                    "feature_count": connection.execute(
                        "SELECT rows FROM keypoints WHERE image_id=?", (int(image_id),)
                    ).fetchone()[0],
                }
            )
        connection.commit()
    finally:
        connection.close()

    output.mkdir(parents=True, exist_ok=True)
    (output / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (output / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (output / "points3D.txt").write_text("# Empty seed model; COLMAP point_triangulator fills this file.\n", encoding="utf-8")
    transform_bytes = (data / "transforms.json").read_bytes()
    manifest = {
        "schema_version": 1,
        "data": str(data),
        "database": str(database),
        "image_count": len(rows_manifest),
        "split": args.split,
        "camera_model": "OPENCV",
        "coordinate_conversion": "world_to_camera_cv = diag(1,-1,-1,1) @ inv(camera_to_world_nerfstudio)",
        "database_pose_priors_written": writes_pose_priors,
        "transforms_sha256": hashlib.sha256(transform_bytes).hexdigest(),
        "images": rows_manifest,
    }
    (output / "seed_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = [int(row["feature_count"]) for row in rows_manifest]
    print(
        f"seeded images={len(rows_manifest)} features_min={min(counts)} "
        f"features_mean={sum(counts) / len(counts):.1f} features_max={max(counts)} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
