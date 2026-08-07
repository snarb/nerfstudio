#!/usr/bin/env python3
"""Build and atomically install the full-resolution raw temporal EXR dataset.

The protected JPEG dataset at ``/home/brans/temporal_perframe_stride7_45f`` is
read-only input for integrity checks.  All generated files are first written to
a sibling staging tree on ``/mnt/data``.  Installation renames the current
``/mnt/data`` JPEG copy to a backup and promotes the already-verified staging
tree, so partially converted data is never exposed as the active dataset.

Run with the package versions pinned by ``requirements-leader-jpeg.txt``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import OpenEXR
import PIL
from convert_temporal_exr_to_leader_jpeg import (
    CHUNK_ROWS,
    DISPLAY_BRIGHTNESS,
    HD_SIZE,
    QHD_SIZE,
    auto_exposure,
    center_crop_box,
    grade_chunk,
    linear_to_srgb,
)
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROTECTED_ROOT = Path("/home/brans/temporal_perframe_stride7_45f")
DEFAULT_SOURCE_ROOT = Path("/mnt/data/6A_4_EXR")
DEFAULT_DATASET_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f")
DEFAULT_STAGING_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f_exr_staging_20260807")
DEFAULT_BACKUP_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f_jpeg_backup_20260807")
EXPECTED_WIDTH = 6144
EXPECTED_HEIGHT = 3072
EXPECTED_FRAME_COUNT = 45
EXPECTED_CAMERA_COUNT = 69
EXPECTED_TRAIN_COUNT = 66
EXPECTED_EVAL_COUNT = 3
EXPECTED_EVAL_STEMS = ("D004_A014", "E004_B014", "I004_D014")
SCHEMA_VERSION = 1
STATE_DIR_NAME = ".exr_conversion_state"
FINAL_MANIFEST_NAME = "exr_conversion_manifest.json"
CONTENT_MANIFEST_NAME = "dataset_content_manifest_exr_20260807.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def protect_paths(*write_paths: Path) -> None:
    protected = resolved(PROTECTED_ROOT)
    for raw in write_paths:
        path = resolved(raw)
        if path == protected or is_below(path, protected):
            raise RuntimeError(f"Refusing write target inside protected dataset: {path}")
        if path == Path("/") or path == Path("/mnt/data") or path == Path("/home"):
            raise RuntimeError(f"Refusing dangerously broad write target: {path}")


def require_mnt_child(path: Path, label: str) -> Path:
    result = resolved(path)
    mnt = Path("/mnt/data").resolve()
    if result == mnt or not is_below(result, mnt):
        raise RuntimeError(f"{label} must be a child of {mnt}: {result}")
    return result


def runtime_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "OpenEXR": str(OpenEXR.__version__).removeprefix("b'").removesuffix("'"),
        "Pillow": PIL.__version__,
    }


def read_root_manifest(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "perframe_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [int(value) for value in data.get("frames", [])]
    if len(frames) != EXPECTED_FRAME_COUNT or len(set(frames)) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_FRAME_COUNT} unique frames in {path}")
    if frames != list(range(7740, 8049, 7)):
        raise RuntimeError(f"Unexpected temporal frame sequence in {path}")
    mapping = data.get("camera_file_mapping", {})
    if len(mapping) != EXPECTED_CAMERA_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_CAMERA_COUNT} camera mappings in {path}")
    return data


def camera_mapping(manifest: dict[str, Any]) -> dict[str, str]:
    """Return physical camera stem -> EXR target filename."""
    result: dict[str, str] = {}
    for old_name, physical_stem in manifest["camera_file_mapping"].items():
        target = f"{Path(old_name).stem}.exr"
        if physical_stem in result:
            raise RuntimeError(f"Duplicate physical camera in mapping: {physical_stem}")
        result[str(physical_stem)] = target
    if len(result) != EXPECTED_CAMERA_COUNT:
        raise RuntimeError("Camera mapping is not one-to-one")
    eval_physical = tuple(
        manifest["camera_file_mapping"][f"frame_eval_{index:05d}.jpg"] for index in range(1, EXPECTED_EVAL_COUNT + 1)
    )
    if eval_physical != EXPECTED_EVAL_STEMS:
        raise RuntimeError(f"Eval-camera mapping changed: {eval_physical}")
    return result


def protected_tree_stat_fingerprint(root: Path) -> dict[str, Any]:
    records: list[list[Any]] = []
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        relative = str(path.relative_to(root))
        kind = "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file"
        target = os.readlink(path) if path.is_symlink() else None
        records.append([relative, kind, stat.st_mode, stat.st_size, stat.st_mtime_ns, target])
    return {
        "root": str(root),
        "entry_count": len(records),
        "sha256": sha256_json(records),
    }


def essential_content_fingerprint(root: Path) -> dict[str, Any]:
    paths: list[Path] = []
    for frame_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit()):
        transforms = frame_dir / "transforms.json"
        if transforms.is_file():
            paths.append(transforms)
        images = frame_dir / "images"
        if images.is_dir():
            paths.extend(sorted(images.glob("frame_*.jpg")))
    paths.extend(sorted(path for path in root.iterdir() if path.is_file()))
    records = []
    aggregate = hashlib.sha256()
    for path in paths:
        relative = str(path.relative_to(root))
        digest = sha256_file(path)
        size = path.stat().st_size
        records.append([relative, size, digest])
        aggregate.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {
        "root": str(root),
        "file_count": len(records),
        "aggregate_sha256": aggregate.hexdigest(),
    }


def source_header_profile(path: Path) -> dict[str, Any]:
    source = OpenEXR.InputFile(str(path))
    header = source.header()
    window = header["dataWindow"]
    width = int(window.max.x - window.min.x + 1)
    height = int(window.max.y - window.min.y + 1)
    channels = sorted(header["channels"])
    pixel_types = {name: str(channel.type) for name, channel in header["channels"].items()}
    compression = str(header["compression"])
    source.close()
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "pixel_types": pixel_types,
        "compression": compression,
    }


def validate_source_inventory(source_root: Path, frames: Iterable[int], mapping: dict[str, str]) -> dict[str, Any]:
    expected = set(mapping)
    size_total = 0
    profiles: dict[str, int] = {}
    for frame in frames:
        directory = source_root / f"{frame:06d}"
        actual = {path.stem for path in directory.glob("*.exr")}
        if actual != expected:
            raise RuntimeError(
                f"Source camera mismatch in {directory}: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        for stem in sorted(expected):
            path = directory / f"{stem}.exr"
            size_total += path.stat().st_size
            profile = source_header_profile(path)
            key = json.dumps(profile, sort_keys=True)
            profiles[key] = profiles.get(key, 0) + 1
            if profile["width"] != EXPECTED_WIDTH or profile["height"] != EXPECTED_HEIGHT:
                raise RuntimeError(f"Unexpected source dimensions: {path}: {profile}")
            if profile["channels"] != ["B", "G", "R"]:
                raise RuntimeError(f"Unexpected source channels: {path}: {profile['channels']}")
            if set(profile["pixel_types"].values()) != {"HALF"}:
                raise RuntimeError(f"Unexpected source pixel type: {path}: {profile['pixel_types']}")
    return {
        "frame_count": len(list(frames)) if not isinstance(frames, list) else len(frames),
        "camera_count": len(expected),
        "file_count": len(expected) * (len(list(frames)) if not isinstance(frames, list) else len(frames)),
        "source_bytes": size_total,
        "header_profiles": {key: count for key, count in sorted(profiles.items())},
    }


def validate_jpeg_dataset(dataset_root: Path, frames: list[int], mapping: dict[str, str]) -> None:
    expected_jpegs = {f"{Path(name).stem}.jpg" for name in mapping.values()}
    transform_hash: str | None = None
    for frame in frames:
        directory = dataset_root / f"{frame:06d}"
        transforms_path = directory / "transforms.json"
        images = directory / "images"
        if not transforms_path.is_file() or not images.is_dir():
            raise RuntimeError(f"Incomplete JPEG dataset frame: {directory}")
        disk = {path.name for path in images.glob("frame_*.jpg")}
        data = json.loads(transforms_path.read_text(encoding="utf-8"))
        bound = {Path(row["file_path"]).name for row in data.get("frames", [])}
        if disk != expected_jpegs or bound != expected_jpegs:
            raise RuntimeError(f"JPEG binding mismatch in {directory}")
        digest = sha256_file(transforms_path)
        transform_hash = transform_hash or digest
        if digest != transform_hash:
            raise RuntimeError(f"Transforms differ between temporal frames: {transforms_path}")


def clone_dataset_without_jpegs(dataset_root: Path, staging_root: Path) -> None:
    staging_root.mkdir(parents=False, exist_ok=False)
    for source in sorted(dataset_root.rglob("*")):
        relative = source.relative_to(dataset_root)
        target = staging_root / relative
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.parent.name == "images" and source.suffix.lower() == ".jpg":
            continue
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def read_exr_rgb_and_header(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    source = OpenEXR.File(str(path), separate_channels=False)
    channels = source.channels()
    if "RGB" in channels:
        image = channels["RGB"].pixels
    elif "RGBA" in channels:
        image = channels["RGBA"].pixels[..., :3]
    else:
        keys = {key.upper(): key for key in channels}
        try:
            image = np.stack([channels[keys[name]].pixels for name in ("R", "G", "B")], axis=-1)
        except KeyError as exc:
            raise RuntimeError(f"No RGB channels in {path}; found {sorted(channels)}") from exc
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB layout in {path}: {image.shape}")
    return image, source.header().copy()


def graded_display_u8(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Create the historical JPEG display transform for visual auditing only."""
    exposure_gain = auto_exposure(image)
    height, width, channels = image.shape
    if channels != 3:
        raise RuntimeError(f"Expected RGB image, got {image.shape}")
    display_u8 = np.empty((height, width, 3), dtype=np.uint8)
    center_x, center_y = width / 2.0, height / 2.0
    for y0 in range(0, height, CHUNK_ROWS):
        y1 = min(y0 + CHUNK_ROWS, height)
        graded = grade_chunk(
            image[y0:y1],
            exposure_gain,
            center_x,
            center_y,
            width / 2.0,
            height / 2.0,
            y0,
            y1,
            width,
        )
        display = linear_to_srgb(np.clip(graded * DISPLAY_BRIGHTNESS, 0.0, 1.0))
        display_u8[y0:y1] = np.clip(display * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return display_u8, exposure_gain


def jpeg_correspondence_metrics(image: np.ndarray, reference_path: Path) -> tuple[dict[str, float], float]:
    display_u8, exposure_gain = graded_display_u8(image)
    image = Image.fromarray(display_u8, "RGB")
    image = image.crop(center_crop_box(EXPECTED_WIDTH, EXPECTED_HEIGHT, QHD_SIZE))
    image = image.resize(QHD_SIZE, Image.Resampling.LANCZOS)
    image = image.resize(HD_SIZE, Image.Resampling.LANCZOS)
    preview = np.asarray(image, dtype=np.float32)
    reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float32)
    delta = preview - reference
    mse = float(np.mean(delta * delta, dtype=np.float64))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    return (
        {
            "preview_vs_jpeg_psnr": psnr,
            "preview_vs_jpeg_mae": float(np.mean(np.abs(delta), dtype=np.float64)),
            "preview_vs_jpeg_max_abs": float(np.max(np.abs(delta))),
        },
        exposure_gain,
    )


def convert_one(task: tuple[str, str, str]) -> dict[str, Any]:
    source_path = Path(task[0])
    output_path = Path(task[1])
    reference_path = Path(task[2])
    started = time.monotonic()
    image, _source_header = read_exr_rgb_and_header(source_path)
    if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        raise RuntimeError(f"Unexpected decoded source shape: {source_path}: {image.shape}")
    if not np.isfinite(image).all():
        raise RuntimeError(f"Non-finite source pixels: {source_path}")
    pixel_min = float(image.min())
    pixel_max = float(image.max())
    metrics, visual_preview_exposure_gain = jpeg_correspondence_metrics(image, reference_path)
    del image
    source_sha256 = sha256_file(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.exr")
    try:
        shutil.copyfile(source_path, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_sha256 = sha256_file(output_path)
    if output_sha256 != source_sha256:
        raise RuntimeError(f"Raw EXR copy is not byte-exact: {source_path} -> {output_path}")
    return {
        "source": str(source_path),
        "source_sha256": source_sha256,
        "source_bytes": source_path.stat().st_size,
        "output_relative": str(output_path),
        "output_sha256": output_sha256,
        "output_bytes": output_path.stat().st_size,
        "reference_jpeg": str(reference_path),
        "reference_jpeg_sha256": sha256_file(reference_path),
        "source_byte_exact": True,
        "visual_preview_exposure_gain": visual_preview_exposure_gain,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "seconds": time.monotonic() - started,
        **metrics,
    }


def frame_state_path(staging_root: Path, frame: int) -> Path:
    return staging_root / STATE_DIR_NAME / f"{frame:06d}.json"


def verify_frame_state(staging_root: Path, state: dict[str, Any], mapping: dict[str, str]) -> bool:
    if state.get("schema_version") != SCHEMA_VERSION or len(state.get("outputs", [])) != len(mapping):
        return False
    expected = set(mapping.values())
    actual = {Path(row["output_relative"]).name for row in state["outputs"]}
    if actual != expected:
        return False
    for row in state["outputs"]:
        path = staging_root / row["output_relative"]
        if not path.is_file() or path.stat().st_size != row["output_bytes"]:
            return False
        if sha256_file(path) != row["output_sha256"]:
            return False
    return True


def update_transform_data(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    crop_left, crop_top, crop_right, crop_bottom = center_crop_box(EXPECTED_WIDTH, EXPECTED_HEIGHT, QHD_SIZE)
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    sx = crop_width / HD_SIZE[0]
    sy = crop_height / HD_SIZE[1]
    max_ray_error = 0.0
    for row in data["frames"]:
        old = {key: row[key] for key in ("fl_x", "fl_y", "cx", "cy", "w", "h")}
        if int(old["w"]) != HD_SIZE[0] or int(old["h"]) != HD_SIZE[1]:
            raise RuntimeError(f"Unexpected processed camera dimensions: {old}")
        row["file_path"] = str(Path(row["file_path"]).with_suffix(".exr"))
        row["w"] = EXPECTED_WIDTH
        row["h"] = EXPECTED_HEIGHT
        row["fl_x"] = float(old["fl_x"]) * sx
        row["fl_y"] = float(old["fl_y"]) * sy
        row["cx"] = crop_left + float(old["cx"]) * sx
        row["cy"] = crop_top + float(old["cy"]) * sy
        for u, v in ((0.0, 0.0), (960.0, 540.0), (1919.0, 1079.0)):
            old_x = (u - float(old["cx"])) / float(old["fl_x"])
            old_y = (v - float(old["cy"])) / float(old["fl_y"])
            new_u = crop_left + u * sx
            new_v = crop_top + v * sy
            new_x = (new_u - row["cx"]) / row["fl_x"]
            new_y = (new_v - row["cy"]) / row["fl_y"]
            max_ray_error = max(max_ray_error, abs(old_x - new_x), abs(old_y - new_y))
    return data, {
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "intrinsics_scale_x": sx,
        "intrinsics_scale_y": sy,
        "max_normalized_ray_error": max_ray_error,
    }


def finalize_stage(  # noqa: PLR0917
    staging_root: Path,
    dataset_root: Path,
    source_root: Path,
    root_manifest: dict[str, Any],
    mapping: dict[str, str],
    protected_before: dict[str, Any],
    essential_before: dict[str, Any],
) -> dict[str, Any]:
    frames = [int(value) for value in root_manifest["frames"]]
    states = []
    for frame in frames:
        state_path = frame_state_path(staging_root, frame)
        if not state_path.is_file():
            raise RuntimeError(f"Missing completed frame state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not verify_frame_state(staging_root, state, mapping):
            raise RuntimeError(f"Invalid completed frame state: {state_path}")
        states.append(state)

    original_transforms_hashes: dict[str, str] = {}
    transformed_hashes: dict[str, str] = {}
    geometry: dict[str, float] | None = None
    invariant_records = []
    for frame in frames:
        old_path = dataset_root / f"{frame:06d}" / "transforms.json"
        new_path = staging_root / f"{frame:06d}" / "transforms.json"
        old_data = json.loads(old_path.read_text(encoding="utf-8"))
        old_invariants = [
            {
                "file_stem": Path(row["file_path"]).stem,
                "colmap_im_id": row.get("colmap_im_id"),
                "transform_matrix": row["transform_matrix"],
                "camera_model": row.get("camera_model"),
                "distortion": {key: row.get(key) for key in ("k1", "k2", "k3", "k4", "p1", "p2")},
            }
            for row in old_data["frames"]
        ]
        new_data, frame_geometry = update_transform_data(old_data)
        geometry = geometry or frame_geometry
        if frame_geometry != geometry:
            raise RuntimeError("Geometry conversion changed between frames")
        new_invariants = [
            {
                "file_stem": Path(row["file_path"]).stem,
                "colmap_im_id": row.get("colmap_im_id"),
                "transform_matrix": row["transform_matrix"],
                "camera_model": row.get("camera_model"),
                "distortion": {key: row.get(key) for key in ("k1", "k2", "k3", "k4", "p1", "p2")},
            }
            for row in new_data["frames"]
        ]
        if old_invariants != new_invariants:
            raise RuntimeError(f"COLMAP invariant changed in frame {frame:06d}")
        atomic_json(new_path, new_data)
        original_transforms_hashes[f"{frame:06d}"] = sha256_file(old_path)
        transformed_hashes[f"{frame:06d}"] = sha256_file(new_path)
        invariant_records.extend(new_invariants)

    updated_root = json.loads(json.dumps(root_manifest))
    updated_root["camera_file_mapping"] = {
        str(Path(name).with_suffix(".exr")): stem for name, stem in root_manifest["camera_file_mapping"].items()
    }
    updated_root["auto_exposure_policy"] = (
        "none for EXR data; source pixels are copied byte-for-byte. Per-frame/per-camera exposure "
        "is used only for temporary visual-audit previews."
    )
    updated_root["geometry"] = (
        "full uncropped 6144x3072 source frame; central 5461x3072 overlap maps to the former 1920x1080 JPEG crop"
    )
    if "frozen_exposure_gains" in updated_root:
        updated_root["legacy_007740_exposure_gains_not_used"] = updated_root.pop("frozen_exposure_gains")
    if "grade_config" in updated_root:
        updated_root["legacy_jpeg_grade_config_not_applied"] = updated_root.pop("grade_config")
    if "grade_script" in updated_root:
        updated_root["legacy_jpeg_grade_script_not_applied"] = updated_root.pop("grade_script")
    updated_root["source_root"] = str(source_root)
    updated_root["image_format"] = {
        "container": "OpenEXR",
        "channels": "RGB",
        "dtype": "float16",
        "compression": "ZIPS",
        "color_encoding": "linear-sRGB",
        "source_byte_exact": True,
        "color_or_exposure_correction": "none",
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
    }
    updated_root["exr_conversion_manifest"] = FINAL_MANIFEST_NAME
    atomic_json(staging_root / "perframe_manifest.json", updated_root)

    all_outputs = [row for state in states for row in state["outputs"]]
    metrics = {
        "preview_vs_jpeg_psnr_min": min(row["preview_vs_jpeg_psnr"] for row in all_outputs),
        "preview_vs_jpeg_psnr_mean": float(
            np.mean([row["preview_vs_jpeg_psnr"] for row in all_outputs], dtype=np.float64)
        ),
        "preview_vs_jpeg_mae_max": max(row["preview_vs_jpeg_mae"] for row in all_outputs),
        "preview_vs_jpeg_mae_mean": float(
            np.mean([row["preview_vs_jpeg_mae"] for row in all_outputs], dtype=np.float64)
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "runtime": runtime_fingerprint(),
        "source_root": str(source_root),
        "source_reference_family_frame": "007390",
        "source_frames": frames,
        "source_camera_mapping": mapping,
        "dataset_root_before_install": str(dataset_root),
        "staging_root": str(staging_root),
        "protected_root": str(PROTECTED_ROOT),
        "protected_tree_before": protected_before,
        "protected_essential_content_before": essential_before,
        "image_contract": updated_root["image_format"],
        "geometry_conversion": geometry,
        "colmap_invariant_aggregate_sha256": sha256_json(invariant_records),
        "original_transforms_hashes": original_transforms_hashes,
        "transformed_hashes": transformed_hashes,
        "frame_count": len(states),
        "image_count": len(all_outputs),
        "preview_correspondence": metrics,
        "frames": states,
    }
    atomic_json(staging_root / FINAL_MANIFEST_NAME, manifest)

    content_records = [
        {
            "path": row["output_relative"],
            "bytes": row["output_bytes"],
            "sha256": row["output_sha256"],
            "source_sha256": row["source_sha256"],
        }
        for row in all_outputs
    ]
    content = {
        "schema_version": 1,
        "created_at": utc_now(),
        "file_count": len(content_records),
        "aggregate_sha256": sha256_json(content_records),
        "files": content_records,
    }
    atomic_json(staging_root / CONTENT_MANIFEST_NAME, content)
    shutil.rmtree(staging_root / STATE_DIR_NAME)
    return manifest


def stage(args: argparse.Namespace) -> int:
    dataset_root = require_mnt_child(args.dataset_root, "--dataset-root")
    source_root = require_mnt_child(args.source_root, "--source-root")
    staging_root = require_mnt_child(args.staging_root, "--staging-root")
    protect_paths(dataset_root, staging_root)
    protected_root = resolved(PROTECTED_ROOT)
    root_manifest = read_root_manifest(dataset_root)
    frames = [int(value) for value in root_manifest["frames"]]
    selected = sorted(set(args.frame or frames))
    if any(frame not in frames for frame in selected):
        raise RuntimeError(f"Selected frame is outside the canonical sequence: {selected}")
    mapping = camera_mapping(root_manifest)
    validate_jpeg_dataset(dataset_root, frames, mapping)

    if not staging_root.exists():
        free = shutil.disk_usage(staging_root.parent).free
        if free < args.minimum_free_gib * (1 << 30):
            raise RuntimeError(f"Insufficient free space: {free / (1 << 30):.1f} GiB < {args.minimum_free_gib} GiB")
        print("preflight protected stat fingerprint", flush=True)
        protected_before = protected_tree_stat_fingerprint(protected_root)
        print("preflight protected essential-content fingerprint", flush=True)
        essential_before = essential_content_fingerprint(protected_root)
        target_essential = essential_content_fingerprint(dataset_root)
        if target_essential["aggregate_sha256"] != essential_before["aggregate_sha256"]:
            raise RuntimeError("/mnt/data JPEG copy does not match protected dataset essentials")
        print("preflight source inventory", flush=True)
        source_inventory = validate_source_inventory(source_root, frames, mapping)
        print(f"initialize staging={staging_root}", flush=True)
        clone_dataset_without_jpegs(dataset_root, staging_root)
        state = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "script_sha256": sha256_file(Path(__file__)),
            "dataset_root": str(dataset_root),
            "source_root": str(source_root),
            "staging_root": str(staging_root),
            "frames": frames,
            "mapping": mapping,
            "protected_tree_before": protected_before,
            "protected_essential_content_before": essential_before,
            "target_essential_before": target_essential,
            "source_inventory": source_inventory,
        }
        atomic_json(staging_root / STATE_DIR_NAME / "campaign.json", state)
    else:
        campaign_path = staging_root / STATE_DIR_NAME / "campaign.json"
        if not campaign_path.is_file():
            raise RuntimeError(f"Existing staging tree has no resumable campaign state: {staging_root}")
        state = json.loads(campaign_path.read_text(encoding="utf-8"))
        if state.get("script_sha256") != sha256_file(Path(__file__)):
            raise RuntimeError("Staging campaign was created by a different script revision")
        if state.get("mapping") != mapping or state.get("frames") != frames:
            raise RuntimeError("Staging campaign contract changed")

    for frame in selected:
        state_path = frame_state_path(staging_root, frame)
        if state_path.is_file():
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            if verify_frame_state(staging_root, prior, mapping):
                print(f"frame={frame:06d} already verified; skip", flush=True)
                continue
            raise RuntimeError(f"Frame state exists but does not verify: {state_path}")
        frame_started = time.monotonic()
        tasks = []
        for physical_stem, target_name in sorted(mapping.items(), key=lambda item: item[1]):
            source = source_root / f"{frame:06d}" / f"{physical_stem}.exr"
            output = staging_root / f"{frame:06d}" / "images" / target_name
            reference = dataset_root / f"{frame:06d}" / "images" / f"{Path(target_name).stem}.jpg"
            tasks.append((str(source), str(output), str(reference)))
        print(f"frame={frame:06d} start cameras={len(tasks)} workers={args.workers}", flush=True)
        outputs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(convert_one, task): task for task in tasks}
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                output_path = Path(row["output_relative"])
                row["output_relative"] = str(output_path.relative_to(staging_root))
                row["source_relative"] = str(Path(row["source"]).relative_to(source_root))
                row["reference_jpeg_relative"] = str(Path(row["reference_jpeg"]).relative_to(dataset_root))
                outputs.append(row)
                print(
                    f"frame={frame:06d} done={completed:02d}/{len(tasks)} "
                    f"image={output_path.name} psnr={row['preview_vs_jpeg_psnr']:.2f} "
                    f"seconds={row['seconds']:.1f}",
                    flush=True,
                )
        outputs.sort(key=lambda row: row["output_relative"])
        frame_state = {
            "schema_version": SCHEMA_VERSION,
            "frame": frame,
            "frame_name": f"{frame:06d}",
            "created_at": utc_now(),
            "seconds": time.monotonic() - frame_started,
            "outputs": outputs,
        }
        atomic_json(state_path, frame_state)
        if not verify_frame_state(staging_root, frame_state, mapping):
            raise RuntimeError(f"Fresh frame state did not verify: {frame:06d}")
        print(f"frame={frame:06d} complete seconds={frame_state['seconds']:.1f}", flush=True)

    completed = [frame for frame in frames if frame_state_path(staging_root, frame).is_file()]
    print(f"campaign completed_frames={len(completed)}/{len(frames)}", flush=True)
    if len(completed) == len(frames):
        campaign = json.loads((staging_root / STATE_DIR_NAME / "campaign.json").read_text(encoding="utf-8"))
        manifest = finalize_stage(
            staging_root,
            dataset_root,
            source_root,
            root_manifest,
            mapping,
            campaign["protected_tree_before"],
            campaign["protected_essential_content_before"],
        )
        print(
            f"finalized images={manifest['image_count']} manifest={staging_root / FINAL_MANIFEST_NAME}",
            flush=True,
        )
    return 0


def output_header_only_profile(path: Path) -> dict[str, Any]:
    input_file = OpenEXR.InputFile(str(path))
    header = input_file.header()
    window = header["dataWindow"]
    result = {
        "width": int(window.max.x - window.min.x + 1),
        "height": int(window.max.y - window.min.y + 1),
        "channels": sorted(header["channels"]),
        "pixel_types": {name: str(channel.type) for name, channel in header["channels"].items()},
        "compression": str(header["compression"]),
    }
    input_file.close()
    return result


def verify_complete(root: Path, manifest: dict[str, Any], full_hash: bool = True) -> dict[str, Any]:
    expected_names = set(manifest["source_camera_mapping"].values())
    errors = []
    checked = 0
    bytes_total = 0
    for frame_state in manifest["frames"]:
        frame_name = frame_state["frame_name"]
        image_dir = root / frame_name / "images"
        disk_exr = {path.name for path in image_dir.glob("frame_*.exr")}
        disk_jpg = {path.name for path in image_dir.glob("frame_*.jpg")}
        if disk_exr != expected_names or disk_jpg:
            errors.append(f"inventory:{frame_name}:exr={len(disk_exr)}:jpg={len(disk_jpg)}")
        transform_data = json.loads((root / frame_name / "transforms.json").read_text(encoding="utf-8"))
        bound = {Path(row["file_path"]).name for row in transform_data["frames"]}
        if bound != expected_names:
            errors.append(f"binding:{frame_name}")
        for row in frame_state["outputs"]:
            path = root / row["output_relative"]
            if not path.is_file() or path.stat().st_size != row["output_bytes"]:
                errors.append(f"missing-or-size:{row['output_relative']}")
                continue
            profile = output_header_only_profile(path)
            if profile != {
                "width": EXPECTED_WIDTH,
                "height": EXPECTED_HEIGHT,
                "channels": ["B", "G", "R"],
                "pixel_types": {"B": "HALF", "G": "HALF", "R": "HALF"},
                "compression": "ZIPS_COMPRESSION",
            }:
                errors.append(f"profile:{row['output_relative']}:{profile}")
            if full_hash and sha256_file(path) != row["output_sha256"]:
                errors.append(f"hash:{row['output_relative']}")
            bytes_total += path.stat().st_size
            checked += 1
    if errors:
        raise RuntimeError(f"EXR verification failed ({len(errors)}): {errors[:20]}")
    return {"checked": checked, "bytes": bytes_total, "full_hash": full_hash, "errors": errors}


def verify(args: argparse.Namespace) -> int:
    root = require_mnt_child(args.root, "--root")
    manifest_path = root / FINAL_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = verify_complete(root, manifest, full_hash=not args.skip_full_hash)
    protected_stat = protected_tree_stat_fingerprint(resolved(PROTECTED_ROOT))
    protected_content = essential_content_fingerprint(resolved(PROTECTED_ROOT))
    if protected_stat != manifest["protected_tree_before"]:
        raise RuntimeError("Protected dataset stat fingerprint changed")
    if protected_content != manifest["protected_essential_content_before"]:
        raise RuntimeError("Protected dataset essential-content fingerprint changed")
    print(json.dumps({"dataset": result, "protected_unchanged": True}, indent=2), flush=True)
    return 0


def install(args: argparse.Namespace) -> int:
    dataset_root = require_mnt_child(args.dataset_root, "--dataset-root")
    staging_root = require_mnt_child(args.staging_root, "--staging-root")
    backup_root = require_mnt_child(args.backup_root, "--backup-root")
    protect_paths(dataset_root, staging_root, backup_root)
    if not args.execute:
        raise RuntimeError("Installation is gated; pass --execute after reviewing paths")
    if backup_root.exists():
        raise RuntimeError(f"Backup target already exists: {backup_root}")
    manifest = json.loads((staging_root / FINAL_MANIFEST_NAME).read_text(encoding="utf-8"))
    verify_result = verify_complete(staging_root, manifest, full_hash=True)
    protected_stat = protected_tree_stat_fingerprint(resolved(PROTECTED_ROOT))
    protected_content = essential_content_fingerprint(resolved(PROTECTED_ROOT))
    if (
        protected_stat != manifest["protected_tree_before"]
        or protected_content != manifest["protected_essential_content_before"]
    ):
        raise RuntimeError("Protected dataset changed; refusing installation")
    old_fingerprint = essential_content_fingerprint(dataset_root)
    if old_fingerprint["aggregate_sha256"] != protected_content["aggregate_sha256"]:
        raise RuntimeError("Active JPEG copy changed since staging; refusing installation")

    print(f"install rename {dataset_root} -> {backup_root}", flush=True)
    os.rename(dataset_root, backup_root)
    promoted = False
    try:
        print(f"install rename {staging_root} -> {dataset_root}", flush=True)
        os.rename(staging_root, dataset_root)
        promoted = True
        quick = verify_complete(dataset_root, manifest, full_hash=False)
        install_record = {
            "schema_version": 1,
            "installed_at": utc_now(),
            "dataset_root": str(dataset_root),
            "backup_root": str(backup_root),
            "preinstall_full_verification": verify_result,
            "postinstall_quick_verification": quick,
            "protected_unchanged": True,
            "backup_essential_fingerprint": essential_content_fingerprint(backup_root),
        }
        atomic_json(dataset_root / "exr_install_manifest.json", install_record)
        atomic_json(backup_root / "exr_backup_manifest.json", install_record)
    except BaseException:
        if promoted:
            os.rename(dataset_root, staging_root)
        os.rename(backup_root, dataset_root)
        raise
    print(
        f"installed images={verify_result['checked']} dataset={dataset_root} backup={backup_root}",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    stage_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    stage_parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    stage_parser.add_argument("--frame", type=int, action="append")
    stage_parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    stage_parser.add_argument("--minimum-free-gib", type=int, default=380)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, default=DEFAULT_STAGING_ROOT)
    verify_parser.add_argument("--skip-full-hash", action="store_true")

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    install_parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    install_parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    install_parser.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "stage":
        return stage(args)
    if args.command == "verify":
        return verify(args)
    if args.command == "install":
        return install(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
