"""Shared fail-closed helpers for the DEC5 fixed-pose PatchMatch-TSDF campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Iterable


FRAME_COUNT = 50
EVAL_PHYSICAL_CAMERA = "F004_B005_1210O9"
EVAL_SOURCE_FILENAME = "frame_eval_00001.exr"
EXCLUDED_EVAL_CAMERAS = ("J004_D005_1210TA", "L004_B005_12106A")
CALIBRATION_FIELDS = (
    "transform_matrix", "fl_x", "fl_y", "cx", "cy", "w", "h",
    "k1", "k2", "p1", "p2", "camera_model",
)
CSV_FIELDS = (
    "frame_id", "source_dataset", "eval_physical_camera", "train_camera_count",
    "texture_camera_count", "face_psnr", "face_ssim", "face_lpips",
    "depth_coverage_mean", "depth_coverage_min", "mesh_vertices", "mesh_triangles",
    "mesh_components", "render_path", "mesh_path", "render_sha256", "mesh_sha256",
    "metric_status", "visual_status", "ear_artifact", "lipstick_artifact",
    "visual_notes", "status",
)
RECIPE = {
    "geometry_train_camera_count": 62,
    "eval_camera_count": 1,
    "texture_camera_count": 16,
    "image_size": 1920,
    "source_count": 12,
    "patchmatch_iterations_per_pass": 3,
    "patchmatch_passes": ["photometric", "geometric"],
    "depth_range": [4.5, 20.0],
    "geometric_consistency_gates": [6.0, 2.0],
    "filter_min_ncc": 0.1,
    "filter_min_consistent_views": 2,
    "filter_min_triangulation_angle_degrees": 1.0,
    "tsdf_backend": "open3d_tensor_cuda",
    "voxel_length": 0.0005,
    "sdf_trunc": 0.004,
    "extraction_weight": 2.0,
    "depth_trunc": 4.0,
    "normalized_crop_aabb": [-0.15, -0.15, -0.15, 0.15, 0.15, 0.15],
    "component_threshold": "max(100, 0.002 * largest_component_triangles)",
    "rgb": "hard_nearest_fill_global_color_order_no_average",
    "nearest_fill_color_continuity": True,
    "nearest_fill_color_continuity_mode": "global",
    "nearest_fill_rank_penalty": 0.0,
    "depth_hole_fill_max_area": 1000,
    "target_depth_component_min_area": 1000,
    "target_depth_component_max_log_jump": 0.0075,
    "masks_in_geometry_texture_prediction": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalized_name(value: str) -> str:
    name = PurePosixPath(value).as_posix()
    while name.startswith("./"):
        name = name[2:]
    return name


def discover_frames(source_root: Path, count: int = FRAME_COUNT) -> list[Path]:
    candidates = sorted(
        (path for path in source_root.iterdir() if path.is_dir() and len(path.name) == 6 and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    selected = candidates[:count]
    if len(selected) != count or len({path.name for path in selected}) != count:
        raise ValueError(f"Expected {count} unique numeric frame directories in {source_root}")
    return selected


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def frame_stem(frame: dict) -> str:
    return Path(normalized_name(str(frame["file_path"]))).stem


def validate_source_frame(source: Path) -> dict:
    transforms = source / "transforms.json"
    payload = load_json(transforms)
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != 65:
        raise ValueError(f"{source} must contain exactly 65 transform frames")
    if any(not isinstance(frame, dict) or "mask_path" in frame for frame in frames):
        raise ValueError(f"Invalid or mask-bearing source frame in {source}")
    names = [frame_stem(frame) for frame in frames]
    physical = [frame.get("physical_camera") for frame in frames]
    train = [frame for frame in frames if frame_stem(frame).startswith("frame_train_")]
    eval_frames = [frame for frame in frames if frame_stem(frame).startswith("frame_eval_")]
    target = [frame for frame in eval_frames if Path(str(frame["file_path"])).name == EVAL_SOURCE_FILENAME]
    if len(set(names)) != 65 or len(set(physical)) != 65 or len(train) != 62 or len(eval_frames) != 3:
        raise ValueError(f"{source} must contain 62 train and 3 uniquely identified eval cameras")
    if len(target) != 1 or target[0].get("physical_camera") != EVAL_PHYSICAL_CAMERA:
        raise ValueError(f"{source} has no unique {EVAL_SOURCE_FILENAME}/{EVAL_PHYSICAL_CAMERA} eval match")
    eval_physical = {frame.get("physical_camera") for frame in eval_frames}
    if eval_physical != {EVAL_PHYSICAL_CAMERA, *EXCLUDED_EVAL_CAMERAS}:
        raise ValueError(f"Unexpected eval physical cameras in {source}: {sorted(eval_physical)}")
    for frame in frames:
        image = source / normalized_name(str(frame["file_path"]))
        if image.suffix.lower() != ".exr" or not image.is_file():
            raise FileNotFoundError(image)
    return {
        "source_dataset": str(source.resolve()),
        "source_transforms_sha256": sha256(transforms),
        "source_exr_count": len(frames),
        "train_camera_count": len(train),
        "eval_camera_count": len(eval_frames),
        "eval_physical_camera": EVAL_PHYSICAL_CAMERA,
        "eval_source_filename": EVAL_SOURCE_FILENAME,
    }


def calibration_by_physical_camera(template: Path) -> dict[str, dict]:
    payload = load_json(template)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Calibration template contains no frames")
    result: dict[str, dict] = {}
    for frame in frames:
        camera = frame.get("physical_camera") if isinstance(frame, dict) else None
        if not isinstance(camera, str) or camera in result:
            raise ValueError("Calibration physical_camera values must be unique strings")
        missing = [field for field in CALIBRATION_FIELDS if field not in frame]
        if missing:
            raise ValueError(f"Calibration camera {camera} lacks {missing}")
        result[camera] = frame
    return result


def stage_fixed_calibration_dataset(source: Path, converted: Path, template: Path, output: Path) -> dict:
    """Hard-link 62+1 JPEGs and replace camera fields by physical_camera."""

    source_payload = load_json(source / "transforms.json")
    converted_payload = load_json(converted / "transforms.json")
    converted_manifest = load_json(converted / "conversion_manifest.json")
    calibration = calibration_by_physical_camera(template)
    source_by_camera = {frame["physical_camera"]: frame for frame in source_payload["frames"]}
    converted_by_camera = {frame["physical_camera"]: frame for frame in converted_payload["frames"]}
    if set(source_by_camera) != set(converted_by_camera):
        raise ValueError("EXR and converted JPEG physical_camera inventories differ")
    train_cameras = [
        frame["physical_camera"] for frame in source_payload["frames"]
        if frame_stem(frame).startswith("frame_train_")
    ]
    selected_cameras = train_cameras + [EVAL_PHYSICAL_CAMERA]
    if len(train_cameras) != 62 or len(set(selected_cameras)) != 63:
        raise ValueError("Selected camera inventory must be exactly 62 train plus one eval")
    missing = sorted(set(selected_cameras) - set(calibration))
    if missing:
        raise ValueError(f"Calibration template lacks selected cameras: {missing}")
    stage = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if output.exists() or stage.exists():
        raise FileExistsError(output if output.exists() else stage)
    (stage / "images").mkdir(parents=True)
    frames: list[dict] = []
    rows_by_original = {
        normalized_name(str(row["frame_file_path"])): row for row in converted_manifest["images"]
    }
    selected_conversion_rows: list[dict] = []
    for camera in selected_cameras:
        source_frame = source_by_camera[camera]
        jpeg_frame = converted_by_camera[camera]
        source_name = normalized_name(str(source_frame["file_path"]))
        jpeg_name = normalized_name(str(jpeg_frame["file_path"]))
        source_jpeg = converted / jpeg_name
        destination_jpeg = stage / "images" / Path(jpeg_name).name
        if not source_jpeg.is_file():
            raise FileNotFoundError(source_jpeg)
        os.link(source_jpeg, destination_jpeg)
        result = json.loads(json.dumps(jpeg_frame))
        result["file_path"] = f"images/{destination_jpeg.name}"
        result["source_file_path"] = source_name
        for field in CALIBRATION_FIELDS:
            result[field] = json.loads(json.dumps(calibration[camera][field]))
        result.pop("colmap_im_id", None)
        frames.append(result)
        if source_name not in rows_by_original:
            raise ValueError(f"Conversion manifest has no row for {source_name}")
        row = dict(rows_by_original[source_name])
        row["physical_camera"] = camera
        selected_conversion_rows.append(row)
    train_names = [frame["file_path"] for frame in frames if frame["physical_camera"] in set(train_cameras)]
    eval_names = [frame["file_path"] for frame in frames if frame["physical_camera"] == EVAL_PHYSICAL_CAMERA]
    result_payload = {
        key: json.loads(json.dumps(value))
        for key, value in converted_payload.items()
        if key not in {"frames", "train_filenames", "val_filenames", "test_filenames", "ply_file_path"}
    }
    result_payload.update(
        {
            "frames": frames,
            "train_filenames": train_names,
            "val_filenames": eval_names,
            "test_filenames": eval_names,
            "fixed_calibration": {
                "template": str(template.resolve()),
                "template_sha256": sha256(template),
                "mapping_key": "physical_camera",
                "copied_fields": list(CALIBRATION_FIELDS),
                "selected_eval_physical_camera": EVAL_PHYSICAL_CAMERA,
                "excluded_eval_physical_cameras": list(EXCLUDED_EVAL_CAMERAS),
            },
        }
    )
    atomic_json(stage / "transforms.json", result_payload)
    audit = {
        "schema_version": 1,
        "source_dataset": str(source.resolve()),
        "source_transforms_sha256": sha256(source / "transforms.json"),
        "calibration_template": str(template.resolve()),
        "calibration_template_sha256": sha256(template),
        "train_camera_count": len(train_names),
        "eval_camera_count": len(eval_names),
        "physical_camera_count": len(frames),
        "train_filenames": train_names,
        "val_filenames": eval_names,
        "test_filenames": eval_names,
        "excluded_eval_physical_cameras": list(EXCLUDED_EVAL_CAMERAS),
        "conversion_rows": selected_conversion_rows,
    }
    atomic_json(stage / "staging_manifest.json", audit)
    os.replace(stage, output)
    return audit


def validate_hash_manifest(root: Path, manifest: dict) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Retained manifest has no files")
    seen: set[str] = set()
    for row in files:
        relative = normalized_name(str(row.get("path")))
        if relative in seen or relative.startswith("../") or Path(relative).is_absolute():
            raise ValueError(f"Invalid duplicate/escaping retained path: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"Retained artifact hash/size mismatch: {path}")


def robust_initial_thresholds(rows: list[dict]) -> dict[str, float]:
    if len(rows) != 3:
        raise ValueError("Initial regression baseline requires exactly three accepted frames")
    result: dict[str, float] = {}
    for key, floor in (("face_psnr", 1.0), ("face_ssim", 0.03), ("face_lpips", 0.05)):
        values = [float(row[key]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite initial {key}")
        median = sorted(values)[1]
        absolute = sorted(abs(value - median) for value in values)[1]
        result[key] = max(floor, 3.0 * 1.4826 * absolute)
    return result


def copy_or_validate_immutable(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256(source) != expected_sha256:
        raise ValueError(f"Unexpected SHA-256 for {source}")
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Immutable config differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    # Some campaign roots are writable shared mounts whose directory entries
    # are owned by the storage service.  Copy bytes only: attempting to carry
    # the source mtime with copy2/copy_stat can be rejected even though the
    # payload itself is writable.  The required identity is the content hash.
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Copied immutable config failed hash verification: {destination}")
    os.replace(temporary, destination)
