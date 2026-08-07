#!/usr/bin/env python3
"""Create a separate 1920x1080 linear-sRGB EXR temporal dataset.

The full-resolution EXR dataset and the protected JPEG dataset are read-only
inputs. Each 6144x3072 EXR is center-cropped with the frozen JPEG crop box and
resized with the same two Pillow Lanczos stages (2560x1440, then 1920x1080).
No exposure, grading, transfer function, or color correction is applied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import OpenEXR
from convert_temporal_exr_to_leader_jpeg import HD_SIZE, QHD_SIZE, center_crop_box
from convert_temporal_raw_exr_dataset import (
    EXPECTED_CAMERA_COUNT,
    EXPECTED_FRAME_COUNT,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    PROTECTED_ROOT,
    atomic_json,
    camera_mapping,
    essential_content_fingerprint,
    output_header_only_profile,
    protected_tree_stat_fingerprint,
    read_exr_rgb_and_header,
    read_root_manifest,
    require_mnt_child,
    runtime_fingerprint,
    sha256_file,
    sha256_json,
    utc_now,
)
from PIL import Image

FULL_EXR_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f")
JPEG_REFERENCE_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f_jpeg_backup_20260807")
DEFAULT_TARGET_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f_exr_1920x1080")
DEFAULT_STAGING_ROOT = Path("/mnt/data/temporal_perframe_stride7_45f_exr_1920x1080_staging_20260807")
STATE_DIR = ".exr_1920_conversion_state"
FINAL_MANIFEST = "exr_1920x1080_conversion_manifest.json"
CONTENT_MANIFEST = "dataset_content_manifest_exr_1920x1080_20260807.json"
INSTALL_MANIFEST = "exr_1920x1080_install_manifest.json"
SCHEMA_VERSION = 1
GENERATED_MANIFESTS = {
    "dataset_content_manifest_exr_20260807.json",
    "exr_backup_manifest.json",
    "exr_conversion_manifest.json",
    "exr_install_manifest.json",
}


def ensure_safe_paths(*write_paths: Path) -> None:
    protected = (PROTECTED_ROOT.resolve(), FULL_EXR_ROOT.resolve(), JPEG_REFERENCE_ROOT.resolve())
    for write_path in write_paths:
        candidate = write_path.resolve()
        for read_only in protected:
            if candidate == read_only or read_only in candidate.parents or candidate in read_only.parents:
                raise RuntimeError(f"Write path overlaps read-only input: {candidate} vs {read_only}")


def source_tree_fingerprint(root: Path) -> dict[str, Any]:
    records = []
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


def clone_template(template_root: Path, staging_root: Path) -> None:
    staging_root.mkdir(parents=False, exist_ok=False)
    for source in sorted(template_root.rglob("*")):
        relative = source.relative_to(template_root)
        target = staging_root / relative
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.parent.name == "images" and source.suffix.lower() == ".jpg":
            continue
        elif relative.as_posix() in GENERATED_MANIFESTS:
            continue
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def resize_float_channel(channel: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.ascontiguousarray(channel, dtype=np.float32), mode="F")
    image = image.resize(QHD_SIZE, Image.Resampling.LANCZOS)
    image = image.resize(HD_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def resize_linear_exr(image: np.ndarray) -> np.ndarray:
    if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        raise RuntimeError(f"Unexpected full EXR shape: {image.shape}")
    crop_box = center_crop_box(EXPECTED_WIDTH, EXPECTED_HEIGHT, QHD_SIZE)
    left, top, right, bottom = crop_box
    cropped = image[top:bottom, left:right]
    resized = np.stack([resize_float_channel(cropped[..., channel]) for channel in range(3)], axis=-1)
    if resized.shape != (HD_SIZE[1], HD_SIZE[0], 3) or not np.isfinite(resized).all():
        raise RuntimeError(f"Invalid resized image: shape={resized.shape}")
    return resized.astype(np.float16)


def output_header(source_header: dict[str, Any]) -> dict[str, Any]:
    header = source_header.copy()
    for key in ("channels", "dataWindow", "displayWindow"):
        header.pop(key, None)
    header["compression"] = OpenEXR.ZIPS_COMPRESSION
    header["type"] = OpenEXR.scanlineimage
    header["FrameWidth"] = HD_SIZE[0]
    header["FrameHeight"] = HD_SIZE[1]
    header["datasetColorEncoding"] = "linear-sRGB"
    header["datasetGeometry"] = "center crop 341,0,5802,3072; Lanczos 2560x1440 then 1920x1080"
    header["datasetColorOrExposureCorrection"] = "none"
    return header


def write_exr_atomic(path: Path, header: dict[str, Any], pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    OpenEXR.File(header, {"RGB": pixels}).write(str(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def convert_one(task: tuple[str, str]) -> dict[str, Any]:
    source_path = Path(task[0])
    output_path = Path(task[1])
    started = time.monotonic()
    pixels, source_header = read_exr_rgb_and_header(source_path)
    source_min = float(np.min(pixels))
    source_max = float(np.max(pixels))
    resized = resize_linear_exr(pixels)
    output_min = float(np.min(resized))
    output_max = float(np.max(resized))
    write_exr_atomic(output_path, output_header(source_header), resized)
    decoded, _ = read_exr_rgb_and_header(output_path)
    if decoded.shape != resized.shape or not np.array_equal(decoded, resized):
        raise RuntimeError(f"EXR round-trip mismatch: {output_path}")
    return {
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "source_pixel_min": source_min,
        "source_pixel_max": source_max,
        "output_pixel_min": output_min,
        "output_pixel_max": output_max,
        "seconds": time.monotonic() - started,
    }


def frame_state_path(staging_root: Path, frame: int) -> Path:
    return staging_root / STATE_DIR / f"{frame:06d}.json"


def verify_frame_state(staging_root: Path, state: dict[str, Any], expected_names: set[str]) -> bool:
    outputs = state.get("outputs", [])
    if state.get("schema_version") != SCHEMA_VERSION or len(outputs) != EXPECTED_CAMERA_COUNT:
        return False
    if {Path(row["output_relative"]).name for row in outputs} != expected_names:
        return False
    for row in outputs:
        path = staging_root / row["output_relative"]
        if not path.is_file() or path.stat().st_size != row["output_bytes"]:
            return False
        if sha256_file(path) != row["output_sha256"]:
            return False
    return True


def update_metadata(
    staging_root: Path,
    template_root: Path,
    frames: list[int],
    mapping: dict[str, str],
) -> tuple[dict[str, str], str]:
    transform_hashes = {}
    invariants = []
    for frame in frames:
        source_path = template_root / f"{frame:06d}" / "transforms.json"
        target_path = staging_root / f"{frame:06d}" / "transforms.json"
        source_data = json.loads(source_path.read_text(encoding="utf-8"))
        target_data = json.loads(json.dumps(source_data))
        for row in target_data["frames"]:
            row["file_path"] = str(Path(row["file_path"]).with_suffix(".exr"))
            if int(row["w"]) != HD_SIZE[0] or int(row["h"]) != HD_SIZE[1]:
                raise RuntimeError(f"Unexpected JPEG camera size: {source_path}")
            invariants.append(
                {
                    "frame": frame,
                    "stem": Path(row["file_path"]).stem,
                    "colmap_im_id": row.get("colmap_im_id"),
                    "transform_matrix": row["transform_matrix"],
                    "camera_model": row.get("camera_model"),
                    "intrinsics": {key: row.get(key) for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")},
                    "distortion": {key: row.get(key) for key in ("k1", "k2", "k3", "k4", "p1", "p2")},
                }
            )
        atomic_json(target_path, target_data)
        transform_hashes[f"{frame:06d}"] = sha256_file(target_path)

    root_manifest = read_root_manifest(template_root)
    root_manifest["camera_file_mapping"] = {
        str(Path(filename).with_suffix(".exr")): physical
        for filename, physical in root_manifest["camera_file_mapping"].items()
    }
    root_manifest["geometry"] = (
        "linear source center crop [341:5802, 0:3072], then Pillow Lanczos "
        "resize 2560x1440 and 1920x1080"
    )
    root_manifest["image_format"] = {
        "container": "OpenEXR",
        "channels": "RGB",
        "dtype": "float16",
        "compression": "ZIPS",
        "color_encoding": "linear-sRGB",
        "color_or_exposure_correction": "none",
        "width": HD_SIZE[0],
        "height": HD_SIZE[1],
    }
    root_manifest["source_root"] = str(FULL_EXR_ROOT)
    root_manifest["exr_conversion_manifest"] = FINAL_MANIFEST
    if "frozen_exposure_gains" in root_manifest:
        root_manifest["legacy_jpeg_exposure_gains_not_used"] = root_manifest.pop("frozen_exposure_gains")
    if "grade_config" in root_manifest:
        root_manifest["legacy_jpeg_grade_config_not_applied"] = root_manifest.pop("grade_config")
    if "grade_script" in root_manifest:
        root_manifest["legacy_jpeg_grade_script_not_applied"] = root_manifest.pop("grade_script")
    atomic_json(staging_root / "perframe_manifest.json", root_manifest)
    return transform_hashes, sha256_json(invariants)


def finalize(staging_root: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    frames = campaign["frames"]
    mapping = campaign["mapping"]
    expected_names = set(mapping.values())
    states = []
    for frame in frames:
        state = json.loads(frame_state_path(staging_root, frame).read_text(encoding="utf-8"))
        if not verify_frame_state(staging_root, state, expected_names):
            raise RuntimeError(f"Invalid frame state: {frame:06d}")
        states.append(state)
    transform_hashes, invariant_hash = update_metadata(
        staging_root,
        Path(campaign["template_root"]),
        frames,
        mapping,
    )
    outputs = [row for state in states for row in state["outputs"]]
    crop = center_crop_box(EXPECTED_WIDTH, EXPECTED_HEIGHT, QHD_SIZE)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "runtime": runtime_fingerprint(),
        "full_exr_source_root": str(FULL_EXR_ROOT),
        "jpeg_reference_root": str(JPEG_REFERENCE_ROOT),
        "target_root": str(DEFAULT_TARGET_ROOT),
        "protected_root": str(PROTECTED_ROOT),
        "protected_tree_before": campaign["protected_tree_before"],
        "protected_content_before": campaign["protected_content_before"],
        "full_source_tree_before": campaign["full_source_tree_before"],
        "jpeg_reference_tree_before": campaign["jpeg_reference_tree_before"],
        "frames": states,
        "frame_count": len(states),
        "image_count": len(outputs),
        "camera_mapping": mapping,
        "image_contract": {
            "container": "OpenEXR",
            "channels": "RGB",
            "dtype": "float16",
            "compression": "ZIPS",
            "color_encoding": "linear-sRGB",
            "color_or_exposure_correction": "none",
            "width": HD_SIZE[0],
            "height": HD_SIZE[1],
        },
        "geometry": {
            "input_size": [EXPECTED_WIDTH, EXPECTED_HEIGHT],
            "crop_box": list(crop),
            "crop_size": [crop[2] - crop[0], crop[3] - crop[1]],
            "intermediate_size": list(QHD_SIZE),
            "output_size": list(HD_SIZE),
            "resampler": "Pillow LANCZOS at both stages",
        },
        "transforms_sha256": transform_hashes,
        "colmap_camera_invariant_sha256": invariant_hash,
        "pixel_range": {
            "minimum": min(row["output_pixel_min"] for row in outputs),
            "maximum": max(row["output_pixel_max"] for row in outputs),
        },
    }
    atomic_json(staging_root / FINAL_MANIFEST, manifest)
    records = [
        {"path": row["output_relative"], "bytes": row["output_bytes"], "sha256": row["output_sha256"]}
        for row in outputs
    ]
    atomic_json(
        staging_root / CONTENT_MANIFEST,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "file_count": len(records),
            "aggregate_sha256": sha256_json(records),
            "files": records,
        },
    )
    shutil.rmtree(staging_root / STATE_DIR)
    return manifest


def stage(args: argparse.Namespace) -> int:
    staging_root = require_mnt_child(args.staging_root, "--staging-root")
    target_root = require_mnt_child(args.target_root, "--target-root")
    ensure_safe_paths(staging_root, target_root)
    if target_root.exists():
        raise RuntimeError(f"Target already exists: {target_root}")
    root_manifest = read_root_manifest(JPEG_REFERENCE_ROOT)
    frames = [int(frame) for frame in root_manifest["frames"]]
    selected = sorted(set(args.frame or frames))
    if any(frame not in frames for frame in selected):
        raise RuntimeError(f"Frame outside canonical sequence: {selected}")
    mapping = camera_mapping(root_manifest)
    expected_names = set(mapping.values())

    if not staging_root.exists():
        free = shutil.disk_usage(staging_root.parent).free
        if free < args.minimum_free_gib * (1 << 30):
            raise RuntimeError(f"Insufficient free space: {free / (1 << 30):.1f} GiB")
        print("fingerprint read-only inputs", flush=True)
        campaign = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "script_sha256": sha256_file(Path(__file__)),
            "frames": frames,
            "mapping": mapping,
            "template_root": str(JPEG_REFERENCE_ROOT),
            "protected_tree_before": protected_tree_stat_fingerprint(PROTECTED_ROOT),
            "protected_content_before": essential_content_fingerprint(PROTECTED_ROOT),
            "full_source_tree_before": source_tree_fingerprint(FULL_EXR_ROOT),
            "jpeg_reference_tree_before": source_tree_fingerprint(JPEG_REFERENCE_ROOT),
        }
        print(f"initialize staging={staging_root}", flush=True)
        clone_template(JPEG_REFERENCE_ROOT, staging_root)
        atomic_json(staging_root / STATE_DIR / "campaign.json", campaign)
    else:
        campaign_path = staging_root / STATE_DIR / "campaign.json"
        if not campaign_path.is_file():
            raise RuntimeError(f"Existing staging is not resumable: {staging_root}")
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if campaign["script_sha256"] != sha256_file(Path(__file__)):
            raise RuntimeError("Staging was created by another script revision")

    for frame in selected:
        state_path = frame_state_path(staging_root, frame)
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if verify_frame_state(staging_root, state, expected_names):
                print(f"frame={frame:06d} already verified; skip", flush=True)
                continue
            raise RuntimeError(f"Invalid existing frame state: {frame:06d}")
        tasks = []
        for target_name in sorted(mapping.values()):
            source = FULL_EXR_ROOT / f"{frame:06d}" / "images" / target_name
            output = staging_root / f"{frame:06d}" / "images" / target_name
            tasks.append((str(source), str(output)))
        started = time.monotonic()
        outputs = []
        print(f"frame={frame:06d} start cameras={len(tasks)} workers={args.workers}", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(convert_one, task) for task in tasks]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                row["source_relative"] = str(Path(row["source"]).relative_to(FULL_EXR_ROOT))
                row["output_relative"] = str(Path(row["output"]).relative_to(staging_root))
                outputs.append(row)
                print(
                    f"frame={frame:06d} done={completed:02d}/{len(tasks)} "
                    f"image={Path(row['output']).name} seconds={row['seconds']:.1f}",
                    flush=True,
                )
        outputs.sort(key=lambda row: row["output_relative"])
        state = {
            "schema_version": SCHEMA_VERSION,
            "frame": frame,
            "frame_name": f"{frame:06d}",
            "created_at": utc_now(),
            "seconds": time.monotonic() - started,
            "outputs": outputs,
        }
        atomic_json(state_path, state)
        if not verify_frame_state(staging_root, state, expected_names):
            raise RuntimeError(f"Fresh state failed verification: {frame:06d}")
        print(f"frame={frame:06d} complete seconds={state['seconds']:.1f}", flush=True)

    completed = [frame for frame in frames if frame_state_path(staging_root, frame).is_file()]
    print(f"campaign completed_frames={len(completed)}/{EXPECTED_FRAME_COUNT}", flush=True)
    if len(completed) == EXPECTED_FRAME_COUNT:
        manifest = finalize(staging_root, campaign)
        print(f"finalized images={manifest['image_count']} manifest={staging_root / FINAL_MANIFEST}", flush=True)
    return 0


def verify_complete(root: Path, full_hash: bool) -> dict[str, Any]:
    manifest = json.loads((root / FINAL_MANIFEST).read_text(encoding="utf-8"))
    expected_names = set(manifest["camera_mapping"].values())
    errors = []
    checked = 0
    total_bytes = 0
    for state in manifest["frames"]:
        frame_name = state["frame_name"]
        image_dir = root / frame_name / "images"
        if {path.name for path in image_dir.glob("frame_*.exr")} != expected_names:
            errors.append(f"inventory:{frame_name}")
        if list(image_dir.glob("*.jpg")):
            errors.append(f"jpeg-present:{frame_name}")
        reference_transform = json.loads(
            (JPEG_REFERENCE_ROOT / frame_name / "transforms.json").read_text(encoding="utf-8")
        )
        actual_transform = json.loads((root / frame_name / "transforms.json").read_text(encoding="utf-8"))
        for row in reference_transform["frames"]:
            row["file_path"] = str(Path(row["file_path"]).with_suffix(".exr"))
        if reference_transform != actual_transform:
            errors.append(f"transform:{frame_name}")
        for row in state["outputs"]:
            path = root / row["output_relative"]
            if not path.is_file() or path.stat().st_size != row["output_bytes"]:
                errors.append(f"missing-or-size:{row['output_relative']}")
                continue
            profile = output_header_only_profile(path)
            expected_profile = {
                "width": HD_SIZE[0],
                "height": HD_SIZE[1],
                "channels": ["B", "G", "R"],
                "pixel_types": {"B": "HALF", "G": "HALF", "R": "HALF"},
                "compression": "ZIPS_COMPRESSION",
            }
            if profile != expected_profile:
                errors.append(f"profile:{row['output_relative']}:{profile}")
            if full_hash and sha256_file(path) != row["output_sha256"]:
                errors.append(f"hash:{row['output_relative']}")
            total_bytes += path.stat().st_size
            checked += 1
    if errors:
        raise RuntimeError(f"Verification errors ({len(errors)}): {errors[:20]}")
    return {"checked": checked, "bytes": total_bytes, "full_hash": full_hash, "errors": []}


def verify_read_only_inputs(manifest: dict[str, Any]) -> None:
    if protected_tree_stat_fingerprint(PROTECTED_ROOT) != manifest["protected_tree_before"]:
        raise RuntimeError("Protected dataset changed")
    if essential_content_fingerprint(PROTECTED_ROOT) != manifest["protected_content_before"]:
        raise RuntimeError("Protected dataset content changed")
    if source_tree_fingerprint(FULL_EXR_ROOT) != manifest["full_source_tree_before"]:
        raise RuntimeError("Full EXR source dataset changed")
    if source_tree_fingerprint(JPEG_REFERENCE_ROOT) != manifest["jpeg_reference_tree_before"]:
        raise RuntimeError("JPEG reference dataset changed")


def verify(args: argparse.Namespace) -> int:
    root = require_mnt_child(args.root, "--root")
    ensure_safe_paths(root)
    manifest = json.loads((root / FINAL_MANIFEST).read_text(encoding="utf-8"))
    result = verify_complete(root, full_hash=not args.skip_full_hash)
    verify_read_only_inputs(manifest)
    print(json.dumps({"dataset": result, "read_only_inputs_unchanged": True}, indent=2), flush=True)
    return 0


def install(args: argparse.Namespace) -> int:
    staging_root = require_mnt_child(args.staging_root, "--staging-root")
    target_root = require_mnt_child(args.target_root, "--target-root")
    ensure_safe_paths(staging_root, target_root)
    if not args.execute:
        raise RuntimeError("Pass --execute after reviewing target paths")
    if target_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing target: {target_root}")
    manifest = json.loads((staging_root / FINAL_MANIFEST).read_text(encoding="utf-8"))
    full_result = verify_complete(staging_root, full_hash=True)
    verify_read_only_inputs(manifest)
    os.rename(staging_root, target_root)
    quick_result = verify_complete(target_root, full_hash=False)
    atomic_json(
        target_root / INSTALL_MANIFEST,
        {
            "schema_version": 1,
            "installed_at": utc_now(),
            "target_root": str(target_root),
            "preinstall_full_verification": full_result,
            "postinstall_quick_verification": quick_result,
            "read_only_inputs_unchanged": True,
        },
    )
    print(f"installed images={full_result['checked']} target={target_root}", flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    stage_parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    stage_parser.add_argument("--frame", type=int, action="append")
    stage_parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    stage_parser.add_argument("--minimum-free-gib", type=int, default=40)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, default=DEFAULT_STAGING_ROOT)
    verify_parser.add_argument("--skip-full-hash", action="store_true")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    install_parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
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
