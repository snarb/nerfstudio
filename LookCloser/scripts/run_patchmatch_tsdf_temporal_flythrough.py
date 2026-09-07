#!/usr/bin/env python3
"""Build a 150-frame DEC5 temporal PatchMatch-TSDF fly-through.

Each output video frame uses one chronological source instant, one TSDF mesh,
and the corresponding pose on a closed calibration-only camera path.  The
controller can be run concurrently on clever-shadow and dev3 with disjoint
atomic frame claims.  Target RGB is never read.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Sequence

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np
from PIL import Image, ImageDraw

from colmap_patchmatch_tsdf_campaign_common import (
    append_jsonl,
    atomic_json,
    canonical_sha256,
    discover_frames,
    sha256,
    stage_fixed_calibration_dataset,
    validate_hash_manifest,
    validate_source_frame,
)
from render_patchmatch_camera_path import calibration_path_intervals


FRAME_COUNT = 150
FPS = 30
SOURCE_ROOT = Path("/mnt/data/dec5_5a3_nerfstudio_exr_1920x1080")
OUTPUT_ROOT = Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150")
EXISTING_ROOT = Path("/mnt/data/lookcloser_dec5_5a3_patchmatch_tsdf_50")
CALIBRATION = EXISTING_ROOT / "config/calibration/transforms.json"
RECONSTRUCTION_CODE = EXISTING_ROOT / "config/code"
LOCAL_PYTHON = Path("/home/brans/repos/nerfstudio/.venv/bin/python")
LOCAL_COLMAP = Path("/home/brans/lookcloser_temp/colmap_5509fffe_dev3_bundle/colmap_pinned")
REMOTE_PYTHON = Path("/home/ubuntu/anaconda3/envs/nerfstudio/bin/python")
REMOTE_COLMAP = Path("/usr/local/bin/colmap")
REMOTE_ROOT = Path("/fsx/oregon/lookcloser_dec5_5a3_patchmatch_tsdf_flythrough_150_scratch")

PATH_ANCHORS = (
    "D004_A005_12103C", "D004_B005_1210N4", "D004_C005_1210ES",
    "D004_D005_1210LZ", "D004_E005_1210GX", "E004_E005_1210WX",
    "F004_E005_1210FP", "G004_E005_1211KO", "H004_E005_1210YK",
    "I004_E005_1210FW", "J004_E005_1210N7", "K004_E005_1210UF",
    "L004_E005_1210RM", "L004_D005_1210T4", "L004_C005_1210QQ",
    "L004_B005_12106A", "L004_A005_1210YO", "K004_A005_1210EF",
    "J004_A005_121014", "I004_A005_121062", "H004_A005_1210M6",
    "G004_A005_121071", "F004_A005_12103K", "E004_A005_1210QB",
    "D004_A005_12103C",
)
PATH_INTERVALS = (8, 8, 8, 9, 6, 5, 5, 6, 4, 5, 5, 6, 8, 8, 9, 7, 5, 5, 5, 5, 5, 5, 5, 7)
RENDER_CODE_NAMES = (
    "build_angular_camera_subset.py",
    "colmap_patchmatch_tsdf_campaign_common.py",
    "hard_texture_seam_cut.py",
    "mesh_texture_visibility.py",
    "patchmatch_color_calibration.py",
    "render_mesh_image_blend.py",
    "render_patchmatch_camera_path.py",
    "render_tsdf_mesh_depth.py",
)
GEOMETRY_WORKER_NAME = "run_colmap_patchmatch_tsdf_geometry_worker.py"
RENDER_RECIPE = {
    "neighbors": 8,
    "aggregation_mode": "seam-cut",
    "seam_cut_rank_penalty": 0.001,
    "primary_angular_camera_count": 16,
    "pixel_center_offset": 0.5,
    "exact_mesh_visibility": True,
    "depth_log_tolerance": 0.01,
    "depth_hole_fill_max_area": 1000,
    "target_depth_component_min_area": 1000,
    "target_depth_component_max_log_jump": 0.0075,
    "source_rgb_averaging": False,
    "target_rgb_read": False,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def script_sha_rows(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def verify_sha_rows(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    expected_names: set[str] = set()
    for row in rows:
        name = str(row["name"])
        if name in expected_names or Path(name).name != name:
            raise ValueError(f"Invalid or duplicate frozen code name: {name}")
        expected_names.add(name)
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"Frozen campaign code changed: {path}")
    actual_names = {path.name for path in root.glob("*.py")}
    if actual_names != expected_names:
        raise ValueError(f"Frozen code inventory changed under {root}")


def verify_conversion(
    source: Path, jpeg: Path, staged: Path, expected_source: dict[str, Any],
    calibration_sha256: str,
) -> None:
    conversion = load_json(jpeg / "conversion_manifest.json")
    tone_map = conversion.get("tone_map", {})
    if (
        conversion.get("source") != str(source.resolve())
        or conversion.get("image_count") != 65
        or tone_map.get("curve") != "global_exposure_then_reinhard_then_srgb"
        or tone_map.get("exposure_mode") != "per-image"
        or not math.isclose(float(tone_map.get("middle_gray", -1)), 0.18, abs_tol=1e-12)
        or int(tone_map.get("jpeg_quality", -1)) != 98
        or tone_map.get("jpeg_subsampling") != "4:4:4"
    ):
        raise ValueError(f"Unexpected JPEG ingest receipt for {source.name}")
    expected_images = {row["file_path"]: row for row in expected_source["source_images"]}
    rows = conversion.get("images", [])
    if len(rows) != 65 or len(expected_images) != 65:
        raise ValueError(f"Incomplete image inventory for {source.name}")
    for row in rows:
        relative = str(row["frame_file_path"])
        expected = expected_images.get(relative)
        output = Path(str(row["output"]))
        if (
            expected is None
            or row.get("source_sha256") != expected["sha256"]
            or not output.is_file()
            or output.stat().st_size != int(row.get("bytes", output.stat().st_size))
            or sha256(output) != row.get("sha256")
        ):
            raise ValueError(f"Changed EXR/JPEG conversion row: {source.name}/{relative}")
    staging = load_json(staged / "staging_manifest.json")
    if (
        staging.get("source_transforms_sha256") != expected_source["source_transforms_sha256"]
        or staging.get("calibration_template_sha256") != calibration_sha256
        or len(staging.get("conversion_rows", [])) != 63
    ):
        raise ValueError(f"Invalid staged calibration receipt for {source.name}")


def source_image_rows(frame_root: Path) -> list[dict[str, Any]]:
    payload = load_json(frame_root / "transforms.json")
    rows: list[dict[str, Any]] = []
    for frame in payload["frames"]:
        relative = str(frame["file_path"])
        path = (frame_root / relative).resolve()
        rows.append({
            "file_path": relative,
            "physical_camera": frame["physical_camera"],
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def grid_position(name: str) -> tuple[int, int]:
    parts = name.split("_")
    if len(parts) < 2 or len(parts[0]) != 4 or len(parts[1]) != 4:
        raise ValueError(f"Cannot parse DEC5 grid camera: {name}")
    return ord(parts[0][0]) - ord("A"), ord(parts[1][0]) - ord("A")


def validate_outer_path(calibration: dict[str, Any], path: list[dict[str, Any]]) -> None:
    if len(path) != FRAME_COUNT or sum(PATH_INTERVALS) + 1 != FRAME_COUNT:
        raise ValueError("The outer path must contain exactly 150 poses")
    by_camera = {row["physical_camera"]: row for row in calibration["frames"]}
    if len(by_camera) != len(calibration["frames"]) or any(name not in by_camera for name in PATH_ANCHORS):
        raise ValueError("Outer path anchors are not a unique subset of calibration cameras")
    center = grid_position("H004_C005_center")
    distances = [max(abs(x - center[0]), abs(y - center[1])) for x, y in map(grid_position, PATH_ANCHORS)]
    if min(distances) < 2:
        raise ValueError("Outer path approaches within two camera-grid steps of the center")
    for segment, intervals in enumerate(PATH_INTERVALS):
        start = np.asarray(grid_position(PATH_ANCHORS[segment]), dtype=np.float64)
        end = np.asarray(grid_position(PATH_ANCHORS[segment + 1]), dtype=np.float64)
        for step in range(intervals + 1):
            point = start + (end - start) * (step / intervals)
            if np.max(np.abs(point - np.asarray(center))) < 2.0 - 1e-12:
                raise ValueError("An interpolated pose approaches within two grid rows of the center")
    first = np.asarray(path[0]["transform_matrix"], dtype=np.float64)
    last = np.asarray(path[-1]["transform_matrix"], dtype=np.float64)
    if not np.allclose(first, last, rtol=0.0, atol=1e-12):
        raise ValueError("Camera path is not exactly closed")
    cumulative = np.cumsum((0, *PATH_INTERVALS)).tolist()
    for index, anchor in zip(cumulative, PATH_ANCHORS):
        expected = np.asarray(by_camera[anchor]["transform_matrix"], dtype=np.float64)
        if not np.allclose(np.asarray(path[index]["transform_matrix"]), expected, atol=1e-12):
            raise ValueError(f"Path misses calibrated anchor {anchor} at index {index}")


def build_request(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = discover_frames(args.source_root, count=FRAME_COUNT)
    source_rows = []
    for index, frame_root in enumerate(frames, 1):
        row = validate_source_frame(frame_root)
        row["source_images"] = source_image_rows(frame_root)
        source_rows.append(row)
        print(f"request_source_hashes={index}/{FRAME_COUNT} frame={frame_root.name}", flush=True)
    calibration = load_json(args.calibration)
    path = calibration_path_intervals(calibration, list(PATH_ANCHORS), list(PATH_INTERVALS))
    validate_outer_path(calibration, path)
    old_request = load_json(args.existing_root / "campaign_request.json")
    old_ids = list(old_request.get("ordered_frame_ids", []))
    new_ids = [row.name for row in frames]
    if old_ids != new_ids[:50]:
        raise ValueError("The reusable 50-frame campaign is not the exact prefix of this campaign")
    request: dict[str, Any] = {
        "schema_version": 1,
        "campaign": "dec5_5a3_patchmatch_tsdf_temporal_outer_flythrough_150",
        "created_at": now(),
        "source_root": str(args.source_root),
        "ordered_frame_ids": new_ids,
        "source_frames": source_rows,
        "calibration": str(args.calibration),
        "calibration_sha256": sha256(args.calibration),
        "existing_geometry_root": str(args.existing_root),
        "existing_geometry_request_sha256": old_request["request_sha256"],
        "existing_geometry_frame_count": 50,
        "new_geometry_frame_count": 100,
        "geometry_recipe": old_request["recipe"],
        "geometry_policy": "reuse_first50_hash_verified_then_same_frozen_recipe",
        "geometry_worker": "geometry_only_same_commands_through_fuse_tsdf_no_discarded_eval_render",
        "surface_refinement": {
            "enabled": False,
            "reason": "native-plane canary is explicitly marked accepted_surface_recipe=false and failed 3/3 views",
        },
        "path": {
            "anchors": list(PATH_ANCHORS),
            "intervals": list(PATH_INTERVALS),
            "frame_count": len(path),
            "path_sha256": canonical_sha256(path),
            "interpolation": "linear_translation_intrinsics_plus_rotation_slerp",
            "closed": True,
            "inside_calibrated_camera_hull": True,
            "inside_hull_proof": "each emitted translation is a convex combination of two calibrated anchors",
            "minimum_grid_radius_from_H_C": 2,
        },
        "render_recipe": {
            **RENDER_RECIPE,
            "texture_source_semantics": (
                "rank0_from_calibration_only_angular16_then_up_to_7_nearest_fallbacks_from_all_62_train; "
                "hard seam labels; no RGB averaging"
            ),
        },
        "video": {"fps": FPS, "resolution": [1920, 1080], "frames": FRAME_COUNT},
        "hosts": {
            "local": {"python": str(args.local_python), "colmap": str(args.local_colmap), "gpu_index": "0"},
            "dev3": {
                "ssh": args.remote_host, "python": str(args.remote_python),
                "colmap": str(args.remote_colmap), "scratch": str(args.remote_root), "gpu_index": "0",
            },
        },
        "resource_policy": {
            "one_gpu_stage_per_host": True,
            "local_scratch": "/dev/shm/lookcloser_patchmatch_tsdf_flythrough_150",
            "local_scratch_min_free_gib": args.local_scratch_min_free_gib,
        },
    }
    request["request_sha256"] = canonical_sha256(request)
    return request, path


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    request_path = args.output_root / "campaign_request.json"
    if request_path.is_file():
        request = require_request(args)
        print(f"status=reused request={request['request_sha256']}")
        return request
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in ("config", "frames", ".work", "claims", "quarantine", "contact_sheets"):
        incomplete = args.output_root / name
        if incomplete.exists():
            quarantine = args.output_root / f".{name.lstrip('.')}.incomplete-{int(time.time())}"
            os.replace(incomplete, quarantine)
    request, path = build_request(args)
    config_stage = args.output_root / f".config.tmp-{os.getpid()}"
    if config_stage.exists():
        raise FileExistsError(config_stage)
    (config_stage / "calibration").mkdir(parents=True)
    (config_stage / "reconstruct_code").mkdir()
    (config_stage / "render_code").mkdir()
    shutil.copyfile(args.calibration, config_stage / "calibration/transforms.json")
    for source in sorted(args.reconstruction_code.glob("*.py")):
        shutil.copyfile(source, config_stage / "reconstruct_code" / source.name)
    scripts = Path(__file__).resolve().parent
    shutil.copyfile(scripts / GEOMETRY_WORKER_NAME, config_stage / "reconstruct_code" / GEOMETRY_WORKER_NAME)
    for name in RENDER_CODE_NAMES:
        shutil.copyfile(scripts / name, config_stage / "render_code" / name)
    request["code"] = {
        "reconstruction": script_sha_rows(config_stage / "reconstruct_code", [p.name for p in sorted((config_stage / "reconstruct_code").glob("*.py"))]),
        "render": script_sha_rows(config_stage / "render_code", RENDER_CODE_NAMES),
        "controller_sha256": sha256(Path(__file__)),
    }
    request.pop("request_sha256")
    request["request_sha256"] = canonical_sha256(request)
    atomic_json(config_stage / "camera_path.json", {"schema_version": 1, "frames": path})
    atomic_json(config_stage / "config_manifest.json", {
        "schema_version": 1, "request_sha256": request["request_sha256"],
        "calibration_sha256": sha256(config_stage / "calibration/transforms.json"),
        "camera_path_sha256": sha256(config_stage / "camera_path.json"),
    })
    os.replace(config_stage, args.output_root / "config")
    for name in ("frames", ".work", "claims", "quarantine", "contact_sheets"):
        (args.output_root / name).mkdir(exist_ok=True)
    atomic_json(request_path, request)
    atomic_json(args.output_root / "campaign_manifest.json", {
        "schema_version": 1, "request_sha256": request["request_sha256"],
        "status": "initialized", "completed": 0, "failed": 0, "updated_at": now(),
    })
    print(f"status=initialized request={request['request_sha256']} frames={FRAME_COUNT}")
    return request


def require_request(args: argparse.Namespace) -> dict[str, Any]:
    request = load_json(args.output_root / "campaign_request.json")
    request_hash = request.get("request_sha256")
    unhashed = dict(request)
    unhashed.pop("request_sha256", None)
    if request_hash != canonical_sha256(unhashed):
        raise ValueError("Campaign request self-hash mismatch")
    runtime = {
        "source_root": str(args.source_root),
        "existing_geometry_root": str(args.existing_root),
        "calibration": str(args.calibration),
    }
    for key, value in runtime.items():
        if request.get(key) != value:
            raise ValueError(f"Runtime {key} disagrees with the immutable request")
    hosts = request.get("hosts", {})
    expected_host_args = {
        "local": {"python": str(args.local_python), "colmap": str(args.local_colmap)},
        "dev3": {
            "ssh": args.remote_host, "python": str(args.remote_python),
            "colmap": str(args.remote_colmap), "scratch": str(args.remote_root),
        },
    }
    for host, expected in expected_host_args.items():
        if any(hosts.get(host, {}).get(key) != value for key, value in expected.items()):
            raise ValueError(f"Runtime {host} configuration disagrees with the immutable request")
    if not math.isclose(
        float(request.get("resource_policy", {}).get("local_scratch_min_free_gib", -1)),
        args.local_scratch_min_free_gib,
        abs_tol=1e-12,
    ):
        raise ValueError("Runtime local scratch policy disagrees with the immutable request")
    config = args.output_root / "config"
    calibration = config / "calibration/transforms.json"
    if sha256(calibration) != request["calibration_sha256"] or sha256(args.calibration) != request["calibration_sha256"]:
        raise ValueError("Campaign calibration changed")
    verify_sha_rows(config / "reconstruct_code", request["code"]["reconstruction"])
    verify_sha_rows(config / "render_code", request["code"]["render"])
    if sha256(Path(__file__)) != request["code"]["controller_sha256"]:
        raise ValueError("Campaign controller changed after initialization")
    config_manifest = load_json(config / "config_manifest.json")
    camera_path_file = config / "camera_path.json"
    camera_path = load_json(camera_path_file).get("frames")
    if (
        config_manifest.get("request_sha256") != request_hash
        or config_manifest.get("calibration_sha256") != sha256(calibration)
        or config_manifest.get("camera_path_sha256") != sha256(camera_path_file)
        or not isinstance(camera_path, list)
        or canonical_sha256(camera_path) != request["path"]["path_sha256"]
    ):
        raise ValueError("Frozen campaign config/path manifest changed")
    validate_outer_path(load_json(calibration), camera_path)
    discovered = [path.name for path in discover_frames(args.source_root, count=FRAME_COUNT)]
    if discovered != request.get("ordered_frame_ids"):
        raise ValueError("Source frame inventory/order changed")
    old_request = load_json(args.existing_root / "campaign_request.json")
    if old_request.get("request_sha256") != request.get("existing_geometry_request_sha256"):
        raise ValueError("Reusable campaign request changed")
    return request


def run_logged(
    command: list[str], log: Path, *, env: dict[str, str],
    check: Callable[[subprocess.Popen[Any]], None] | None = None,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, text=True, env=env,
            start_new_session=True,
        )

        def safe_check() -> None:
            if check is None:
                return
            try:
                check(process)
            except Exception as error:  # Monitoring must never orphan or duplicate a worker.
                stream.write(f"monitor_warning={error!r}\n")
                stream.flush()

        try:
            next_check = time.monotonic()
            while process.poll() is None:
                if time.monotonic() >= next_check:
                    safe_check()
                    next_check = time.monotonic() + 300
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            safe_check()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, 15)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    process.wait()
            raise


def rsync(source: str, destination: str, *, delete: bool = False) -> None:
    # The campaign output is on FUSE/NTFS, where ownership and timestamp
    # preservation are rejected even though byte writes are valid.
    command = ["rsync", "-rl"]
    if delete:
        command.append("--delete")
    subprocess.run([*command, source, destination], check=True)


def ssh(host: str, command: Sequence[object], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", host, shlex.join([str(value) for value in command])],
        check=True, text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def record_check(args: argparse.Namespace, frame_id: str, host: str, process: subprocess.Popen[Any]) -> None:
    if host == "local":
        gpu = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.strip()
        disk = {
            "output_free_bytes": shutil.disk_usage(args.output_root).free,
            "local_scratch_free_bytes": shutil.disk_usage("/dev/shm").free,
        }
    else:
        gpu = ssh(args.remote_host, ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture=True).stdout.strip()
        disk = ssh(args.remote_host, ["df", "-Pk", args.remote_root], capture=True).stdout.strip()
    append_jsonl(args.output_root / "campaign_checks.jsonl", {
        "timestamp": now(), "frame_id": frame_id, "host": host,
        "local_child_pid": process.pid, "worker_alive": process.poll() is None,
        "gpu_processes": gpu, "disk": disk,
    })


def stage_frame(args: argparse.Namespace, request: dict[str, Any], frame_id: str, work: Path) -> Path:
    source = args.source_root / frame_id
    jpeg = work / "jpeg65"
    staged = work / "staged63"
    recon_code = args.output_root / "config/reconstruct_code"
    env = dict(os.environ, PYTHONPATH=f"/home/brans/repos/nerfstudio:{recon_code}", OPENCV_IO_ENABLE_OPENEXR="1")
    conversion_was_present = (jpeg / "conversion_manifest.json").is_file()
    if not conversion_was_present:
        run_logged([
            str(args.local_python), str(recon_code / "convert_exr_nerfstudio_to_jpeg.py"),
            "--input", str(source), "--output", str(jpeg), "--middle-gray", "0.18",
            "--exposure-mode", "per-image", "--quality", "98",
        ], work / "convert.log", env=env)
    if not (staged / "staging_manifest.json").is_file():
        stage_fixed_calibration_dataset(
            source, jpeg, args.output_root / "config/calibration/transforms.json", staged
        )
    payload = load_json(staged / "transforms.json")
    if len(payload.get("train_filenames", [])) != 62 or len(payload.get("val_filenames", [])) != 1:
        raise ValueError(f"Staged frame {frame_id} is not 62/1")
    expected = next(row for row in request["source_frames"] if Path(row["source_dataset"]).name == frame_id)
    if sha256(source / "transforms.json") != expected["source_transforms_sha256"]:
        raise ValueError(f"Source transforms changed for {frame_id}")
    verify_conversion(source, jpeg, staged, expected, request["calibration_sha256"])
    if conversion_was_present:
        for row in expected["source_images"]:
            source_image = (source / row["file_path"]).resolve()
            if source_image.stat().st_size != int(row["bytes"]) or sha256(source_image) != row["sha256"]:
                raise ValueError(f"Source EXR changed before resume: {source_image}")
    return staged


def render_command(
    python: Path, render_code: Path, data: Path, mesh: Path, metadata: Path,
    calibration: Path, output: Path, index: int,
) -> list[str]:
    return [
        str(python), str(render_code / "render_patchmatch_camera_path.py"),
        "--data", str(data), "--mesh", str(mesh), "--mesh-metadata", str(metadata),
        "--calibration", str(calibration), "--output", str(output),
        "--anchors", *PATH_ANCHORS, "--segment-intervals", *map(str, PATH_INTERVALS),
        "--target-indices", str(index), "--allow-source-anchors",
        "--neighbors", "8", "--aggregation-mode", "seam-cut",
        "--seam-cut-rank-penalty", "0.001", "--primary-angular-camera-count", "16",
        "--pixel-center-offset", "0.5", "--exact-mesh-visibility",
        "--depth-log-tolerance", "0.01", "--depth-hole-fill-max-area", "1000",
        "--target-depth-component-min-area", "1000",
        "--target-depth-component-max-log-jump", "0.0075",
    ]


def adopted_geometry(args: argparse.Namespace, frame_id: str, work: Path) -> tuple[Path, Path, dict[str, Any]]:
    original = args.existing_root / "frames" / frame_id
    result = load_json(original / "result.json")
    old_request = load_json(args.existing_root / "campaign_request.json")
    if (
        result.get("frame_id") != frame_id
        or result.get("request_sha256") != old_request.get("request_sha256")
        or frame_id not in old_request.get("ordered_frame_ids", [])
    ):
        raise ValueError(f"Existing frame provenance mismatch: {frame_id}")
    validate_hash_manifest(original, load_json(original / "retained_manifest.json"))
    mesh = original / "mesh/colmap_patchmatch_tsdf.ply"
    metadata = original / "mesh/colmap_patchmatch_tsdf.json"
    mesh_meta = load_json(metadata)
    if (
        sha256(mesh) != result["mesh_sha256"]
        or mesh_meta.get("output_sha256") != result["mesh_sha256"]
        or int(mesh_meta.get("vertices", 0)) != int(result["mesh_vertices"])
        or int(mesh_meta.get("triangles", 0)) != int(result["mesh_triangles"])
        or int(mesh_meta.get("connected_components", 0)) != int(result["mesh_components"])
        or result["mesh_components"] != 1
    ):
        raise ValueError(f"Existing frame geometry failed verification: {frame_id}")
    destination = work / "geometry"
    destination.mkdir(exist_ok=True)
    shutil.copyfile(mesh, destination / mesh.name)
    shutil.copyfile(metadata, destination / metadata.name)
    visual = load_json(original / "visual_review.json") if (original / "visual_review.json").is_file() else {}
    return destination / mesh.name, destination / metadata.name, {
        "provenance": "adopted_hash_verified_first50",
        "source_campaign_request_sha256": old_request["request_sha256"],
        "source_campaign_status": result.get("status"),
        "source_campaign_visual_status": visual.get("visual_status", result.get("visual_status")),
        "source_campaign_visual_notes": visual.get("visual_notes", result.get("visual_notes")),
        "mesh_metadata_sha256": sha256(metadata),
        "depth_coverage_mean": result["depth_coverage_mean"],
        "depth_coverage_min": result["depth_coverage_min"],
        "mesh_vertices": result["mesh_vertices"], "mesh_triangles": result["mesh_triangles"],
        "mesh_components": result["mesh_components"], "mesh_sha256": result["mesh_sha256"],
    }


def local_geometry(
    args: argparse.Namespace, request: dict[str, Any], frame_id: str, staged: Path,
    work: Path, attempt: int,
) -> tuple[Path, Path, dict[str, Any]]:
    recon_code = args.output_root / "config/reconstruct_code"
    workspace = Path("/dev/shm/lookcloser_patchmatch_tsdf_flythrough_150") / request["request_sha256"] / frame_id / f"attempt_{attempt}"
    free = shutil.disk_usage("/dev/shm").free
    if free < args.local_scratch_min_free_gib * (1 << 30):
        raise RuntimeError(
            f"Local reconstruction scratch has only {free / (1 << 30):.1f} GiB free; "
            f"requires {args.local_scratch_min_free_gib:.1f} GiB"
        )
    probe = subprocess.run(
        [str(args.local_colmap), "-h"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    if not all(marker in probe for marker in ("COLMAP 3.13.0.dev0", "Commit 5509fffe", "with CUDA")):
        raise RuntimeError("Local COLMAP build is not the verified CUDA commit")
    env = dict(
        os.environ,
        PYTHONPATH=f"/home/brans/repos/nerfstudio:{recon_code}", CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="16", OPENBLAS_NUM_THREADS="16", OPENCV_IO_ENABLE_OPENEXR="1",
    )
    run_logged([
        str(args.local_python), str(recon_code / GEOMETRY_WORKER_NAME),
        "--frame-id", frame_id, "--data", str(staged), "--workspace", str(workspace),
        "--colmap-bin", str(args.local_colmap), "--gpu-index", "0",
    ], work / "reconstruct.log", env=env, check=lambda process: record_check(args, frame_id, "local", process))
    retained = workspace / "retained"
    validate_hash_manifest(retained, load_json(retained / "retained_manifest.json"))
    result = load_json(retained / "remote_result.json")
    return (
        retained / "mesh/colmap_patchmatch_tsdf.ply",
        retained / "mesh/colmap_patchmatch_tsdf.json",
        {"provenance": "new_frozen_recipe_local", **result},
    )


def prepare_remote(args: argparse.Namespace, request: dict[str, Any]) -> tuple[Path, Path, Path]:
    base = args.remote_root / request["request_sha256"]
    recon = base / "reconstruct_code"
    render = base / "render_code"
    deps = base / "python_deps"
    ssh(args.remote_host, ["mkdir", "-p", recon, render, deps, base / "calibration"])
    rsync(str(args.output_root / "config/reconstruct_code") + "/", f"{args.remote_host}:{recon}/", delete=True)
    rsync(str(args.output_root / "config/render_code") + "/", f"{args.remote_host}:{render}/", delete=True)
    rsync(str(args.output_root / "config/calibration/transforms.json"), f"{args.remote_host}:{base / 'calibration/transforms.json'}")
    probe = ssh(args.remote_host, [args.remote_colmap, "-h"], capture=True).stdout
    if not all(marker in probe for marker in ("COLMAP 3.13.0.dev0", "Commit 5509fffe", "with CUDA")):
        raise RuntimeError("dev3 COLMAP build is not the verified CUDA commit")
    maxflow_env = f"{deps}:/home/ubuntu/repos/nerfstudio"
    check = subprocess.run(
        ["ssh", args.remote_host, shlex.join(["env", f"PYTHONPATH={maxflow_env}", str(args.remote_python), "-c", "import maxflow"])],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check.returncode:
        ssh(args.remote_host, [args.remote_python, "-m", "pip", "install", "--no-deps", "--target", deps, "PyMaxflow==1.3.2"])
    version = ssh(
        args.remote_host,
        ["env", f"PYTHONPATH={maxflow_env}", args.remote_python, "-c", "import importlib.metadata,maxflow;print(importlib.metadata.version('PyMaxflow'))"],
        capture=True,
    ).stdout.strip()
    if version != "1.3.2":
        raise RuntimeError(f"Unexpected remote PyMaxflow version: {version}")
    append_jsonl(args.output_root / "campaign_checks.jsonl", {
        "timestamp": now(), "host": "dev3", "check_status": "preflight_pass",
        "colmap": "COLMAP 3.13.0.dev0 commit 5509fffe with CUDA", "PyMaxflow": version,
    })
    return base, recon, render


def remote_geometry_and_render(
    args: argparse.Namespace, request: dict[str, Any], frame_id: str, index: int,
    staged: Path, work: Path, remote_layout: tuple[Path, Path, Path], attempt: int,
) -> tuple[Path, Path, dict[str, Any], Path]:
    base, recon_code, render_code = remote_layout
    remote_frame = base / "frames" / frame_id / f"attempt_{attempt}"
    ssh(args.remote_host, [
        args.remote_python, "-c",
        (
            "import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
            "p.exists() and p.rename(p.with_name(p.name+'.stale-'+str(int(time.time())))); "
            "p.mkdir(parents=True)"
        ),
        remote_frame,
    ])
    rsync(str(staged) + "/", f"{args.remote_host}:{remote_frame / 'data'}/", delete=True)
    remote_log = work / "remote.log"
    pythonpath = f"/home/ubuntu/repos/nerfstudio:{recon_code}"
    command = [
        "ssh", args.remote_host,
        shlex.join([str(value) for value in [
            "env", f"PYTHONPATH={pythonpath}", "CUDA_VISIBLE_DEVICES=0", "OMP_NUM_THREADS=4",
            "OPENBLAS_NUM_THREADS=4", "OPENCV_IO_ENABLE_OPENEXR=1", args.remote_python,
            recon_code / GEOMETRY_WORKER_NAME, "--frame-id", frame_id,
            "--data", remote_frame / "data", "--workspace", remote_frame / "reconstruction",
            "--colmap-bin", args.remote_colmap, "--gpu-index", "0",
        ]]),
    ]
    run_logged(command, remote_log, env=dict(os.environ), check=lambda process: record_check(args, frame_id, "dev3", process))
    render_output = remote_frame / "path_render"
    render_env = f"/home/ubuntu/repos/nerfstudio:{render_code}:{base / 'python_deps'}"
    remote_render = render_command(
        args.remote_python, render_code, remote_frame / "data",
        remote_frame / "reconstruction/retained/mesh/colmap_patchmatch_tsdf.ply",
        remote_frame / "reconstruction/retained/mesh/colmap_patchmatch_tsdf.json",
        base / "calibration/transforms.json", render_output, index,
    )
    run_logged(
        ["ssh", args.remote_host, shlex.join([str(value) for value in ["env", f"PYTHONPATH={render_env}", "CUDA_VISIBLE_DEVICES=0", *remote_render]])],
        remote_log, env=dict(os.environ), check=lambda process: record_check(args, frame_id, "dev3-render", process),
    )
    package = remote_frame / "fly_retained"
    ssh(args.remote_host, [
        args.remote_python, "-c",
        (
            "import hashlib,json,os,pathlib,shutil,sys; s=pathlib.Path(sys.argv[1]); p=pathlib.Path(sys.argv[2]); i=sys.argv[3]; "
            "(p/'mesh').mkdir(parents=True,exist_ok=True); (p/'render').mkdir(exist_ok=True); "
            "shutil.copy2(s/'reconstruction/retained/mesh/colmap_patchmatch_tsdf.ply',p/'mesh/colmap_patchmatch_tsdf.ply'); "
            "shutil.copy2(s/'reconstruction/retained/mesh/colmap_patchmatch_tsdf.json',p/'mesh/colmap_patchmatch_tsdf.json'); "
            "shutil.copy2(s/'reconstruction/retained/remote_result.json',p/'geometry_result.json'); "
            "shutil.copy2(s/f'path_render/view_{i}.png',p/'render/frame.png'); "
            "shutil.copy2(s/'path_render/path_request.json',p/'render/path_request.json'); "
            "r=s/f'path_render/scratch/{i}/render'; "
            "shutil.copy2(r/'seam_cut8/eval_pred_0000.exr',p/'render/frame.exr'); "
            "shutil.copy2(r/'seam_cut8/source_selection.png',p/'render/source_selection.png'); "
            "shutil.copy2(r/'reprojection_audit.json',p/'render/reprojection_audit.json'); "
            "h=lambda q:hashlib.sha256(q.read_bytes()).hexdigest(); "
            "rows=[{'path':str(q.relative_to(p)),'bytes':q.stat().st_size,'sha256':h(q)} for q in sorted(p.rglob('*')) if q.is_file()]; "
            "tmp=p/'.retained_manifest.json.tmp'; tmp.write_text(json.dumps({'schema_version':1,'files':rows},sort_keys=True,indent=2)+'\\n'); "
            "os.replace(tmp,p/'retained_manifest.json')"
        ),
        remote_frame, package, f"{index:04d}",
    ])
    incoming = work / "remote_fly_retained"
    incoming.mkdir(exist_ok=True)
    rsync(f"{args.remote_host}:{package}/", str(incoming) + "/")
    validate_hash_manifest(incoming, load_json(incoming / "retained_manifest.json"))
    result = load_json(incoming / "geometry_result.json")
    mesh_meta = load_json(incoming / "mesh/colmap_patchmatch_tsdf.json")
    if (
        sha256(incoming / "mesh/colmap_patchmatch_tsdf.ply") != result.get("mesh_sha256")
        or mesh_meta.get("output_sha256") != result.get("mesh_sha256")
    ):
        raise ValueError("Remote geometry package metadata/hash mismatch")
    return (
        incoming / "mesh/colmap_patchmatch_tsdf.ply",
        incoming / "mesh/colmap_patchmatch_tsdf.json",
        {"provenance": "new_frozen_recipe_dev3", **result},
        incoming / "render",
    )


def local_render(
    args: argparse.Namespace, frame_id: str, index: int, staged: Path,
    mesh: Path, metadata: Path, work: Path,
) -> Path:
    render_code = args.output_root / "config/render_code"
    output = work / "path_render"
    env = dict(
        os.environ,
        PYTHONPATH=f"/home/brans/repos/nerfstudio:{render_code}:/home/brans/lookcloser_temp/surface_repair_python_deps",
        CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        OPENCV_IO_ENABLE_OPENEXR="1",
    )
    command = render_command(
        args.local_python, render_code, staged, mesh, metadata,
        args.output_root / "config/calibration/transforms.json", output, index,
    )
    run_logged(command, work / "render.log", env=env, check=lambda process: record_check(args, frame_id, "local-render", process))
    retained_render = output / f"scratch/{index:04d}/render"
    shutil.copyfile(output / "path_request.json", retained_render / "path_request.json")
    return retained_render


def validate_render(
    render: Path, mesh: Path, geometry: dict[str, Any], index: int,
    campaign_request: dict[str, Any],
) -> dict[str, Any]:
    png = render / "frame.png" if (render / "frame.png").is_file() else render / "seam_cut8/eval_pred_0000.png"
    exr = render / "frame.exr" if (render / "frame.exr").is_file() else render / "seam_cut8/eval_pred_0000.exr"
    audit_path = render / "reprojection_audit.json"
    request_path = render / "path_request.json"
    image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    linear = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (1080, 1920, 3):
        raise ValueError(f"Invalid 1920x1080 video frame: {png}")
    if linear is None or linear.shape != (1080, 1920, 3) or not np.isfinite(linear).all():
        raise ValueError(f"Invalid finite EXR video frame: {exr}")
    expected_png = np.clip(linear, 0.0, 1.0) * 255.0 + 0.5
    if not np.array_equal(image, expected_png.astype(np.uint8)):
        raise ValueError("PNG is not the exact display conversion of the retained EXR")
    nonblack = float((image.max(axis=-1) > 4).mean())
    audit = load_json(audit_path)
    if audit.get("uses_eval_rgb_for_prediction") is not False or audit.get("eval_rgb_use") != "not_read":
        raise ValueError("Fly-through renderer did not prove target RGB exclusion")
    target_filter = audit.get("target_depth_component_filter", {})
    seam_cut = audit.get("seam_cut", {})
    if (
        audit.get("uses_masks") is not False
        or audit.get("aggregation_modes") != ["seam-cut"]
        or len(audit.get("sources", [])) != RENDER_RECIPE["neighbors"]
        or audit.get("exact_mesh_visibility") is not True
        or not math.isclose(float(audit.get("pixel_center_offset", -1)), RENDER_RECIPE["pixel_center_offset"], abs_tol=1e-12)
        or not math.isclose(float(audit.get("depth_log_tolerance", -1)), RENDER_RECIPE["depth_log_tolerance"], abs_tol=1e-12)
        or int(audit.get("primary_angular_camera_count", -1)) != RENDER_RECIPE["primary_angular_camera_count"]
        or int(audit.get("depth_hole_fill", {}).get("max_area", -1)) != RENDER_RECIPE["depth_hole_fill_max_area"]
        or int(target_filter.get("min_area", -1)) != RENDER_RECIPE["target_depth_component_min_area"]
        or not math.isclose(float(target_filter.get("max_log_jump", -1)), RENDER_RECIPE["target_depth_component_max_log_jump"], abs_tol=1e-12)
        or seam_cut.get("enabled") is not True
        or seam_cut.get("color_averaging") is not False
        or not math.isclose(float(seam_cut.get("rank_penalty", -1)), RENDER_RECIPE["seam_cut_rank_penalty"], abs_tol=1e-12)
    ):
        raise ValueError("Render audit does not match the frozen fly-through recipe")
    path_request = load_json(request_path)
    if (
        path_request.get("target_indices") != [index]
        or path_request.get("complete_path_size") != FRAME_COUNT
        or path_request.get("anchors") != list(PATH_ANCHORS)
        or path_request.get("segment_intervals") != list(PATH_INTERVALS)
        or path_request.get("calibration_sha256") != campaign_request["calibration_sha256"]
        or len(path_request.get("targets", [])) != 1
    ):
        raise ValueError("Rendered target index/path receipt mismatch")
    if geometry.get("mesh_components") != 1 or int(geometry.get("mesh_triangles", 0)) < 1000:
        raise ValueError("Catastrophic mesh topology/size failure")
    if sha256(mesh) != geometry.get("mesh_sha256") or nonblack < 0.03:
        raise ValueError(f"Catastrophic mesh/render validation failed; nonblack={nonblack:.6f}")
    return {
        "png": png, "exr": exr, "audit": audit_path, "path_request": request_path,
        "nonblack_fraction": nonblack,
    }


def publish_frame(
    args: argparse.Namespace, request: dict[str, Any], frame_id: str, index: int,
    host: str, mesh: Path, metadata: Path, geometry: dict[str, Any], render: Path,
) -> None:
    final = args.output_root / "frames" / frame_id
    if final.exists():
        raise FileExistsError(final)
    checked = validate_render(render, mesh, geometry, index, request)
    stage = final.with_name(f".{frame_id}.tmp-{os.getpid()}")
    (stage / "mesh").mkdir(parents=True)
    (stage / "render").mkdir()
    shutil.copyfile(mesh, stage / "mesh/colmap_patchmatch_tsdf.ply")
    shutil.copyfile(metadata, stage / "mesh/colmap_patchmatch_tsdf.json")
    shutil.copyfile(checked["png"], stage / "render/frame.png")
    shutil.copyfile(checked["exr"], stage / "render/frame.exr")
    shutil.copyfile(checked["audit"], stage / "render/reprojection_audit.json")
    shutil.copyfile(checked["path_request"], stage / "render/path_request.json")
    selection = render / "source_selection.png" if (render / "source_selection.png").is_file() else render / "seam_cut8/source_selection.png"
    shutil.copyfile(selection, stage / "render/source_selection.png")
    path_frame = load_json(args.output_root / "config/camera_path.json")["frames"][index]
    result = {
        "schema_version": 1, "request_sha256": request["request_sha256"],
        "frame_id": frame_id, "temporal_index": index, "host": host,
        "source_dataset": str(args.source_root / frame_id), "camera": path_frame,
        "geometry": geometry, "render_recipe": RENDER_RECIPE,
        "render_png": str(final / "render/frame.png"), "render_exr": str(final / "render/frame.exr"),
        "render_sha256": sha256(stage / "render/frame.png"),
        "render_exr_sha256": sha256(stage / "render/frame.exr"),
        "mesh_sha256": sha256(stage / "mesh/colmap_patchmatch_tsdf.ply"),
        "nonblack_fraction": checked["nonblack_fraction"],
        "catastrophic_status": "pass", "visual_status": "pending", "completed_at": now(),
    }
    atomic_json(stage / "result.json", result)
    files = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(stage / "retained_manifest.json", {"schema_version": 1, "files": files})
    validate_hash_manifest(stage, load_json(stage / "retained_manifest.json"))
    os.replace(stage, final)


def validate_finished(path: Path, request_hash: str) -> bool:
    try:
        result = load_json(path / "result.json")
        if result.get("request_sha256") != request_hash or result.get("catastrophic_status") != "pass":
            return False
        validate_hash_manifest(path, load_json(path / "retained_manifest.json"))
        return True
    except Exception:
        return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def gpu_lock(args: argparse.Namespace, host: str, frame_id: str) -> Iterator[None]:
    lock_root = args.output_root / ".locks"
    lock_root.mkdir(exist_ok=True)
    path = lock_root / f"{host}.gpu0.lock"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        append_jsonl(args.output_root / "campaign_checks.jsonl", {
            "timestamp": now(), "frame_id": frame_id, "host": host,
            "check_status": "gpu_lock_acquired", "owner_pid": os.getpid(),
        })
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def quarantine_path(args: argparse.Namespace, path: Path, label: str) -> Path:
    destination = args.output_root / "quarantine" / f"{label}.{int(time.time())}.{os.getpid()}"
    destination.parent.mkdir(exist_ok=True)
    if path.stat().st_dev == destination.parent.stat().st_dev:
        os.replace(path, destination)
    else:
        # Large reconstruction scratch lives on tmpfs.  Copying it to the
        # campaign's FUSE volume is both slow and can fail when copy2 attempts
        # unsupported ownership/timestamp operations.  Preserve it atomically
        # on its native filesystem and leave a durable pointer in the campaign.
        native_root = path.parent / ".quarantine"
        native_root.mkdir(exist_ok=True)
        native_destination = native_root / destination.name
        os.replace(path, native_destination)
        atomic_json(destination.with_suffix(".json"), {
            "schema_version": 1,
            "label": label,
            "quarantined_at": now(),
            "native_path": str(native_destination),
            "source_device": int(native_destination.stat().st_dev),
            "campaign_device": int(destination.parent.stat().st_dev),
        })
        destination = native_destination
    return destination


def claim(args: argparse.Namespace, request: dict[str, Any], frame_id: str, host: str) -> Path | None:
    lock_path = args.output_root / "claims/.claim.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        final = args.output_root / "frames" / frame_id
        if validate_finished(final, request["request_sha256"]):
            return None
        if final.exists():
            quarantine_path(args, final, f"invalid_final.{frame_id}")
        path = args.output_root / "claims" / frame_id
        if path.exists():
            receipt_path = path / "claim.json"
            receipt = load_json(receipt_path) if receipt_path.is_file() else {}
            owner_pid = int(receipt.get("pid", -1))
            same_machine = receipt.get("owner_hostname") == socket.gethostname()
            live = same_machine and owner_pid > 0 and pid_alive(owner_pid)
            claimed_at = datetime.fromisoformat(receipt.get("claimed_at", "1970-01-01T00:00:00+00:00"))
            age_hours = (datetime.now(timezone.utc) - claimed_at).total_seconds() / 3600.0
            immediate_failed_retry = args.retry_failed and (path / "failure.json").is_file()
            if live or (not immediate_failed_retry and age_hours < args.reclaim_stale_hours):
                return None
            quarantine_path(args, path, f"stale_claim.{frame_id}")
        path.mkdir()
        atomic_json(path / "claim.json", {
            "schema_version": 1, "request_sha256": request["request_sha256"],
            "frame_id": frame_id, "host": host, "pid": os.getpid(),
            "owner_hostname": socket.gethostname(), "claimed_at": now(),
        })
        return path


def process_one(
    args: argparse.Namespace, request: dict[str, Any], frame_id: str, index: int,
    host: str, remote_layout: tuple[Path, Path, Path] | None, attempt: int,
) -> None:
    work = args.output_root / ".work" / f"{frame_id}.{host}"
    work.mkdir(parents=True, exist_ok=True)
    staged = stage_frame(args, request, frame_id, work)
    with gpu_lock(args, host, frame_id):
        if index < 50:
            if host != "local":
                raise ValueError("Verified first-50 adoption is local-only")
            mesh, metadata, geometry = adopted_geometry(args, frame_id, work)
            render = local_render(args, frame_id, index, staged, mesh, metadata, work)
        elif host == "local":
            mesh, metadata, geometry = local_geometry(args, request, frame_id, staged, work, attempt)
            render = local_render(args, frame_id, index, staged, mesh, metadata, work)
        else:
            if remote_layout is None:
                raise RuntimeError("Remote layout was not prepared")
            mesh, metadata, geometry, render = remote_geometry_and_render(
                args, request, frame_id, index, staged, work, remote_layout, attempt
            )
        publish_frame(args, request, frame_id, index, host, mesh, metadata, geometry, render)


def cleanup_after_publish(
    args: argparse.Namespace, request: dict[str, Any], frame_id: str, index: int,
    host: str, remote_layout: tuple[Path, Path, Path] | None, attempt: int,
) -> None:
    try:
        if host == "dev3":
            if remote_layout is None:
                raise RuntimeError("Missing remote layout during cleanup")
            remote_frame = remote_layout[0] / "frames" / frame_id / f"attempt_{attempt}"
            ssh(args.remote_host, [
                args.remote_python, "-c",
                "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); p.is_dir() and shutil.rmtree(p)",
                remote_frame,
            ])
        elif index >= 50:
            local_scratch = Path("/dev/shm/lookcloser_patchmatch_tsdf_flythrough_150") / request["request_sha256"] / frame_id / f"attempt_{attempt}"
            if local_scratch.exists():
                shutil.rmtree(local_scratch)
        work = args.output_root / ".work" / f"{frame_id}.{host}"
        if work.exists():
            shutil.rmtree(work)
    except Exception as error:
        append_jsonl(args.output_root / "campaign_checks.jsonl", {
            "timestamp": now(), "frame_id": frame_id, "host": host,
            "attempt": attempt, "check_status": "post_publish_cleanup_warning", "error": repr(error),
        })


def process(args: argparse.Namespace) -> None:
    request = require_request(args)
    host = args.host
    remote_layout = prepare_remote(args, request) if host == "dev3" else None
    ordered = request["ordered_frame_ids"]
    stop = FRAME_COUNT if args.end_index is None else args.end_index
    failures = 0
    completed = 0
    for index in range(args.start_index, stop):
        frame_id = ordered[index]
        claimed = claim(args, request, frame_id, host)
        if claimed is None:
            continue
        success = False
        for attempt in range(2):
            try:
                process_one(args, request, frame_id, index, host, remote_layout, attempt)
                if not validate_finished(args.output_root / "frames" / frame_id, request["request_sha256"]):
                    raise RuntimeError("Atomic publication did not validate")
                atomic_json(claimed / "complete.json", {"completed_at": now(), "attempt": attempt})
                success = True
                completed += 1
                print(f"frame={frame_id} index={index:03d} host={host} status=complete", flush=True)
                cleanup_after_publish(args, request, frame_id, index, host, remote_layout, attempt)
                break
            except Exception as error:
                append_jsonl(args.output_root / "campaign_checks.jsonl", {
                    "timestamp": now(), "frame_id": frame_id, "host": host,
                    "attempt": attempt, "check_status": "attempt_failed", "error": repr(error),
                })
                if validate_finished(args.output_root / "frames" / frame_id, request["request_sha256"]):
                    try:
                        atomic_json(claimed / "complete.json", {
                            "completed_at": now(), "attempt": attempt,
                            "recovered_after_commit_error": repr(error),
                        })
                    except Exception:
                        pass
                    success = True
                    completed += 1
                    cleanup_after_publish(args, request, frame_id, index, host, remote_layout, attempt)
                    print(f"frame={frame_id} index={index:03d} host={host} status=complete_recovered", flush=True)
                    break
                work = args.output_root / ".work" / f"{frame_id}.{host}"
                if work.exists():
                    quarantine_path(args, work, f"{frame_id}.{host}.attempt{attempt}.work")
                if host == "local" and index >= 50:
                    local_scratch = Path("/dev/shm/lookcloser_patchmatch_tsdf_flythrough_150") / request["request_sha256"] / frame_id / f"attempt_{attempt}"
                    if local_scratch.exists():
                        quarantine_path(args, local_scratch, f"{frame_id}.local.attempt{attempt}.scratch")
                if attempt == 0:
                    print(f"frame={frame_id} host={host} status=retry_clean error={error}", flush=True)
        if not success:
            failures += 1
            atomic_json(claimed / "failure.json", {"failed_at": now(), "attempts": 2})
            print(f"frame={frame_id} index={index:03d} host={host} status=failed", flush=True)
    print(f"worker={host} claimed_complete={completed} failures={failures}", flush=True)


def rebuild_manifest(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    rows = []
    failed = []
    for index, frame_id in enumerate(request["ordered_frame_ids"]):
        path = args.output_root / "frames" / frame_id
        if validate_finished(path, request["request_sha256"]):
            rows.append(load_json(path / "result.json"))
        elif (args.output_root / "claims" / frame_id / "failure.json").is_file():
            failed.append(frame_id)
    manifest = {
        "schema_version": 1, "request_sha256": request["request_sha256"],
        "status": "complete" if len(rows) == FRAME_COUNT and not failed else "running",
        "completed": len(rows), "failed": len(failed), "failed_frame_ids": failed,
        "completed_frame_ids": [row["frame_id"] for row in rows], "updated_at": now(),
    }
    atomic_json(args.output_root / "campaign_manifest.json", manifest)
    return manifest


def make_contact_sheets(args: argparse.Namespace, request: dict[str, Any], batch_size: int = 25) -> list[str]:
    outputs = []
    for start in range(0, FRAME_COUNT, batch_size):
        ids = request["ordered_frame_ids"][start:start + batch_size]
        if not all((args.output_root / "frames" / frame_id / "render/frame.png").is_file() for frame_id in ids):
            continue
        sheet = Image.new("RGB", (1920, 1080), "black")
        draw = ImageDraw.Draw(sheet)
        thumb_w, thumb_h = 384, 216
        for offset, frame_id in enumerate(ids):
            image = Image.open(args.output_root / "frames" / frame_id / "render/frame.png").convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x, y = (offset % 5) * thumb_w, (offset // 5) * thumb_h
            sheet.paste(image, (x, y))
            draw.text((x + 5, y + 5), f"{start + offset:03d} {frame_id}", fill="white", stroke_width=2, stroke_fill="black")
        output = args.output_root / "contact_sheets" / f"frames_{start:03d}_{start + len(ids) - 1:03d}.png"
        sheet.save(output, compress_level=3)
        outputs.append(str(output))
    return outputs


def ffmpeg_hashes(path: Path, *, fps_input: bool = False) -> list[str]:
    command = ["ffmpeg", "-v", "error"]
    if fps_input:
        command += ["-framerate", str(FPS)]
    command += ["-i", str(path), "-frames:v", str(FRAME_COUNT), "-vf", "format=rgb24", "-f", "framemd5", "-"]
    output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in output.splitlines() if line and not line.startswith("#")]


def encode(args: argparse.Namespace) -> None:
    request = require_request(args)
    manifest = rebuild_manifest(args, request)
    if manifest["completed"] != FRAME_COUNT or manifest["failed"]:
        raise RuntimeError(f"Cannot encode incomplete campaign: {manifest}")
    sequence = args.output_root / "video_frames"
    sequence.mkdir(exist_ok=True)
    for index, frame_id in enumerate(request["ordered_frame_ids"]):
        source = args.output_root / "frames" / frame_id / "render/frame.png"
        destination = sequence / f"{index:05d}.png"
        if not destination.exists():
            os.link(source, destination)
        elif sha256(destination) != sha256(source):
            raise ValueError(f"Stale video frame link: {destination}")
    environment = dict(os.environ)
    preload = Path("/lib/x86_64-linux-gnu/libmpg123.so.0")
    if preload.is_file():
        environment["LD_PRELOAD"] = str(preload)
    lossless = args.output_root / "dec5_patchmatch_tsdf_flythrough_150_lossless_ffv1.mkv"
    h264 = args.output_root / "dec5_patchmatch_tsdf_flythrough_150_hq_h264.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-framerate", str(FPS),
        "-i", str(sequence / "%05d.png"), "-frames:v", str(FRAME_COUNT), "-an",
        "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1", "-g", "1",
        "-slicecrc", "1", "-pix_fmt", "bgr0", str(lossless),
    ], check=True, env=environment)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-framerate", str(FPS),
        "-i", str(sequence / "%05d.png"), "-frames:v", str(FRAME_COUNT), "-an",
        "-c:v", "libx264", "-preset", "veryslow", "-crf", "10", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace", "bt709", str(h264),
    ], check=True, env=environment)
    source_hashes = ffmpeg_hashes(sequence / "%05d.png", fps_input=True)
    decoded_hashes = ffmpeg_hashes(lossless)
    if len(source_hashes) != FRAME_COUNT or source_hashes != decoded_hashes:
        raise RuntimeError("FFV1 RGB round-trip validation failed")
    sheets = make_contact_sheets(args, request)
    atomic_json(args.output_root / "video_manifest.json", {
        "schema_version": 1, "request_sha256": request["request_sha256"], "frames": FRAME_COUNT,
        "fps": FPS, "duration_seconds": FRAME_COUNT / FPS,
        "lossless": str(lossless), "lossless_sha256": sha256(lossless), "lossless_rgb_roundtrip": True,
        "h264": str(h264), "h264_sha256": sha256(h264), "contact_sheets": sheets,
    })
    print(f"status=encoded frames={FRAME_COUNT} fps={FPS} h264={h264}")


def review(args: argparse.Namespace) -> None:
    request = require_request(args)
    indices = range(args.review_start, args.review_end)
    for index in indices:
        frame_id = request["ordered_frame_ids"][index]
        result_path = args.output_root / "frames" / frame_id / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = load_json(result_path)
        atomic_json(args.output_root / "frames" / frame_id / "visual_review.json", {
            "schema_version": 1, "frame_id": frame_id, "temporal_index": index,
            "render_sha256": result["render_sha256"],
            "visual_status": args.visual_status, "visual_notes": args.visual_notes,
            "reviewed_at": now(),
        })
    atomic_json(args.output_root / "contact_sheets" / f"review_{args.review_start:03d}_{args.review_end - 1:03d}.json", {
        "schema_version": 1, "indices": [args.review_start, args.review_end - 1],
        "visual_status": args.visual_status, "visual_notes": args.visual_notes, "reviewed_at": now(),
    })


def audit(args: argparse.Namespace) -> None:
    request = require_request(args)
    manifest = rebuild_manifest(args, request)
    if manifest["completed"] != FRAME_COUNT or manifest["failed"]:
        raise RuntimeError(f"Incomplete frame inventory: {manifest}")
    results = [load_json(args.output_root / "frames" / frame_id / "result.json") for frame_id in request["ordered_frame_ids"]]
    camera_path = load_json(args.output_root / "config/camera_path.json")["frames"]
    for index, row in enumerate(results):
        if (
            row.get("frame_id") != request["ordered_frame_ids"][index]
            or row.get("temporal_index") != index
            or row.get("camera") != camera_path[index]
        ):
            raise ValueError(f"Temporal ordering/camera mismatch at index {index}")
    pending = []
    visual_fail = []
    for row in results:
        review_path = args.output_root / "frames" / row["frame_id"] / "visual_review.json"
        if not review_path.is_file():
            pending.append(row["frame_id"])
            continue
        review_receipt = load_json(review_path)
        if review_receipt.get("render_sha256") != row["render_sha256"]:
            pending.append(row["frame_id"])
            continue
        verdict = review_receipt.get("visual_status")
        if verdict == "fail":
            visual_fail.append(row["frame_id"])
        elif verdict != "pass":
            pending.append(row["frame_id"])
    video = load_json(args.output_root / "video_manifest.json")
    if (
        video.get("request_sha256") != request["request_sha256"]
        or video.get("frames") != FRAME_COUNT
        or video.get("fps") != FPS
        or video.get("lossless_rgb_roundtrip") is not True
    ):
        raise ValueError("Video manifest does not match the campaign")
    for key in ("lossless", "h264"):
        if sha256(Path(video[key])) != video[f"{key}_sha256"]:
            raise ValueError(f"Video hash mismatch: {key}")
    status = "pass"
    if pending:
        status = "pending_visual_review"
    elif visual_fail:
        status = "complete_with_visual_failures"
    receipt = {
        "schema_version": 1, "status": status,
        "request_sha256": request["request_sha256"], "frames": FRAME_COUNT,
        "pending_visual_frame_ids": pending, "visual_fail_frame_ids": visual_fail,
        "catastrophic_failures": len(visual_fail),
        "min_nonblack_fraction": min(row["nonblack_fraction"] for row in results),
        "verified_at": now(),
    }
    atomic_json(args.output_root / "audit.json", receipt)
    print(json.dumps(receipt, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--existing-root", type=Path, default=EXISTING_ROOT)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--reconstruction-code", type=Path, default=RECONSTRUCTION_CODE)
    parser.add_argument("--local-python", type=Path, default=LOCAL_PYTHON)
    parser.add_argument("--local-colmap", type=Path, default=LOCAL_COLMAP)
    parser.add_argument("--remote-host", default="ubuntu@dev3")
    parser.add_argument("--remote-python", type=Path, default=REMOTE_PYTHON)
    parser.add_argument("--remote-colmap", type=Path, default=REMOTE_COLMAP)
    parser.add_argument("--remote-root", type=Path, default=REMOTE_ROOT)
    parser.add_argument("--local-scratch-min-free-gib", type=float, default=20.0)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    process_parser = sub.add_parser("process")
    process_parser.add_argument("--host", choices=("local", "dev3"), required=True)
    process_parser.add_argument("--start-index", type=int, default=0)
    process_parser.add_argument("--end-index", type=int, default=None)
    process_parser.add_argument("--reclaim-stale-hours", type=float, default=1.0)
    process_parser.add_argument("--retry-failed", action="store_true")
    sub.add_parser("status")
    sub.add_parser("contact-sheets")
    sub.add_parser("encode")
    review_parser = sub.add_parser("review")
    review_parser.add_argument("--review-start", type=int, required=True)
    review_parser.add_argument("--review-end", type=int, required=True)
    review_parser.add_argument("--visual-status", choices=("pass", "fail"), required=True)
    review_parser.add_argument("--visual-notes", required=True)
    sub.add_parser("audit")
    args = parser.parse_args(argv)
    for name in ("source_root", "output_root", "existing_root", "calibration", "reconstruction_code"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in ("local_python", "local_colmap", "remote_python", "remote_colmap", "remote_root"):
        path = getattr(args, name).expanduser()
        setattr(args, name, path if path.is_absolute() else path.absolute())
    if hasattr(args, "start_index"):
        stop = FRAME_COUNT if args.end_index is None else args.end_index
        if not 0 <= args.start_index < stop <= FRAME_COUNT:
            parser.error("Processing indices must satisfy 0 <= start < end <= 150")
        if args.host == "dev3" and args.start_index < 50:
            parser.error("dev3 must not claim the locally adopted first 50 frames")
        if args.reclaim_stale_hours < 0:
            parser.error("--reclaim-stale-hours must be nonnegative")
    if args.local_scratch_min_free_gib <= 0:
        parser.error("--local-scratch-min-free-gib must be positive")
    if hasattr(args, "review_start") and not 0 <= args.review_start < args.review_end <= FRAME_COUNT:
        parser.error("Review indices must satisfy 0 <= start < end <= 150")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "init":
        initialize(args)
    elif args.action == "process":
        process(args)
    elif args.action == "status":
        print(json.dumps(rebuild_manifest(args, require_request(args)), indent=2))
    elif args.action == "contact-sheets":
        request = require_request(args)
        print(json.dumps({"contact_sheets": make_contact_sheets(args, request)}, indent=2))
    elif args.action == "encode":
        encode(args)
    elif args.action == "review":
        review(args)
    elif args.action == "audit":
        audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
