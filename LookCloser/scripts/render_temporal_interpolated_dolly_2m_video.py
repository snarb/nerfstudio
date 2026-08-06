#!/usr/bin/env python3
"""Render the original interpolated train-camera path with a gradual 2 m dolly-in.

The base trajectory is the validated closed path through the medium train
cameras from ``render_temporal_snapshot_videos.py``. At the same time, the
camera moves radially toward a 3D point between the actors' faces, reaches two
meters at ``--peak-index``, and returns to the unmodified path endpoint. Dataset
files are only read; generated artifacts are written under ``--output-root``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import render_temporal_dolly_videos as dolly
import render_temporal_face_dolly_videos as face_dolly
import render_temporal_snapshot_videos as base
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


DEFAULT_OUTPUT_ROOT = Path(
    "/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_interpolated_dolly_2m"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default=None, help="Comma-separated indices, e.g. 0,5,11,22,44.")
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=65536)
    parser.add_argument("--closest-distance-meters", type=float, default=2.0)
    parser.add_argument(
        "--peak-index",
        type=int,
        default=11,
        help="Temporal index where the combined path reaches two meters.",
    )
    parser.add_argument(
        "--left-face-pixel",
        type=face_dolly.parse_pixel,
        default=(755.0, 395.0),
        help="Left actor face center x,y at --peak-index in the central-camera frame.",
    )
    parser.add_argument(
        "--right-face-pixel",
        type=face_dolly.parse_pixel,
        default=(895.0, 465.0),
        help="Right actor face center x,y at --peak-index in the central-camera frame.",
    )
    parser.add_argument("--depth-resolution-scale", type=float, default=0.25)
    parser.add_argument("--depth-patch-radius", type=int, default=2)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def dolly_ease(frame_count: int, peak_index: int) -> torch.Tensor:
    if not 0 < peak_index < frame_count - 1:
        raise ValueError(f"peak-index must be within [1, {frame_count - 2}]")
    ease = torch.empty(frame_count, dtype=torch.float32)
    approach_time = torch.linspace(0.0, 1.0, peak_index + 1, dtype=torch.float32)
    return_time = torch.linspace(0.0, 1.0, frame_count - peak_index, dtype=torch.float32)
    ease[: peak_index + 1] = torch.sin(0.5 * torch.pi * approach_time).square()
    ease[peak_index:] = torch.cos(0.5 * torch.pi * return_time).square()
    return ease


def build_combined_path(
    moving_path: Cameras,
    calibration: dict[str, Any],
    closest_distance_meters: float,
    peak_index: int,
) -> tuple[Cameras, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = dolly.clone_cameras(moving_path.to("cpu"))
    target = torch.tensor(calibration["target_world"], dtype=torch.float32)
    scene_units_per_meter = float(calibration["scene_units_per_meter"])
    closest_distance = closest_distance_meters * scene_units_per_meter
    if closest_distance <= 0.0:
        raise ValueError("closest-distance-meters must be positive")

    original_positions = path.camera_to_worlds[:, :, 3].clone()
    target_vectors = original_positions - target[None, :]
    original_distances = torch.linalg.norm(target_vectors, dim=-1)
    if torch.any(original_distances <= closest_distance):
        minimum = float(torch.min(original_distances) / scene_units_per_meter)
        raise ValueError(
            f"The original interpolated path already comes within {minimum:.3f} m of the target"
        )

    ease = dolly_ease(path.size, peak_index)
    combined_distances = original_distances - (original_distances - closest_distance) * ease
    directions_from_target = target_vectors / original_distances[:, None]
    path.camera_to_worlds[:, :, 3] = target[None, :] + directions_from_target * combined_distances[:, None]

    position_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, 3] - path.camera_to_worlds[-1, :, 3])
    ).item()
    rotation_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, :3] - path.camera_to_worlds[-1, :, :3])
    ).item()
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise RuntimeError(f"Combined camera path is not closed: position={position_error}, rotation={rotation_error}")
    measured_peak = float(combined_distances[peak_index] / scene_units_per_meter)
    if abs(measured_peak - closest_distance_meters) > 1e-5:
        raise RuntimeError(
            f"Combined path peak is {measured_peak:.6f} m, expected {closest_distance_meters:.6f} m"
        )
    return path, ease, original_distances, combined_distances


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if base.is_relative_to(output_root, data_root):
        raise RuntimeError(f"Output root must be outside the immutable dataset: {output_root}")
    if not (0.0 < args.resolution_scale <= 1.0):
        raise ValueError("resolution-scale must be within (0, 1]")
    if not (0.0 < args.depth_resolution_scale <= 1.0):
        raise ValueError("depth-resolution-scale must be within (0, 1]")
    if args.depth_patch_radius < 0:
        raise ValueError("depth-patch-radius must be non-negative")
    if not 0 < args.peak_index < base.EXPECTED_FRAMES - 1:
        raise ValueError(f"peak-index must be within [1, {base.EXPECTED_FRAMES - 2}]")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume after inspection: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free space is required")

    frames = base.discover_frames(data_root)
    indices = base.parse_indices(args.indices, len(frames))
    _, camera_to_file = base.load_camera_mapping(data_root)
    if not args.skip_encode and indices != list(range(base.EXPECTED_FRAMES)):
        raise RuntimeError("Encoding requires the complete 45-frame sequence")
    if not args.skip_encode and args.resolution_scale != 1.0:
        raise RuntimeError("Final video encoding requires resolution-scale=1.0")
    base.validate_ffmpeg()

    input_before = base.snapshot_tree(data_root)
    before_path = output_root / "input_tree_before.json"
    if before_path.exists():
        if json.loads(before_path.read_text(encoding="utf-8")) != input_before:
            raise RuntimeError("Dataset metadata differs from the existing pre-render manifest")
    else:
        base.atomic_write_json(before_path, input_before)

    target_width = round(1920 * args.resolution_scale)
    target_height = round(1080 * args.resolution_scale)
    target_width -= target_width % 2
    target_height -= target_height % 2
    output_dir = output_root / "interpolated_dolly_to_2m_png"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "data_root": str(data_root),
        "resolution": [target_width, target_height],
        "resolution_scale": args.resolution_scale,
        "selected_indices": indices,
        "base_path_mode": "interpolated_train_cameras_plus_metric_face_dolly",
        "path_anchors": list(base.PATH_ANCHORS),
        "path_intervals": list(base.PATH_INTERVALS),
        "closest_distance_meters": args.closest_distance_meters,
        "peak_index": args.peak_index,
        "face_pixels": [list(args.left_face_pixel), list(args.right_face_pixel)],
        "depth_resolution_scale": args.depth_resolution_scale,
        "depth_patch_radius": args.depth_patch_radius,
        "eval_num_rays_per_chunk": args.eval_num_rays_per_chunk,
    }
    manifest_path = output_root / "manifest.json"
    existing_by_index = (
        face_dolly.validate_resume_manifest(manifest_path, settings) if args.resume else {}
    )

    calibration_record = frames[args.peak_index]
    print(
        f"calibrating_target frame={calibration_record['frame']} index={args.peak_index:02d} "
        f"config={calibration_record['config']}",
        flush=True,
    )
    _, calibration_pipeline, calibration_checkpoint, calibration_step = eval_setup(
        calibration_record["config"],
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        test_mode="test",
    )
    calibration_pipeline.eval()
    calibration_dataset = calibration_pipeline.datamanager.train_dataset
    if calibration_dataset is None or len(calibration_dataset) != 66:
        raise RuntimeError(f"Expected 66 train images for calibration frame {calibration_record['frame']}")
    central_camera = base.camera_from_dataset(
        calibration_dataset, camera_to_file[base.CENTRAL_CAMERA]
    )
    calibration = face_dolly.calibrate_target(
        calibration_pipeline,
        central_camera,
        [args.left_face_pixel, args.right_face_pixel],
        args.depth_resolution_scale,
        args.depth_patch_radius,
    )
    moving_path = base.build_moving_path(calibration_dataset, camera_to_file)
    combined_path, ease, original_distances, combined_distances = build_combined_path(
        moving_path,
        calibration,
        args.closest_distance_meters,
        args.peak_index,
    )
    scene_units_per_meter = float(calibration["scene_units_per_meter"])
    print(
        f"calibrated original_peak_distance={float(original_distances[args.peak_index] / scene_units_per_meter):.3f}m "
        f"combined_peak_distance={float(combined_distances[args.peak_index] / scene_units_per_meter):.3f}m",
        flush=True,
    )
    del calibration_pipeline
    torch.cuda.empty_cache()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        **settings,
        "output_root": str(output_root),
        "dataset_policy": "read_only",
        "frame_count": len(frames),
        "fps": f"{base.FPS_NUMERATOR}/{base.FPS_DENOMINATOR}",
        "duration_seconds": base.EXPECTED_FRAMES * base.FPS_DENOMINATOR / base.FPS_NUMERATOR,
        "calibration_temporal_index": args.peak_index,
        "calibration_frame": calibration_record["frame"],
        "calibration_checkpoint": str(calibration_checkpoint),
        "calibration_step": int(calibration_step),
        "calibration": calibration,
        "dolly_profile": "piecewise sine-squared approach and cosine-squared return",
        "meter_scale_assumption": "Original reconstruction coordinates are metric; dataparser_scale converts meters to scene units.",
        "dolly_ease": ease.tolist(),
        "original_target_distances_meters": (
            original_distances / scene_units_per_meter
        ).tolist(),
        "combined_target_distances_meters": (
            combined_distances / scene_units_per_meter
        ).tolist(),
        "original_camera_to_worlds": [pose.tolist() for pose in moving_path.camera_to_worlds],
        "combined_camera_to_worlds": [pose.tolist() for pose in combined_path.camera_to_worlds],
        "frames": [],
        "videos": {},
    }

    frame_entries: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for temporal_index, record in enumerate(frames):
        if temporal_index not in indices:
            continue
        frame_started = time.perf_counter()
        output_path = output_dir / f"{temporal_index:05d}.png"
        valid = base.validate_png(output_path, target_width, target_height)
        if output_path.exists() and not valid:
            raise RuntimeError(
                f"Invalid existing PNG for index {temporal_index}; refusing to overwrite"
            )

        checkpoint_loaded = record["checkpoint"]
        loaded_step = record["selected_step"]
        pipeline = None
        if not valid:
            print(
                f"frame={record['frame']} index={temporal_index:02d}/{base.EXPECTED_FRAMES - 1:02d} "
                f"loading={record['config']}",
                flush=True,
            )
            _, pipeline, checkpoint_loaded, loaded_step = eval_setup(
                record["config"],
                eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
                test_mode="test",
            )
            pipeline.eval()
            train_dataset = pipeline.datamanager.train_dataset
            if train_dataset is None or len(train_dataset) != 66:
                raise RuntimeError(f"Expected 66 train images for {record['frame']}")
            image = base.render_rgb(
                pipeline,
                combined_path[temporal_index : temporal_index + 1],
                args.resolution_scale,
            )
            base.save_png(output_path, image)

        elapsed = time.perf_counter() - frame_started
        entry = {
            "temporal_index": temporal_index,
            "frame": record["frame"],
            "config": str(record["config"]),
            "checkpoint": str(checkpoint_loaded),
            "selected_step": int(loaded_step),
            "png": str(output_path),
            "dolly_ease": float(ease[temporal_index]),
            "original_target_distance_meters": float(
                original_distances[temporal_index] / scene_units_per_meter
            ),
            "combined_target_distance_meters": float(
                combined_distances[temporal_index] / scene_units_per_meter
            ),
            "elapsed_seconds": elapsed,
        }
        if temporal_index in existing_by_index:
            entry["previous_elapsed_seconds"] = existing_by_index[temporal_index].get(
                "elapsed_seconds"
            )
        frame_entries.append(entry)
        manifest["frames"] = sorted(frame_entries, key=lambda item: item["temporal_index"])
        manifest["render_elapsed_seconds"] = time.perf_counter() - run_started
        base.atomic_write_json(manifest_path, manifest)
        print(f"frame={record['frame']} done elapsed={elapsed:.2f}s", flush=True)
        if pipeline is not None:
            del pipeline
            torch.cuda.empty_cache()

    if not args.skip_encode:
        files = sorted(output_dir.glob("*.png"))
        if len(files) != base.EXPECTED_FRAMES:
            raise RuntimeError(
                f"Expected {base.EXPECTED_FRAMES} PNGs in {output_dir}, found {len(files)}"
            )
        manifest["videos"] = base.encode_video_sequence(
            output_dir,
            output_root / "moving_camera_interpolated_dolly_to_2m",
            base.EXPECTED_FRAMES,
        )

    input_after = base.snapshot_tree(data_root)
    base.atomic_write_json(output_root / "input_tree_after.json", input_after)
    if input_before != input_after:
        raise RuntimeError("Dataset metadata changed during rendering")
    manifest["dataset_unchanged"] = True
    manifest["total_elapsed_seconds"] = time.perf_counter() - run_started
    base.atomic_write_json(manifest_path, manifest)
    print(json.dumps({"output_root": str(output_root), "dataset_unchanged": True}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
