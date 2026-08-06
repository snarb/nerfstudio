#!/usr/bin/env python3
"""Render loopable temporal dolly-ins ending 2 m and 1 m from the actors.

The camera starts at the central training pose, moves toward a 3D target between
the two faces, reaches the requested distance at ``--peak-index``, and returns
to the exact training pose. Dataset files are only read; all generated artifacts
are written under ``--output-root``.
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
import render_temporal_snapshot_videos as base
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


DEFAULT_OUTPUT_ROOT = Path(
    "/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_face_dolly_1m_2m"
)
VARIANTS = ("2m", "1m")


def parse_pixel(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("pixel must be specified as x,y") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default=None, help="Comma-separated indices, e.g. 0,11,22,33,44.")
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=65536)
    parser.add_argument("--two-meter-distance", type=float, default=2.0)
    parser.add_argument("--one-meter-distance", type=float, default=1.0)
    parser.add_argument(
        "--peak-index",
        type=int,
        default=11,
        help="Temporal index where the camera reaches its closest point.",
    )
    parser.add_argument(
        "--left-face-pixel",
        type=parse_pixel,
        default=(755.0, 395.0),
        help="Left actor face center x,y at --peak-index in the central-camera frame.",
    )
    parser.add_argument(
        "--right-face-pixel",
        type=parse_pixel,
        default=(895.0, 465.0),
        help="Right actor face center x,y at --peak-index in the central-camera frame.",
    )
    parser.add_argument("--depth-resolution-scale", type=float, default=0.25)
    parser.add_argument("--depth-patch-radius", type=int, default=2)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sample_depth(depth: torch.Tensor, pixel: tuple[float, float], scale: float, radius: int) -> float:
    array = depth.detach().cpu().numpy().squeeze(-1)
    x = int(round(pixel[0] * scale))
    y = int(round(pixel[1] * scale))
    y0, y1 = max(0, y - radius), min(array.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(array.shape[1], x + radius + 1)
    patch = array[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch) & (patch > 0.0)]
    if finite.size == 0:
        raise RuntimeError(f"No finite positive depth around pixel {pixel}")
    return float(np.median(finite))


@torch.no_grad()
def calibrate_target(
    pipeline: Any,
    camera: Cameras,
    face_pixels: list[tuple[float, float]],
    depth_scale: float,
    patch_radius: int,
) -> dict[str, Any]:
    depth_camera = dolly.clone_cameras(camera.to("cpu"))
    depth_camera.rescale_output_resolution(depth_scale)
    outputs = pipeline.model.get_outputs_for_camera(depth_camera.to(pipeline.device))
    if "depth" not in outputs:
        raise RuntimeError(f"Model output has no depth channel: {sorted(outputs)}")
    depths = [sample_depth(outputs["depth"], pixel, depth_scale, patch_radius) for pixel in face_pixels]

    coords = torch.tensor([[pixel[1], pixel[0]] for pixel in face_pixels], dtype=torch.float32)
    rays = camera.to("cpu").generate_rays(camera_indices=0, coords=coords)
    origins = rays.origins.detach().cpu()
    directions = torch.nn.functional.normalize(rays.directions.detach().cpu(), dim=-1)
    points = origins + directions * torch.tensor(depths, dtype=torch.float32)[:, None]
    target = points.mean(dim=0)
    origin = origins[0]
    target_vector = target - origin
    target_distance = float(torch.linalg.norm(target_vector))
    target_direction = target_vector / target_distance

    dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
    scene_units_per_meter = float(dataparser_outputs.dataparser_scale)
    if not np.isfinite(scene_units_per_meter) or scene_units_per_meter <= 0.0:
        raise RuntimeError(f"Invalid dataparser scale: {scene_units_per_meter}")
    return {
        "face_pixels": [list(pixel) for pixel in face_pixels],
        "face_depths_scene_units": depths,
        "face_points_world": points.tolist(),
        "target_world": target.tolist(),
        "target_direction_world": target_direction.tolist(),
        "base_distance_scene_units": target_distance,
        "base_distance_meters": target_distance / scene_units_per_meter,
        "scene_units_per_meter": scene_units_per_meter,
        "depth_resolution_scale": depth_scale,
        "depth_patch_radius": patch_radius,
    }


def build_distance_path(
    base_camera: Cameras,
    frame_count: int,
    calibration: dict[str, Any],
    closest_distance_meters: float,
    peak_index: int,
) -> tuple[Cameras, torch.Tensor, torch.Tensor]:
    path = base.concatenate_cameras([base_camera for _ in range(frame_count)])
    base_distance = float(calibration["base_distance_scene_units"])
    closest_distance = closest_distance_meters * float(calibration["scene_units_per_meter"])
    if closest_distance <= 0.0 or closest_distance >= base_distance:
        raise ValueError(
            f"Closest distance must be within (0, {calibration['base_distance_meters']:.3f}) meters"
        )
    if not 0 < peak_index < frame_count - 1:
        raise ValueError(f"peak-index must be within [1, {frame_count - 2}]")
    ease = torch.empty(frame_count, dtype=torch.float32)
    approach_time = torch.linspace(0.0, 1.0, peak_index + 1, dtype=torch.float32)
    return_time = torch.linspace(0.0, 1.0, frame_count - peak_index, dtype=torch.float32)
    ease[: peak_index + 1] = torch.sin(0.5 * torch.pi * approach_time).square()
    ease[peak_index:] = torch.cos(0.5 * torch.pi * return_time).square()
    distances = base_distance - (base_distance - closest_distance) * ease
    offsets = base_distance - distances
    direction = torch.tensor(calibration["target_direction_world"], dtype=torch.float32)
    path.camera_to_worlds[:, :, 3] += offsets[:, None] * direction[None, :]

    position_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, 3] - path.camera_to_worlds[-1, :, 3])
    ).item()
    rotation_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, :3] - path.camera_to_worlds[-1, :, :3])
    ).item()
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise RuntimeError(f"Camera path is not closed: position={position_error}, rotation={rotation_error}")
    return path, offsets, distances


def validate_resume_manifest(manifest_path: Path, settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not manifest_path.is_file():
        return {}
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in settings.items():
        if existing.get(key) != expected:
            raise RuntimeError(
                f"Existing manifest setting mismatch for {key}: {existing.get(key)!r} != {expected!r}"
            )
    return {int(item["temporal_index"]): item for item in existing.get("frames", [])}


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
    if args.one_meter_distance >= args.two_meter_distance:
        raise ValueError("one-meter-distance must be smaller than two-meter-distance")
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
    output_dirs = {name: output_root / f"dolly_to_{name}_png" for name in VARIANTS}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    variant_distances = {
        "2m": args.two_meter_distance,
        "1m": args.one_meter_distance,
    }
    settings = {
        "data_root": str(data_root),
        "resolution": [target_width, target_height],
        "resolution_scale": args.resolution_scale,
        "selected_indices": indices,
        "base_path_mode": "central_face_target_distance_dolly",
        "variant_distances_meters": variant_distances,
        "peak_index": args.peak_index,
        "face_pixels": [list(args.left_face_pixel), list(args.right_face_pixel)],
        "depth_resolution_scale": args.depth_resolution_scale,
        "depth_patch_radius": args.depth_patch_radius,
        "eval_num_rays_per_chunk": args.eval_num_rays_per_chunk,
    }
    manifest_path = output_root / "manifest.json"
    existing_by_index = validate_resume_manifest(manifest_path, settings) if args.resume else {}

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
    calibration = calibrate_target(
        calibration_pipeline,
        central_camera,
        [args.left_face_pixel, args.right_face_pixel],
        args.depth_resolution_scale,
        args.depth_patch_radius,
    )
    paths: dict[str, Cameras] = {}
    offsets: dict[str, torch.Tensor] = {}
    distances: dict[str, torch.Tensor] = {}
    for name in VARIANTS:
        paths[name], offsets[name], distances[name] = build_distance_path(
            central_camera,
            base.EXPECTED_FRAMES,
            calibration,
            variant_distances[name],
            args.peak_index,
        )
    print(
        f"calibrated base_distance={calibration['base_distance_meters']:.3f}m "
        f"scene_units_per_meter={calibration['scene_units_per_meter']:.9f}",
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
        "base_camera": base.CENTRAL_CAMERA,
        "base_camera_file": camera_to_file[base.CENTRAL_CAMERA],
        "calibration_temporal_index": args.peak_index,
        "calibration_frame": calibration_record["frame"],
        "calibration_checkpoint": str(calibration_checkpoint),
        "calibration_step": int(calibration_step),
        "calibration": calibration,
        "camera_to_worlds": {
            name: [pose.tolist() for pose in paths[name].camera_to_worlds]
            for name in VARIANTS
        },
        "offsets_scene_units": {name: offsets[name].tolist() for name in VARIANTS},
        "target_distances_scene_units": {
            name: distances[name].tolist() for name in VARIANTS
        },
        "dolly_profile": "piecewise sine-squared approach and cosine-squared return, with zero velocity at the endpoints and peak",
        "meter_scale_assumption": "Original reconstruction coordinates are metric; dataparser_scale converts meters to scene units.",
        "frames": [],
        "videos": {},
    }

    frame_entries: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for temporal_index, record in enumerate(frames):
        if temporal_index not in indices:
            continue
        frame_started = time.perf_counter()
        output_paths = {
            name: output_dirs[name] / f"{temporal_index:05d}.png" for name in VARIANTS
        }
        valid = {
            name: base.validate_png(path, target_width, target_height)
            for name, path in output_paths.items()
        }
        for name, path in output_paths.items():
            if path.exists() and not valid[name]:
                raise RuntimeError(f"Invalid existing {name} PNG for index {temporal_index}; refusing to overwrite")

        pipeline = None
        checkpoint_loaded = record["checkpoint"]
        loaded_step = record["selected_step"]
        if not all(valid.values()):
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
            for name in VARIANTS:
                if not valid[name]:
                    image = base.render_rgb(
                        pipeline,
                        paths[name][temporal_index : temporal_index + 1],
                        args.resolution_scale,
                    )
                    base.save_png(output_paths[name], image)

        elapsed = time.perf_counter() - frame_started
        entry = {
            "temporal_index": temporal_index,
            "frame": record["frame"],
            "config": str(record["config"]),
            "checkpoint": str(checkpoint_loaded),
            "selected_step": int(loaded_step),
            "dolly_to_2m_png": str(output_paths["2m"]),
            "dolly_to_1m_png": str(output_paths["1m"]),
            "offset_2m_scene_units": float(offsets["2m"][temporal_index]),
            "offset_1m_scene_units": float(offsets["1m"][temporal_index]),
            "target_distance_2m_meters": float(
                distances["2m"][temporal_index]
                / manifest["calibration"]["scene_units_per_meter"]
            ),
            "target_distance_1m_meters": float(
                distances["1m"][temporal_index]
                / manifest["calibration"]["scene_units_per_meter"]
            ),
            "elapsed_seconds": elapsed,
        }
        if temporal_index in existing_by_index:
            entry["previous_elapsed_seconds"] = existing_by_index[temporal_index].get("elapsed_seconds")
        frame_entries.append(entry)
        manifest["frames"] = sorted(frame_entries, key=lambda item: item["temporal_index"])
        manifest["render_elapsed_seconds"] = time.perf_counter() - run_started
        base.atomic_write_json(manifest_path, manifest)
        print(f"frame={record['frame']} done elapsed={elapsed:.2f}s", flush=True)
        if pipeline is not None:
            del pipeline
            torch.cuda.empty_cache()

    if not args.skip_encode:
        for name, directory in output_dirs.items():
            files = sorted(directory.glob("*.png"))
            if len(files) != base.EXPECTED_FRAMES:
                raise RuntimeError(f"Expected {base.EXPECTED_FRAMES} PNGs in {directory}, found {len(files)}")
        manifest["videos"] = {
            name: base.encode_video_sequence(
                output_dirs[name],
                output_root / f"moving_camera_dolly_to_{name}",
                base.EXPECTED_FRAMES,
            )
            for name in VARIANTS
        }

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
