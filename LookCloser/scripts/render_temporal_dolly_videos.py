#!/usr/bin/env python3
"""Render two temporally synchronized moving-camera videos with smooth dolly-in motion.

This utility reuses the validated temporal snapshot renderer while keeping the
dataset immutable. It renders a moderate and a closer physical camera move,
then returns to the initial pose so the camera trajectory remains loopable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

import render_temporal_snapshot_videos as base
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


DEFAULT_OUTPUT_ROOT = Path(
    "/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_dolly"
)
VARIANTS = ("moderate", "close")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default=None, help="Comma-separated temporal indices, e.g. 0,11,22,33,44.")
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=65536)
    parser.add_argument("--moderate-start", type=float, default=0.08)
    parser.add_argument("--moderate-peak", type=float, default=0.28)
    parser.add_argument("--close-start", type=float, default=0.14)
    parser.add_argument("--close-peak", type=float, default=0.42)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def clone_cameras(cameras: Cameras) -> Cameras:
    distortion = cameras.distortion_params.clone() if cameras.distortion_params is not None else None
    return Cameras(
        fx=cameras.fx.clone(),
        fy=cameras.fy.clone(),
        cx=cameras.cx.clone(),
        cy=cameras.cy.clone(),
        width=cameras.width.clone(),
        height=cameras.height.clone(),
        camera_to_worlds=cameras.camera_to_worlds.clone(),
        camera_type=cameras.camera_type.clone(),
        distortion_params=distortion,
    )


def dolly_offsets(frame_count: int, start: float, peak: float) -> torch.Tensor:
    if start < 0.0 or peak <= start:
        raise ValueError(f"Expected 0 <= start < peak, got start={start}, peak={peak}")
    normalized_time = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
    ease = torch.sin(torch.pi * normalized_time).square()
    return start + (peak - start) * ease


def build_dolly_path(base_path: Cameras, start: float, peak: float) -> tuple[Cameras, torch.Tensor]:
    path = clone_cameras(base_path)
    offsets = dolly_offsets(path.size, start, peak)
    forward = -path.camera_to_worlds[:, :, 2]
    path.camera_to_worlds[:, :, 3] += forward * offsets[:, None]
    position_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, 3] - path.camera_to_worlds[-1, :, 3])
    ).item()
    rotation_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, :3] - path.camera_to_worlds[-1, :, :3])
    ).item()
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise RuntimeError(f"Dolly path is not closed: position={position_error}, rotation={rotation_error}")
    return path, offsets


def build_central_base_path(dataset: Any, camera_to_file: dict[str, str]) -> Cameras:
    central = base.camera_from_dataset(dataset, camera_to_file[base.CENTRAL_CAMERA])
    return base.concatenate_cameras([central for _ in range(base.EXPECTED_FRAMES)])


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
    if args.resolution_scale <= 0.0 or args.resolution_scale > 1.0:
        raise ValueError("resolution-scale must be within (0, 1]")
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume after inspection: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free space is required")

    variant_settings = {
        "moderate": {"start": args.moderate_start, "peak": args.moderate_peak},
        "close": {"start": args.close_start, "peak": args.close_peak},
    }
    if args.close_start <= args.moderate_start or args.close_peak <= args.moderate_peak:
        raise ValueError("The close variant must be closer than the moderate variant at start and peak")

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
    output_dirs = {name: output_root / f"dolly_{name}_png" for name in VARIANTS}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    settings = {
        "data_root": str(data_root),
        "resolution": [target_width, target_height],
        "resolution_scale": args.resolution_scale,
        "selected_indices": indices,
        "base_path_mode": "central_dolly",
        "variant_settings": variant_settings,
        "eval_num_rays_per_chunk": args.eval_num_rays_per_chunk,
    }
    manifest_path = output_root / "manifest.json"
    existing_by_index = validate_resume_manifest(manifest_path, settings) if args.resume else {}
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
        "dolly_profile": "start + (peak - start) * sin(pi * normalized_time)^2",
        "frames": [],
        "videos": {},
    }

    paths: dict[str, Cameras] | None = None
    offsets: dict[str, torch.Tensor] | None = None
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
        if paths is None or not all(valid.values()):
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
            if paths is None:
                base_path = build_central_base_path(train_dataset, camera_to_file)
                paths = {}
                offsets = {}
                for name in VARIANTS:
                    paths[name], offsets[name] = build_dolly_path(
                        base_path,
                        variant_settings[name]["start"],
                        variant_settings[name]["peak"],
                    )
                manifest["camera_to_worlds"] = {
                    name: [pose.tolist() for pose in paths[name].camera_to_worlds]
                    for name in VARIANTS
                }
                manifest["offsets"] = {
                    name: offsets[name].tolist() for name in VARIANTS
                }
            assert paths is not None
            for name in VARIANTS:
                if not valid[name]:
                    image = base.render_rgb(
                        pipeline,
                        paths[name][temporal_index : temporal_index + 1],
                        args.resolution_scale,
                    )
                    base.save_png(output_paths[name], image)

        assert offsets is not None
        elapsed = time.perf_counter() - frame_started
        entry = {
            "temporal_index": temporal_index,
            "frame": record["frame"],
            "config": str(record["config"]),
            "checkpoint": str(checkpoint_loaded),
            "selected_step": int(loaded_step),
            "dolly_moderate_png": str(output_paths["moderate"]),
            "dolly_close_png": str(output_paths["close"]),
            "moderate_offset": float(offsets["moderate"][temporal_index]),
            "close_offset": float(offsets["close"][temporal_index]),
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
                output_root / f"moving_camera_dolly_{name}",
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
