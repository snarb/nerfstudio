#!/usr/bin/env python3
"""Render chronological comparison and moving-camera videos from per-frame snapshots.

The temporal dataset is treated as immutable input. All frames, logs, manifests,
and videos are written below an explicitly separate output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nerfstudio.cameras.camera_utils import get_interpolated_poses
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


DEFAULT_DATA_ROOT = Path("/home/brans/temporal_perframe_stride7_45f")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/brans/lookcloser_temporal_runs/videos/snapshot_per_frame_45f_final"
)
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
EXPECTED_FRAMES = 45
CENTRAL_CAMERA = "H004_C016"
PATH_ANCHORS = (
    "H004_B014",
    "H004_D014",
    "J004_E014",
    "L004_E014",
    "L004_B014",
    "J004_A014",
    "H004_B014",
)
PATH_INTERVALS = (11, 5, 6, 11, 5, 6)
FPS_NUMERATOR = 60
FPS_DENOMINATOR = 7
FFMPEG_MPG123 = Path("/lib/x86_64-linux-gnu/libmpg123.so.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--indices", default=None, help="Comma-separated temporal indices, e.g. 0,22,44.")
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=65536)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> dict[str, dict[str, int]]:
    """Capture path/size/mtime metadata without opening dataset files for writing."""
    snapshot: dict[str, dict[str, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return snapshot


def parse_indices(value: str | None, frame_count: int) -> list[int]:
    if value is None:
        return list(range(frame_count))
    indices = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not indices or indices[0] < 0 or indices[-1] >= frame_count:
        raise ValueError(f"Indices must be within [0, {frame_count - 1}]")
    return indices


def discover_frames(data_root: Path) -> list[dict[str, Any]]:
    frame_dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit())
    if len(frame_dirs) != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} numeric frame directories, found {len(frame_dirs)}")

    transform_hashes: set[str] = set()
    records: list[dict[str, Any]] = []
    for frame_dir in frame_dirs:
        config = frame_dir / "snapshot" / "config.yml"
        transforms = frame_dir / "transforms.json"
        selection = frame_dir / "snapshot" / "selection.json"
        checkpoints = sorted(
            (frame_dir / "snapshot" / "lookcloser" / "final" / "nerfstudio_models").glob("step-*.ckpt")
        )
        if not config.is_file() or not transforms.is_file() or not selection.is_file():
            raise FileNotFoundError(f"Incomplete snapshot inputs for {frame_dir.name}")
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one selected checkpoint for {frame_dir.name}, found {len(checkpoints)}")
        selection_data = json.loads(selection.read_text(encoding="utf-8"))
        selected_step = int(selection_data["selected_step"])
        checkpoint_step = int(checkpoints[0].stem.split("-")[-1])
        if selected_step != checkpoint_step:
            raise RuntimeError(
                f"Selection/checkpoint step mismatch for {frame_dir.name}: {selected_step} != {checkpoint_step}"
            )
        transform_hash = sha256_file(transforms)
        transform_hashes.add(transform_hash)
        records.append(
            {
                "frame": frame_dir.name,
                "data_dir": frame_dir,
                "config": config,
                "checkpoint": checkpoints[0],
                "selected_step": selected_step,
                "transforms_sha256": transform_hash,
            }
        )
    if len(transform_hashes) != 1:
        raise RuntimeError(f"Camera geometry differs across temporal frames: {sorted(transform_hashes)}")
    return records


def load_camera_mapping(data_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest_path = data_root / "perframe_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    mapping = json.loads(manifest_path.read_text(encoding="utf-8"))["camera_file_mapping"]
    file_to_camera = {str(filename): str(camera) for filename, camera in mapping.items()}
    camera_to_file = {camera: filename for filename, camera in file_to_camera.items()}
    required = {CENTRAL_CAMERA, *PATH_ANCHORS}
    missing = sorted(required - camera_to_file.keys())
    if missing:
        raise RuntimeError(f"Camera names missing from perframe manifest: {missing}")
    return file_to_camera, camera_to_file


def camera_from_dataset(dataset: Any, filename: str) -> Cameras:
    filename_to_index = {Path(path).name: index for index, path in enumerate(dataset.image_filenames)}
    if filename not in filename_to_index:
        raise RuntimeError(f"Training camera {filename} not found in loaded dataset")
    return dataset.cameras[filename_to_index[filename] : filename_to_index[filename] + 1]


def scalar(camera: Cameras, field: str) -> torch.Tensor:
    value = getattr(camera, field)
    return value.reshape(-1)[0].detach().cpu()


def interpolate_field(a: torch.Tensor, b: torch.Tensor, count: int) -> torch.Tensor:
    weights = torch.linspace(0.0, 1.0, count, dtype=torch.float32)
    return (1.0 - weights) * a.float() + weights * b.float()


def interpolate_camera_segment(camera_a: Cameras, camera_b: Cameras, intervals: int) -> Cameras:
    count = intervals + 1
    poses = get_interpolated_poses(
        camera_a.camera_to_worlds[0].detach().cpu().numpy(),
        camera_b.camera_to_worlds[0].detach().cpu().numpy(),
        steps=count,
    )
    distortion = None
    if camera_a.distortion_params is not None and camera_b.distortion_params is not None:
        weights = torch.linspace(0.0, 1.0, count, dtype=torch.float32)[:, None]
        distortion = (
            (1.0 - weights) * camera_a.distortion_params[0].detach().cpu().float()
            + weights * camera_b.distortion_params[0].detach().cpu().float()
        )
    camera_type_value = int(camera_a.camera_type.reshape(-1)[0].item())
    return Cameras(
        fx=interpolate_field(scalar(camera_a, "fx"), scalar(camera_b, "fx"), count),
        fy=interpolate_field(scalar(camera_a, "fy"), scalar(camera_b, "fy"), count),
        cx=interpolate_field(scalar(camera_a, "cx"), scalar(camera_b, "cx"), count),
        cy=interpolate_field(scalar(camera_a, "cy"), scalar(camera_b, "cy"), count),
        width=interpolate_field(scalar(camera_a, "width"), scalar(camera_b, "width"), count).round().long(),
        height=interpolate_field(scalar(camera_a, "height"), scalar(camera_b, "height"), count).round().long(),
        camera_to_worlds=torch.tensor(np.stack(poses), dtype=torch.float32),
        camera_type=torch.full((count,), camera_type_value, dtype=torch.long),
        distortion_params=distortion,
    )


def concatenate_cameras(cameras: Iterable[Cameras]) -> Cameras:
    camera_list = list(cameras)
    distortion = None
    if camera_list[0].distortion_params is not None:
        distortion = torch.cat([camera.distortion_params for camera in camera_list], dim=0)
    return Cameras(
        fx=torch.cat([camera.fx.reshape(-1) for camera in camera_list]),
        fy=torch.cat([camera.fy.reshape(-1) for camera in camera_list]),
        cx=torch.cat([camera.cx.reshape(-1) for camera in camera_list]),
        cy=torch.cat([camera.cy.reshape(-1) for camera in camera_list]),
        width=torch.cat([camera.width.reshape(-1) for camera in camera_list]),
        height=torch.cat([camera.height.reshape(-1) for camera in camera_list]),
        camera_to_worlds=torch.cat([camera.camera_to_worlds for camera in camera_list], dim=0),
        camera_type=torch.cat([camera.camera_type.reshape(-1) for camera in camera_list]),
        distortion_params=distortion,
    )


def build_moving_path(dataset: Any, camera_to_file: dict[str, str]) -> Cameras:
    anchors = [camera_from_dataset(dataset, camera_to_file[name]) for name in PATH_ANCHORS]
    segments: list[Cameras] = []
    for segment_index, intervals in enumerate(PATH_INTERVALS):
        segment = interpolate_camera_segment(anchors[segment_index], anchors[segment_index + 1], intervals)
        if segment_index > 0:
            segment = segment[1:]
        segments.append(segment)
    path = concatenate_cameras(segments)
    if path.size != EXPECTED_FRAMES:
        raise RuntimeError(f"Expected {EXPECTED_FRAMES} moving-camera poses, got {path.size}")
    position_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, 3] - path.camera_to_worlds[-1, :, 3])
    ).item()
    rotation_error = torch.max(
        torch.abs(path.camera_to_worlds[0, :, :3] - path.camera_to_worlds[-1, :, :3])
    ).item()
    if position_error > 1e-6 or rotation_error > 1e-6:
        raise RuntimeError(f"Camera path is not closed: position={position_error}, rotation={rotation_error}")
    return path


def render_rgb(pipeline: Any, camera: Cameras, resolution_scale: float) -> np.ndarray:
    render_camera = camera.to("cpu")
    if resolution_scale != 1.0:
        render_camera.rescale_output_resolution(resolution_scale)
    with torch.no_grad():
        rgb = pipeline.model.get_outputs_for_camera(render_camera.to(pipeline.device))["rgb"]
    return np.clip(rgb.detach().cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="PNG", compress_level=3)


def validate_png(path: Path, width: int, height: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGB" and image.size == (width, height)
    except Exception:
        return False


def make_comparison(
    ground_truth_path: Path,
    rendered_path: Path,
    output_path: Path,
    width: int,
    height: int,
    font_path: Path,
) -> None:
    with Image.open(ground_truth_path) as source_gt, Image.open(rendered_path) as source_render:
        ground_truth = source_gt.convert("RGB")
        rendered = source_render.convert("RGB")
        if ground_truth.size != (width, height):
            ground_truth = ground_truth.resize((width, height), Image.Resampling.LANCZOS)
        if rendered.size != (width, height):
            rendered = rendered.resize((width, height), Image.Resampling.LANCZOS)
        split = width // 2
        comparison = Image.new("RGB", (width, height))
        comparison.paste(ground_truth.crop((0, 0, split, height)), (0, 0))
        comparison.paste(rendered.crop((split, 0, width, height)), (split, 0))

    font_size = max(18, round(48 * height / 1080))
    stroke_width = max(2, round(3 * height / 1080))
    font = ImageFont.truetype(str(font_path), size=font_size)
    draw = ImageDraw.Draw(comparison)
    y = max(font_size, round(52 * height / 1080))
    draw.text(
        (width * 0.25, y),
        "GT",
        font=font,
        fill="white",
        stroke_width=stroke_width,
        stroke_fill="black",
        anchor="mm",
    )
    draw.text(
        (width * 0.75, y),
        "RENDERED",
        font=font,
        fill="white",
        stroke_width=stroke_width,
        stroke_fill="black",
        anchor="mm",
    )
    draw.line((width // 2, 0, width // 2, height), fill="white", width=max(1, round(2 * width / 1920)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, format="PNG", compress_level=3)


def ffmpeg_environment() -> dict[str, str]:
    environment = dict(os.environ)
    if FFMPEG_MPG123.is_file():
        previous = environment.get("LD_PRELOAD")
        environment["LD_PRELOAD"] = str(FFMPEG_MPG123) if not previous else f"{FFMPEG_MPG123}:{previous}"
    return environment


def run_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=ffmpeg_environment(),
    )


def validate_ffmpeg() -> None:
    encoders = run_command(["ffmpeg", "-hide_banner", "-encoders"], capture=True).stdout
    for encoder in ("ffv1", "libx264"):
        if encoder not in encoders:
            raise RuntimeError(f"Required FFmpeg encoder is unavailable: {encoder}")


def video_probe(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(result.stdout)


def frame_hashes(command: list[str]) -> list[str]:
    output = run_command(command, capture=True).stdout
    hashes: list[str] = []
    for line in output.splitlines():
        if line and not line.startswith("#"):
            hashes.append(line.rsplit(",", 1)[-1].strip())
    return hashes


def encode_video_sequence(images_dir: Path, output_base: Path, frame_count: int) -> dict[str, Any]:
    fps = f"{FPS_NUMERATOR}/{FPS_DENOMINATOR}"
    lossless = output_base.with_name(f"{output_base.name}_lossless_ffv1.mkv")
    compatible = output_base.with_name(f"{output_base.name}_hq_h264.mp4")
    if not lossless.exists():
        run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-n",
                "-framerate",
                fps,
                "-i",
                str(images_dir / "%05d.png"),
                "-frames:v",
                str(frame_count),
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-g",
                "1",
                "-slicecrc",
                "1",
                "-pix_fmt",
                "bgr0",
                str(lossless),
            ]
        )
    if not compatible.exists():
        run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-n",
                "-framerate",
                fps,
                "-i",
                str(images_dir / "%05d.png"),
                "-frames:v",
                str(frame_count),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryslow",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                str(compatible),
            ]
        )

    source_hashes = frame_hashes(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            fps,
            "-i",
            str(images_dir / "%05d.png"),
            "-frames:v",
            str(frame_count),
            "-vf",
            "format=rgb24",
            "-f",
            "framemd5",
            "-",
        ]
    )
    lossless_hashes = frame_hashes(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(lossless),
            "-frames:v",
            str(frame_count),
            "-vf",
            "format=rgb24",
            "-f",
            "framemd5",
            "-",
        ]
    )
    if source_hashes != lossless_hashes or len(source_hashes) != frame_count:
        raise RuntimeError(f"FFV1 RGB round-trip validation failed for {lossless}")
    return {
        "lossless": str(lossless),
        "lossless_probe": video_probe(lossless),
        "lossless_rgb_roundtrip": True,
        "compatible": str(compatible),
        "compatible_probe": video_probe(compatible),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if is_relative_to(output_root, data_root):
        raise RuntimeError(f"Output root must be outside the immutable dataset: {output_root}")
    if args.resolution_scale <= 0.0 or args.resolution_scale > 1.0:
        raise ValueError("resolution-scale must be within (0, 1]")
    if not DEFAULT_FONT.is_file():
        raise FileNotFoundError(DEFAULT_FONT)
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; use --resume after inspection: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    if free_bytes < 4 * 1024**3:
        raise RuntimeError(f"At least 4 GiB free space is required, found {free_bytes / 1024**3:.2f} GiB")

    frames = discover_frames(data_root)
    indices = parse_indices(args.indices, len(frames))
    _, camera_to_file = load_camera_mapping(data_root)
    if not args.skip_encode and indices != list(range(EXPECTED_FRAMES)):
        raise RuntimeError("Encoding requires the complete 45-frame sequence")
    if not args.skip_encode and args.resolution_scale != 1.0:
        raise RuntimeError("Final video encoding requires resolution-scale=1.0")
    validate_ffmpeg()

    input_before = snapshot_tree(data_root)
    before_path = output_root / "input_tree_before.json"
    if before_path.exists():
        recorded_before = json.loads(before_path.read_text(encoding="utf-8"))
        if recorded_before != input_before:
            raise RuntimeError("Dataset metadata differs from the existing pre-render manifest")
    else:
        atomic_write_json(before_path, input_before)

    target_width = round(1920 * args.resolution_scale)
    target_height = round(1080 * args.resolution_scale)
    if target_width % 2:
        target_width -= 1
    if target_height % 2:
        target_height -= 1
    raw_central_dir = output_root / "raw_central_png"
    comparison_dir = output_root / "comparison_png"
    moving_dir = output_root / "moving_png"
    for directory in (raw_central_dir, comparison_dir, moving_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "dataset_policy": "read_only",
        "frame_count": len(frames),
        "selected_indices": indices,
        "resolution": [target_width, target_height],
        "resolution_scale": args.resolution_scale,
        "fps": f"{FPS_NUMERATOR}/{FPS_DENOMINATOR}",
        "duration_seconds": EXPECTED_FRAMES * FPS_DENOMINATOR / FPS_NUMERATOR,
        "central_camera": CENTRAL_CAMERA,
        "central_camera_file": camera_to_file[CENTRAL_CAMERA],
        "path_anchors": list(PATH_ANCHORS),
        "path_intervals": list(PATH_INTERVALS),
        "eval_num_rays_per_chunk": args.eval_num_rays_per_chunk,
        "frames": [],
        "videos": {},
    }
    existing_by_index: dict[int, dict[str, Any]] = {}
    if args.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_by_index = {int(item["temporal_index"]): item for item in existing.get("frames", [])}

    moving_path: Cameras | None = None
    run_started = time.perf_counter()
    frame_entries: list[dict[str, Any]] = []
    for temporal_index, record in enumerate(frames):
        if temporal_index not in indices:
            continue
        frame_started = time.perf_counter()
        raw_central_path = raw_central_dir / f"{temporal_index:05d}.png"
        comparison_path = comparison_dir / f"{temporal_index:05d}.png"
        moving_path_file = moving_dir / f"{temporal_index:05d}.png"
        raw_ok = validate_png(raw_central_path, target_width, target_height)
        comparison_ok = validate_png(comparison_path, target_width, target_height)
        moving_ok = validate_png(moving_path_file, target_width, target_height)
        if (raw_central_path.exists() and not raw_ok) or (comparison_path.exists() and not comparison_ok):
            raise RuntimeError(f"Invalid existing PNG for temporal index {temporal_index}; refusing to overwrite")
        if moving_path_file.exists() and not moving_ok:
            raise RuntimeError(f"Invalid existing moving PNG for temporal index {temporal_index}; refusing to overwrite")

        pipeline = None
        checkpoint_loaded = record["checkpoint"]
        loaded_step = record["selected_step"]
        if not raw_ok or not moving_ok or moving_path is None:
            print(
                f"frame={record['frame']} index={temporal_index:02d}/{EXPECTED_FRAMES - 1:02d} loading={record['config']}",
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
            if moving_path is None:
                moving_path = build_moving_path(train_dataset, camera_to_file)
                pose_manifest = [pose.tolist() for pose in moving_path.camera_to_worlds]
                manifest["moving_camera_to_worlds"] = pose_manifest
            if not raw_ok:
                central_camera = camera_from_dataset(train_dataset, camera_to_file[CENTRAL_CAMERA])
                save_png(raw_central_path, render_rgb(pipeline, central_camera, args.resolution_scale))
                raw_ok = True
            if not moving_ok:
                assert moving_path is not None
                save_png(
                    moving_path_file,
                    render_rgb(pipeline, moving_path[temporal_index : temporal_index + 1], args.resolution_scale),
                )
                moving_ok = True

        if not comparison_ok:
            ground_truth = record["data_dir"] / "images" / camera_to_file[CENTRAL_CAMERA]
            make_comparison(
                ground_truth,
                raw_central_path,
                comparison_path,
                target_width,
                target_height,
                DEFAULT_FONT,
            )

        elapsed = time.perf_counter() - frame_started
        entry = {
            "temporal_index": temporal_index,
            "frame": record["frame"],
            "config": str(record["config"]),
            "checkpoint": str(checkpoint_loaded),
            "selected_step": int(loaded_step),
            "ground_truth": str(record["data_dir"] / "images" / camera_to_file[CENTRAL_CAMERA]),
            "raw_central_png": str(raw_central_path),
            "comparison_png": str(comparison_path),
            "moving_png": str(moving_path_file),
            "elapsed_seconds": elapsed,
        }
        if temporal_index in existing_by_index:
            entry["previous_elapsed_seconds"] = existing_by_index[temporal_index].get("elapsed_seconds")
        frame_entries.append(entry)
        manifest["frames"] = sorted(frame_entries, key=lambda item: item["temporal_index"])
        manifest["render_elapsed_seconds"] = time.perf_counter() - run_started
        atomic_write_json(manifest_path, manifest)
        print(f"frame={record['frame']} done elapsed={elapsed:.2f}s", flush=True)

        if pipeline is not None:
            del pipeline
            torch.cuda.empty_cache()

    if not args.skip_encode:
        for directory in (comparison_dir, moving_dir):
            files = sorted(directory.glob("*.png"))
            if len(files) != EXPECTED_FRAMES:
                raise RuntimeError(f"Expected {EXPECTED_FRAMES} PNGs in {directory}, found {len(files)}")
        manifest["videos"] = {
            "comparison": encode_video_sequence(comparison_dir, output_root / "gt_vs_rendered", EXPECTED_FRAMES),
            "moving": encode_video_sequence(moving_dir, output_root / "moving_camera_temporal", EXPECTED_FRAMES),
        }

    input_after = snapshot_tree(data_root)
    atomic_write_json(output_root / "input_tree_after.json", input_after)
    if input_before != input_after:
        raise RuntimeError("Dataset metadata changed during rendering")
    manifest["dataset_unchanged"] = True
    manifest["total_elapsed_seconds"] = time.perf_counter() - run_started
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"output_root": str(output_root), "dataset_unchanged": True}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
